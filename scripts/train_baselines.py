"""Phase 3: baselines, and the decisive test for commitment V2.

Every model here is topology-blind. Their job is to establish what can be
achieved *without* a graph, so that Phase 4's GNN has something honest to beat.

The comparison is three-way, not two-way:

    flow          the 31 flow features + one-hot categoricals
    flow+host     the same, plus 9 hand-crafted per-host aggregates computed
                  within each window -- exactly what one round of message
                  passing would aggregate, written out by hand

    (GNN, Phase 4)

A two-way comparison would only show that *some* context helps. The middle arm
is what makes the result specific: if the GNN cannot beat hand-crafted
aggregates, it is an expensive `groupby` and the report must say so.

All models see identical rows, identical splits, identical metrics. Thresholds
are chosen on validation, never test (D16). Headline numbers are mean +/- std
over several seeds.

Usage:
    python scripts/train_baselines.py
    python scripts/train_baselines.py --models mlp --seeds 1
"""

from __future__ import annotations

import os

# Must precede the torch and xgboost imports. Both ship their own OpenMP
# runtime, and loading two into one process segfaults on macOS (exit 139).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnids.data.host_features import FEATURE_NAMES, standardise, windowed_host_features  # noqa: E402
from gnnids.eval.metrics import aggregate_seeds, choose_threshold  # noqa: E402
from gnnids.eval.prevalence import report_both, subsample_to_prevalence  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "metrics" / "baselines"

# Bumped when the meaning of the numbers changes, so a merged file can never
# present two incompatible reporting regimes as one table (C17).
SCHEMA = "dual-prevalence-v1"
STALE_SCHEMA = "pre-c17-native-only"


def one_hot(cat: np.ndarray, cardinalities: list[int]) -> np.ndarray:
    """Expand categorical indices so every model sees the same representation.

    Tree models could take the integers directly and the MLP could embed them,
    but then the two would be seeing different inputs and any difference between
    them would be partly an encoding artefact. Total width is only ~50 columns.
    """
    blocks = []
    for j, card in enumerate(cardinalities):
        block = np.zeros((len(cat), card), dtype=np.float32)
        block[np.arange(len(cat)), np.clip(cat[:, j], 0, card - 1)] = 1.0
        blocks.append(block)
    return np.hstack(blocks)


def train_torch_mlp(
    Xtr, ytr, ymtr, Xva, yva, cfg, seed, device, n_blocks: int | None = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train FlowMLP and return (score_fn, info).

    `n_blocks=0` degenerates to logistic regression -- a single linear layer on
    the same inputs, trained by the same loop. Used instead of sklearn's
    LogisticRegression, which upcasts to float64 and would need 1.1 GB for the
    training split alone on a 16 GB machine.
    """
    import torch
    from gnnids.models.mlp import FlowMLP, FocalLoss

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = FlowMLP(
        Xtr.shape[1], cfg["hidden_dim"],
        cfg["n_blocks"] if n_blocks is None else n_blocks,
        n_classes=int(ymtr.max()) + 1, dropout=cfg["dropout"],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    focal = FocalLoss(cfg["focal_alpha"], cfg["focal_gamma"])
    ce = torch.nn.CrossEntropyLoss()

    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr).float()
    ymtr_t = torch.from_numpy(ymtr).long()
    Xva_t = torch.from_numpy(Xva).to(device)

    n, bs = len(Xtr), cfg["batch_size"]
    best_ap, best_state, patience = -1.0, None, 0

    for epoch in range(cfg["max_epochs"]):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr_t[idx].to(device, non_blocking=True)
            logit_b, logit_m = model(xb)
            loss = focal(logit_b, ytr_t[idx].to(device))
            # Auxiliary multi-class objective (D3), down-weighted so the binary
            # head -- the one that survives leave-one-attack-out -- dominates.
            loss = loss + cfg["multiclass_weight"] * ce(logit_m, ymtr_t[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)

        model.eval()
        with torch.no_grad():
            va_scores = torch.sigmoid(model(Xva_t)[0]).cpu().numpy()
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(yva, va_scores)

        if ap > best_ap + 1e-5:
            best_ap, patience = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    model.eval()

    def score(X: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 65536):
                xb = torch.from_numpy(X[i:i + 65536]).to(device)
                out.append(torch.sigmoid(model(xb)[0]).cpu().numpy())
        return np.concatenate(out)

    return score, {"best_val_pr_auc": round(float(best_ap), 5), "epochs_run": epoch + 1}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "baselines.yaml")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--preprocess-config", type=Path,
                    default=REPO_ROOT / "configs" / "preprocess.yaml",
                    help="selects the dataset; its dataset_config is followed")
    ap.add_argument("--out", type=Path, default=None,
                    help="results file (defaults to baselines.json in OUT_DIR)")
    ap.add_argument("--target-prevalence", type=float, default=None,
                    help="override the standardised prevalence (D23; default 0.04)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    n_seeds = args.seeds or cfg["n_seeds"]
    models_to_run = args.models or cfg["models"]

    # D23 -- the same standardised prevalence Phase 4 reports at. Held in the
    # config, not hardcoded here, and written into the output alongside the
    # metrics so no downstream reader has to assume it (C17).
    target_prev = args.target_prevalence or cfg["eval"]["target_prevalence"]
    prev_seed = cfg["eval"]["prevalence_seed"]
    prev_key = f"at_{target_prev:.0%}"

    pre_cfg = yaml.safe_load(args.preprocess_config.read_text())
    ds_cfg = yaml.safe_load((REPO_ROOT / pre_cfg["dataset_config"]).read_text())
    print(f"dataset: {ds_cfg['name']}")
    stem = Path(ds_cfg["source"]["filename"]).stem
    proc = REPO_ROOT / pre_cfg["output"]["dir"]

    print("Loading preprocessed features ...")
    npz = np.load(proc / "features.npz")
    cont, cat = npz["continuous"], npz["categorical"]
    pipeline_meta = json.loads((proc / "pipeline.json").read_text())
    splits = json.loads((proc / "splits.json").read_text())
    families = json.loads((proc / "attack_families.json").read_text())

    # Written by preprocess.py for exactly the rows in features.npz. Reading the
    # source parquet instead would misalign whenever preprocessing was capped.
    df = pd.read_parquet(proc / "meta.parquet")
    if len(df) != len(cont):
        raise SystemExit(
            f"meta.parquet has {len(df):,} rows but features.npz has {len(cont):,}. "
            f"Re-run scripts/preprocess.py --config {args.preprocess_config}.")
    y = df["Label"].to_numpy().astype(np.int64)
    ymc = df["Attack"].map(families).to_numpy().astype(np.int64)

    tr = slice(splits["train"]["start"], splits["train"]["stop"])
    va = slice(splits["val"]["start"], splits["val"]["stop"])
    te = slice(splits["test"]["start"], splits["test"]["stop"])

    print("Building feature blocks ...")
    onehot = one_hot(cat, pipeline_meta["cardinalities"])
    del cat
    host_raw = windowed_host_features(df, pre_cfg["graph"]["window_size"])
    host, host_stats = standardise(host_raw, tr)
    del host_raw

    # Blocks are kept separate and the model input is assembled per split on
    # demand. Materialising both full feature matrices at once needs ~1.6 GB
    # before any model allocates anything, which is what was exhausting a 16 GB
    # machine -- the process was being killed rather than erroring.
    def make(rows: slice, with_host: bool) -> np.ndarray:
        parts = [cont[rows], onehot[rows]] + ([host[rows]] if with_host else [])
        return np.ascontiguousarray(np.hstack(parts), dtype=np.float32)

    # Names in the same order as the assembled matrix, so tree feature
    # importances can be read back against real columns rather than indices.
    feature_names = (
        pipeline_meta["continuous_names"]
        + [f"{n}={i}" for n, c in zip(pipeline_meta["categorical_names"],
                                      pipeline_meta["cardinalities"]) for i in range(c)]
        + list(FEATURE_NAMES)
    )
    n_flow = cont.shape[1] + onehot.shape[1]
    feature_sets = {"flow": False, "flow+host": True}
    print(f"  flow      ({len(df):,}, {n_flow})")
    print(f"  flow+host ({len(df):,}, {n_flow + host.shape[1]})  "
          f"(+{len(FEATURE_NAMES)} host aggregates)")
    print(f"  train {splits['train']['n_rows']:,} / val {splits['val']['n_rows']:,} "
          f"/ test {splits['test']['n_rows']:,}")
    print(f"  attack rate  train {y[tr].mean():.4%}  val {y[va].mean():.4%}  "
          f"test {y[te].mean():.4%}\n")

    # Imported lazily and only when a torch model is actually requested: torch
    # and xgboost each bundle an OpenMP runtime, and loading both into one
    # process segfaults on macOS regardless of KMP_DUPLICATE_LIB_OK.
    if {"mlp", "logreg"} & set(models_to_run):
        import torch
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = None
    results: dict = {}

    for fs_name, with_host in feature_sets.items():
        Xtr, Xva, Xte = make(tr, with_host), make(va, with_host), make(te, with_host)
        for model_name in models_to_run:
            key = f"{model_name}::{fs_name}"
            print(f"--- {key} ---")
            runs, infos = [], []

            for seed in range(n_seeds):
                t0 = time.time()
                if model_name == "logreg":
                    # Torch linear model rather than sklearn: same loop, same
                    # dtype, no float64 upcast. n_blocks=0 is a bare linear layer.
                    sc, info = train_torch_mlp(
                        Xtr, y[tr], ymc[tr], Xva, y[va], cfg["mlp"], seed, device,
                        n_blocks=0)
                elif model_name == "random_forest":
                    m = RandomForestClassifier(
                        n_estimators=cfg["random_forest"]["n_estimators"],
                        max_depth=cfg["random_forest"]["max_depth"],
                        min_samples_leaf=cfg["random_forest"]["min_samples_leaf"],
                        class_weight="balanced_subsample",
                        # NOT -1: each worker gets its own copy of the training
                        # matrix, so full parallelism multiplies a 600 MB array
                        # by the core count and the process is killed.
                        n_jobs=cfg["random_forest"]["n_jobs"], random_state=seed)
                    m.fit(Xtr, y[tr])
                    sc = lambda Z, m=m: m.predict_proba(Z)[:, 1]
                    info = {"feature_importance": m.feature_importances_.tolist()}
                elif model_name == "xgboost":
                    from xgboost import XGBClassifier
                    pos = max(int(y[tr].sum()), 1)
                    m = XGBClassifier(
                        n_estimators=cfg["xgboost"]["n_estimators"],
                        max_depth=cfg["xgboost"]["max_depth"],
                        learning_rate=cfg["xgboost"]["learning_rate"],
                        subsample=0.8, colsample_bytree=0.8,
                        scale_pos_weight=(len(y[tr]) - pos) / pos,
                        tree_method="hist", n_jobs=cfg["xgboost"]["n_jobs"],
                        random_state=seed, eval_metric="aucpr")
                    m.fit(Xtr, y[tr])
                    sc = lambda Z, m=m: m.predict_proba(Z)[:, 1]
                    info = {"feature_importance": m.feature_importances_.tolist()}
                elif model_name == "mlp":
                    sc, info = train_torch_mlp(
                        Xtr, y[tr], ymc[tr], Xva, y[va], cfg["mlp"], seed, device)
                else:
                    raise SystemExit(f"unknown model {model_name!r}")

                val_scores = sc(Xva)
                # Select the threshold on validation subsampled to the target
                # prevalence, not on native validation. This mirrors
                # scripts/train_gnn.py line-for-line, and it is the half of C17
                # that is easy to miss: a threshold tuned at 63% attack traffic
                # does not transfer to a 4% network, so a baseline evaluated at
                # 4% with a natively-tuned threshold is handicapped and the GNN
                # beats a strawman. PR-AUC is threshold-free and unaffected;
                # F1, precision and recall are not.
                vidx = subsample_to_prevalence(y[va], target_prev, prev_seed)
                thr = choose_threshold(y[va][vidx], val_scores[vidx], cfg["threshold_mode"])
                test_scores = sc(Xte)

                # Both prevalences, from one set of predictions (D23).
                run = report_both(y[te], test_scores, thr, target_prev,
                                  ymc[te], families, prev_seed)
                from sklearn.metrics import average_precision_score
                # At the target prevalence, matching train_gnn.py's model
                # selection criterion. The native figure would sit against a
                # 0.63 floor and say nothing.
                run["val_pr_auc_at_target"] = round(float(
                    average_precision_score(y[va][vidx], val_scores[vidx])), 5)
                run["threshold_selected_at"] = target_prev
                run["fit_seconds"] = round(time.time() - t0, 1)
                runs.append(run)
                infos.append(info)
                adj, nat = run[prev_key], run["native"]
                print(f"  seed {seed}  PR-AUC {adj['pr_auc']:.4f} (at "
                      f"{adj['prevalence']:.3f}, native {nat['pr_auc']:.4f})  "
                      f"F1 {adj['f1']:.4f}  recall {adj['recall']:.4f}  "
                      f"FPR@95 {adj['fpr_at_95_recall']:.5f}  "
                      f"({run['fit_seconds']:.0f}s)")

            results[key] = {
                "model": model_name, "feature_set": fs_name,
                "n_seeds": n_seeds,
                # "aggregate" is the standardised figure, because that is the
                # one comparable with Phase 4 and across datasets. The native
                # block is kept beside it and named, never left implicit --
                # three numbers in this project have been misread because the
                # PR-AUC floor moved underneath them.
                "reported_at_prevalence": target_prev,
                "schema": SCHEMA,
                "aggregate": aggregate_seeds([r[prev_key] for r in runs]),
                "aggregate_native": aggregate_seeds([r["native"] for r in runs]),
                "runs": runs, "info": infos,
            }
            agg, agg_n = results[key]["aggregate"], results[key]["aggregate_native"]
            print(f"  => PR-AUC {agg['pr_auc']['mean']:.4f} +/- {agg['pr_auc']['std']:.4f} "
                  f"at {target_prev:.0%} (floor {target_prev:.2f})   "
                  f"native {agg_n['pr_auc']['mean']:.4f}\n")
        del Xtr, Xva, Xte   # free ~850 MB before assembling the next feature set

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Merge rather than overwrite: torch models and xgboost must run in separate
    # processes (each bundles its own OpenMP runtime and loading both segfaults
    # on macOS), so a full sweep is several invocations.
    out_path = args.out or (OUT_DIR / f"baselines_{ds_cfg['name']}.json")
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        merged = prev.get("results", {})
        # Entries written before C17 was closed report at NATIVE prevalence and
        # carry no schema tag. Tag them on sight rather than leaving them to be
        # read as though they were 4% figures -- a merged file mixing the two
        # silently is precisely the failure C17 was.
        for k, v in merged.items():
            if isinstance(v, dict) and "schema" not in v:
                v["schema"] = STALE_SCHEMA
                v["reported_at_prevalence"] = "native"
        merged.update(results); results = merged
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "host_feature_names": FEATURE_NAMES,
        "host_feature_scaling": host_stats,
        "n_features": {"flow": n_flow, "flow+host": n_flow + host.shape[1]},
        "feature_names": feature_names,
        "split_prevalence": {
            "train": round(float(y[tr].mean()), 6),
            "val": round(float(y[va].mean()), 6),
            "test": round(float(y[te].mean()), 6),
        },
        "results": results,
    }
    out_path.write_text(json.dumps(report, indent=2))

    # ---- the comparison that matters -------------------------------------
    print("=" * 74)
    print(f"  PHASE 3 SUMMARY -- test PR-AUC at {target_prev:.0%} prevalence "
          f"(floor {target_prev:.2f}), mean +/- std")
    print("=" * 74)
    print(f"  {'model':<16} {'flow':>18} {'flow+host':>18} {'host gain':>12}")
    current = {k: v for k, v in results.items() if v.get("schema") == SCHEMA}
    stale = sorted(k for k in results if k not in current)
    for model_name in sorted({k.split("::")[0] for k in current}):
        a = current.get(f"{model_name}::flow", {}).get("aggregate", {}).get("pr_auc")
        b = current.get(f"{model_name}::flow+host", {}).get("aggregate", {}).get("pr_auc")
        if a and b:
            print(f"  {model_name:<16} {a['mean']:>9.4f} +/-{a['std']:<6.4f} "
                  f"{b['mean']:>9.4f} +/-{b['std']:<6.4f} {b['mean'] - a['mean']:>+12.4f}")
    print("\n  'host gain' is what hand-crafted context is worth. Phase 4's GNN")
    print("  must beat the flow+host column to justify message passing at all.")
    if stale:
        print(f"\n  NOT SHOWN -- {len(stale)} entries predate C17 and report at NATIVE")
        print("  prevalence, so they do not belong in the table above. Re-run them")
        print("  to bring them onto the standardised scale:")
        for k in stale:
            print(f"    {k}")
    shown = out_path.relative_to(REPO_ROOT) if out_path.is_relative_to(REPO_ROOT) else out_path
    print(f"\nwritten -> {shown}\n")


if __name__ == "__main__":
    main()
