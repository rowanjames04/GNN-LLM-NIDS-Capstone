"""Phase 9, the decisive half: does the TREE transfer either? (D18, D32)

The GNN collapses to the prevalence floor on an unseen network. That result has
two completely different readings, and the entire Discussion section depends on
which one is true:

  - **If XGBoost also collapses**, the finding is about the *datasets*. These
    benchmarks are separable by capture-specific artefacts, nothing crosses
    between them, and no model of any kind trained on one predicts the other.
    That is a finding about NIDS benchmarking, and it is the stronger paper.
  - **If XGBoost survives**, the finding is about the *model*. The graph model
    learned something less portable than a tree did, and the architecture --
    not the data -- is what fails to generalise.

Nothing else in the project distinguishes these, and the experiment is cheap:
one tree fit plus inference.

The protocol is identical to `transfer_eval.py` by construction -- same frozen
source pipeline, same D23 reporting, same threshold treatment -- because a
comparison run through a different path would answer nothing (that was C17).

**Safe to run in one process with xgboost**: nothing imported here pulls torch,
which is checked at import time below. The two bundle separate OpenMP runtimes
and segfault together on macOS (C3).

Usage:
    python scripts/transfer_baselines.py --smoke
    python scripts/transfer_baselines.py \
        --source-preprocess configs/preprocess_toniot.yaml \
        --target-preprocess configs/preprocess.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnids.data.schema import build_schema  # noqa: E402
from gnnids.eval.metrics import choose_threshold  # noqa: E402
from gnnids.eval.prevalence import report_both, subsample_to_prevalence  # noqa: E402
from gnnids.eval.transfer import transfer_inputs  # noqa: E402
from gnnids.graph.inputs import one_hot  # noqa: E402

assert "torch" not in sys.modules, (
    "torch was imported into an xgboost process -- both bundle OpenMP and this "
    "segfaults on macOS (C3). Keep this script's import graph torch-free.")


def source_matrix(proc_dir: Path, split: dict, max_rows: int | None):
    """The source dataset's own features, assembled exactly as transfer does.

    Uses the same `one_hot` and the same column order as
    `gnnids.eval.transfer.transfer_inputs`, so the model is fitted on the
    representation it will be scored on. Assembling these two differently is
    the failure mode that produced C17.
    """
    npz = np.load(proc_dir / "features.npz")
    meta = json.loads((proc_dir / "pipeline.json").read_text())
    lo, hi = split["start"], split["stop"]
    cont = npz["continuous"][lo:hi]
    cat = npz["categorical"][lo:hi]
    if max_rows is not None and len(cont) > max_rows:
        idx = np.linspace(0, len(cont) - 1, max_rows).astype(int)
        cont, cat = cont[idx], cat[idx]
    else:
        idx = np.arange(len(cont))
    X = np.hstack([cont, one_hot(cat, meta["cardinalities"])]).astype(np.float32)
    y = pd.read_parquet(proc_dir / "meta.parquet", columns=["Label"]
                        ).Label.to_numpy()[lo:hi][idx].astype(np.int64)
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "transfer.yaml")
    ap.add_argument("--baselines-config", type=Path,
                    default=REPO_ROOT / "configs" / "baselines.yaml")
    ap.add_argument("--source-preprocess", type=Path, default=None)
    ap.add_argument("--target-preprocess", type=Path, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-train-rows", type=int, default=1_500_000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    bcfg = yaml.safe_load(args.baselines_config.read_text())
    gnn_cfg = yaml.safe_load((REPO_ROOT / cfg["gnn_config"]).read_text())
    src_pre = yaml.safe_load(
        (args.source_preprocess or REPO_ROOT / cfg["source"]["preprocess_config"]).read_text())
    tgt_pre = yaml.safe_load(
        (args.target_preprocess or REPO_ROOT / cfg["target"]["preprocess_config"]).read_text())
    src_ds = yaml.safe_load((REPO_ROOT / src_pre["dataset_config"]).read_text())
    tgt_ds = yaml.safe_load((REPO_ROOT / tgt_pre["dataset_config"]).read_text())

    n_seeds = 1 if args.smoke else args.seeds
    max_target = cfg["smoke"]["max_rows"] if args.smoke else cfg["max_rows"]
    max_train = 40_000 if args.smoke else args.max_train_rows
    target_prev = gnn_cfg["eval"]["target_prevalence"]
    prev_seed = gnn_cfg["output"]["seed"]

    print(f"train on: {src_ds['name']}   ->   score on: {tgt_ds['name']}"
          f"{'   [SMOKE]' if args.smoke else ''}")
    print("model: xgboost   pipeline: FROZEN from source (same protocol as the GNN)\n")

    src_proc = REPO_ROOT / src_pre["output"]["dir"]
    splits = json.loads((src_proc / "splits.json").read_text())
    Xtr, ytr = source_matrix(src_proc, splits["train"], max_train)
    Xva, yva = source_matrix(src_proc, splits["val"], max_train)
    print(f"  source train {Xtr.shape}  attack {ytr.mean():.2%}")

    target_parquet = (REPO_ROOT / "data" / "interim" /
                      tgt_ds["source"]["filename"].replace(".csv", ".parquet"))
    probe = pd.read_parquet(target_parquet).head(2000)
    schema = build_schema(src_pre, probe, REPO_ROOT)
    del probe
    tgt = transfer_inputs(src_proc / "pipeline.json", target_parquet, schema,
                          max_rows=max_target)
    Xte = np.asarray(tgt["edge_features"], dtype=np.float32)
    yte = tgt["y"]
    print(f"  target       {Xte.shape}  attack {tgt['prevalence']:.2%}  "
          f"worst OOV {tgt['oov']['_worst_oov_rate']:.2%}")
    if Xtr.shape[1] != Xte.shape[1]:
        raise SystemExit(f"feature mismatch: source {Xtr.shape[1]} vs target {Xte.shape[1]}")
    if tgt["oov"]["_worst_oov_rate"] >= cfg["oov_warn_above"]:
        print(f"  WARNING: this transfer partly measures a degraded input, not "
              f"only a different network.\n")

    from xgboost import XGBClassifier

    key = f"at_{target_prev:.0%}"
    runs = []
    for seed in range(n_seeds):
        pos = max(int(ytr.sum()), 1)
        m = XGBClassifier(
            n_estimators=bcfg["xgboost"]["n_estimators"],
            max_depth=bcfg["xgboost"]["max_depth"],
            learning_rate=bcfg["xgboost"]["learning_rate"],
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(len(ytr) - pos) / pos,
            tree_method="hist", n_jobs=bcfg["xgboost"]["n_jobs"],
            random_state=seed, eval_metric="aucpr")
        m.fit(Xtr, ytr)

        # Threshold chosen on the SOURCE validation split at the target
        # prevalence -- the same rule train_baselines.py uses (C18). This is
        # what an operator deploying the model would actually carry across.
        vs = m.predict_proba(Xva)[:, 1]
        vidx = subsample_to_prevalence(yva, target_prev, prev_seed)
        thr = choose_threshold(yva[vidx], vs[vidx], bcfg["threshold_mode"])

        scores = m.predict_proba(Xte)[:, 1]
        tidx = subsample_to_prevalence(yte, target_prev, prev_seed)
        thr_recal = choose_threshold(yte[tidx], scores[tidx], bcfg["threshold_mode"])

        run = {
            "seed": seed,
            "source_threshold": report_both(yte, scores, thr, target_prev, seed=prev_seed),
            "recalibrated_threshold": report_both(yte, scores, thr_recal, target_prev,
                                                  seed=prev_seed),
            "thresholds": {"source": round(float(thr), 5),
                           "recalibrated_on_target": round(float(thr_recal), 5)},
        }
        runs.append(run)
        a = run["source_threshold"][key]
        print(f"  seed {seed}  PR-AUC {a['pr_auc']:.4f} (floor {target_prev:.2f})  "
              f"F1 {a['f1']:.4f}  recall {a['recall']:.4f}")

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "cross-dataset transfer, no retraining (D18/D32), xgboost arm",
        "model": "xgboost::flow", "smoke": args.smoke,
        "source_dataset": src_ds["name"], "target_dataset": tgt_ds["name"],
        "pipeline": "frozen from source; identical protocol to transfer_eval.py",
        "reported_at_prevalence": target_prev,
        "target_rows": int(tgt["n_rows"]),
        "target_native_prevalence": tgt["prevalence"],
        "out_of_vocabulary": tgt["oov"],
        "n_seeds": n_seeds, "runs": runs,
    }
    out_dir = REPO_ROOT / cfg["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    name = (f"{'smoke_' if args.smoke else ''}transfer_xgboost_{src_ds['name']}"
            f"_to_{tgt_ds['name']}.json")
    (out_dir / name).write_text(json.dumps(out, indent=2))

    prs = [r["source_threshold"][key]["pr_auc"] for r in runs]
    print("\n" + "=" * 74)
    print(f"  PHASE 9 (xgboost) -- {src_ds['name']} -> {tgt_ds['name']}")
    print("=" * 74)
    print(f"  PR-AUC @{target_prev:.0%}: {np.mean(prs):.4f} "
          f"+/-{np.std(prs, ddof=1) if len(prs) > 1 else 0:.4f}   floor {target_prev:.2f}")
    print("\n  Compare with the GNN's number for the SAME pair. If the tree also")
    print("  collapses, the finding is about the DATASETS. If it survives, the")
    print("  finding is about the MODEL. These are different reports.")
    print(f"\nwritten -> {(out_dir / name).relative_to(REPO_ROOT)}\n")


if __name__ == "__main__":
    main()
