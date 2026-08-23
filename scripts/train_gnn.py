"""Phase 4: train and evaluate the dual-channel GNN.

This is the measurement V2 has been waiting for. Channel 1 is the Phase 3 MLP
baseline class, unchanged, so the gap between `channel1_only` and `full` is the
value of learned message passing and not an artefact of architecture.

Reported at a standardised 4% prevalence (D23) by subsampling scores after
inference, so the graph stays intact while the metrics stay meaningful.

Usage:
    python scripts/train_gnn.py --ablation full --seeds 1
    python scripts/train_gnn.py                       # all three ablations
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnids.data.splits import Split  # noqa: E402
from gnnids.eval.metrics import aggregate_seeds, choose_threshold  # noqa: E402
from gnnids.eval.prevalence import report_both, subsample_to_prevalence  # noqa: E402
from gnnids.graph.dataset import SnapshotDataset  # noqa: E402
from gnnids.models.dual_channel import DualChannelGNN, channel_attribution  # noqa: E402
from gnnids.models.mlp import FocalLoss  # noqa: E402


def one_hot(cat: np.ndarray, cardinalities: list[int]) -> np.ndarray:
    """Identical expansion to Phase 3, so Channel 1 sees the same input the
    baseline did."""
    blocks = []
    for j, card in enumerate(cardinalities):
        block = np.zeros((len(cat), card), dtype=np.float32)
        block[np.arange(len(cat)), np.clip(cat[:, j], 0, card - 1)] = 1.0
        blocks.append(block)
    return np.hstack(blocks)


def pick_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def infer(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score every edge. Returns (scores, y, y_multiclass, fusion weights)."""
    model.eval()
    scores, ys, yms, alphas = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        logit_b, _, alpha = model(batch.x, batch.edge_index, batch.edge_attr)
        scores.append(torch.sigmoid(logit_b).cpu().numpy())
        ys.append(batch.y.cpu().numpy())
        yms.append(batch.y_multiclass.cpu().numpy())
        if alpha is not None:
            alphas.append(alpha.cpu().numpy())
    return (
        np.concatenate(scores), np.concatenate(ys), np.concatenate(yms),
        np.concatenate(alphas) if alphas else None,
    )


def train_one(cfg, datasets, edge_dim, n_classes, ablation, seed, device) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    tcfg = cfg["train"]

    model = DualChannelGNN(
        edge_dim=edge_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        n_mlp_blocks=cfg["model"]["n_mlp_blocks"],
        n_gnn_layers=cfg["model"]["n_gnn_layers"],
        n_classes=n_classes,
        dropout=cfg["model"]["dropout"],
        bidirectional=cfg["model"]["bidirectional"],
        **ablation,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"])
    focal = FocalLoss(tcfg["focal_alpha"], tcfg["focal_gamma"])
    ce = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(datasets["train"], batch_size=tcfg["batch_size"], shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=tcfg["batch_size"])

    target_prev = cfg["eval"]["target_prevalence"]
    best_ap, best_state, patience, epochs_run = -1.0, None, 0, 0

    for epoch in range(tcfg["max_epochs"]):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            logit_b, logit_m, _ = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = focal(logit_b, batch.y.float())
            loss = loss + tcfg["multiclass_weight"] * ce(logit_m, batch.y_multiclass)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            total += float(loss.detach())

        val_scores, val_y, _, _ = infer(model, val_loader, device)
        # Select on PR-AUC at the *target* prevalence, not native. Selecting at
        # 64% attack would optimise for a regime we do not report.
        idx = subsample_to_prevalence(val_y, target_prev, cfg["output"]["seed"])
        ap = average_precision_score(val_y[idx], val_scores[idx])
        epochs_run = epoch + 1

        if ap > best_ap + 1e-5:
            best_ap, patience = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= tcfg["patience"]:
                break

    model.load_state_dict(best_state)

    val_scores, val_y, _, _ = infer(model, val_loader, device)
    vidx = subsample_to_prevalence(val_y, target_prev, cfg["output"]["seed"])
    threshold = choose_threshold(val_y[vidx], val_scores[vidx], cfg["eval"]["threshold_mode"])

    test_loader = DataLoader(datasets["test"], batch_size=tcfg["batch_size"])
    scores, y, ymc, alpha = infer(model, test_loader, device)

    out = report_both(y, scores, threshold, target_prev, ymc, datasets["families"],
                      cfg["output"]["seed"])
    out["val_pr_auc_at_target"] = round(float(best_ap), 5)
    out["epochs_run"] = epochs_run
    out["n_parameters"] = sum(p.numel() for p in model.parameters())
    if alpha is not None:
        out["channel_attribution"] = channel_attribution(torch.from_numpy(alpha))
        # The per-detection form of the V2 claim: on flows the model got right,
        # how much of the decision came from neighbourhood context?
        tp = (y == 1) & (scores >= threshold)
        if tp.any():
            out["channel_attribution"]["topological_weight_on_true_positives"] = round(
                float(alpha[tp, 1].mean()), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "gnn.yaml")
    ap.add_argument("--ablation", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    pre_cfg = yaml.safe_load((REPO_ROOT / cfg["preprocess_config"]).read_text())
    ds_cfg = yaml.safe_load((REPO_ROOT / pre_cfg["dataset_config"]).read_text())
    proc = REPO_ROOT / pre_cfg["output"]["dir"]
    window = pre_cfg["graph"]["window_size"]
    n_seeds = args.seeds or cfg["train"]["n_seeds"]

    print(f"dataset: {ds_cfg['name']}   window: {window:,}")
    npz = np.load(proc / "features.npz")
    meta = json.loads((proc / "pipeline.json").read_text())
    splits_raw = json.loads((proc / "splits.json").read_text())
    families = json.loads((proc / "attack_families.json").read_text())

    # float16 storage, upcast to float32 per window inside build_snapshot. The
    # full array at float32 is 2.6 GB, which pushed a 16 GB machine into swap:
    # the first attempt ran at 23% CPU in uninterruptible I/O wait with 2.9 GB
    # of swap in use. Precision is irrelevant here -- these are standardised
    # features and one-hot indicators.
    edge_feats = np.hstack([
        npz["continuous"], one_hot(npz["categorical"], meta["cardinalities"])
    ]).astype(np.float16)
    del npz

    df = pd.read_parquet(proc / "meta.parquet",
                         columns=["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label", "Attack"])
    if len(df) != len(edge_feats):
        raise SystemExit(
            f"meta.parquet has {len(df):,} rows, features have {len(edge_feats):,}. "
            f"Re-run scripts/preprocess.py.")

    # Factorise addresses to int32 once. As numpy object arrays these are
    # 0.63 GB of Python string objects for 2,556 distinct hosts; as codes they
    # are 47 MB. np.unique inside build_snapshot is also far faster on ints.
    n = len(df)
    codes = pd.factorize(
        np.concatenate([df.IPV4_SRC_ADDR.to_numpy(), df.IPV4_DST_ADDR.to_numpy()])
    )[0].astype(np.int32)
    src, dst = codes[:n], codes[n:]
    y = df.Label.to_numpy().astype(np.int64)
    ymc = df.Attack.map(families).to_numpy().astype(np.int64)
    del df, codes

    datasets = {"families": families}
    for name, s in splits_raw.items():
        datasets[name] = SnapshotDataset(
            src, dst, edge_feats, y, ymc,
            Split(name, s["start"], s["stop"]), window)
        print(f"  {name:<6} {len(datasets[name]):>5} windows  "
              f"{datasets[name].n_edges:>10,} edges  "
              f"attack {datasets[name].attack_rate():.2%}")

    device = pick_device(cfg["train"]["device"])
    edge_dim = edge_feats.shape[1]
    n_classes = len(families)
    print(f"\ndevice: {device}   edge features: {edge_dim}   classes: {n_classes}")
    print(f"target: beat XGBoost PR-AUC 0.9552 / F1 0.8963 at "
          f"{cfg['eval']['target_prevalence']:.0%} prevalence\n")

    to_run = args.ablation or list(cfg["ablations"].keys() - {"gnn_layer_sweep"})
    order = [a for a in ("channel1_only", "channel2_only", "full") if a in to_run]

    results = {}
    for ablation in order:
        spec = cfg["ablations"][ablation]
        print(f"--- {ablation} ---")
        runs = []
        for seed in range(n_seeds):
            t0 = time.time()
            r = train_one(cfg, datasets, edge_dim, n_classes, spec, seed, device)
            r["fit_seconds"] = round(time.time() - t0, 1)
            runs.append(r)
            adj = r[f"at_{cfg['eval']['target_prevalence']:.0%}"]
            print(f"  seed {seed}  PR-AUC {adj['pr_auc']:.4f} (base "
                  f"{adj['prevalence']:.3f})  F1 {adj['f1']:.4f}  "
                  f"recall {adj['recall']:.4f}  {r['epochs_run']}ep "
                  f"({r['fit_seconds']:.0f}s)")
        key = f"at_{cfg['eval']['target_prevalence']:.0%}"
        results[ablation] = {
            "spec": spec, "n_seeds": n_seeds, "runs": runs,
            "aggregate": aggregate_seeds([r[key] for r in runs]),
        }
        agg = results[ablation]["aggregate"]
        print(f"  => PR-AUC {agg['pr_auc']['mean']:.4f} +/- {agg['pr_auc']['std']:.4f}"
              f"   F1 {agg['f1']['mean']:.4f}\n")

    out_dir = REPO_ROOT / cfg["output"]["metrics"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"gnn_{ds_cfg['name']}.json"
    (out_dir / out_name).write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": ds_cfg["name"], "config": cfg,
        "edge_dim": edge_dim, "window_size": window,
        "baseline_to_beat": {"model": "xgboost", "pr_auc": 0.9552, "f1": 0.8963},
        "results": results,
    }, indent=2))

    print("=" * 72)
    print(f"  PHASE 4 -- test metrics at {cfg['eval']['target_prevalence']:.0%} prevalence")
    print("=" * 72)
    print(f"  {'model':<16} {'PR-AUC':>16} {'F1':>10}")
    print(f"  {'XGBoost (P3)':<16} {0.9552:>16.4f} {0.8963:>10.4f}   <- to beat")
    for a in order:
        g = results[a]["aggregate"]
        print(f"  {a:<16} {g['pr_auc']['mean']:>9.4f} +/-{g['pr_auc']['std']:<5.4f} "
              f"{g['f1']['mean']:>10.4f}")
    if "full" in results and "channel1_only" in results:
        gap = (results["full"]["aggregate"]["pr_auc"]["mean"]
               - results["channel1_only"]["aggregate"]["pr_auc"]["mean"])
        print(f"\n  topology gain (full - channel1_only): {gap:+.4f}")
        print("  This is the V2 measurement.")
    print(f"\nwritten -> {(out_dir / out_name).relative_to(REPO_ROOT)}\n")


if __name__ == "__main__":
    main()
