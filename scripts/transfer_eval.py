"""Phase 9: cross-dataset generalisation -- a different network, no retraining.

Take a model trained on one NF-v2 dataset and score it on another. No
retraining, no fine-tuning, and the source dataset's feature pipeline is applied
unchanged to the target (refitting it there would be test-set access).

**Why this is Tier 1 (D18).** NF-UNSW-NB15-v2 is saturated and full of
provenance fingerprints, and removing the ones we measured does not prove none
remain. A different network is the only test that distinguishes a model that
learned attack *behaviour* from one that learned *which generator produced the
flow*. Expect the numbers to fall; the size of the fall is the finding.

**Why it is cheap: D1.** The NF-v2 family shares an identical 43-feature schema,
so transfer is a config change. On the original UNSW-NB15's bespoke Argus
features this experiment would have cost weeks and would not have been
attempted -- which makes D1 a methodological decision, not a convenience.

Usage:
    python scripts/transfer_eval.py --smoke
    python scripts/transfer_eval.py --checkpoint results/checkpoints/X.pt \
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
import torch
import yaml
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnids.data.schema import build_schema  # noqa: E402
from gnnids.data.splits import Split  # noqa: E402
from gnnids.eval.metrics import choose_threshold  # noqa: E402
from gnnids.eval.prevalence import report_both, subsample_to_prevalence  # noqa: E402
from gnnids.eval.transfer import transfer_inputs  # noqa: E402
from gnnids.graph.dataset import SnapshotDataset  # noqa: E402
from gnnids.models.dual_channel import DualChannelGNN  # noqa: E402
from gnnids.training.loop import infer, pick_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "transfer.yaml")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--source-preprocess", type=Path, default=None)
    ap.add_argument("--target-preprocess", type=Path, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    gnn_cfg = yaml.safe_load((REPO_ROOT / cfg["gnn_config"]).read_text())
    src_pre = yaml.safe_load(
        (args.source_preprocess or REPO_ROOT / cfg["source"]["preprocess_config"]).read_text())
    tgt_pre = yaml.safe_load(
        (args.target_preprocess or REPO_ROOT / cfg["target"]["preprocess_config"]).read_text())
    src_ds = yaml.safe_load((REPO_ROOT / src_pre["dataset_config"]).read_text())
    tgt_ds = yaml.safe_load((REPO_ROOT / tgt_pre["dataset_config"]).read_text())

    max_rows = args.max_rows or (cfg["smoke"]["max_rows"] if args.smoke else cfg["max_rows"])
    window = cfg["smoke"]["window_size"] if args.smoke else src_pre["graph"]["window_size"]
    ckpt_path = args.checkpoint or (REPO_ROOT / cfg["checkpoint"])
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path}. Train one first.")

    print(f"train on: {src_ds['name']}   ->   score on: {tgt_ds['name']}"
          f"{'   [SMOKE]' if args.smoke else ''}")
    print(f"checkpoint: {ckpt_path.name}   pipeline: FROZEN from {src_ds['name']}\n")

    target_parquet = REPO_ROOT / "data" / "interim" / tgt_ds["source"]["filename"].replace(".csv", ".parquet")
    if not target_parquet.exists():
        raise SystemExit(f"no target data at {target_parquet}")

    # The schema is built from the SOURCE config -- same drops, same roles -- so
    # the model receives columns in the order it was trained on.
    head = pd.read_parquet(target_parquet, columns=None).head(1) if False else None
    probe = pd.read_parquet(target_parquet).head(2000)
    schema = build_schema(src_pre, probe, REPO_ROOT)
    del probe

    inputs = transfer_inputs(
        REPO_ROOT / src_pre["output"]["dir"] / "pipeline.json",
        target_parquet, schema, max_rows=max_rows)
    print(f"  target rows {inputs['n_rows']:,}   hosts {inputs['n_hosts']:,}   "
          f"attack rate {inputs['prevalence']:.2%}")
    worst = inputs["oov"]["_worst_oov_rate"]
    print(f"  worst categorical out-of-vocabulary rate: {worst:.2%}")
    if worst >= cfg["oov_warn_above"]:
        print(f"  WARNING: above {cfg['oov_warn_above']:.0%} of one column's values")
        print(f"  fell outside the source vocabulary. The transfer number below")
        print(f"  partly measures a degraded input, not a different network.")

    device = pick_device(gnn_cfg["train"]["device"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt["edge_dim"] != inputs["edge_features"].shape[1]:
        raise SystemExit(
            f"checkpoint expects {ckpt['edge_dim']} features, the frozen "
            f"pipeline produced {inputs['edge_features'].shape[1]}. The source "
            f"and target must share a schema (D1).")
    model = DualChannelGNN(edge_dim=ckpt["edge_dim"], n_classes=ckpt["n_classes"],
                           **ckpt["model_cfg"], **ckpt["ablation"])
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    ds = SnapshotDataset(inputs["src"], inputs["dst"], inputs["edge_features"],
                         inputs["y"], inputs["y_multiclass"],
                         Split("transfer", 0, inputs["n_rows"]), window)
    print(f"  {len(ds)} windows of {window:,} edges\n")
    scores, y, _, alpha = infer(model, DataLoader(ds, batch_size=8), device)

    target_prev = gnn_cfg["eval"]["target_prevalence"]
    seed = gnn_cfg["output"]["seed"]

    # Two thresholds, and the difference between them IS a result.
    #  - the source threshold, transferred untouched: what an operator would
    #    actually get if they deployed this model on a new network today.
    #  - a threshold re-chosen on the target: what the model could do if someone
    #    calibrated it there. The design allows recalibration of the threshold
    #    and nothing else.
    idx = subsample_to_prevalence(y, target_prev, seed)
    recalibrated = choose_threshold(y[idx], scores[idx], gnn_cfg["eval"]["threshold_mode"])
    results = {
        "source_threshold": report_both(y, scores, ckpt["threshold"], target_prev, seed=seed),
        "recalibrated_threshold": report_both(y, scores, recalibrated, target_prev, seed=seed),
    }

    key = f"at_{target_prev:.0%}"
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "cross-dataset transfer, no retraining (D18, Phase 9)",
        "smoke": args.smoke,
        "source_dataset": src_ds["name"], "target_dataset": tgt_ds["name"],
        "checkpoint": ckpt_path.name,
        "pipeline": "frozen from source; refitting on target would be test-set access",
        "thresholds": {"source": round(float(ckpt["threshold"]), 5),
                       "recalibrated_on_target": round(float(recalibrated), 5)},
        "target_rows": inputs["n_rows"], "target_hosts": inputs["n_hosts"],
        "target_native_prevalence": inputs["prevalence"],
        "out_of_vocabulary": inputs["oov"],
        "reported_at_prevalence": target_prev,
        "results": results,
    }
    out_dir = REPO_ROOT / cfg["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    name = (f"{'smoke_' if args.smoke else ''}transfer_{src_ds['name']}"
            f"_to_{tgt_ds['name']}.json")
    (out_dir / name).write_text(json.dumps(out, indent=2))

    print("=" * 76)
    print(f"  PHASE 9 -- {src_ds['name']} -> {tgt_ds['name']} at "
          f"{target_prev:.0%} prevalence (floor {target_prev:.2f})")
    print("=" * 76)
    print(f"  {'threshold':<28} {'PR-AUC':>9} {'F1':>9} {'recall':>9} {'FPR@95':>10}")
    for label, r in results.items():
        m = r[key]
        def f(v, spec=".4f"):
            return format(v, spec) if isinstance(v, (int, float)) else "n/a"
        print(f"  {label:<28} {f(m['pr_auc']):>9} {f(m['f1']):>9} "
              f"{f(m['recall']):>9} {f(m['fpr_at_95_recall'], '.5f'):>10}")
    print("\n  Expect these to be LOWER than the in-dataset numbers. That is the")
    print("  point: the gap is what the in-dataset result owed to this network's")
    print("  own fingerprints rather than to attack behaviour.")
    print(f"\nwritten -> {(out_dir / name).relative_to(REPO_ROOT)}\n")


if __name__ == "__main__":
    main()
