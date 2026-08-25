"""Phase 8: headless replay -- drives the demo and measures what batch cannot.

Streams the held-out split in window order through a trained checkpoint,
measuring end-to-end latency and trigger efficiency (D7). Headless by design:
the Streamlit app is a view over the same engine, so the logic is testable and a
demo failure the night before the showcase is a rendering bug rather than a
logic one.

**No real-time claim is being made.** Every reported metric in this project comes
from offline batch evaluation. Replay measures the operational qualities batch
evaluation cannot show -- how long an alert takes to produce, and how many
alerts an analyst receives per unit of traffic.

Usage:
    python scripts/replay.py --smoke
    python scripts/replay.py --windows 50
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnids.data.splits import Split  # noqa: E402
from gnnids.graph.dataset import SnapshotDataset  # noqa: E402
from gnnids.graph.inputs import load_graph_inputs  # noqa: E402
from gnnids.models.dual_channel import DualChannelGNN  # noqa: E402
from gnnids.training.loop import pick_device  # noqa: E402
from gnnids.ui.replay import ReplayEngine  # noqa: E402


def build_engine(cfg, gnn_cfg, pre_cfg, ds_cfg, ckpt_path: Path, window: int,
                 max_windows: int | None):
    """Everything the UI and this script both need, assembled once."""
    inputs = load_graph_inputs(REPO_ROOT / pre_cfg["output"]["dir"], ds_cfg["name"])
    device = pick_device(gnn_cfg["train"]["device"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = DualChannelGNN(edge_dim=ckpt["edge_dim"], n_classes=ckpt["n_classes"],
                           **ckpt["model_cfg"], **ckpt["ablation"])
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    s = inputs.splits["test"]
    ds = SnapshotDataset(inputs.src, inputs.dst, inputs.edge_features,
                         inputs.y, inputs.y_multiclass,
                         Split("test", s["start"], s["stop"]), window,
                         max_windows=max_windows)

    @torch.no_grad()
    def score_window(i: int):
        batch = ds[i].to(device)
        logit, _, _ = model(batch.x, batch.edge_index, batch.edge_attr)
        return (torch.sigmoid(logit).cpu().numpy(), batch.y.cpu().numpy())

    families_inv = {v: k for k, v in inputs.families.items()}
    return ReplayEngine(ds, score_window, ckpt["threshold"], families_inv), ckpt, inputs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "replay.yaml")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--windows", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    gnn_cfg = yaml.safe_load((REPO_ROOT / cfg["gnn_config"]).read_text())
    pre_cfg = yaml.safe_load((REPO_ROOT / gnn_cfg["preprocess_config"]).read_text())
    ds_cfg = yaml.safe_load((REPO_ROOT / pre_cfg["dataset_config"]).read_text())

    window = cfg["smoke"]["window_size"] if args.smoke else pre_cfg["graph"]["window_size"]
    n_windows = args.windows or (cfg["smoke"]["windows"] if args.smoke else cfg["windows"])
    ckpt_path = args.checkpoint or (REPO_ROOT / cfg["checkpoint"])
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path}. Train one first.")

    print(f"replay: {ds_cfg['name']} test split, window {window:,}"
          f"{'   [SMOKE]' if args.smoke else ''}")
    engine, ckpt, _ = build_engine(cfg, gnn_cfg, pre_cfg, ds_cfg, ckpt_path,
                                   window, n_windows)
    print(f"threshold {engine.threshold:.4f} (from the checkpoint)   "
          f"{len(engine)} windows\n")

    top = []
    for i, detections in engine.run(n_windows):
        if detections:
            d = detections[0]
            print(f"  window {i:>3}  {len(detections):>4} alerts  "
                  f"top score {d.score:.4f} ({'attack' if d.true_label else 'BENIGN'})"
                  f"  {d.latency_ms['scoring']:.0f}ms")
        top.extend(detections[:3])

    stats = engine.stats.as_dict()
    out_dir = REPO_ROOT / cfg["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{'smoke_' if args.smoke else ''}replay_{ds_cfg['name']}.json"
    (out_dir / name).write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": ds_cfg["name"], "checkpoint": ckpt_path.name,
        "note": ("replay drives the demo and measures latency and trigger "
                 "efficiency; it is NOT a real-time claim (D7)"),
        "smoke": args.smoke, "window_size": window,
        "threshold": round(float(engine.threshold), 5),
        "stats": stats,
    }, indent=2))

    print("\n" + "=" * 68)
    print(f"  PHASE 8 -- replay{' [SMOKE]' if args.smoke else ''}")
    print("=" * 68)
    for k, v in stats.items():
        print(f"  {k:<24} {v}")
    print("\n  Latency and trigger rate are what replay adds. Precision and")
    print("  recall here are NOT the reported figures -- those come from the")
    print("  batch evaluation at a standardised 4% prevalence (D23), and this")
    print(f"  split runs at its native rate.")
    print(f"\nwritten -> {(out_dir / name).relative_to(REPO_ROOT)}\n")


if __name__ == "__main__":
    main()
