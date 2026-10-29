"""Deletion test of the attributions in the evidence packs (D56).

For each of the study's evidence packs, takes the five features the pack gives
as reasons (the ones the facts layer passes to the language model), removes
those that raised the score, and rescores the flow in its own window. The same
number of other features is removed at random for comparison. If the
attributions are right, removing the stated reasons should lower the score far
more than removing random features. See `gnnids.explain.deletion`.

Torch only; reads the packs, the checkpoint that made them and the test split.

    python scripts/attribution_deletion.py
    python scripts/attribution_deletion.py --n-random 20

Writes `results/metrics/evidence/deletion_<dataset>.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gnnids.data.splits import Split  # noqa: E402
from gnnids.data.version import check_checkpoint  # noqa: E402
from gnnids.explain.deletion import deletion_check  # noqa: E402
from gnnids.explain.units import feature_names  # noqa: E402
from gnnids.graph.dataset import SnapshotDataset  # noqa: E402
from gnnids.graph.inputs import load_graph_inputs  # noqa: E402
from gnnids.training.loop import pick_device, release  # noqa: E402
from make_evidence import load_model  # noqa: E402

TOP_K = 5     # gnnids.llm.facts.reasons passes the first five to the model


def sigmoid(v: float) -> float:
    return float(1 / (1 + np.exp(-v)))


def summarise(rows: list[dict], threshold: float) -> dict:
    tested = [r for r in rows if r["n_raising"]]
    drop_r = np.array([r["drop_without_raising"] for r in tested])
    drop_x = np.array([r["drop_without_random"] for r in tested])
    still = np.array([sigmoid(r["logit_without_raising"]) >= threshold for r in tested])
    still_x = np.array([np.mean([sigmoid(v) >= threshold for v in r["random_logits"]])
                        for r in tested])
    rise = np.array([r["rise_without_lowering"] for r in rows if r["n_lowering"]])
    return {
        "packs": len(rows),
        "packs_with_a_raising_reason": len(tested),
        "median_logit_drop_reasons_removed": round(float(np.median(drop_r)), 4),
        "median_logit_drop_random_removed": round(float(np.median(drop_x)), 4),
        "share_reasons_drop_more_than_random": round(float(np.mean(drop_r > drop_x)), 4),
        "share_still_flagged_reasons_removed": round(float(np.mean(still)), 4),
        "share_still_flagged_random_removed": round(float(np.mean(still_x)), 4),
        "packs_with_a_lowering_reason": int(len(rise)),
        "median_logit_rise_lowering_removed": (round(float(np.median(rise)), 4)
                                               if len(rise) else None),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "evidence.yaml")
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    gnn_cfg = yaml.safe_load((REPO_ROOT / cfg["gnn_config"]).read_text())
    pre_cfg = yaml.safe_load((REPO_ROOT / gnn_cfg["preprocess_config"]).read_text())
    ds_cfg = yaml.safe_load((REPO_ROOT / pre_cfg["dataset_config"]).read_text())
    name = ds_cfg["name"]

    packs = json.loads((REPO_ROOT / cfg["output"]["dir"] / f"evidence_{name}.json").read_text())
    proc = REPO_ROOT / pre_cfg["output"]["dir"]
    inputs = load_graph_inputs(proc, name)
    names = feature_names(json.loads((proc / "pipeline.json").read_text()))
    index = {n: i for i, n in enumerate(names)}

    device = pick_device(gnn_cfg["train"]["device"])
    ckpt_path = REPO_ROOT / cfg["checkpoint"]
    model, ckpt = load_model(ckpt_path, device)
    check_checkpoint(ckpt, gnn_cfg["preprocess_config"], "attribution_deletion")
    threshold = float(ckpt["threshold"])

    s = inputs.splits["test"]
    ds = SnapshotDataset(inputs.src, inputs.dst, inputs.edge_features,
                         inputs.y, inputs.y_multiclass,
                         Split("test", s["start"], s["stop"]),
                         pre_cfg["graph"]["window_size"], max_windows=cfg["max_windows"])

    by_window: dict[int, list[dict]] = {}
    for p in packs:
        by_window.setdefault(p["provenance"]["window_index"], []).append(p)

    rng = np.random.default_rng(args.seed)
    rows, worst_mismatch = [], 0.0
    print(f"{name}: {len(packs)} packs in {len(by_window)} windows, "
          f"{args.n_random} random draws each, device {device}")
    for wi in sorted(by_window):
        batch = ds[wi].to(device)
        lo, hi = ds.windows[wi]
        win_rows = np.arange(lo, hi) if ds.row_index is None else ds.row_index[lo:hi]
        for p in by_window[wi]:
            edge = int(np.flatnonzero(win_rows == p["provenance"]["row_index"])[0])
            reasons = p["attribution"]["top_features"][:TOP_K]
            raising = [index[f["name"]] for f in reasons if f["contribution"] > 0]
            lowering = [index[f["name"]] for f in reasons if f["contribution"] < 0]
            r = deletion_check(model, batch.x, batch.edge_index, batch.edge_attr, edge,
                               raising, lowering, [index[f["name"]] for f in reasons],
                               args.n_random, rng)
            # The pack's score was computed by the same checkpoint on the same
            # window; a gap beyond MPS noise would mean the rows do not line up.
            worst_mismatch = max(worst_mismatch,
                                 abs(sigmoid(r["base_logit"]) - p["detection"]["score"]))
            rows.append({"detection_id": p["detection_id"],
                         "true_label": p["flow"].get("true_label"),
                         "predicted_class": p["detection"]["predicted_class"], **r})
        release(device)
    if worst_mismatch > 1e-3:
        raise SystemExit(f"rescored packs differ from their stored score by up to "
                         f"{worst_mismatch:.5f}: the windows do not match the packs")

    families = sorted({r["true_label"] for r in rows})
    summary = summarise(rows, threshold)
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": name,
        "checkpoint": ckpt_path.name,
        "data_version": ckpt.get("data_version"),
        "threshold": round(threshold, 5),
        "reasons_per_pack": TOP_K,
        "n_random": args.n_random,
        "seed": args.seed,
        "removed_to": "integrated-gradients baseline (0 in standardised units, the mean flow)",
        "max_score_mismatch_with_pack": round(worst_mismatch, 6),
        "summary": summary,
        "by_true_label": {f: summarise([r for r in rows if r["true_label"] == f], threshold)
                          for f in families},
        "per_pack": [{k: v for k, v in r.items() if k != "random_logits"} for r in rows],
    }
    path = REPO_ROOT / "results" / "metrics" / "evidence" / f"deletion_{name}.json"
    path.write_text(json.dumps(out, indent=2))
    for k, v in summary.items():
        print(f"  {k:42s} {v}")
    print(f"\n  wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
