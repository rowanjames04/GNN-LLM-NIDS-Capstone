"""Phase 2: flows in, PyG graph snapshots out, reproducibly.

Runs the whole preprocessing path and records what it did:

  1. resolve the feature schema from config, validated against the real columns
  2. split contiguously with a purge gap (D5)
  3. fit transforms on TRAIN ONLY, apply to all splits
  4. build graph snapshots and verify they preserve the source rows exactly
  5. sweep window size, since that is the aperture through which the GNN sees
     structure and the right value is an empirical question

Writes the fitted pipeline and transformed arrays to data/processed/ so Phases
3-5 load them directly instead of re-deriving, and statistics to
results/metrics/preprocess.json.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --no-sweep
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

from gnnids.data.schema import LABEL_BINARY, LABEL_MULTICLASS, build_schema  # noqa: E402
from gnnids.data.splits import assert_no_overlap, contiguous_split  # noqa: E402
from gnnids.data.transforms import FeaturePipeline  # noqa: E402
from gnnids.graph.build import GlobalNodeMap, build_snapshot, snapshot_stats  # noqa: E402
from gnnids.graph.windows import count_windows, fixed_count_windows  # noqa: E402


def summarise(values: list[float]) -> dict:
    a = np.asarray(values, dtype="float64")
    if not len(a):
        return {}
    return {
        "mean": round(float(a.mean()), 4),
        "std": round(float(a.std()), 4),
        "min": round(float(a.min()), 4),
        "max": round(float(a.max()), 4),
        "cv": round(float(a.std() / a.mean()), 4) if a.mean() else None,
    }


def sweep_window_sizes(df, feats, splits, sizes, families) -> dict:
    """How does snapshot structure change with the aperture?

    Reported per size so the choice of window is made on measured structure
    rather than convention. The number that matters is the coefficient of
    variation of fan-out: if it is zero, every window looks the same and
    message passing has no variation to exploit.
    """
    src_all = df["IPV4_SRC_ADDR"].to_numpy()
    dst_all = df["IPV4_DST_ADDR"].to_numpy()
    y_all = df[LABEL_BINARY].to_numpy()
    ymc_all = df[LABEL_MULTICLASS].map(families).to_numpy()

    out = {}
    train = splits["train"]
    for size in sizes:
        # Sample windows spread evenly across the split, not the first N. The
        # first 30% of this dataset contains no attacks at all (a separate
        # capture session), so taking a prefix would report attack statistics
        # from a region that is unrepresentative of the split as a whole.
        all_windows = list(fixed_count_windows(train, size))
        if len(all_windows) > 200:
            idx = np.linspace(0, len(all_windows) - 1, 200).astype(int)
            sampled = [all_windows[i] for i in idx]
        else:
            sampled = all_windows

        stats = []
        for lo, hi in sampled:
            snap = build_snapshot(
                src_all[lo:hi], dst_all[lo:hi],
                feats["continuous"][lo:hi], feats["categorical"][lo:hi],
                y_all[lo:hi], ymc_all[lo:hi],
            )
            stats.append(snapshot_stats(snap))

        out[str(size)] = {
            "n_windows_total": count_windows(train, size),
            "n_windows_measured": len(stats),
            "nodes": summarise([s["n_nodes"] for s in stats]),
            "unique_pairs": summarise([s["n_unique_pairs"] for s in stats]),
            "density": summarise([s["density"] for s in stats]),
            # max_fanout saturates at 10 and is therefore uninformative; the
            # spread across hosts is what message passing can exploit.
            "max_fanout": summarise([s["max_fanout"] for s in stats]),
            "mean_fanout": summarise([s["mean_fanout"] for s in stats]),
            "fanout_std_within_window": summarise([s["fanout_std"] for s in stats]),
            "n_active_senders": summarise([s["n_active_senders"] for s in stats]),
            "flows_per_node_std": summarise([s["flows_per_node_std"] for s in stats]),
            "attack_rate": summarise([s["attack_rate"] for s in stats]),
            "windows_with_no_attack": sum(1 for s in stats if s["n_attack_edges"] == 0),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "preprocess.yaml")
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    ds_cfg = yaml.safe_load((REPO_ROOT / cfg["dataset_config"]).read_text())
    stem = Path(ds_cfg["source"]["filename"]).stem
    path = REPO_ROOT / ds_cfg["paths"]["interim"] / f"{stem}.parquet"

    print(f"Loading {path.name} ...")
    df = pd.read_parquet(path)
    print(f"  {len(df):,} flows\n")

    # ---- 1. schema -------------------------------------------------------
    schema = build_schema(cfg, df, REPO_ROOT)
    print("Feature schema")
    print(f"  dropped (shortcuts):    {schema.dropped['shortcuts']}")
    print(f"  dropped (uninformative):{schema.dropped['uninformative']}")
    print(f"  continuous:  {len(schema.continuous)}  "
          f"({len(schema.log_transform)} log-transformed)")
    print(f"  categorical: {len(schema.categorical)} + "
          f"{len(schema.ports)} port buckets + "
          f"{len(schema.conditional)} presence indicators\n")

    # ---- 2. splits -------------------------------------------------------
    sp = cfg["split"]
    splits = contiguous_split(len(df), sp["train"], sp["val"], sp["test"], sp["purge_gap"])
    assert_no_overlap(splits)
    print("Splits (contiguous, purge gap "
          f"{sp['purge_gap']:,} rows either side of each boundary)")
    for s in splits.values():
        rows = df.iloc[s.start:s.stop]
        print(f"  {s.name:<6} rows {s.start:>9,}-{s.stop:>9,}  "
              f"n={len(s):>9,}  attack rate={rows[LABEL_BINARY].mean():.4%}")
    purged = len(df) - sum(len(s) for s in splits.values())
    print(f"  purged: {purged:,} rows ({purged / len(df):.2%})\n")

    # ---- 3. transforms, fitted on TRAIN ONLY ------------------------------
    print(f"Fitting transforms on {sp['fit_on']} only ...")
    pipeline = FeaturePipeline(
        schema,
        cfg["features"]["categorical_min_frequency"],
        cfg["features"]["categorical_max_cardinality"],
    )
    pipeline.fit(df.iloc[splits[sp["fit_on"]].start:splits[sp["fit_on"]].stop])
    feats = pipeline.transform(df)
    print(f"  continuous  {feats['continuous'].shape}")
    print(f"  categorical {feats['categorical'].shape}  "
          f"cardinalities {pipeline.cardinalities()}\n")

    families = {v: i for i, v in enumerate(sorted(df[LABEL_MULTICLASS].unique()))}

    # ---- 4. build a snapshot and verify it preserves the source ------------
    g = cfg["graph"]
    node_map = GlobalNodeMap() if g["global_node_map"] else None
    lo, hi = next(fixed_count_windows(splits["train"], g["window_size"]))
    snap = build_snapshot(
        df["IPV4_SRC_ADDR"].to_numpy()[lo:hi], df["IPV4_DST_ADDR"].to_numpy()[lo:hi],
        feats["continuous"][lo:hi], feats["categorical"][lo:hi],
        df[LABEL_BINARY].to_numpy()[lo:hi],
        df[LABEL_MULTICLASS].map(families).to_numpy()[lo:hi],
        global_map=node_map, window_index=0,
    )
    src_rows = df.iloc[lo:hi]
    checks = {
        "edge_count_equals_flow_count": int(snap.edge_index.shape[1]) == (hi - lo),
        "node_count_equals_unique_ips": int(snap.num_nodes) == len(
            set(src_rows.IPV4_SRC_ADDR) | set(src_rows.IPV4_DST_ADDR)),
        "attack_count_preserved": int(snap.y.sum()) == int(src_rows[LABEL_BINARY].sum()),
        "node_indices_are_bijective": int(snap.edge_index.max()) == int(snap.num_nodes) - 1,
        "no_nan_in_features": bool(np.isfinite(snap.edge_attr.numpy()).all()),
    }
    print("Gate: snapshot preserves the source rows exactly")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    if not all(checks.values()):
        raise SystemExit("\nGate failed -- not writing artefacts.")
    print(f"\n  {snapshot_stats(snap)}\n")

    # ---- 5. window-size sweep --------------------------------------------
    sweep = {}
    if not args.no_sweep:
        print("Sweeping window size ...")
        sweep = sweep_window_sizes(df, feats, splits, g["window_sizes_to_sweep"], families)
        print(f"  {'window':>7} {'nodes':>7} {'pairs':>7} {'senders':>8} "
              f"{'mean fo':>8} {'fo spread':>10} {'fo cv':>7} {'no-attack':>12}")
        for size, s in sweep.items():
            print(f"  {size:>7} {s['nodes']['mean']:>7.1f} "
                  f"{s['unique_pairs']['mean']:>7.1f} "
                  f"{s['n_active_senders']['mean']:>8.1f} "
                  f"{s['mean_fanout']['mean']:>8.2f} "
                  f"{s['fanout_std_within_window']['mean']:>10.2f} "
                  f"{str(s['mean_fanout']['cv']):>7} "
                  f"{s['windows_with_no_attack']:>5}/{s['n_windows_measured']}")
        print("  'fo spread' = std of fan-out across hosts within a window --")
        print("  the variation message passing can actually exploit.\n")

    # ---- 6. persist -------------------------------------------------------
    out_dir = REPO_ROOT / cfg["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline.save(out_dir / "pipeline.json")
    np.savez_compressed(
        out_dir / "features.npz",
        continuous=feats["continuous"], categorical=feats["categorical"],
    )
    (out_dir / "splits.json").write_text(json.dumps(
        {k: v.as_dict() for k, v in splits.items()}, indent=2))
    (out_dir / "attack_families.json").write_text(json.dumps(families, indent=2))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "n_rows": int(len(df)),
        "schema": schema.summary(),
        "splits": {k: v.as_dict() for k, v in splits.items()},
        "split_attack_rates": {
            k: round(float(df.iloc[v.start:v.stop][LABEL_BINARY].mean()), 6)
            for k, v in splits.items()
        },
        "rows_purged": int(purged),
        "categorical_cardinalities": dict(
            zip(pipeline.categorical_names_, pipeline.cardinalities())),
        "gate_checks": checks,
        "window_sweep": sweep,
        "attack_families": families,
    }
    stats_path = REPO_ROOT / cfg["output"]["stats"]
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(report, indent=2))

    print(f"pipeline  -> {(out_dir / 'pipeline.json').relative_to(REPO_ROOT)}")
    print(f"features  -> {(out_dir / 'features.npz').relative_to(REPO_ROOT)}")
    print(f"stats     -> {stats_path.relative_to(REPO_ROOT)}\n")


if __name__ == "__main__":
    main()
