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
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnids.data.splits import Split  # noqa: E402
from gnnids.eval.metrics import aggregate_seeds  # noqa: E402
from gnnids.graph.dataset import SnapshotDataset  # noqa: E402
from gnnids.graph.inputs import load_graph_inputs  # noqa: E402
from gnnids.training.loop import pick_device, train_one  # noqa: E402


def _run_seeds_in_subprocesses(args) -> int:
    """Re-invoke this script once per seed, then merge the partial results."""
    import subprocess

    cfg = yaml.safe_load(args.config.read_text())
    pre_cfg = yaml.safe_load((REPO_ROOT / cfg["preprocess_config"]).read_text())
    ds_cfg = yaml.safe_load((REPO_ROOT / pre_cfg["dataset_config"]).read_text())
    n_seeds = args.seeds or cfg["train"]["n_seeds"]
    out_dir = REPO_ROOT / cfg["output"]["metrics"]
    out_dir.mkdir(parents=True, exist_ok=True)

    merged, failed, completed = {}, [], []
    for seed in range(n_seeds):
        cmd = [sys.executable, __file__, "--config", str(args.config),
               "--single-seed", str(seed)]
        if args.ablation:
            cmd += ["--ablation"] + list(args.ablation)
        if args.max_windows:
            cmd += ["--max-windows", str(args.max_windows)]
        print(f"\n===== seed {seed} (fresh process) =====", flush=True)
        if subprocess.run(cmd).returncode != 0:
            print(f"  seed {seed} FAILED -- continuing with the rest", flush=True)
            failed.append(seed)
            continue
        part = out_dir / f"partial_{ds_cfg['name']}_seed{seed}.json"
        if part.exists():
            for ab, payload in json.loads(part.read_text())["results"].items():
                merged.setdefault(ab, {"spec": payload["spec"], "runs": []})
                merged[ab]["runs"].extend(payload["runs"])
            # C19: the partial is NOT deleted here. Until this fix, completed
            # seeds lived only in this process's memory and the merged file was
            # written after the last seed -- so a parent killed in hour 9 of a
            # 9-hour campaign lost all nine hours of finished work. Partials are
            # cheap; a lost campaign is not. They are cleaned up only after the
            # merged file is successfully written, at the end of this function.
            completed.append(part)

    target_prev = cfg["eval"]["target_prevalence"]
    key = f"at_{target_prev:.0%}"
    for ab, payload in merged.items():
        payload["n_seeds"] = len(payload["runs"])
        payload["aggregate"] = aggregate_seeds([r[key] for r in payload["runs"]])

    comparators = load_comparators(ds_cfg["name"], target_prev)
    out_path = out_dir / f"gnn_{ds_cfg['name']}.json"
    out_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": ds_cfg["name"], "config": cfg,
        "seeds_failed": failed,
        "phase3_comparators": comparators,
        "results": merged,
    }, indent=2))

    # Only now is the work durable, so the partials can go.
    for part in completed:
        part.unlink(missing_ok=True)

    print()
    print_comparison(comparators, merged,
                     ("channel1_only", "channel2_only", "full"), target_prev)
    if "full" in merged and "channel1_only" in merged:
        gap = (merged["full"]["aggregate"]["pr_auc"]["mean"]
               - merged["channel1_only"]["aggregate"]["pr_auc"]["mean"])
        print(f"\n  topology gain (full - channel1_only): {gap:+.4f}")
        print("  This is the V2 measurement.")
    if failed:
        print(f"\n  WARNING: seeds {failed} failed and are excluded.")
    print(f"\nwritten -> {out_path.relative_to(REPO_ROOT)}\n")
    return 0


BASELINE_SCHEMA = "dual-prevalence-v1"


def load_comparators(dataset_name: str, target_prevalence: float) -> dict:
    """Phase 3 baselines, read from disk rather than hardcoded (C17).

    Until 2026-08-25 this file carried `baseline_to_beat = 0.9552 / 0.8963` as a
    literal in five places. That figure came from `audit_dataset.py`, which
    resampled *rows* to 4% and trained there, while the GNN trains at native
    prevalence and subsamples its *scores* (D23) -- roughly 16x more positives.
    It was never a comparison; it was a decoration, and everything of the form
    "the GNN beat the baseline" rested on it.

    Reading the file has a second benefit: a baseline that has not been re-run
    on the current reporting path is *absent* rather than silently wrong, so a
    missing comparator shows up as a missing row.
    """
    path = REPO_ROOT / "results" / "metrics" / "baselines" / f"baselines_{dataset_name}.json"
    if not path.exists():
        return {"error": f"no baselines at {path.relative_to(REPO_ROOT)}", "entries": {}}

    results = json.loads(path.read_text()).get("results", {})
    entries, stale = {}, []
    for key, payload in results.items():
        if payload.get("schema") != BASELINE_SCHEMA:
            stale.append(key)
            continue
        if payload.get("reported_at_prevalence") != target_prevalence:
            stale.append(key)
            continue
        agg = payload["aggregate"]
        entries[key] = {
            "pr_auc": round(agg["pr_auc"]["mean"], 5),
            "pr_auc_std": round(agg["pr_auc"]["std"], 5),
            "f1": round(agg["f1"]["mean"], 5),
            "f1_std": round(agg["f1"]["std"], 5),
            "fpr_at_95_recall": round(agg["fpr_at_95_recall"]["mean"], 5),
            "n_seeds": payload["n_seeds"],
        }
    return {
        "source": str(path.relative_to(REPO_ROOT)),
        "reported_at_prevalence": target_prevalence,
        "entries": entries,
        "excluded_not_on_this_reporting_path": sorted(stale),
    }


def print_comparison(comparators: dict, ablations: dict, order, target_prevalence: float) -> None:
    """One table, one prevalence, every number carrying its own std."""
    print("=" * 72)
    print(f"  PHASE 4 -- test metrics at {target_prevalence:.0%} prevalence "
          f"(PR-AUC floor {target_prevalence:.2f})")
    print("=" * 72)
    print(f"  {'model':<22} {'PR-AUC':>18} {'F1':>18} {'FPR@95':>9}")
    for key, c in sorted(comparators.get("entries", {}).items()):
        print(f"  {'P3 ' + key:<22} {c['pr_auc']:>9.4f} +/-{c['pr_auc_std']:<7.4f} "
              f"{c['f1']:>9.4f} +/-{c['f1_std']:<7.4f} {c['fpr_at_95_recall']:>9.5f}")
    if not comparators.get("entries"):
        print("  (no Phase 3 baseline on this reporting path -- run "
              "scripts/train_baselines.py)")
    for a in order:
        if a not in ablations:
            continue
        g = ablations[a]["aggregate"]
        print(f"  {'GNN ' + a:<22} {g['pr_auc']['mean']:>9.4f} +/-{g['pr_auc']['std']:<7.4f} "
              f"{g['f1']['mean']:>9.4f} +/-{g['f1']['std']:<7.4f} "
              f"{g['fpr_at_95_recall']['mean']:>9.5f}")
    for key in comparators.get("excluded_not_on_this_reporting_path", []):
        print(f"  excluded (not on this reporting path): {key}")

    # State the verdict rather than leaving it to be eyeballed, and state it
    # against the STRONGEST baseline. Picking the weakest is how a result gets
    # overclaimed without anyone lying.
    best = max(comparators.get("entries", {}).items(),
               key=lambda kv: kv[1]["pr_auc"], default=None)
    if best and "full" in ablations:
        name, c = best
        g = ablations["full"]["aggregate"]
        d_pr = g["pr_auc"]["mean"] - c["pr_auc"]
        d_f1 = g["f1"]["mean"] - c["f1"]
        print(f"\n  vs strongest baseline ({name}): "
              f"PR-AUC {d_pr:+.4f}, F1 {d_f1:+.4f}")
        if d_pr > 0 and d_f1 < 0:
            print("  Split verdict: better ranking, worse operating point. Not")
            print("  'the GNN beat the baseline' -- say both halves.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "gnn.yaml")
    ap.add_argument("--ablation", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to prove the code path: few windows, few epochs")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--single-seed", type=int, default=None,
                    help="internal: run exactly one seed and write a partial file")
    ap.add_argument("--in-process", action="store_true",
                    help="run seeds in this process instead of spawning one each")
    args = ap.parse_args()

    # One seed per process unless told otherwise. PyTorch's Metal allocator
    # caches freed blocks and our graphs are variable-sized, so running several
    # seeds in one process fragments it: the heaviest ablation degraded
    # 51 -> 77 -> 206 s/epoch across three identical seeds, and a later attempt
    # was OOM-killed outright. A fresh process per seed removes the whole class
    # of problem, and means one seed failing no longer takes the campaign with
    # it. Costs a few seconds of startup each.
    if (args.single_seed is None and not args.in_process and not args.smoke
            and (args.seeds or 3) > 1):
        raise SystemExit(_run_seeds_in_subprocesses(args))

    cfg = yaml.safe_load(args.config.read_text())
    pre_cfg = yaml.safe_load((REPO_ROOT / cfg["preprocess_config"]).read_text())
    ds_cfg = yaml.safe_load((REPO_ROOT / pre_cfg["dataset_config"]).read_text())
    proc = REPO_ROOT / pre_cfg["output"]["dir"]
    window = pre_cfg["graph"]["window_size"]
    n_seeds = args.seeds or cfg["train"]["n_seeds"]
    max_windows = args.max_windows
    if args.smoke:
        # Seconds, not hours. Proves every code path -- construction, training,
        # early stopping, inference, prevalence adjustment, checkpointing --
        # without waiting. A component that has never executed is not built.
        cfg["train"]["max_epochs"] = 2
        cfg["train"]["patience"] = 99
        n_seeds, max_windows = 1, max_windows or 12
        print("SMOKE RUN -- 12 windows, 2 epochs, 1 seed. Results are meaningless.")

    print(f"dataset: {ds_cfg['name']}   window: {window:,}")
    inputs = load_graph_inputs(proc, ds_cfg["name"])
    src, dst = inputs.src, inputs.dst
    edge_feats = inputs.edge_features
    y, ymc = inputs.y, inputs.y_multiclass
    splits_raw, families = inputs.splits, inputs.families

    datasets = {"families": families}
    for name, s in splits_raw.items():
        datasets[name] = SnapshotDataset(
            src, dst, edge_feats, y, ymc,
            Split(name, s["start"], s["stop"]), window,
            max_windows=max_windows)
        print(f"  {name:<6} {len(datasets[name]):>5} windows  "
              f"{datasets[name].n_edges:>10,} edges  "
              f"attack {datasets[name].attack_rate():.2%}")

    device = pick_device(cfg["train"]["device"])
    edge_dim = edge_feats.shape[1]
    n_classes = len(families)
    print(f"\ndevice: {device}   edge features: {edge_dim}   classes: {n_classes}")
    _c = load_comparators(ds_cfg["name"], cfg["eval"]["target_prevalence"])
    _best = max(_c["entries"].items(), key=lambda kv: kv[1]["pr_auc"], default=None)
    if _best:
        print(f"target: beat {_best[0]} PR-AUC {_best[1]['pr_auc']:.4f} / "
              f"F1 {_best[1]['f1']:.4f} at "
              f"{cfg['eval']['target_prevalence']:.0%} prevalence "
              f"({_c['source']})\n")
    else:
        print("target: no Phase 3 baseline on this reporting path -- run "
              "scripts/train_baselines.py first\n")

    to_run = args.ablation or list(cfg["ablations"].keys() - {"gnn_layer_sweep"})
    order = [a for a in ("channel1_only", "channel2_only", "full") if a in to_run]

    results = {}
    for ablation in order:
        spec = cfg["ablations"][ablation]
        print(f"--- {ablation} ---")
        runs = []
        seeds = [args.single_seed] if args.single_seed is not None else range(n_seeds)
        for seed in seeds:
            t0 = time.time()
            ckpt = (None if args.smoke else
                    REPO_ROOT / cfg["output"]["checkpoints"] /
                    f"{ds_cfg['name']}_{ablation}_seed{seed}.pt")
            r = train_one(cfg, datasets, edge_dim, n_classes, spec, seed, device, ckpt)
            r["fit_seconds"] = round(time.time() - t0, 1)
            runs.append(r)
            adj = r[f"at_{cfg['eval']['target_prevalence']:.0%}"]
            es = r["epoch_seconds"]
            print(f"  seed {seed}  PR-AUC {adj['pr_auc']:.4f} (base "
                  f"{adj['prevalence']:.3f})  F1 {adj['f1']:.4f}  "
                  f"recall {adj['recall']:.4f}  {r['epochs_run']}ep "
                  f"({r['fit_seconds']:.0f}s, epoch {es[0]:.1f}->{es[-1]:.1f}s)")
        key = f"at_{cfg['eval']['target_prevalence']:.0%}"
        results[ablation] = {
            "spec": spec, "n_seeds": len(runs), "runs": runs,
            "aggregate": aggregate_seeds([r[key] for r in runs]),
        }
        agg = results[ablation]["aggregate"]
        print(f"  => PR-AUC {agg['pr_auc']['mean']:.4f} +/- {agg['pr_auc']['std']:.4f}"
              f"   F1 {agg['f1']['mean']:.4f}\n")

    out_dir = REPO_ROOT / cfg["output"]["metrics"]
    out_dir.mkdir(parents=True, exist_ok=True)
    # Smoke results are meaningless by construction and must never overwrite a
    # real run. The first smoke test clobbered the Phase 4 results file; they
    # were only recoverable because they had been committed.
    out_name = (f"smoke_gnn_{ds_cfg['name']}.json" if args.smoke
                else f"partial_{ds_cfg['name']}_seed{args.single_seed}.json"
                if args.single_seed is not None
                else f"gnn_{ds_cfg['name']}.json")
    comparators = load_comparators(ds_cfg["name"], cfg["eval"]["target_prevalence"])
    (out_dir / out_name).write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": ds_cfg["name"], "config": cfg,
        "edge_dim": edge_dim, "window_size": window,
        "phase3_comparators": comparators,
        "results": results,
    }, indent=2))

    print_comparison(comparators, results, order, cfg["eval"]["target_prevalence"])
    if "full" in results and "channel1_only" in results:
        gap = (results["full"]["aggregate"]["pr_auc"]["mean"]
               - results["channel1_only"]["aggregate"]["pr_auc"]["mean"])
        print(f"\n  topology gain (full - channel1_only): {gap:+.4f}")
        print("  This is the V2 measurement.")
    print(f"\nwritten -> {(out_dir / out_name).relative_to(REPO_ROOT)}\n")


if __name__ == "__main__":
    main()
