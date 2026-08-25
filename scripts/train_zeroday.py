"""Phase 5: leave-one-attack-out -- the zero-day measurement (D4, Protocol 2).

For each attack family in turn: remove it entirely from training and validation,
train, then test on benign traffic plus that family alone. Recall on the held-out
family is the fraction of a genuinely unseen attack the model catches.

This is the claim the project is built on, and it is the one most often measured
badly in the literature -- a random split of a dataset containing every family
is not a zero-day result. See [[Zero-Day Evaluation Protocol]].

**Read the per-family numbers, not the mean.** The families are not equally
novel relative to each other, so the *pattern* of which ones transfer says what
the model actually learned; the mean says almost nothing. And on NF-ToN-IoT-v2
three families are thin enough (`ransomware` 3,425, `mitm` 7,723, `backdoor`
16,809) that their recall carries error bars several times wider than
`scanning`'s. They are reported with their positive counts for that reason.

Usage:
    python scripts/train_zeroday.py --smoke               # ~seconds, proves the path
    python scripts/train_zeroday.py --families scanning   # one family
    python scripts/train_zeroday.py --seeds 3             # the real campaign
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

from gnnids.data.splits import Split  # noqa: E402
from gnnids.eval.holdout import assert_holdout_is_clean, holdout_plan  # noqa: E402
from gnnids.eval.metrics import aggregate_seeds  # noqa: E402
from gnnids.graph.dataset import SnapshotDataset  # noqa: E402
from gnnids.graph.inputs import load_graph_inputs  # noqa: E402
from gnnids.training.loop import pick_device, train_one  # noqa: E402


def build_holdout_datasets(inputs, plan, window, max_windows=None) -> dict:
    """Datasets for one held-out family, filtered inside each split."""
    datasets = {"families": inputs.families}
    for name, s in inputs.splits.items():
        datasets[name] = SnapshotDataset(
            inputs.src, inputs.dst, inputs.edge_features,
            inputs.y, inputs.y_multiclass,
            Split(name, s["start"], s["stop"]), window,
            max_windows=max_windows,
            row_index=plan["splits"][name]["row_index"],
        )
    return datasets


def held_out_recall(run: dict, family: str, target_key: str) -> dict | None:
    """Pull the held-out family's row out of the per-family breakdown.

    Returned with its positive count attached, never bare. A recall computed
    over 40 flows and one computed over 750,000 are not the same measurement
    and must not appear in a table as though they were.
    """
    for block in (run.get(target_key, {}), run.get("native", {})):
        per_family = block.get("per_family") or block.get("recall_by_family")
        if isinstance(per_family, dict) and family in per_family:
            entry = per_family[family]
            if isinstance(entry, dict):
                return entry
            return {"recall": entry}
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "zeroday.yaml")
    ap.add_argument("--families", nargs="+", default=None,
                    help="families to hold out (default: every attack family)")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="seconds-long proof the code path runs; numbers are meaningless")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    gnn_cfg = yaml.safe_load((REPO_ROOT / cfg["gnn_config"]).read_text())
    pre_cfg = yaml.safe_load((REPO_ROOT / gnn_cfg["preprocess_config"]).read_text())
    ds_cfg = yaml.safe_load((REPO_ROOT / pre_cfg["dataset_config"]).read_text())

    n_seeds = args.seeds or cfg["n_seeds"]
    window = pre_cfg["graph"]["window_size"]
    max_windows = None
    if args.smoke:
        n_seeds = 1
        window = cfg["smoke"]["window_size"]
        max_windows = cfg["smoke"]["max_windows"]
        gnn_cfg["train"] = {**gnn_cfg["train"], **cfg["smoke"]["train"]}

    print(f"dataset: {ds_cfg['name']}   window {window:,}"
          f"{'   [SMOKE -- numbers are meaningless]' if args.smoke else ''}")
    inputs = load_graph_inputs(REPO_ROOT / pre_cfg["output"]["dir"], ds_cfg["name"])

    attack_families = [f for f in inputs.families if f != cfg["benign_class_name"]]
    to_run = args.families or attack_families
    unknown = [f for f in to_run if f not in inputs.families]
    if unknown:
        raise SystemExit(f"unknown families {unknown}; have {attack_families}")

    device = pick_device(gnn_cfg["train"]["device"])
    target_key = f"at_{gnn_cfg['eval']['target_prevalence']:.0%}"
    ablation = gnn_cfg["ablations"][cfg["ablation"]]
    print(f"device: {device}   ablation: {cfg['ablation']}   "
          f"{len(to_run)} families x {n_seeds} seed(s)\n")

    out_dir = REPO_ROOT / cfg["output"]["metrics"]
    out_dir.mkdir(parents=True, exist_ok=True)
    results, skipped = {}, {}

    for family in to_run:
        plan = holdout_plan(inputs.y_multiclass, inputs.splits,
                            inputs.families, family, inputs.families[cfg["benign_class_name"]])
        # Asserted rather than inspected: a leak of the held-out family into
        # training produces plausible metrics, not an error.
        try:
            assert_holdout_is_clean(plan)
        except AssertionError as e:
            print(f"--- {family} --- SKIPPED: {e}\n")
            skipped[family] = str(e)
            continue

        tr, te = plan["splits"]["train"], plan["splits"]["test"]
        print(f"--- holding out {family} ---")
        print(f"  train {tr['n_rows']:>9,} rows ({tr['removed']:,} removed)   "
              f"test {te['n_rows']:>9,} rows, {te['n_held_out']:,} held-out "
              f"positives ({te['n_held_out'] / max(te['n_rows'], 1):.2%})")

        # A hard gate, not a warning. Training a model to compute a recall over
        # three flows costs an hour and produces a number that would be quoted.
        if te["n_held_out"] < cfg["skip_below_test_positives"]:
            why = (f"only {te['n_held_out']:,} held-out flows reach the test "
                   f"split (need {cfg['skip_below_test_positives']:,}); "
                   f"{tr['removed']:,} were removed from train, so this family "
                   f"is concentrated in one split and cannot be held out here")
            print(f"  SKIPPED -- {why}\n")
            skipped[family] = why
            continue
        if te["n_held_out"] < cfg["thin_below_test_positives"]:
            print(f"  THIN: under {cfg['thin_below_test_positives']:,} positives "
                  f"-- recall must be quoted with its count, never bare")

        datasets = build_holdout_datasets(inputs, plan, window, max_windows)
        runs = []
        for seed in range(n_seeds):
            run = train_one(gnn_cfg, datasets, inputs.edge_dim, inputs.n_classes,
                            ablation, seed, device)
            run["held_out_test_positives"] = te["n_held_out"]
            runs.append(run)
            adj = run[target_key]

            def fmt(v, spec=".4f"):
                # Metrics come back None when a subsample has no positives at
                # all. Printing "None" is right; formatting it as 0.0000 would
                # put a fabricated zero in a results table.
                return format(v, spec) if isinstance(v, (int, float)) else "  n/a"
            print(f"  seed {seed}  recall {fmt(adj['recall'])}  "
                  f"PR-AUC {fmt(adj['pr_auc'])}  "
                  f"FPR@95 {fmt(adj['fpr_at_95_recall'], '.5f')}  "
                  f"({run['epochs_run']} epochs)")

        results[family] = {
            "held_out": family,
            "n_seeds": n_seeds,
            "test_positives": te["n_held_out"],
            "train_rows": tr["n_rows"],
            "train_rows_removed": tr["removed"],
            "aggregate": aggregate_seeds([r[target_key] for r in runs]),
            "runs": runs,
        }
        agg = results[family]["aggregate"]
        print(f"  => recall {agg['recall']['mean']:.4f} +/- {agg['recall']['std']:.4f} "
              f"on {te['n_held_out']:,} unseen flows\n")

    name = f"{'smoke_' if args.smoke else ''}zeroday_{ds_cfg['name']}.json"
    (out_dir / name).write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": ds_cfg["name"],
        "protocol": "leave-one-attack-out (D4, Protocol 2)",
        "smoke": args.smoke,
        "ablation": cfg["ablation"],
        "reported_at_prevalence": gnn_cfg["eval"]["target_prevalence"],
        "config": cfg, "gnn_config": gnn_cfg,
        "skipped": skipped,
        "results": results,
    }, indent=2))

    print("=" * 78)
    print(f"  PHASE 5 -- leave-one-attack-out recall on the UNSEEN family"
          f"{' [SMOKE]' if args.smoke else ''}")
    print("=" * 78)
    print(f"  {'held-out family':<16} {'recall':>18} {'PR-AUC':>10} {'test +ve':>12}")

    def agg_fmt(a: dict, key: str, spec: str = ".4f", with_std: bool = False) -> str:
        """Absent means absent. `aggregate_seeds` drops a metric entirely when
        every seed returned None -- which happens when a subsample contains no
        positives -- and a table that printed 0.0000 there would be stating a
        measured zero that was never measured."""
        stat = a.get(key)
        if not isinstance(stat, dict) or not isinstance(stat.get("mean"), (int, float)):
            return f"{'n/a':>9}" + (f" {'':<8}" if with_std else "")
        if with_std:
            return f"{stat['mean']:>9{spec}} +/-{stat['std']:<7{spec}}"
        return f"{stat['mean']:>9{spec}}"

    for family, r in sorted(results.items(),
                            key=lambda kv: -kv[1]["test_positives"]):
        a = r["aggregate"]
        thin = "  (thin)" if r["test_positives"] < cfg["thin_below_test_positives"] else ""
        print(f"  {family:<16} {agg_fmt(a, 'recall', with_std=True)} "
              f"{agg_fmt(a, 'pr_auc')} {r['test_positives']:>12,}{thin}")
    if results:
        print("\n  The MEAN of this column is not the finding. Which families "
              "transfer\n  and which do not is what says what the model learned.")
    for family, why in skipped.items():
        print(f"  skipped {family}: {why}")
    print(f"\nwritten -> {(out_dir / name).relative_to(REPO_ROOT)}\n")


if __name__ == "__main__":
    main()
