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

**One process per run, and every run saved as it finishes.** The campaign is
8 families x 3 seeds, about five hours. Until 2026-10-03 it trained all 24 in
one process and wrote a single file at the end -- the U1 pattern (one process
degraded 51 -> 77 -> 206 s/epoch across seeds) and the C19 pattern (a crash in
hour four loses everything). Each (family, seed) is now its own process writing
its own partial, the merged file is rewritten after every family, and
re-running the same command skips whatever is already done.

Usage:
    python scripts/train_zeroday.py --smoke               # ~seconds, proves the path
    python scripts/train_zeroday.py --families scanning   # one family
    python scripts/train_zeroday.py --seeds 3             # the real campaign
    python scripts/train_zeroday.py --seeds 3             # again: resumes
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnids.data.splits import Split  # noqa: E402
from gnnids.eval.holdout import assert_holdout_is_clean, holdout_plan  # noqa: E402
from gnnids.eval.metrics import aggregate_seeds  # noqa: E402
from gnnids.graph.dataset import SnapshotDataset  # noqa: E402
from gnnids.graph.inputs import load_graph_inputs  # noqa: E402
from gnnids.training.campaign import fingerprint, read_if_matching  # noqa: E402
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


def load_context(args) -> dict:
    """Configs and run parameters, shared by the parent and every child."""
    cfg = yaml.safe_load(args.config.read_text())
    gnn_cfg = yaml.safe_load((REPO_ROOT / cfg["gnn_config"]).read_text())
    pre_cfg = yaml.safe_load((REPO_ROOT / gnn_cfg["preprocess_config"]).read_text())
    ds_cfg = yaml.safe_load((REPO_ROOT / pre_cfg["dataset_config"]).read_text())

    ctx = {
        "cfg": cfg, "gnn_cfg": gnn_cfg, "pre_cfg": pre_cfg, "ds_cfg": ds_cfg,
        "n_seeds": args.seeds or cfg["n_seeds"],
        "window": pre_cfg["graph"]["window_size"],
        # Absent in the real config. A cap makes the per-process path testable
        # in seconds without going through --smoke's in-process shortcut.
        "max_windows": cfg.get("max_windows"),
    }
    if args.smoke:
        ctx["n_seeds"] = 1
        ctx["window"] = cfg["smoke"]["window_size"]
        ctx["max_windows"] = cfg["smoke"]["max_windows"]
        gnn_cfg["train"] = {**gnn_cfg["train"], **cfg["smoke"]["train"]}
    ctx["target_key"] = f"at_{gnn_cfg['eval']['target_prevalence']:.0%}"
    # What makes two zero-day runs the same experiment: the model and training
    # config, plus the protocol's own gates and the ablation it trains.
    ctx["fingerprint"] = fingerprint(gnn_cfg, extra={
        "protocol": "leave-one-attack-out", "ablation": cfg["ablation"],
        "skip_below": cfg["skip_below_test_positives"],
        "window": ctx["window"], "max_windows": ctx["max_windows"]})
    ctx["out_dir"] = REPO_ROOT / cfg["output"]["metrics"]
    return ctx


def plan_family(ctx: dict, inputs, family: str, quiet: bool = False):
    """The hold-out plan for one family, or the reason it cannot be measured.

    Returns (plan, None) or (None, reason). Both gates are hard: a leak of the
    held-out family into training produces plausible metrics rather than an
    error, and a recall over three flows costs an hour and would be quoted.
    """
    cfg = ctx["cfg"]
    say = (lambda *a: None) if quiet else print
    plan = holdout_plan(inputs.y_multiclass, inputs.splits, inputs.families,
                        family, inputs.families[cfg["benign_class_name"]])
    try:
        assert_holdout_is_clean(plan)
    except AssertionError as e:
        say(f"--- {family} --- SKIPPED: {e}\n")
        return None, str(e)

    tr, te = plan["splits"]["train"], plan["splits"]["test"]
    say(f"--- holding out {family} ---")
    say(f"  train {tr['n_rows']:>9,} rows ({tr['removed']:,} removed)   "
        f"test {te['n_rows']:>9,} rows, {te['n_held_out']:,} held-out "
        f"positives ({te['n_held_out'] / max(te['n_rows'], 1):.2%})")

    if te["n_held_out"] < cfg["skip_below_test_positives"]:
        why = (f"only {te['n_held_out']:,} held-out flows reach the test "
               f"split (need {cfg['skip_below_test_positives']:,}); "
               f"{tr['removed']:,} were removed from train, so this family "
               f"is concentrated in one split and cannot be held out here")
        say(f"  SKIPPED -- {why}\n")
        return None, why
    if te["n_held_out"] < cfg["thin_below_test_positives"]:
        say(f"  THIN: under {cfg['thin_below_test_positives']:,} positives "
            f"-- recall must be quoted with its count, never bare")
    return plan, None


def train_family_seed(ctx: dict, inputs, plan: dict, seed: int, device) -> dict:
    cfg, gnn_cfg = ctx["cfg"], ctx["gnn_cfg"]
    datasets = build_holdout_datasets(inputs, plan, ctx["window"], ctx["max_windows"])
    run = train_one(gnn_cfg, datasets, inputs.edge_dim, inputs.n_classes,
                    gnn_cfg["ablations"][cfg["ablation"]], seed, device)
    run["held_out_test_positives"] = plan["splits"]["test"]["n_held_out"]
    return run


def describe(run: dict, seed: int, target_key: str) -> str:
    def fmt(v, spec=".4f"):
        # Metrics come back None when a subsample has no positives at all.
        # Printing "n/a" is right; formatting it as 0.0000 would put a
        # fabricated zero in a results table.
        return format(v, spec) if isinstance(v, (int, float)) else "  n/a"
    adj = run[target_key]
    return (f"  seed {seed}  recall {fmt(adj['recall'])}  "
            f"PR-AUC {fmt(adj['pr_auc'])}  "
            f"FPR@95 {fmt(adj['fpr_at_95_recall'], '.5f')}  "
            f"({run['epochs_run']} epochs)")


def mistaken_for(runs: list[dict], family: str) -> dict:
    """When the unseen family is scored by the attack-family head, what is it
    called? Summed over seeds, as shares.

    The head has never seen this family, so it cannot be right -- but *which*
    wrong answer it gives is what an evidence pack would tell the language
    model about a genuinely novel attack, and it says which known family the
    model thinks the unseen one resembles.
    """
    total: dict[str, int] = {}
    for run in runs:
        row = (run.get("multiclass") or {}).get("confusion", {}).get(family, {})
        for name, n in row.items():
            total[name] = total.get(name, 0) + n
    n = sum(total.values())
    return ({k: round(v / n, 4) for k, v in sorted(total.items(), key=lambda kv: -kv[1])}
            if n else {})


def family_entry(ctx: dict, plan: dict, family: str, runs: list[dict]) -> dict:
    tr, te = plan["splits"]["train"], plan["splits"]["test"]
    runs = sorted(runs, key=lambda r: r.get("seed", 0))
    return {
        "held_out": family, "n_seeds": len(runs),
        "test_positives": te["n_held_out"],
        "train_rows": tr["n_rows"], "train_rows_removed": tr["removed"],
        "aggregate": aggregate_seeds([r[ctx["target_key"]] for r in runs]),
        "held_out_mistaken_for": mistaken_for(runs, family),
        "runs": runs,
    }


def write_merged(ctx: dict, path: Path, results: dict, skipped: dict,
                 smoke: bool, failed: list) -> None:
    cfg, gnn_cfg = ctx["cfg"], ctx["gnn_cfg"]
    path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": ctx["ds_cfg"]["name"],
        "protocol": "leave-one-attack-out (D4, Protocol 2)",
        "smoke": smoke, "fingerprint": ctx["fingerprint"],
        "ablation": cfg["ablation"],
        "reported_at_prevalence": gnn_cfg["eval"]["target_prevalence"],
        "config": cfg, "gnn_config": gnn_cfg,
        "skipped": skipped, "runs_failed": failed,
        "results": results,
    }, indent=2))


def partial_path(ctx: dict, family: str, seed: int) -> Path:
    return ctx["out_dir"] / (f"partial_zeroday_{ctx['ds_cfg']['name']}_"
                             f"{family}_seed{seed}.json")


def run_child(args, ctx: dict) -> None:
    """One (family, seed), in its own process. Writes a partial and exits."""
    family, seed = args.single_family, args.single_seed
    inputs = load_graph_inputs(REPO_ROOT / ctx["pre_cfg"]["output"]["dir"],
                               ctx["ds_cfg"]["name"])
    plan, why = plan_family(ctx, inputs, family, quiet=True)
    if plan is None:
        raise SystemExit(f"{family} cannot be held out: {why}")
    device = pick_device(ctx["gnn_cfg"]["train"]["device"])
    run = train_family_seed(ctx, inputs, plan, seed, device)
    print(describe(run, seed, ctx["target_key"]), flush=True)
    ctx["out_dir"].mkdir(parents=True, exist_ok=True)
    partial_path(ctx, family, seed).write_text(json.dumps({
        "fingerprint": ctx["fingerprint"], "family": family, "seed": seed,
        "run": run}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "zeroday.yaml")
    ap.add_argument("--families", nargs="+", default=None,
                    help="families to hold out (default: every attack family)")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="seconds-long proof the code path runs; numbers are meaningless")
    ap.add_argument("--fresh", action="store_true",
                    help="discard finished runs from an earlier campaign and start over")
    ap.add_argument("--single-family", default=None,
                    help="internal: train exactly one family (with --single-seed)")
    ap.add_argument("--single-seed", type=int, default=None, help="internal")
    args = ap.parse_args()

    ctx = load_context(args)
    if args.single_family is not None:
        run_child(args, ctx)
        return

    import subprocess

    cfg, gnn_cfg, ds_cfg = ctx["cfg"], ctx["gnn_cfg"], ctx["ds_cfg"]
    n_seeds, target_key = ctx["n_seeds"], ctx["target_key"]
    out_dir = ctx["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{'smoke_' if args.smoke else ''}zeroday_{ds_cfg['name']}.json"

    print(f"dataset: {ds_cfg['name']}   window {ctx['window']:,}"
          f"{'   [SMOKE -- numbers are meaningless]' if args.smoke else ''}")
    inputs = load_graph_inputs(REPO_ROOT / ctx["pre_cfg"]["output"]["dir"], ds_cfg["name"])

    attack_families = [f for f in inputs.families if f != cfg["benign_class_name"]]
    to_run = args.families or attack_families
    unknown = [f for f in to_run if f not in inputs.families]
    if unknown:
        raise SystemExit(f"unknown families {unknown}; have {attack_families}")

    # Start from what an earlier invocation of the same experiment finished.
    # Without this, `--families scanning` after a full campaign would rewrite
    # the results file holding eight families with a file holding one (C20).
    results, skipped, failed = {}, {}, []
    if not args.smoke and not args.fresh:
        prior = read_if_matching(out_path, ctx["fingerprint"])
        if prior is None and out_path.exists():
            raise SystemExit(
                f"{out_path.name} holds results from a different configuration. "
                f"Move it aside, or pass --fresh to replace it.")
        if prior:
            results, skipped = prior["results"], prior.get("skipped", {})

    device = None if not args.smoke else pick_device(gnn_cfg["train"]["device"])
    print(f"ablation: {cfg['ablation']}   {len(to_run)} families x {n_seeds} seed(s)"
          f"   selection: {'early stopping' if gnn_cfg['train'].get('early_stopping', True) else 'fixed budget, best validation'}\n")

    for family in to_run:
        plan, why = plan_family(ctx, inputs, family)
        if plan is None:
            skipped[family] = why
            continue

        done = {r["seed"]: r for r in results.get(family, {}).get("runs", [])
                if "seed" in r}
        for seed in range(n_seeds):
            if seed in done:
                print(f"  seed {seed}  already done, reused")
                continue
            if args.smoke:
                # In-process: a smoke run is seconds long and exists to prove
                # the code path, which a subprocess per run would only slow.
                done[seed] = train_family_seed(ctx, inputs, plan, seed, device)
                print(describe(done[seed], seed, target_key))
                continue

            part = partial_path(ctx, family, seed)
            left = None if args.fresh else read_if_matching(part, ctx["fingerprint"])
            if left is None:
                cmd = [sys.executable, __file__, "--config", str(args.config),
                       "--single-family", family, "--single-seed", str(seed)]
                if subprocess.run(cmd).returncode != 0:
                    print(f"  seed {seed}  FAILED -- continuing with the rest",
                          flush=True)
                    failed.append([family, seed])
                    continue
                left = read_if_matching(part, ctx["fingerprint"])
            else:
                print(f"  seed {seed}  resumed from {part.name}")
            if left:
                done[seed] = left["run"]

        if not done:
            continue
        results[family] = family_entry(ctx, plan, family, list(done.values()))
        agg = results[family]["aggregate"]
        te = plan["splits"]["test"]
        print(f"  => recall {agg['recall']['mean']:.4f} +/- {agg['recall']['std']:.4f} "
              f"on {te['n_held_out']:,} unseen flows")
        wrong = results[family]["held_out_mistaken_for"]
        if wrong:
            top = ", ".join(f"{k} {v:.0%}" for k, v in list(wrong.items())[:3])
            print(f"     the family head calls it: {top}")
        print()
        # Durable after every family, so the most a crash can cost is the
        # family in progress -- and its finished seeds are in partials anyway.
        write_merged(ctx, out_path, results, skipped, args.smoke, failed)

    write_merged(ctx, out_path, results, skipped, args.smoke, failed)
    if not args.smoke:
        for part in out_dir.glob(f"partial_zeroday_{ds_cfg['name']}_*.json"):
            if read_if_matching(part, ctx["fingerprint"]):
                part.unlink(missing_ok=True)

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
        short = f"  ({r['n_seeds']} of {n_seeds} seeds)" if r["n_seeds"] < n_seeds else ""
        print(f"  {family:<16} {agg_fmt(a, 'recall', with_std=True)} "
              f"{agg_fmt(a, 'pr_auc')} {r['test_positives']:>12,}{thin}{short}")
    if results:
        print("\n  The MEAN of this column is not the finding. Which families "
              "transfer\n  and which do not is what says what the model learned.")
    for family, why in skipped.items():
        print(f"  skipped {family}: {why}")
    if failed:
        print(f"\n  WARNING: these runs failed and are excluded: {failed}")
    shown = out_path.relative_to(REPO_ROOT) if out_path.is_relative_to(REPO_ROOT) else out_path
    print(f"\nwritten -> {shown}\n")


if __name__ == "__main__":
    main()
