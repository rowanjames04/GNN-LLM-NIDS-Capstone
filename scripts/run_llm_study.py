"""Phase 7b: run the whole model roster over one fixed pack set.

The study is a **fully crossed, single-factor design**: every model sees every
pack, through the same prompt, and the model is the only variable. Running
providers one at a time by hand does not give that -- it gives several samples
that happen to share a name, and any difference between them confounds the model
with whatever else changed between invocations.

This script fixes the pack set once, then sweeps the roster.

**Money.** No `paid` roster entry is called without `--confirm-spend`. Without
it the script prints the projected cost of the whole sweep and stops, which is
the number worth seeing before committing: a roster of five models over 200
packs is five times the per-model estimate. `free_tier` and `self_hosted`
entries run without the flag (see `billing` in configs/llm.yaml).

**The study runs on Ollama's hosted service** (D44): open-weight models, called
from the Mac with `OLLAMA_API_KEY` set, on the account's free usage credits
(D45), so those entries are `free_tier` and need no spend flag. `--only` selects by billing class or provider name.
The closed-weight cloud adapters and the local-server roster remain, disabled.

Usage:
    python scripts/run_llm_study.py --smoke                          # stub only, free
    python scripts/run_llm_study.py --estimate                       # cost, then stop
    python scripts/run_llm_study.py --only paid free_tier --confirm-spend
    python scripts/run_llm_study.py --only ollama                      # the study
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnids.llm.adapters import LIST_PRICES  # noqa: E402
from gnnids.llm.order import balanced_order
from gnnids.llm.prompts import build_prompt  # noqa: E402

BILLING_CLASSES = ("free", "paid", "free_tier", "self_hosted")


def billing(entry: dict) -> str:
    """The entry's billing class. Missing means paid, never free.

    The stub is the one exception, because it cannot spend money by
    construction. Anything else that forgot to declare itself is guarded: the
    cost of wrongly gating a free model is one extra flag, while the cost of
    wrongly waving through a paid one is real money.
    """
    if entry["provider"] == "stub":
        return "free"
    b = entry.get("billing", "paid")
    if b not in BILLING_CLASSES:
        raise SystemExit(f"roster entry {entry} has billing {b!r}; "
                         f"expected one of {BILLING_CLASSES}")
    return b


def select(roster: list[dict], only: list[str] | None) -> list[dict]:
    """Enabled entries, narrowed by billing class or provider name if asked."""
    enabled = [e for e in roster if e.get("enabled", True)]
    if not only:
        return enabled
    return [e for e in enabled if billing(e) in only or e["provider"] in only]


def estimate(entry: dict, packs: list, version: str) -> tuple[float | None, str]:
    """Projected spend for one roster entry over the whole pack set.

    Returns what would actually be *charged*: a free-tier entry projects zero,
    with its list price shown beside it so the reference figure is not lost.
    Output is guessed at 350 tokens per report, which excludes any thinking a
    model does before answering. Most of the hosted roster thinks, and thinking
    is billed as output, so treat the figure as a floor: the pilot gives the
    real number.
    """
    provider, model = entry["provider"], entry.get("model")
    if provider == "stub":
        return 0.0, "free"
    prices = LIST_PRICES.get(provider, {})
    if model not in prices:
        return None, "unknown (no published per-token price for this provider)"
    words = sum(len(build_prompt(p, version)[1].split()) for p in packs)
    n_in = int(words * 1.35) + 400 * len(packs)
    n_out = 350 * len(packs)
    pin, pout = prices[model]
    usd = n_in / 1e6 * pin + n_out / 1e6 * pout
    if billing(entry) == "free_tier":
        return 0.0, f"$0 on the free tier (~${usd:.4f} at list price)"
    return usd, f"${usd:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "llm.yaml")
    ap.add_argument("--packs", type=Path, default=None)
    ap.add_argument("--prompt-versions", nargs="+", default=None,
                    help="run each listed version (the prompt-sensitivity ablation)")
    ap.add_argument("-n", "--n-reports", type=int, default=None)
    ap.add_argument("--estimate", action="store_true", help="print the cost and stop")
    ap.add_argument("--confirm-spend", action="store_true",
                    help="required before any paid provider is called")
    ap.add_argument("--only", nargs="+", default=None, metavar="CLASS_OR_PROVIDER",
                    help="run only these billing classes (paid, free_tier, "
                         "self_hosted) or providers (e.g. gemini)")
    ap.add_argument("--round-size", type=int, default=None,
                    help="packs per round (default: round_size in the config, "
                         "or everything in one round)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    roster = select(cfg["roster"], args.only)
    if not roster:
        raise SystemExit(f"--only {' '.join(args.only)} "
                         "matches no enabled roster entry")
    if args.smoke:
        roster = [e for e in roster if e["provider"] == "stub"] or [{"provider": "stub"}]

    packs_path = args.packs or (REPO_ROOT / (cfg["smoke"]["packs"] if args.smoke
                                             else cfg["packs"]))
    if not packs_path.exists():
        raise SystemExit(f"no evidence packs at {packs_path}. "
                         f"Run scripts/make_evidence.py first.")
    packs = balanced_order(json.loads(packs_path.read_text()))
    n = args.n_reports or (cfg["smoke"]["n_reports"] if args.smoke else cfg["n_reports"])
    packs = packs[:n]
    versions = args.prompt_versions or [cfg.get("prompt_version", "v1")]

    print(f"packs: {packs_path.name}  ({len(packs)} detections)")
    print(f"prompt versions: {', '.join(versions)}")
    print(f"roster: {len(roster)} model(s) x {len(versions)} version(s) = "
          f"{len(roster) * len(versions)} runs\n")

    total, unknown = 0.0, []
    for entry in roster:
        for v in versions:
            usd, label = estimate(entry, packs, v)
            name = f"{entry['provider']}/{entry.get('model', 'default')} [{v}]"
            print(f"  {name:<44} {billing(entry):<12} {label}")
            if usd is None:
                unknown.append(name)
            else:
                total += usd
    print(f"\n  PROJECTED TOTAL: ${total:.4f}"
          + (f"  (plus {len(unknown)} run(s) with no published price)" if unknown else ""))

    paid = [e for e in roster if billing(e) == "paid"]
    if args.estimate:
        print("\n  --estimate given; stopping before any call.\n")
        return
    if paid and not args.confirm_spend:
        print("\n  This sweep would call paid providers. Re-run with "
              "--confirm-spend to proceed,\n  or --estimate to see the cost "
              "without running anything.\n")
        return
    if any(billing(e) == "free_tier" for e in roster):
        print("\n  free_tier entries are only free if the API key's project has "
              "no billing attached.")

    # Each run is a separate process invoking generate_reports.py, so one
    # provider failing (a refusal, an expired key, an unreachable local server)
    # cannot take the sweep down with it. A partial roster is still a result;
    # a crashed sweep is not.
    out_dir = REPO_ROOT / cfg["output"]["dir"]
    # Rounds (D46). The sweep goes prompt version by prompt version, and inside
    # a version it takes the packs a round at a time through EVERY model before
    # starting the next round. Wherever it stops -- free credit, a rate limit, a
    # closed laptop -- every model has reports for the same packs, so what
    # exists is a valid comparison on fewer packs rather than a full result for
    # some models and nothing for the rest. The pack order is balanced, so each
    # round is itself a fair sample. The first listed version is the study; a
    # later one (the prompt-sensitivity ablation) starts only once it is done.
    size = args.round_size or cfg.get("round_size") or len(packs)
    targets = list(range(size, len(packs), size)) + [len(packs)]
    status: dict[tuple, dict] = {}
    stopped = False
    for v in versions:
        for rnd, target in enumerate(targets, 1):
            print(f"\n########## prompt {v} -- round {rnd} of {len(targets)}: "
                  f"packs 1-{target} ##########")
            for entry in roster:
                cmd = [sys.executable, str(REPO_ROOT / "scripts" / "generate_reports.py"),
                       "--provider", entry["provider"], "--prompt-version", v,
                       "--packs", str(packs_path), "-n", str(target),
                       "--config", str(args.config)]
                if entry.get("model"):
                    cmd += ["--model", entry["model"]]
                if args.smoke:
                    cmd += ["--smoke"]
                label = f"{entry['provider']}/{entry.get('model', 'default')} [{v}]"
                print(f"\n===== {label} -- to {target} packs =====")
                rc = subprocess.run(cmd).returncode
                # 75 is generate_reports.py's "stopped, resumable" (a quota or a
                # dead server): not a failure of the model, and not finished.
                state = {0: "complete", 75: "stopped_resumable"}.get(rc, "failed")
                key = (entry["provider"], entry.get("model"), v)
                rec = status.setdefault(key, {
                    "provider": entry["provider"], "model": entry.get("model"),
                    "billing": billing(entry), "prompt_version": v,
                    "packs_completed": 0})
                if rc == 0:
                    rec["packs_completed"] = target
                rec["status"] = state
                rec["ok"] = rc == 0 and target == len(packs)
                if state == "stopped_resumable":
                    # One account serves the whole roster, so the next model
                    # would stop the same way. Ending here keeps the rounds even.
                    print(f"  {label} STOPPED in round {rnd} -- resumable. Ending "
                          f"the sweep here; re-run the same command to continue.")
                    stopped = True
                    break
                if rc != 0:
                    print(f"  {label} FAILED (exit {rc}) -- continuing with the rest")
            if stopped:
                break
        if stopped:
            break
    runs = list(status.values())

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "packs": packs_path.name, "n_detections": len(packs),
        "prompt_versions": versions, "only": args.only,
        "round_size": size, "stopped_early": stopped, "runs": runs,
        "projected_cost_usd": round(total, 4),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{'smoke_' if args.smoke else ''}study_manifest.json"
    (out_dir / name).write_text(json.dumps(manifest, indent=2))

    ok = sum(1 for r in runs if r["ok"])
    print(f"\n{ok}/{len(runs)} runs completed.")
    for r in runs:
        print(f"  {r['provider']}/{r.get('model') or 'default'} [{r['prompt_version']}]"
              f"  {r['packs_completed']} of {len(packs)} packs  ({r['status']})")
    if stopped:
        print("\n  Stopped early. Every model that ran has the same packs up to the "
              "last finished\n  round; the same command resumes from there.")
    # The exit status says what happened, so a campaign runner cannot record a
    # failed sweep as a passed step (pre-mortem, 2026-10-17): 75 = stopped and
    # resumable, 1 = some run failed, 0 = everything asked for is complete.
    if stopped:
        raise SystemExit(75)
    if any(r["status"] == "failed" for r in runs):
        raise SystemExit(1)
    # The Colab notebook writes to Google Drive, outside the repo, where an
    # unconditional relative_to() would crash after the sweep had finished.
    out = out_dir / name
    shown = out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out
    print(f"written -> {shown}")
    print("\nCollate with: python scripts/summarise_llm_study.py\n")


if __name__ == "__main__":
    main()
