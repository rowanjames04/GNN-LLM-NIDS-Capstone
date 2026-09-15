"""Phase 7a: evidence pack -> incident report, scored for groundedness.

The end-to-end LLM path. Reads packs from Phase 6a, renders a versioned prompt,
generates a report through one backend, and scores it automatically.

**Money.** The default provider is `stub` -- deterministic, offline, free, and
the control arm of the study. Any other provider spends Rowan's money or needs a
local server, and must be asked for explicitly:

    python scripts/generate_reports.py --smoke                      # free
    python scripts/generate_reports.py --provider anthropic -n 5    # costs money
    python scripts/generate_reports.py --provider ollama --model llama3.1:8b

`--estimate-only` prints what a run would cost without making a call.

**Resuming.** Each report is appended to a `.partial.jsonl` log as soon as it is
scored, and re-running the same command skips what is already there -- the
self-hosted arm runs in a Colab session that can end at any pack (D35).
`--fresh` discards the log; `--retry-failed` regenerates failed rows and records
how many were retried.
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

from gnnids.llm.adapters import ANTHROPIC_PRICING, build_adapter  # noqa: E402
from gnnids.llm.groundedness import (  # noqa: E402
    check_no_jargon, check_uncertainty_conveyed, score_report,
)
from gnnids.llm.prompts import DEFAULT_VERSION, build_prompt  # noqa: E402
from gnnids.llm.resume import ReportLog, ResumeMismatch  # noqa: E402

# Providers whose every call is charged. Gemini can be (on a billed key), but its
# free tier is how this study uses it (D36); the roster's `billing` field and
# run_llm_study.py's gate are the real control, this only sets the banner.
PAID_PROVIDERS = ("anthropic", "openai")


def summarise(rows: list[dict]) -> dict:
    """Aggregate the metric families that survive aggregation.

    Groundedness is averaged over successful reports only, and the failure count
    is reported beside it -- a model whose errored runs were dropped would look
    better than one that returned a flawed report, which is backwards.
    """
    ok = [r for r in rows if r["ok"]]
    def mean(key, source="scores"):
        vals = [r[source][key] for r in ok
                if isinstance(r[source].get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None

    unc = [r["uncertainty"] for r in ok if r.get("uncertainty")]
    return {
        "n": len(rows), "n_ok": len(ok), "n_failed": len(rows) - len(ok),
        "groundedness_mean": mean("groundedness"),
        "reports_with_fabricated_addresses": sum(
            1 for r in ok if r["scores"]["fabricated_addresses"]),
        "reports_with_jargon": sum(1 for r in ok if not r["jargon"]["clean"]),
        "report_words_mean": mean("report_words"),
        "latency_seconds_mean": round(
            sum(r["latency_seconds"] for r in ok) / len(ok), 3) if ok else None,
        "total_cost_usd": round(sum(r["cost_usd"] or 0.0 for r in ok), 6),
        "ambiguous_detections": len(unc),
        "uncertainty_conveyed_rate": (
            round(sum(u["uncertainty_conveyed"] for u in unc) / len(unc), 4)
            if unc else None),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "llm.yaml")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt-version", default=None)
    ap.add_argument("--packs", type=Path, default=None)
    ap.add_argument("-n", "--n-reports", type=int, default=None)
    ap.add_argument("--estimate-only", action="store_true",
                    help="print the projected cost and stop")
    ap.add_argument("--fresh", action="store_true",
                    help="discard any partial log from an interrupted run")
    ap.add_argument("--retry-failed", action="store_true",
                    help="when resuming, regenerate rows that failed (counted)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    provider = args.provider or cfg["provider"]
    version = args.prompt_version or cfg.get("prompt_version", DEFAULT_VERSION)
    packs_path = args.packs or (REPO_ROOT / (cfg["smoke"]["packs"] if args.smoke
                                             else cfg["packs"]))
    if not packs_path.exists():
        raise SystemExit(
            f"no evidence packs at {packs_path}. Run scripts/make_evidence.py first.")

    packs = json.loads(packs_path.read_text())
    n = args.n_reports or (cfg["smoke"]["n_reports"] if args.smoke else cfg["n_reports"])
    packs = packs[:n]

    opts = dict(cfg.get("provider_options", {}).get(provider, {}))
    if args.model:
        opts["model"] = args.model
    adapter = build_adapter(provider, **opts)

    print(f"provider: {adapter.provider}   model: {adapter.model}   "
          f"prompt: {version}   packs: {len(packs)}")

    if provider != "stub":
        # A deliberate friction point. Every paid run should be a decision.
        est = _estimate(adapter, packs, version)
        paid = provider in PAID_PROVIDERS
        print(f"\n  {'NOT FREE. ' if paid else ''}Estimated cost for "
              f"{len(packs)} reports: {est}")
        if args.estimate_only:
            print("  --estimate-only, stopping before any call.\n")
            return
    elif args.estimate_only:
        print("\n  stub provider is free.\n")
        return

    out_dir = REPO_ROOT / cfg["output"]["dir"]
    name = (f"{'smoke_' if args.smoke else ''}reports_"
            f"{adapter.provider}_{adapter.model.replace(':', '-')}_{version}.json")
    log = ReportLog(out_dir / name.replace(".json", ".partial.jsonl"), {
        "provider": adapter.provider, "model": adapter.model,
        "prompt_version": version, "packs_source": packs_path.name})
    if args.fresh:
        log.discard()
    try:
        done, retried = log.load([p["detection_id"] for p in packs],
                                 retry_failed=args.retry_failed)
    except ResumeMismatch as e:
        raise SystemExit(str(e)) from None
    if done:
        print(f"\n  resuming: {len(done)} of {len(packs)} already on disk"
              + (f", {retried} failed row(s) dropped for retry" if retried else ""))
    log.start(done)

    for pack in packs:
        if pack["detection_id"] in done:
            continue
        system, user = build_prompt(pack, version)
        r = adapter.generate(system, user, version)
        row = {
            "detection_id": pack["detection_id"],
            "true_label": pack["flow"].get("true_label"),
            "predicted_class": pack["detection"]["predicted_class"],
            "ok": r.ok, "error": r.error, "report": r.text,
            "latency_seconds": r.latency_seconds, "cost_usd": r.cost_usd,
            "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
            "scores": score_report(pack, r.text) if r.ok else {},
            "jargon": check_no_jargon(r.text) if r.ok else {},
            "uncertainty": (check_uncertainty_conveyed(pack, r.text)
                            if r.ok else None),
            # What determinism controls this call could actually be given. The
            # roster is not uniform, so this is recorded per response rather
            # than asserted once for the study.
            "determinism": r.extra.get("determinism"),
        }
        log.append(row)
        done[row["detection_id"]] = row
        mark = "ok " if r.ok else "FAIL"
        g = row["scores"].get("groundedness")
        print(f"  {mark} {pack['detection_id'][:8]}  "
              f"groundedness {g if g is not None else 'n/a':<7}  "
              f"{row['scores'].get('report_words', 0):>4} words  "
              f"{r.latency_seconds:.2f}s"
              + (f"  {r.error}" if r.error else ""))

    # Pack order, not completion order, so a resumed run's file is identical in
    # layout to an uninterrupted one.
    rows = [done[p["detection_id"]] for p in packs]
    summary = summarise(rows)
    det = next((r["determinism"] for r in rows if r.get("determinism")), None)
    summary["determinism"] = det or {"supported": False,
                                     "note": "no determinism controls applied"}
    summary["n_retried_after_failure"] = retried
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": adapter.provider, "model": adapter.model,
        "prompt_version": version, "packs_source": str(packs_path.name),
        "smoke": args.smoke, "summary": summary, "reports": rows,
    }, indent=2))
    log.discard()

    print("\n" + "=" * 70)
    print(f"  PHASE 7a -- {adapter.provider}/{adapter.model}, prompt {version}")
    print("=" * 70)
    for k in ("n_ok", "n_failed", "groundedness_mean",
              "reports_with_fabricated_addresses", "reports_with_jargon",
              "report_words_mean", "latency_seconds_mean", "total_cost_usd",
              "ambiguous_detections", "uncertainty_conveyed_rate"):
        print(f"  {k:<38} {summary[k]}")
    print("\n  Groundedness means 'invented nothing', NOT 'reasoned correctly'.")
    print("  A fluent wrong interpretation built from real facts scores 1.0.")
    # Not relative_to(REPO_ROOT) unconditionally: an output dir outside the repo
    # (a Colab session writing to Drive, a test's tmp dir) crashed here AFTER the
    # results were written -- the C2 defect again.
    out = out_dir / name
    shown = out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out
    print(f"\nwritten -> {shown}\n")


def _estimate(adapter, packs, version) -> str:
    """Rough projected cost, from prompt length and a generous output guess."""
    if adapter.provider != "anthropic" or adapter.model not in ANTHROPIC_PRICING:
        return "unknown for this provider (self-hosted models have no per-token price)"
    words = sum(len(build_prompt(p, version)[1].split()) for p in packs)
    n_in = int(words * 1.35) + 400 * len(packs)      # ~1.35 tokens/word + system
    n_out = 350 * len(packs)
    price_in, price_out = ANTHROPIC_PRICING[adapter.model]
    usd = n_in / 1e6 * price_in + n_out / 1e6 * price_out
    return f"~${usd:.4f} (~{n_in:,} in / ~{n_out:,} out tokens)"


if __name__ == "__main__":
    main()
