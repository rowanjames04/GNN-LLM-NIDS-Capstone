"""Phase 7b: run the whole model roster over one fixed pack set.

The study is a **fully crossed, single-factor design**: every model sees every
pack, through the same prompt, and the model is the only variable. Running
providers one at a time by hand does not give that -- it gives several samples
that happen to share a name, and any difference between them confounds the model
with whatever else changed between invocations.

This script fixes the pack set once, then sweeps the roster.

**Money.** Nothing is spent without `--confirm-spend`. Without it the script
prints the projected cost of the whole sweep and stops, which is the number
worth seeing before committing: a roster of five models over 200 packs is five
times the per-model estimate.

Usage:
    python scripts/run_llm_study.py --smoke              # stub only, free
    python scripts/run_llm_study.py --estimate           # cost, then stop
    python scripts/run_llm_study.py --confirm-spend      # actually runs
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

from gnnids.llm.adapters import ANTHROPIC_PRICING  # noqa: E402
from gnnids.llm.prompts import build_prompt  # noqa: E402


def estimate(entry: dict, packs: list, version: str) -> tuple[float | None, str]:
    """Projected cost for one roster entry over the whole pack set."""
    provider, model = entry["provider"], entry.get("model")
    if provider == "stub":
        return 0.0, "free"
    if provider != "anthropic" or model not in ANTHROPIC_PRICING:
        return None, "unknown (no published per-token price for this provider)"
    words = sum(len(build_prompt(p, version)[1].split()) for p in packs)
    n_in = int(words * 1.35) + 400 * len(packs)
    n_out = 350 * len(packs)
    pin, pout = ANTHROPIC_PRICING[model]
    usd = n_in / 1e6 * pin + n_out / 1e6 * pout
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
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    roster = [e for e in cfg["roster"] if e.get("enabled", True)]
    if args.smoke:
        roster = [e for e in roster if e["provider"] == "stub"] or [{"provider": "stub"}]

    packs_path = args.packs or (REPO_ROOT / (cfg["smoke"]["packs"] if args.smoke
                                             else cfg["packs"]))
    if not packs_path.exists():
        raise SystemExit(f"no evidence packs at {packs_path}. "
                         f"Run scripts/make_evidence.py first.")
    packs = json.loads(packs_path.read_text())
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
            print(f"  {name:<44} {label}")
            if usd is None:
                unknown.append(name)
            else:
                total += usd
    print(f"\n  PROJECTED TOTAL: ${total:.4f}"
          + (f"  (plus {len(unknown)} run(s) with no published price)" if unknown else ""))

    paid = [e for e in roster if e["provider"] != "stub"]
    if args.estimate:
        print("\n  --estimate given; stopping before any call.\n")
        return
    if paid and not args.confirm_spend:
        print("\n  This sweep would call paid providers. Re-run with "
              "--confirm-spend to proceed,\n  or --estimate to see the cost "
              "without running anything.\n")
        return

    # Each run is a separate process invoking generate_reports.py, so one
    # provider failing (a refusal, an expired key, an unreachable local server)
    # cannot take the sweep down with it. A partial roster is still a result;
    # a crashed sweep is not.
    out_dir = REPO_ROOT / cfg["output"]["dir"]
    runs = []
    for entry in roster:
        for v in versions:
            cmd = [sys.executable, str(REPO_ROOT / "scripts" / "generate_reports.py"),
                   "--provider", entry["provider"], "--prompt-version", v,
                   "--packs", str(packs_path), "-n", str(len(packs)),
                   "--config", str(args.config)]
            if entry.get("model"):
                cmd += ["--model", entry["model"]]
            if args.smoke:
                cmd += ["--smoke"]
            label = f"{entry['provider']}/{entry.get('model', 'default')} [{v}]"
            print(f"\n===== {label} =====")
            rc = subprocess.run(cmd).returncode
            runs.append({"provider": entry["provider"], "model": entry.get("model"),
                         "prompt_version": v, "ok": rc == 0})
            if rc != 0:
                print(f"  {label} FAILED (exit {rc}) -- continuing with the rest")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "packs": packs_path.name, "n_detections": len(packs),
        "prompt_versions": versions, "runs": runs,
        "projected_cost_usd": round(total, 4),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{'smoke_' if args.smoke else ''}study_manifest.json"
    (out_dir / name).write_text(json.dumps(manifest, indent=2))

    ok = sum(1 for r in runs if r["ok"])
    print(f"\n{ok}/{len(runs)} runs completed.")
    print(f"written -> {(out_dir / name).relative_to(REPO_ROOT)}")
    print("\nCollate with: python scripts/summarise_llm_study.py\n")


if __name__ == "__main__":
    main()
