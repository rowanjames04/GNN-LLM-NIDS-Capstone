"""End-to-end outcome of each detection: from the detector's call to the note.

The language model study scores each note against its evidence, which says
whether the translation was faithful, not whether the analyst ended up with the
right answer. That also depends on the detector. This script joins the two for
every note in the study:

- **Detection outcome** (from the pack): a real attack named as the right
  family, a real attack named as another family, or a false positive, split by
  whether the facts layer called it borderline.
- **Translation** (from the note's scores): passed every automatic check --
  completed, nothing ungrounded, no invented address, the predicted family
  named and none substituted, no direction error, and the detector's doubt
  passed on where the facts layer marked a detection as borderline or its
  family as uncertain -- or failed at least one.

It also gives the per-model running figures (completion, median response time,
median note length, list price per note) beside the outcome, so the models can
be compared on the whole path. The sample is stratified (D33): families are
equally represented and 15% of packs are false positives, so the shares
describe the sample, not the detector's alert stream.

    python scripts/end_to_end.py

Writes `results/metrics/end_to_end_<dataset>.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnids.llm.facts import borderline  # noqa: E402

BENIGN = "Benign"
PROMPTS = ("v3", "v4")


def detection_outcome(pack: dict) -> str:
    true, predicted = pack["flow"]["true_label"], pack["detection"]["predicted_class"]
    if true == BENIGN:
        return "false_positive_borderline" if borderline(pack) else "false_positive_clear"
    return "attack_right_family" if predicted == true else "attack_wrong_family"


def passed_every_check(r: dict) -> bool:
    if not r["ok"] or r.get("degenerate"):
        return False
    s, c = r["scores"], r["classification"]
    checks = [s["n_ungrounded"] == 0, not s["fabricated_addresses"],
              c["names_predicted"], not c["substituted"]]
    if r.get("direction"):
        checks.append(r["direction"]["n_errors"] == 0)
    if r.get("borderline"):
        checks.append(r["borderline"]["borderline_conveyed"])
    if r.get("uncertainty"):
        checks.append(r["uncertainty"]["uncertainty_conveyed"])
    return all(checks)


def model_row(doc: dict, outcomes: dict) -> dict:
    reports = doc["reports"]
    done = [r for r in reports if r["ok"] and not r.get("degenerate")]
    passed = [r for r in done if passed_every_check(r)]
    tally = Counter(outcomes[r["detection_id"]] for r in passed)
    return {
        "model": doc["model"],
        "requested": len(reports),
        "completed": len(done),
        "passed_every_check": len(passed),
        "passed_by_outcome": dict(sorted(tally.items())),
        "correct_and_faithful": tally.get("attack_right_family", 0),
        "median_latency_seconds": (round(float(np.median([r["latency_seconds"] for r in done])), 2)
                                   if done else None),
        "median_note_words": (int(np.median([r["scores"]["report_words"] for r in done]))
                              if done else None),
        "list_price_per_note_usd": (round(sum(r.get("reference_cost_usd") or 0 for r in reports)
                                          / len(done), 5) if done else None),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="NF-ToN-IoT-v2")
    args = ap.parse_args()

    packs = json.loads((REPO_ROOT / "results" / "evidence" /
                        f"evidence_{args.dataset}.json").read_text())
    outcomes = {p["detection_id"]: detection_outcome(p) for p in packs}

    by_prompt = {}
    for v in PROMPTS:
        rows = []
        for path in sorted((REPO_ROOT / "results" / "reports").glob(f"reports_*_{v}.json")):
            doc = json.loads(path.read_text())
            if doc.get("smoke"):
                continue
            rows.append({"provider": doc["provider"], **model_row(doc, outcomes)})
        by_prompt[v] = rows

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "packs": len(packs),
        "detection_outcomes": dict(sorted(Counter(outcomes.values()).items())),
        "checks": ["completed", "nothing ungrounded", "no invented address",
                   "predicted family named", "no family substituted",
                   "no direction error", "borderline conveyed where borderline",
                   "uncertainty conveyed where the family is uncertain"],
        "note": "stratified sample (D33): shares describe the sample, not the alert stream",
        "by_prompt": by_prompt,
    }
    path = REPO_ROOT / "results" / "metrics" / f"end_to_end_{args.dataset}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"  detection outcomes over {len(packs)} packs: {out['detection_outcomes']}")
    for v, rows in by_prompt.items():
        print(f"\n  prompt {v}")
        for r in rows:
            print(f"    {r['model']:22s} {r['completed']:>3}/{r['requested']:<3} passed "
                  f"{r['passed_every_check']:>3}  correct+faithful {r['correct_and_faithful']:>3}  "
                  f"{r['median_latency_seconds']}s  {r['median_note_words']}w  "
                  f"${r['list_price_per_note_usd']}")
    print(f"\n  wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
