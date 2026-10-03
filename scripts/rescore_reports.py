"""Re-apply the automated scores to reports already on disk. Calls no model.

Scoring is arithmetic over the evidence pack and the saved report text, so it
can be repeated for free. That matters twice: a metric added after a paid run
does not require paying again, and a correction to a metric reaches every
existing result rather than only the next one generated.

The reports are matched to their evidence packs by detection id. **If the packs
have been regenerated since the reports were written, the file is refused** --
scoring a report against a pack it was not written from would produce numbers
that parse and mean nothing (the D33 hazard).

Usage:
    python scripts/rescore_reports.py                 # every non-smoke file
    python scripts/rescore_reports.py results/reports/reports_stub_template-v1_v1.json
    python scripts/rescore_reports.py --check         # report what would change
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnids.llm.scoring import SCORING_VERSION, score_all, summarise  # noqa: E402

# Summary fields that describe the run, not the scores, and must survive.
KEEP = ("determinism", "n_retried_after_failure", "n_sessions")


def rescore(doc: dict, packs: dict[str, dict]) -> tuple[dict, list[str]]:
    """The document with every row re-scored, and the ids that had no pack."""
    missing = [r["detection_id"] for r in doc["reports"]
               if r["detection_id"] not in packs]
    if missing:
        return doc, missing
    for row in doc["reports"]:
        row.update(score_all(packs[row["detection_id"]], row.get("report") or "",
                             row["ok"]))
    kept = {k: doc.get("summary", {}).get(k) for k in KEEP
            if k in doc.get("summary", {})}
    doc["summary"] = {**summarise(doc["reports"]), **kept}
    doc["rescored_at"] = datetime.now(timezone.utc).isoformat()
    return doc, []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--evidence-dir", type=Path,
                    default=REPO_ROOT / "results" / "evidence")
    ap.add_argument("--check", action="store_true",
                    help="print what would change and write nothing")
    args = ap.parse_args()

    files = args.files or sorted(
        f for f in (REPO_ROOT / "results" / "reports").glob("reports_*.json"))
    if not files:
        raise SystemExit("no report files to re-score")

    refused = 0
    for path in files:
        doc = json.loads(path.read_text())
        packs_path = args.evidence_dir / doc["packs_source"]
        if not packs_path.exists():
            print(f"  REFUSED {path.name}: evidence file {doc['packs_source']} not found")
            refused += 1
            continue
        packs = {p["detection_id"]: p for p in json.loads(packs_path.read_text())}
        before = doc.get("summary", {})
        doc, missing = rescore(doc, packs)
        if missing:
            print(f"  REFUSED {path.name}: {len(missing)} of {len(doc['reports'])} "
                  f"detections are not in {doc['packs_source']} -- the packs were "
                  f"regenerated after these reports were written")
            refused += 1
            continue
        after = doc["summary"]
        changed = sorted(k for k in after if before.get(k) != after[k]
                         and k != "scoring_version")
        print(f"  {'would update' if args.check else 'updated'} {path.name}: "
              f"{len(doc['reports'])} reports, scoring {SCORING_VERSION}; "
              f"summary fields changed: {', '.join(changed) or 'none'}")
        if not args.check:
            path.write_text(json.dumps(doc, indent=2))
    if refused:
        raise SystemExit(f"{refused} file(s) refused")


if __name__ == "__main__":
    main()
