"""Test whether seed-level differences in PR-AUC are larger than seed noise.

Reads the per-seed PR-AUC at the reporting prevalence from the GNN and baseline
results files and runs an exact two-sided Mann-Whitney U test on each pair:

- full model against channel 1 only (the topology ablation), on both datasets;
- full model against XGBoost with host features, on NF-ToN-IoT-v2.

Mann-Whitney was chosen over a t-test because it assumes nothing about the
shape of the seed distribution, and the exact form because the samples are
small. With n seeds per side the smallest possible two-sided p is
2 / C(n1 + n2, n1), so three seeds against three can never go below 0.1: the
report states that bound rather than claiming significance it cannot have.

    python scripts/seed_significance.py

Writes `results/metrics/significance.json`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from math import comb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = REPO_ROOT / "results" / "metrics"
PREVALENCE_KEY = "at_4%"

COMPARISONS = [
    # (dataset, label, (file, entry), (file, entry))
    ("NF-ToN-IoT-v2", "full vs channel1_only",
     ("gnn/gnn_NF-ToN-IoT-v2.json", "full"), ("gnn/gnn_NF-ToN-IoT-v2.json", "channel1_only")),
    ("NF-UNSW-NB15-v2", "full vs channel1_only",
     ("gnn/gnn_NF-UNSW-NB15-v2.json", "full"), ("gnn/gnn_NF-UNSW-NB15-v2.json", "channel1_only")),
    ("NF-ToN-IoT-v2", "full vs xgboost::flow+host",
     ("gnn/gnn_NF-ToN-IoT-v2.json", "full"),
     ("baselines/baselines_NF-ToN-IoT-v2.json", "xgboost::flow+host")),
]


def seed_values(rel: str, entry: str) -> list[float]:
    """PR-AUC at the reporting prevalence, one value per seed."""
    doc = json.loads((METRICS / rel).read_text())
    return [run[PREVALENCE_KEY]["pr_auc"] for run in doc["results"][entry]["runs"]]


def min_two_sided_p(n1: int, n2: int) -> float:
    """The smallest p an exact two-sided Mann-Whitney test can return."""
    return min(1.0, 2 / comb(n1 + n2, n1))


def compare(a: list[float], b: list[float]) -> dict:
    from scipy.stats import mannwhitneyu

    res = mannwhitneyu(a, b, alternative="two-sided", method="exact")
    return {
        "n": [len(a), len(b)],
        "u": float(res.statistic),
        "p_two_sided": float(res.pvalue),
        "min_possible_p": min_two_sided_p(len(a), len(b)),
        "ranges_separated": min(a) > max(b) or min(b) > max(a),
        "mean_difference": round(sum(a) / len(a) - sum(b) / len(b), 5),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=METRICS / "significance.json")
    args = ap.parse_args()

    rows = []
    for dataset, label, (fa, ea), (fb, eb) in COMPARISONS:
        r = compare(seed_values(fa, ea), seed_values(fb, eb))
        rows.append({"dataset": dataset, "comparison": label,
                     "sources": [f"{fa}::{ea}", f"{fb}::{eb}"], **r})
        print(f"  {dataset:16s} {label:28s} n={r['n']}  diff={r['mean_difference']:+.4f}  "
              f"p={r['p_two_sided']:.2e}  (min possible {r['min_possible_p']:.2e})")

    args.out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test": "Mann-Whitney U, exact, two-sided",
        "metric": f"PR-AUC {PREVALENCE_KEY.replace('at_', 'at ')} prevalence, per seed",
        "comparisons": rows,
    }, indent=2))
    shown = args.out.relative_to(REPO_ROOT) if args.out.is_relative_to(REPO_ROOT) else args.out
    print(f"\n  wrote {shown}")


if __name__ == "__main__":
    main()
