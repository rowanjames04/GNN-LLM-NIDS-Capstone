"""Flows per class and split, for the report's dataset analysis.

`preprocess.py` records the attack rate of each split but not how the attack
families are spread across them, and the report has to show the imbalance
rather than assert it. This script counts every class (Benign and each attack
family) in each split of the preprocessed rows, the same rows the models train
and test on, and reports the ratio between the largest and smallest attack
family.

Read-only: it reads `meta.parquet` and `splits.json` from each processed
dataset and writes one JSON.

    python scripts/class_balance.py

Writes `results/metrics/class_balance.json`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = REPO_ROOT / "results" / "metrics"
DATASETS = {
    "NF-ToN-IoT-v2": REPO_ROOT / "data/processed/toniot",
    "NF-UNSW-NB15-v2": REPO_ROOT / "data/processed",
}
SPLITS = ("train", "val", "test")


def count_classes(attack: pd.Series, splits: dict) -> dict:
    """Flows per class in each split. Rows purged between splits are not counted."""
    out = {}
    for name in SPLITS:
        s = splits[name]
        out[name] = attack.iloc[s["start"]:s["stop"]].value_counts().to_dict()
    classes = sorted(set().union(*(c.keys() for c in out.values())), key=str.lower)
    return {c: {name: int(out[name].get(c, 0)) for name in SPLITS} for c in classes}


def summarise(counts: dict) -> dict:
    total = {c: sum(v.values()) for c, v in counts.items()}
    n = sum(total.values())
    attacks = {c: t for c, t in total.items() if c != "Benign"}
    largest = max(attacks, key=attacks.get)
    smallest = min(attacks, key=attacks.get)
    return {
        "n_flows": n,
        "share_of_flows": {c: round(t / n, 6) for c, t in total.items()},
        "share_of_attacks": {c: round(t / sum(attacks.values()), 6) for c, t in attacks.items()},
        "largest_family": largest,
        "smallest_family": smallest,
        "largest_to_smallest": round(attacks[largest] / attacks[smallest], 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=METRICS / "class_balance.json")
    args = ap.parse_args()

    result = {"generated_at": datetime.now(timezone.utc).isoformat(), "datasets": {}}
    for name, base in DATASETS.items():
        attack = pd.read_parquet(base / "meta.parquet", columns=["Attack"])["Attack"].astype(str)
        splits = json.loads((base / "splits.json").read_text())
        counts = count_classes(attack, splits)
        result["datasets"][name] = {
            "data_version": json.loads((base / "data_version.json").read_text())["data_version"],
            "counts": counts,
            **summarise(counts),
        }
        s = result["datasets"][name]
        print(f"  {name}: {s['n_flows']:,} flows, largest family {s['largest_family']} is "
              f"{s['largest_to_smallest']}x the smallest ({s['smallest_family']})")

    args.out.write_text(json.dumps(result, indent=2))
    shown = args.out.relative_to(REPO_ROOT) if args.out.is_relative_to(REPO_ROOT) else args.out
    print(f"\n  wrote {shown}")


if __name__ == "__main__":
    main()
