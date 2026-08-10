"""Inspect the dataset and answer the questions the design depends on.

Nothing in this project should be built on assumptions about the file's
contents. Three answers in particular gate design decisions that are already
written into the Obsidian notes, and every one of them is currently taken from
the literature rather than from the file itself:

  1. How many unique IP addresses are there, and how dense is the host graph?
     -> Decision D2 (edge-level classification) rests on the claim that the host
        graph is near-degenerate. If there turn out to be thousands of hosts
        with sparse connectivity, that argument weakens and node-level
        classification deserves a second look.

  2. Is there any usable time information?
     -> Decision D6. Absolute timestamps keep the Granger causality module
        alive; ordering alone is enough for temporal splitting (D5); neither
        means Granger is dead and Tier 1 explainability carries the project.

  3. How severe is the class imbalance, per attack family?
     -> Sets focal loss weighting (D8) and calibrates what to expect from
        leave-one-attack-out (D4). Families with only a few hundred examples
        will have noisy held-out recall and that needs saying up front.

Writes everything to results/metrics/dataset_schema.json so the answers are a
committed artefact rather than terminal output that scrolls away.

Usage:
    python scripts/verify_dataset.py
    python scripts/verify_dataset.py --input data/raw/NF-UNSW-NB15-v2.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dataset.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "metrics" / "dataset_schema.json"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def find_input(config: dict, explicit: Path | None) -> Path:
    """Locate the dataset, preferring Parquet over CSV when both exist."""
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"No such file: {explicit}")
        return explicit

    filename = config["source"]["filename"]
    stem = Path(filename).stem
    candidates = [
        REPO_ROOT / config["paths"]["interim"] / f"{stem}.parquet",
        REPO_ROOT / config["paths"]["raw"] / filename,
    ]
    for path in candidates:
        if path.exists():
            return path

    searched = "\n  ".join(str(c) for c in candidates)
    raise SystemExit(
        f"Dataset not found. Looked for:\n  {searched}\n\n"
        f"Download it from {config['source']['landing_page']} (or Kaggle: "
        f"{config['source']['kaggle']}) and place the CSV in "
        f"{config['paths']['raw']}/, then re-run."
    )


def load(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    # low_memory=False: mixed-type inference across chunks silently produces
    # object columns for what should be numerics, which corrupts the stats below.
    return pd.read_csv(path, low_memory=False)


# --------------------------------------------------------------------------
# question 1: graph shape
# --------------------------------------------------------------------------

def analyse_graph_shape(df: pd.DataFrame, cols: dict) -> dict:
    """Measure how much topology actually exists.

    The headline number is density: unique ordered host pairs divided by the
    number possible. Close to 1.0 means every host talks to every other host,
    which is the degeneracy that motivates classifying edges rather than nodes.
    """
    src_col, dst_col = cols["src_ip"], cols["dst_ip"]
    missing = [c for c in (src_col, dst_col) if c not in df.columns]
    if missing:
        return {"error": f"IP columns not found in file: {missing}"}

    src, dst = df[src_col], df[dst_col]
    hosts = pd.unique(pd.concat([src, dst], ignore_index=True))
    n_hosts = len(hosts)

    pairs = df.groupby([src_col, dst_col], observed=True).size()
    possible = n_hosts * (n_hosts - 1)

    out_degree = df.groupby(src_col, observed=True)[dst_col].nunique()
    flows_per_src = src.value_counts()

    return {
        "unique_src_ips": int(src.nunique()),
        "unique_dst_ips": int(dst.nunique()),
        "unique_hosts_total": int(n_hosts),
        "unique_ordered_pairs": int(len(pairs)),
        "possible_ordered_pairs": int(possible),
        "graph_density": round(len(pairs) / possible, 6) if possible else None,
        "flows_per_pair": {
            "mean": round(float(pairs.mean()), 2),
            "median": float(pairs.median()),
            "max": int(pairs.max()),
        },
        "distinct_peers_per_src": {
            "mean": round(float(out_degree.mean()), 2),
            "median": float(out_degree.median()),
            "max": int(out_degree.max()),
        },
        "flows_per_src_ip": {
            "mean": round(float(flows_per_src.mean()), 2),
            "median": float(flows_per_src.median()),
            "max": int(flows_per_src.max()),
        },
        "top_10_talkers": {
            str(k): int(v) for k, v in flows_per_src.head(10).items()
        },
    }


# --------------------------------------------------------------------------
# question 2: time
# --------------------------------------------------------------------------

def analyse_time(df: pd.DataFrame, candidates: list[str]) -> dict:
    """Determine what time information, if any, is recoverable.

    Three outcomes, in descending order of usefulness:
      absolute   - a real timestamp column. Granger causality is viable.
      ordinal    - no timestamp, but row order is usable as a proxy ordering.
                   Temporal splitting works; Granger probably does not.
      none       - neither.

    The ordinal check looks for a monotonically non-decreasing numeric column,
    which would betray a sequence or time-like field under a non-obvious name.
    """
    found = [c for c in candidates if c in df.columns]

    detail: dict = {}
    for col in found:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        detail[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "monotonic_non_decreasing": bool(series.is_monotonic_increasing),
            "n_unique": int(series.nunique()),
        }

    # Probe every numeric column for monotonicity, in case a time-like field
    # exists under a name not in the candidate list.
    monotonic_numeric = []
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].dropna()
        if len(series) > 1 and series.is_monotonic_increasing and series.nunique() > 1:
            monotonic_numeric.append(str(col))

    if detail:
        verdict = "absolute"
        implication = "Granger causality (D6 Tier 2) is viable. Proceed with Phase 6b."
    elif monotonic_numeric:
        verdict = "ordinal"
        implication = (
            f"No timestamp, but {monotonic_numeric} is monotonic and may encode "
            "order. Temporal splitting (D5) is safe; Granger is doubtful."
        )
    else:
        verdict = "none"
        implication = (
            "No time column and no monotonic numeric column. Fall back to row "
            "order for temporal splitting (D5). Granger causality (D6 Tier 2) "
            "is almost certainly not viable -- write up why instead."
        )

    return {
        "verdict": verdict,
        "implication": implication,
        "named_candidates_found": detail,
        "monotonic_numeric_columns": monotonic_numeric,
    }


# --------------------------------------------------------------------------
# question 3: class balance
# --------------------------------------------------------------------------

def analyse_labels(df: pd.DataFrame, cols: dict) -> dict:
    """Class balance overall and per attack family.

    Also computes the inverse-frequency weights that seed focal loss alpha,
    and flags families too small for a stable leave-one-attack-out result.
    """
    out: dict = {}
    n = len(df)

    binary_col = cols["label_binary"]
    if binary_col in df.columns:
        counts = df[binary_col].value_counts()
        n_attack = int(counts.get(1, 0))
        out["binary"] = {
            "column": binary_col,
            "counts": {str(k): int(v) for k, v in counts.items()},
            "attack_prevalence": round(n_attack / n, 6) if n else None,
            "imbalance_ratio": (
                round((n - n_attack) / n_attack, 2) if n_attack else None
            ),
        }

    multi_col = cols["label_multiclass"]
    if multi_col in df.columns:
        counts = df[multi_col].value_counts()
        families = {str(k): int(v) for k, v in counts.items()}
        # Inverse-frequency weights, normalised to mean 1 so the overall loss
        # scale is unchanged. Starting point for focal loss alpha (D8).
        raw = {k: n / (len(counts) * v) for k, v in families.items()}
        mean_w = sum(raw.values()) / len(raw)
        out["multiclass"] = {
            "column": multi_col,
            "n_classes": int(len(counts)),
            "counts": families,
            "proportions": {k: round(v / n, 6) for k, v in families.items()},
            "suggested_class_weights": {
                k: round(v / mean_w, 4) for k, v in raw.items()
            },
            "families_under_1000": [k for k, v in families.items() if v < 1000],
        }

    return out


# --------------------------------------------------------------------------
# general schema
# --------------------------------------------------------------------------

def analyse_schema(df: pd.DataFrame) -> dict:
    nulls = df.isna().sum()
    nunique = df.nunique()
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 1),
        "columns": [
            {
                "name": str(c),
                "dtype": str(df[c].dtype),
                "n_unique": int(nunique[c]),
                "n_null": int(nulls[c]),
            }
            for c in df.columns
        ],
        "columns_with_nulls": {
            str(c): int(v) for c, v in nulls[nulls > 0].items()
        },
        # Constant columns carry no information and should be dropped before
        # training; they also quietly waste a slot in every feature vector.
        "constant_columns": [str(c) for c in df.columns if nunique[c] <= 1],
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def print_report(report: dict) -> None:
    schema, graph = report["schema"], report["graph_shape"]
    time_info, labels = report["time"], report["labels"]

    print(f"\n{'=' * 68}\n  {report['dataset']}  --  {report['input_file']}\n{'=' * 68}")
    print(f"\n{schema['n_rows']:,} rows x {schema['n_columns']} columns "
          f"({schema['memory_mb']:,.1f} MB in memory)")
    if schema["constant_columns"]:
        print(f"  constant columns (drop these): {schema['constant_columns']}")
    if schema["columns_with_nulls"]:
        print(f"  columns with nulls: {schema['columns_with_nulls']}")

    print(f"\n--- Q1: graph shape  (gates D2, edge-level classification) ---")
    if "error" in graph:
        print(f"  {graph['error']}")
    else:
        print(f"  unique hosts:        {graph['unique_hosts_total']:,}")
        print(f"  unique ordered pairs: {graph['unique_ordered_pairs']:,} "
              f"of {graph['possible_ordered_pairs']:,} possible")
        print(f"  graph density:        {graph['graph_density']}")
        print(f"  flows per pair:       mean {graph['flows_per_pair']['mean']:,} "
              f"/ max {graph['flows_per_pair']['max']:,}")
        density = graph["graph_density"]
        if density is not None:
            if density > 0.5:
                print("  => DENSE. Host graph is near-complete; D2 (edge-level) "
                      "is strongly justified.")
            elif density > 0.1:
                print("  => MODERATELY DENSE. D2 still justified; note the "
                      "density in the report.")
            else:
                print("  => SPARSE. Re-examine D2 -- there may be more usable "
                      "topology than assumed.")

    print(f"\n--- Q2: time  (gates D5 splitting, D6 Granger) ---")
    print(f"  verdict: {time_info['verdict'].upper()}")
    print(f"  {time_info['implication']}")

    print(f"\n--- Q3: class balance  (sets D8 loss weighting, D4 expectations) ---")
    if "binary" in labels:
        b = labels["binary"]
        print(f"  attack prevalence: {b['attack_prevalence']:.4%} "
              f"(1 attack per {b['imbalance_ratio']:,.1f} benign)")
    if "multiclass" in labels:
        m = labels["multiclass"]
        print(f"  {m['n_classes']} classes:")
        for name, count in sorted(m["counts"].items(), key=lambda kv: -kv[1]):
            print(f"    {name:<20} {count:>10,}  ({m['proportions'][name]:.4%})")
        if m["families_under_1000"]:
            print(f"  => families under 1000 examples: {m['families_under_1000']}")
            print("     Expect noisy leave-one-attack-out recall for these; say "
                  "so in the report.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    path = find_input(config, args.input)

    print(f"Loading {path} ...")
    df = load(path)

    report = {
        "dataset": config["name"],
        "input_file": str(path.relative_to(REPO_ROOT)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema": analyse_schema(df),
        "graph_shape": analyse_graph_shape(df, config["columns"]),
        "time": analyse_time(df, config["timestamp_candidates"]),
        "labels": analyse_labels(df, config["columns"]),
    }

    print_report(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nWritten to {args.output.relative_to(REPO_ROOT)}\n")


if __name__ == "__main__":
    main()
