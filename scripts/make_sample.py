"""Build a small, committable sample of the dataset.

The full dataset cannot live in git -- GitHub hard-rejects files over 100 MB,
and git stores every version of a binary file forever, so one accidental
re-commit bloats the repository permanently. It stays in data/raw/, gitignored,
fetched by download_data.py and verified by checksum.

A sample is a different matter, and earns its place three times over:

  - tests can run against real data rather than only a synthetic frame
  - the repo can be cloned and run end-to-end without a 1 GB download
  - graph construction can be iterated on in seconds instead of minutes

Sampling is stratified by attack family, not random. At ~4% attack prevalence a
random 50k sample would contain almost no Worms or Shellcode -- the rare
families that matter most for leave-one-attack-out evaluation. Every family is
kept at its true proportion, subject to a floor so the rare ones survive at all.

Contiguity is preserved within each family: rows are taken as a block rather
than scattered, so flows belonging to the same attack episode stay together.
Scattering them would destroy the very neighbourhood structure the GNN exists
to learn, and would make the sample useless for testing graph construction.

Usage:
    python scripts/make_sample.py
    python scripts/make_sample.py --rows 100000 --output data/sample.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dataset.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "sample" / "NF-UNSW-NB15-v2-sample.csv"

# Below this, a family cannot support a meaningful train/test split in the
# sample, so we over-sample it relative to its true proportion.
MIN_ROWS_PER_FAMILY = 200

# GitHub warns at 50 MB and hard-rejects at 100 MB. Stay well clear.
SIZE_WARN_MB = 40


def stratified_contiguous_sample(
    df: pd.DataFrame, label_col: str, target_rows: int
) -> pd.DataFrame:
    """Sample each family proportionally, as a contiguous block, with a floor."""
    total = len(df)
    frames = []

    for family, group in df.groupby(label_col, observed=True, sort=False):
        proportional = int(round(target_rows * len(group) / total))
        take = max(min(proportional, len(group)), min(MIN_ROWS_PER_FAMILY, len(group)))

        # Take from the middle of the family's rows. The head and tail of a
        # dataset are often setup and teardown traffic rather than the steady
        # state we want to develop against.
        start = max(0, (len(group) - take) // 2)
        frames.append(group.iloc[start : start + take])

    # Restore original row order so any ordering the file carries survives --
    # temporal splitting (D5) depends on it.
    return pd.concat(frames).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    stem = Path(config["source"]["filename"]).stem

    source = REPO_ROOT / config["paths"]["interim"] / f"{stem}.parquet"
    if not source.exists():
        source = REPO_ROOT / config["paths"]["raw"] / config["source"]["filename"]
    if not source.exists():
        raise SystemExit(
            f"Dataset not found. Run scripts/download_data.py first."
        )

    print(f"Loading {source.name} ...")
    df = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(
        source, low_memory=False
    )

    label_col = config["columns"]["label_multiclass"]
    if label_col not in df.columns:
        raise SystemExit(
            f"Label column {label_col!r} not in file. Run verify_dataset.py and "
            f"correct configs/dataset.yaml first."
        )

    sample = stratified_contiguous_sample(df, label_col, args.rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False)
    size_mb = args.output.stat().st_size / 1024**2

    print(f"\n{len(sample):,} rows -> {args.output.relative_to(REPO_ROOT)} "
          f"({size_mb:.1f} MB)")
    print("\nfamily composition (sample vs full):")
    full_prop = df[label_col].value_counts(normalize=True)
    for family, count in sample[label_col].value_counts().items():
        print(f"  {str(family):<20} {count:>8,}  "
              f"{count / len(sample):>7.3%}  (full: {full_prop[family]:.3%})")

    if size_mb > SIZE_WARN_MB:
        print(f"\nWARNING: {size_mb:.1f} MB is close to GitHub's 100 MB limit. "
              f"Re-run with a smaller --rows.")
    else:
        print(f"\nSafe to commit ({size_mb:.1f} MB, well under GitHub's limit).")


if __name__ == "__main__":
    main()
