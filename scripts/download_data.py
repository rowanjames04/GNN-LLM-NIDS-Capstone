"""Fetch the dataset, checksum it, and convert it to Parquet.

The NF-v2 collection is distributed through UQ's research data manager and
mirrored on Kaggle. Both routes may require accepting terms in a browser, so
this script does not assume an unattended download is possible: if no URL is
configured it checks for a manually placed file and tells you where to get it.

Parquet conversion is not cosmetic. At ~2.4M rows the CSV takes tens of seconds
to parse on every single run; Parquet is columnar, typed and compressed, and
loads roughly an order of magnitude faster. Over a project with this many
training runs that difference is hours.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --force-reconvert
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dataset.yaml"
CHUNK = 1024 * 1024


def sha256_of(path: Path) -> str:
    """Hash in chunks -- the file is too large to hold in memory twice."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n  -> {dest}")

    def progress(count: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100.0, count * block_size * 100 / total)
            print(f"\r  {pct:5.1f}%  of {total / 1024**2:,.0f} MB", end="")

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()


def to_parquet(csv_path: Path, parquet_path: Path) -> None:
    print(f"Converting to Parquet\n  {csv_path.name} -> {parquet_path.name}")
    df = pd.read_csv(csv_path, low_memory=False)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False, compression="snappy")

    csv_mb = csv_path.stat().st_size / 1024**2
    pq_mb = parquet_path.stat().st_size / 1024**2
    print(f"  {len(df):,} rows x {df.shape[1]} cols")
    print(f"  {csv_mb:,.0f} MB -> {pq_mb:,.0f} MB ({csv_mb / pq_mb:.1f}x smaller)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force-reconvert", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    source = config["source"]

    raw_dir = REPO_ROOT / config["paths"]["raw"]
    interim_dir = REPO_ROOT / config["paths"]["interim"]
    csv_path = raw_dir / source["filename"]
    parquet_path = interim_dir / f"{Path(source['filename']).stem}.parquet"

    if not csv_path.exists():
        if source["url"]:
            download(source["url"], csv_path)
        else:
            raise SystemExit(
                f"\n{source['filename']} not found in {raw_dir}/, and no direct "
                f"URL is configured.\n\n"
                f"Get it from one of:\n"
                f"  UQ:     {source['landing_page']}\n"
                f"  Kaggle: kaggle datasets download -d {source['kaggle']}\n\n"
                f"Place the CSV at {csv_path}, then re-run this script.\n"
                f"(If you find a stable direct URL, put it in configs/dataset.yaml"
                f" so this becomes reproducible.)\n"
            )

    digest = sha256_of(csv_path)
    expected = source.get("sha256")
    if expected is None:
        print(f"\nsha256: {digest}")
        print("Record this in configs/dataset.yaml so future runs are verified.")
    elif digest != expected:
        raise SystemExit(
            f"\nCHECKSUM MISMATCH\n  expected {expected}\n  got      {digest}\n"
            f"The file differs from the one this project was built against."
        )
    else:
        print(f"\nChecksum verified: {digest[:16]}...")

    if args.force_reconvert or not parquet_path.exists():
        to_parquet(csv_path, parquet_path)
    else:
        print(f"Parquet already present at {parquet_path.relative_to(REPO_ROOT)}")

    print("\nNext: python scripts/verify_dataset.py\n")


if __name__ == "__main__":
    main()
