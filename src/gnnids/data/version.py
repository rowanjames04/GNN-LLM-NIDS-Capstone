"""Which preprocessed dataset a result was built from (C32).

The training campaign reuses a finished run when its configuration matches.
Until 2026-10-17 "configuration" meant the YAML only, so re-running
preprocessing with a different feature list left the fingerprint unchanged and
the leave-one-attack-out runs would have been reused from the old data. The
data version closes that: it is a hash of the fitted pipeline and the splits,
and it is folded into every run fingerprint and stamped into every checkpoint
and metrics file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def data_version(proc_dir: Path) -> str:
    """16 hex characters naming the preprocessed data in `proc_dir`.

    pipeline.json holds the feature names and every fitted statistic, and
    splits.json the row ranges, so between them they change whenever the
    model's inputs do. Read as parsed JSON and re-serialised with sorted keys,
    so formatting cannot change the hash.
    """
    proc_dir = Path(proc_dir)
    h = hashlib.sha256()
    for name in ("pipeline.json", "splits.json"):
        path = proc_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} -- run scripts/preprocess.py first")
        h.update(name.encode())
        h.update(json.dumps(json.loads(path.read_text()), sort_keys=True).encode())
    return h.hexdigest()[:16]


def recorded_version(proc_dir: Path) -> str | None:
    """The version preprocess.py stamped, or None for data made before it did."""
    path = Path(proc_dir) / "data_version.json"
    return json.loads(path.read_text()).get("data_version") if path.exists() else None


REPO_ROOT = Path(__file__).resolve().parents[3]


def version_for_preprocess_config(preprocess_config: str | Path) -> str:
    """The data version of the dataset a preprocess config writes.

    Returns "unavailable" when that dataset has not been preprocessed, so a
    fingerprint can still be computed in a test; a real run cannot train
    without the data, so it never sees that value.
    """
    import yaml

    path = Path(preprocess_config)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        cfg = yaml.safe_load(path.read_text())
        return data_version(REPO_ROOT / cfg["output"]["dir"])
    except (FileNotFoundError, KeyError, TypeError):
        return "unavailable"


def check_checkpoint(ckpt: dict, preprocess_config: str | Path, what: str) -> None:
    """Refuse a checkpoint trained on different data from what it will read.

    A model trained before a preprocessing change, loaded against the new
    features, runs and produces numbers whenever the edge dimension happens to
    match. Checkpoints made before 2026-10-17 carry no data version and are
    refused for the same reason.
    """
    want = version_for_preprocess_config(preprocess_config)
    got = ckpt.get("data_version")
    if got != want:
        raise SystemExit(
            f"{what}: checkpoint was trained on data version {got!r}, but the "
            f"data it would read is {want!r}. Retrain it on the current "
            f"preprocessing (scripts/train_gnn.py) rather than mixing the two.")
