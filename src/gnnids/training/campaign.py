"""Bookkeeping for multi-run campaigns: naming, fingerprints, reuse.

A campaign is many training runs whose results are merged into one file. Three
defects in this project came from that bookkeeping rather than from training:

- **C14 / C20** -- a run wrote to a filename an earlier run had used, and the
  earlier result was silently gone.
- **C19** -- finished runs lived only in the parent process's memory until the
  final merge, so a crash late in a campaign lost all of it.

So every run is written to its own partial file the moment it finishes, a
variant run (a different layer count, a tagged experiment) gets its own names,
and a partial is only reused by a campaign with the **same configuration** --
checked by fingerprint, because a partial from last week's config that happens
to share a filename would merge cleanly and mean nothing.

Deliberately free of torch: the parent process of a campaign only schedules.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# Not part of what a run *is*: how many seeds were asked for, and which device
# happened to execute it.
_NOT_IDENTITY = ("n_seeds", "device")


def fingerprint(gnn_cfg: dict, extra: dict | None = None) -> str:
    """A short hash of everything that determines a run's result.

    Two runs with the same fingerprint and seed are the same experiment. The
    seed count and the device are excluded -- asking for ten seeds instead of
    three does not change what seed 0 is.
    """
    ident = {
        "model": gnn_cfg["model"],
        "train": {k: v for k, v in gnn_cfg["train"].items() if k not in _NOT_IDENTITY},
        "eval": gnn_cfg["eval"],
        "preprocess_config": gnn_cfg["preprocess_config"],
        "ablations": {k: v for k, v in gnn_cfg["ablations"].items()
                      if isinstance(v, dict)},
        "extra": extra or {},
    }
    blob = json.dumps(ident, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def stem(dataset: str, tag: str | None) -> str:
    """`NF-ToN-IoT-v2`, or `NF-ToN-IoT-v2__layers3` for a tagged variant."""
    return f"{dataset}__{tag}" if tag else dataset


def checkpoint_name(dataset: str, ablation: str, seed: int, tag: str | None) -> str:
    suffix = f"__{tag}" if tag else ""
    return f"{dataset}_{ablation}_seed{seed}{suffix}.pt"


def read_if_matching(path: Path, fp: str) -> dict | None:
    """The file's contents if it exists and was produced by this configuration."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None                      # torn write from a killed process
    return data if data.get("fingerprint") == fp else None
