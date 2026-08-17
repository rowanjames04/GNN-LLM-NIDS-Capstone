"""Contiguous, order-preserving train/val/test splits (D5).

Never random. Two independent reasons, both measured in Phase 0/1:

  1. Consecutive flows within an attack episode are near-duplicates. Phase 0
     measured a mean label-run length of 14.5, so a random split scatters
     near-identical rows across the boundary and the model is scored on
     examples it has effectively memorised.

  2. Several NetFlow features are windowed aggregates computed over surrounding
     records, so a feature attached to a training row was partly computed from
     rows that would land in test. No amount of de-duplication fixes this --
     only an ordered split does.

Honest naming: this is *contiguous*, not verifiably *temporal*. The dataset
carries no timestamp (confirmed Phase 0), so row order is the only ordering
available and we cannot prove it is capture order. The report states this as a
limitation rather than claiming a chronological split.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    """Half-open row range [start, stop) for one split."""

    name: str
    start: int
    stop: int

    def __len__(self) -> int:
        return self.stop - self.start

    def as_dict(self) -> dict:
        return {"name": self.name, "start": self.start, "stop": self.stop, "n_rows": len(self)}


def contiguous_split(
    n_rows: int,
    train: float = 0.70,
    val: float = 0.10,
    test: float = 0.20,
    purge_gap: int = 0,
) -> dict[str, Split]:
    """Split rows into three contiguous blocks, discarding a gap between them.

    The purge gap drops `purge_gap` rows on *each* side of a boundary. Windowed
    features can only reach a limited distance, so removing a margin removes the
    residual leakage they would otherwise carry across. Standard practice in
    financial time-series validation, and cheap here -- 10k rows out of 2.39M.
    """
    if not 0.99 < train + val + test < 1.01:
        raise ValueError(f"fractions must sum to 1, got {train + val + test}")
    if n_rows <= 4 * purge_gap:
        raise ValueError(f"purge_gap {purge_gap} too large for {n_rows} rows")

    train_end = int(n_rows * train)
    val_end = int(n_rows * (train + val))

    splits = {
        "train": Split("train", 0, train_end - purge_gap),
        "val": Split("val", train_end + purge_gap, val_end - purge_gap),
        "test": Split("test", val_end + purge_gap, n_rows),
    }

    for s in splits.values():
        if len(s) <= 0:
            raise ValueError(f"split {s.name!r} is empty; check fractions and purge_gap")
    return splits


def assert_no_overlap(splits: dict[str, Split]) -> None:
    """Guard against the failure this whole module exists to prevent."""
    ordered = sorted(splits.values(), key=lambda s: s.start)
    for a, b in zip(ordered, ordered[1:]):
        if a.stop > b.start:
            raise AssertionError(f"splits {a.name} and {b.name} overlap: {a.stop} > {b.start}")
