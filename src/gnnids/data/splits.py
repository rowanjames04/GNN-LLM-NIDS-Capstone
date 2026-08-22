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


def window_split(
    n_rows: int,
    window_size: int,
    train: float = 0.70,
    val: float = 0.10,
    test: float = 0.20,
    seed: int = 42,
) -> tuple[dict[str, Split], "np.ndarray"]:
    """Assign whole windows to splits at random, then reorder so each split is
    contiguous again.

    For NF-ToN-IoT-v2 the contiguous split above is actively harmful. Its rows
    are ordered by capture scenario rather than by time, so `xss` occupies
    positions 0.81-0.98 and `ransomware` sits at 0.000. A contiguous 70/10/20
    therefore produced train 53% / val 80% / **test 97%** attack, with the test
    split 90% one family that barely appears in training -- a ragged accidental
    leave-one-attack-out reported as the conventional protocol.

    D5 is not being reversed; its precondition is absent. D5 chose contiguous
    splits because consecutive flows are near-duplicates *in time*. Where the
    ordering is not temporal, that argument does not apply and a contiguous cut
    just partitions by attack type.

    Windows rather than rows, for the D20 reason: a window is assigned as a
    unit, so its internal topology survives and a near-duplicate episode travels
    to one split rather than being scattered across all three.

    Returns the splits and a **permutation** that groups train, then val, then
    test. Applying it to the dataframe makes every split a contiguous range
    again, so everything downstream is unchanged.

    One property worth knowing: because assignment is random over windows, the
    class balance it achieves carries sampling variance scaling as
    1/sqrt(n_windows). At 100 windows the residual imbalance is around 0.13; by
    1,000 it is under 0.03. Check the reported per-split attack rates rather
    than assuming the balance came out even, especially at large window sizes
    where the window count falls.
    """
    import numpy as np

    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if not 0.99 < train + val + test < 1.01:
        raise ValueError(f"fractions must sum to 1, got {train + val + test}")

    n_windows = n_rows // window_size
    if n_windows < 3:
        raise ValueError(
            f"{n_rows} rows at window {window_size} gives {n_windows} windows; "
            f"need at least 3")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_windows)
    n_tr = int(n_windows * train)
    n_va = int(n_windows * val)
    assigned = {
        "train": np.sort(order[:n_tr]),
        "val": np.sort(order[n_tr:n_tr + n_va]),
        "test": np.sort(order[n_tr + n_va:]),
    }

    # Rows of the assigned windows, split by split, in original order within
    # each. Trailing rows beyond the last whole window are dropped.
    pieces, splits, cursor = [], {}, 0
    for name in ("train", "val", "test"):
        idx = np.concatenate([
            np.arange(w * window_size, (w + 1) * window_size) for w in assigned[name]
        ]) if len(assigned[name]) else np.array([], dtype=np.int64)
        pieces.append(idx)
        splits[name] = Split(name, cursor, cursor + len(idx))
        cursor += len(idx)

    return splits, np.concatenate(pieces).astype(np.int64)


def assert_no_overlap(splits: dict[str, Split]) -> None:
    """Guard against the failure this whole module exists to prevent."""
    ordered = sorted(splits.values(), key=lambda s: s.start)
    for a, b in zip(ordered, ordered[1:]):
        if a.stop > b.start:
            raise AssertionError(f"splits {a.name} and {b.name} overlap: {a.stop} > {b.start}")
