"""Chunk flows into graph snapshots.

One graph over 2.39M flows will not fit in 16 GB and would not train usefully
if it did, so flows are chunked and each chunk becomes one training sample.

Window size is not an incidental implementation detail -- it is the aperture
through which the GNN sees structure. Too small and a port scan is spread over
fifty windows so its fan-out never appears in any one of them; too large and
every host has contacted every other host and the structure saturates. Phase 0
measured the consequence directly: at 10k-flow windows the busiest host reaches
exactly 10 peers in every window (zero variance), so the aperture matters.

Fixed-count rather than fixed-duration, because this dataset has no timestamp
(confirmed Phase 0). Fixed-duration is physically the better choice and is kept
in the config for a dataset that supports it.

Windows never straddle a split boundary: they are generated *within* a split,
so no snapshot can mix training and test flows.
"""

from __future__ import annotations

from typing import Iterator

from ..data.splits import Split


def fixed_count_windows(
    split: Split, window_size: int, drop_last: bool = True
) -> Iterator[tuple[int, int]]:
    """Yield half-open [start, stop) row ranges inside one split.

    `drop_last` discards a trailing partial window. A short final window would
    have a systematically smaller node count and edge count than the rest,
    which shows up as an outlier in any per-window statistic.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    start = split.start
    while start < split.stop:
        stop = min(start + window_size, split.stop)
        if stop - start < window_size and drop_last:
            return
        yield start, stop
        start = stop


def count_windows(split: Split, window_size: int, drop_last: bool = True) -> int:
    n = len(split)
    return n // window_size if drop_last else -(-n // window_size)
