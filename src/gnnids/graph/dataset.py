"""Serve graph snapshots to the training loop.

Snapshots are built on demand rather than precomputed and stored. Construction
is cheap -- a `np.unique` over the window's endpoints plus a few tensor copies --
while materialising every window would cost gigabytes for no benefit.

Window size **must** match the value used to build the splits. On
NF-ToN-IoT-v2 the split itself is an assignment of whole windows (D21), so a
different size here would silently cut across split boundaries and leak.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from ..data.splits import Split
from .build import build_snapshot
from .windows import fixed_count_windows


class SnapshotDataset(torch.utils.data.Dataset):
    """Windows of one split, each yielded as a PyG ``Data`` graph."""

    def __init__(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        edge_features: np.ndarray,
        y_binary: np.ndarray,
        y_multiclass: np.ndarray,
        split: Split,
        window_size: int,
    ) -> None:
        self.src, self.dst = src, dst
        self.edge_features = edge_features
        self.y_binary, self.y_multiclass = y_binary, y_multiclass
        self.windows = list(fixed_count_windows(split, window_size))
        if not self.windows:
            raise ValueError(
                f"split {split.name!r} has {len(split)} rows, too few for "
                f"window size {window_size}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> Data:
        lo, hi = self.windows[i]
        # Categorical features were one-hot expanded upstream so that Channel 1
        # sees exactly what the Phase 3 MLP baseline saw; the empty categorical
        # slot keeps build_snapshot's signature honest.
        return build_snapshot(
            self.src[lo:hi], self.dst[lo:hi],
            self.edge_features[lo:hi],
            np.zeros((hi - lo, 0), dtype=np.int64),
            self.y_binary[lo:hi], self.y_multiclass[lo:hi],
            window_index=i,
        )

    @property
    def n_edges(self) -> int:
        return sum(hi - lo for lo, hi in self.windows)

    def attack_rate(self) -> float:
        total = sum(int(self.y_binary[lo:hi].sum()) for lo, hi in self.windows)
        return total / max(self.n_edges, 1)
