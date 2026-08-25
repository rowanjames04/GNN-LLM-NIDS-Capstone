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
        max_windows: int | None = None,
        row_index: np.ndarray | None = None,
    ) -> None:
        self.src, self.dst = src, dst
        self.edge_features = edge_features
        self.y_binary, self.y_multiclass = y_binary, y_multiclass
        # `row_index` selects a subset of this split's rows -- Phase 5 uses it
        # to hold out an attack family. Windows are then counted over the
        # *surviving* rows, so every window still carries `window_size` edges
        # and per-window statistics stay comparable with Phase 4's.
        #
        # The consequence worth knowing: surviving rows either side of a
        # removed block become neighbours in a window, so a window can span a
        # wider stretch of the original capture than it otherwise would. That
        # is the intended counterfactual -- "this attack family never occurred
        # on this network" -- and it is why holding a family out is done by
        # removing its flows rather than by relabelling them benign, which
        # would teach the model that attack traffic is benign.
        self.row_index = None if row_index is None else np.asarray(row_index)
        if self.row_index is not None:
            if self.row_index.ndim != 1:
                raise ValueError("row_index must be 1-D")
            span = Split(split.name, 0, len(self.row_index))
        else:
            span = split
        self.windows = list(fixed_count_windows(span, window_size))
        if max_windows is not None and len(self.windows) > max_windows:
            # Smoke runs take an even spread rather than a prefix: a prefix of a
            # scenario-ordered dataset samples one attack family (B4).
            idx = np.linspace(0, len(self.windows) - 1, max_windows).astype(int)
            self.windows = [self.windows[i] for i in idx]
        if not self.windows:
            raise ValueError(
                f"split {split.name!r} has {len(split)} rows, too few for "
                f"window size {window_size}")

    def __len__(self) -> int:
        return len(self.windows)

    def _rows(self, lo: int, hi: int):
        """Row selector for one window: a slice normally, a gather if filtered.

        The slice path is kept rather than always gathering, because a slice of
        a memory-mapped array is a view and a gather is a copy. Phase 4 reads
        every window of a 1.3 GB mapping.
        """
        return slice(lo, hi) if self.row_index is None else self.row_index[lo:hi]

    def __getitem__(self, i: int) -> Data:
        lo, hi = self.windows[i]
        rows = self._rows(lo, hi)
        # Categorical features were one-hot expanded upstream so that Channel 1
        # sees exactly what the Phase 3 MLP baseline saw; the empty categorical
        # slot keeps build_snapshot's signature honest.
        return build_snapshot(
            self.src[rows], self.dst[rows],
            self.edge_features[rows],
            np.zeros((hi - lo, 0), dtype=np.int64),
            self.y_binary[rows], self.y_multiclass[rows],
            window_index=i,
        )

    @property
    def n_edges(self) -> int:
        return sum(hi - lo for lo, hi in self.windows)

    def attack_rate(self) -> float:
        total = sum(int(self.y_binary[self._rows(lo, hi)].sum())
                    for lo, hi in self.windows)
        return total / max(self.n_edges, 1)
