"""Turn a window of flows into a PyG graph snapshot.

The formulation is D2, edge-level classification in the E-GraphSAGE style:

  nodes  = IP addresses, carrying NO intrinsic features
  edges  = flows, carrying all 30-odd measured features
  target = the edge

Nodes are initialised to a constant vector because there is nothing else to
initialise them with -- a host's representation becomes entirely a function of
the traffic that passes through it. That is the property worth stating in the
report, and it is also what makes the model inductive: embeddings are computed
from features and neighbourhood rather than looked up per address, so unseen
IPs (and an entirely different network, in cross-dataset transfer) work without
retraining.

Snapshots are directed multigraphs. Multi-edges are mandatory rather than
chosen: each flow carries its own label and we classify per flow, so flows
between the same pair cannot be collapsed. Edge count therefore equals flow
count exactly, which is what bounds memory and hence window size.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data


class GlobalNodeMap:
    """Stable IP -> integer identity across all windows.

    Within a window, nodes are renumbered 0..N-1 for PyG. That local index is
    meaningless outside the window, so per-host baselines ("this host's fan-out
    versus its own norm", needed for the LLM evidence pack) and any future
    temporal work need an identity that persists. Maintaining it now costs
    nothing; retrofitting it later means rebuilding every snapshot.
    """

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def get(self, key: str) -> int:
        if key not in self._ids:
            self._ids[key] = len(self._ids)
        return self._ids[key]

    def get_many(self, keys: np.ndarray) -> np.ndarray:
        return np.fromiter((self.get(str(k)) for k in keys), dtype=np.int64, count=len(keys))

    def __len__(self) -> int:
        return len(self._ids)

    def as_dict(self) -> dict[str, int]:
        return dict(self._ids)


def build_snapshot(
    src: np.ndarray,
    dst: np.ndarray,
    continuous: np.ndarray,
    categorical: np.ndarray,
    y_binary: np.ndarray,
    y_multiclass: np.ndarray,
    global_map: GlobalNodeMap | None = None,
    window_index: int | None = None,
) -> Data:
    """Build one PyG snapshot from parallel arrays of a single window.

    All inputs must be the same length; that length becomes the edge count.
    """
    n_flows = len(src)
    for name, arr in (("dst", dst), ("continuous", continuous),
                      ("categorical", categorical), ("y_binary", y_binary),
                      ("y_multiclass", y_multiclass)):
        if len(arr) != n_flows:
            raise ValueError(f"{name} has length {len(arr)}, expected {n_flows}")

    # Renumber the addresses appearing in this window to 0..N-1. np.unique with
    # return_inverse does both the vocabulary and the lookup in one pass, which
    # matters at 10k+ edges per window across hundreds of windows.
    endpoints = np.concatenate([src, dst])
    uniques, inverse = np.unique(endpoints, return_inverse=True)
    local_src, local_dst = inverse[:n_flows], inverse[n_flows:]

    # np.array(..., copy=True) rather than ascontiguousarray: slices of a
    # memory-mapped or read-only source stay read-only, and torch.from_numpy on
    # a read-only array yields a tensor whose writes are undefined behaviour.
    data = Data(
        # Constant node features: all information enters through the edges.
        x=torch.ones((len(uniques), 1), dtype=torch.float32),
        edge_index=torch.from_numpy(np.stack([local_src, local_dst])).long(),
        edge_attr=torch.from_numpy(np.array(continuous, dtype=np.float32, copy=True)),
        edge_cat=torch.from_numpy(np.array(categorical, dtype=np.int64, copy=True)),
        y=torch.from_numpy(np.array(y_binary, dtype=np.int64, copy=True)),
        y_multiclass=torch.from_numpy(np.array(y_multiclass, dtype=np.int64, copy=True)),
    )
    data.num_nodes = len(uniques)

    if global_map is not None:
        data.global_node_id = torch.from_numpy(global_map.get_many(uniques)).long()
    if window_index is not None:
        data.window_index = torch.tensor([window_index], dtype=torch.long)
    return data


def snapshot_stats(data: Data) -> dict:
    """Structural summary of one snapshot, for the Phase 2 window-size sweep.

    Reports the *spread* of fan-out across hosts, not only its maximum. The
    maximum saturates at 10 in this dataset -- every window contains at least
    one host that has contacted all its peers -- so it has zero variance and
    says nothing about whether structure varies. What message passing can
    actually exploit is the difference *between* hosts within a window, which
    is what fanout_std and fanout_p90 capture.
    """
    n, e = int(data.num_nodes), int(data.edge_index.shape[1])
    # Distinct peers per source. Phase 0 measured that this carries real signal
    # under the IP node scheme (AUC 0.749 for predicting an attacking host).
    pairs = torch.unique(data.edge_index.t(), dim=0)
    out_deg = torch.bincount(pairs[:, 0], minlength=n).float()
    active = out_deg[out_deg > 0]
    flows_per_node = torch.bincount(data.edge_index[0], minlength=n).float()

    return {
        "n_nodes": n,
        "n_edges": e,
        "n_unique_pairs": int(pairs.shape[0]),
        "density": round(pairs.shape[0] / (n * (n - 1)), 6) if n > 1 else 0.0,
        "max_fanout": int(out_deg.max().item()) if n else 0,
        "mean_fanout": round(float(active.mean().item()), 3) if len(active) else 0.0,
        "fanout_std": round(float(active.std().item()), 3) if len(active) > 1 else 0.0,
        "fanout_p90": round(float(active.quantile(0.9).item()), 3) if len(active) else 0.0,
        "n_active_senders": int(len(active)),
        "flows_per_node_std": round(float(flows_per_node.std().item()), 2) if n > 1 else 0.0,
        "attack_rate": round(float(data.y.float().mean().item()), 6),
        "n_attack_edges": int(data.y.sum().item()),
    }
