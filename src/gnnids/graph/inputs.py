"""Load the preprocessed dataset into the arrays every graph phase needs.

Extracted from `scripts/train_gnn.py` on 2026-08-25 so Phase 5 and Phase 9 load
data the same way Phase 4 does. The memory discipline here is not incidental --
it is the difference between a run that completes and one that is OOM-killed on
a 16 GB machine (C11, C15, U1):

- **edge features are memory-mapped float16**, never resident. Held resident
  they are 1.3 GB and pushed the original Phase 4 run into swap at 23% CPU.
- **IP addresses are int32 codes**, not Python strings. As object arrays the
  2,556 distinct hosts cost 0.63 GB; as codes, 47 MB.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class GraphInputs:
    """Everything a training script needs, and nothing it has to re-derive."""

    src: np.ndarray            # int32 node codes, per edge
    dst: np.ndarray
    edge_features: np.ndarray  # float16, memory-mapped
    y: np.ndarray              # binary label
    y_multiclass: np.ndarray   # family index
    splits: dict               # name -> {"start", "stop"}
    families: dict             # family name -> index
    dataset_name: str

    @property
    def edge_dim(self) -> int:
        return int(self.edge_features.shape[1])

    @property
    def n_classes(self) -> int:
        return len(self.families)


def one_hot(cat: np.ndarray, cardinalities: list[int]) -> np.ndarray:
    """Identical expansion to Phase 3, so Channel 1 sees the same input the
    baseline did."""
    blocks = []
    for j, card in enumerate(cardinalities):
        block = np.zeros((len(cat), card), dtype=np.float32)
        block[np.arange(len(cat)), np.clip(cat[:, j], 0, card - 1)] = 1.0
        blocks.append(block)
    return np.hstack(blocks)


def load_graph_inputs(processed_dir: Path, dataset_name: str,
                      verbose: bool = True) -> GraphInputs:
    """Read one preprocessed dataset directory into memory-bounded arrays."""
    proc = Path(processed_dir)
    meta = json.loads((proc / "pipeline.json").read_text())
    splits = json.loads((proc / "splits.json").read_text())
    families = json.loads((proc / "attack_families.json").read_text())

    # Built once as a float16 .npy, thereafter paged in by the OS on demand, so
    # the process holds only the windows it is touching.
    feat_path = proc / "edge_features.npy"
    if not feat_path.exists():
        if verbose:
            print("  building memory-mappable edge features (one-off) ...")
        npz = np.load(proc / "features.npz")
        arr = np.hstack([
            npz["continuous"], one_hot(npz["categorical"], meta["cardinalities"])
        ]).astype(np.float16)
        np.save(feat_path, arr)
        del npz, arr
    edge_feats = np.load(feat_path, mmap_mode="r")
    if verbose:
        print(f"  edge features: {edge_feats.shape} float16, memory-mapped")

    df = pd.read_parquet(proc / "meta.parquet",
                         columns=["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label", "Attack"])
    if len(df) != len(edge_feats):
        raise ValueError(
            f"meta.parquet has {len(df):,} rows, features have "
            f"{len(edge_feats):,}. Re-run scripts/preprocess.py.")

    n = len(df)
    codes = pd.factorize(
        np.concatenate([df.IPV4_SRC_ADDR.to_numpy(), df.IPV4_DST_ADDR.to_numpy()])
    )[0].astype(np.int32)
    out = GraphInputs(
        src=codes[:n], dst=codes[n:],
        edge_features=edge_feats,
        y=df.Label.to_numpy().astype(np.int64),
        y_multiclass=df.Attack.map(families).to_numpy().astype(np.int64),
        splits=splits, families=families, dataset_name=dataset_name,
    )
    del df, codes
    return out
