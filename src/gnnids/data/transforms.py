"""Feature transforms, fitted on train only.

Everything stateful here -- scaler statistics, categorical vocabularies, log
shifts -- is learned from the training split and then applied unchanged to val
and test. Fitting on the full dataset is a subtle and very common leak: the
model gains access to the mean and variance of data it is about to be scored
on. It rarely changes results dramatically, which is exactly why it survives
review.

The pipeline is serialisable so replay mode (D7) and any later inference use
identical statistics rather than recomputing them from whatever data happens to
be in front of them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import FeatureSchema

OTHER = 0  # reserved index for rare or unseen categorical values

# Port ranges carry meaning that magnitude does not: 443 is not "more" than 80.
PORT_BUCKETS = ["well_known", "registered", "ephemeral"]


def bucket_ports(values: np.ndarray) -> np.ndarray:
    """0 = well-known (<1024), 1 = registered, 2 = ephemeral (>=49152)."""
    out = np.ones(len(values), dtype=np.int64)
    out[values < 1024] = 0
    out[values >= 49152] = 2
    return out


@dataclass
class FeaturePipeline:
    schema: FeatureSchema
    min_frequency: float = 0.001
    max_cardinality: int = 64
    clip_quantile: float = 0.99999

    # Upper clip bound per continuous column, from a high train quantile.
    clip_hi_: dict[str, float] = field(default_factory=dict)
    log_shift_: dict[str, float] = field(default_factory=dict)
    mean_: dict[str, float] = field(default_factory=dict)
    std_: dict[str, float] = field(default_factory=dict)
    vocab_: dict[str, dict[str, int]] = field(default_factory=dict)
    continuous_names_: list[str] = field(default_factory=list)
    categorical_names_: list[str] = field(default_factory=list)
    fitted_: bool = False

    # ------------------------------------------------------------------ fit

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """Learn transform parameters. Must be given the TRAIN split only."""
        for col in self.schema.continuous:
            x = df[col].to_numpy(dtype="float64")

            # Winsorise the extreme upper tail. NF-ToN-IoT-v2 contains
            # physically impossible throughput values -- DST_TO_SRC_SECOND_BYTES
            # reaches 1.9e219 bytes/sec, an artefact of dividing bytes by a
            # near-zero duration. Left in, they overflow the skew computation
            # (so the column is never flagged for log transform) and then
            # dominate the standard deviation so completely that every real
            # value maps to approximately zero. Silent, and catastrophic.
            #
            # The bound comes from the TRAIN split only, like every other
            # statistic here, and at this quantile it touches roughly one value
            # in 100,000 -- the corrupted ones -- while leaving genuine large
            # flows intact.
            finite = x[np.isfinite(x)]
            self.clip_hi_[col] = (
                float(np.quantile(finite, self.clip_quantile)) if len(finite) else 0.0
            )
            x = np.clip(np.nan_to_num(x, nan=0.0, posinf=self.clip_hi_[col],
                                      neginf=0.0), None, self.clip_hi_[col])

            if col in self.schema.log_transform:
                # log1p needs non-negative input; record the shift so val/test
                # get exactly the same mapping even if their minimum differs.
                shift = -min(0.0, float(np.nanmin(x)))
                self.log_shift_[col] = shift
                x = np.log1p(x + shift)
            self.mean_[col] = float(np.nanmean(x))
            # A zero-variance column would divide by zero; 1.0 leaves it at its
            # centred value, and the column is reported as constant in the stats.
            sd = float(np.nanstd(x))
            self.std_[col] = sd if sd > 1e-12 else 1.0

        for col in self.schema.categorical:
            counts = df[col].value_counts(normalize=True)
            keep = counts[counts >= self.min_frequency].index[: self.max_cardinality - 1]
            # Index 0 is reserved for OTHER, so real values start at 1.
            self.vocab_[col] = {str(v): i + 1 for i, v in enumerate(keep)}

        self.continuous_names_ = list(self.schema.continuous)
        self.categorical_names_ = (
            list(self.schema.categorical)
            + [f"{c}_bucket" for c in self.schema.ports]
            + [f"{c}_present" for c in self.schema.conditional]
        )
        self.fitted_ = True
        return self

    # ------------------------------------------------------------ transform

    def transform(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.fitted_:
            raise RuntimeError("FeaturePipeline.transform called before fit")

        cont = np.empty((len(df), len(self.continuous_names_)), dtype=np.float32)
        for j, col in enumerate(self.continuous_names_):
            x = df[col].to_numpy(dtype="float64")
            hi = self.clip_hi_.get(col)
            if hi is not None:
                x = np.clip(np.nan_to_num(x, nan=0.0, posinf=hi, neginf=0.0), None, hi)
            if col in self.log_shift_:
                x = np.log1p(np.maximum(x + self.log_shift_[col], 0.0))
            cont[:, j] = (x - self.mean_[col]) / self.std_[col]
        np.nan_to_num(cont, copy=False)

        cat_cols = []
        for col in self.schema.categorical:
            vocab = self.vocab_[col]
            # Unseen-in-training values fall to OTHER rather than erroring --
            # required for val/test and for cross-dataset transfer.
            cat_cols.append(
                df[col].astype(str).map(vocab).fillna(OTHER).to_numpy(dtype=np.int64)
            )
        for col in self.schema.ports:
            cat_cols.append(bucket_ports(df[col].to_numpy()))
        for col in self.schema.conditional:
            # The indicator is the point: a zero in these protocol-conditional
            # columns means "not applicable", not "measured zero".
            cat_cols.append((df[col].to_numpy() != 0).astype(np.int64))

        cat = (
            np.stack(cat_cols, axis=1)
            if cat_cols
            else np.zeros((len(df), 0), dtype=np.int64)
        )
        return {"continuous": cont, "categorical": cat}

    def fit_transform(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        return self.fit(df).transform(df)

    # ---------------------------------------------------------- properties

    def cardinalities(self) -> list[int]:
        """Embedding table sizes, in the column order of the categorical array."""
        sizes = [len(self.vocab_[c]) + 1 for c in self.schema.categorical]
        sizes += [len(PORT_BUCKETS)] * len(self.schema.ports)
        sizes += [2] * len(self.schema.conditional)
        return sizes

    # -------------------------------------------------------- persistence

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "clip_hi": self.clip_hi_,
            "clip_quantile": self.clip_quantile,
            "log_shift": self.log_shift_,
            "mean": self.mean_,
            "std": self.std_,
            "vocab": self.vocab_,
            "continuous_names": self.continuous_names_,
            "categorical_names": self.categorical_names_,
            "cardinalities": self.cardinalities(),
            "min_frequency": self.min_frequency,
            "max_cardinality": self.max_cardinality,
        }, indent=2))

    @classmethod
    def load(cls, path: Path, schema: FeatureSchema) -> "FeaturePipeline":
        d = json.loads(path.read_text())
        p = cls(schema, d["min_frequency"], d["max_cardinality"],
                d.get("clip_quantile", 0.99999))
        p.clip_hi_ = d.get("clip_hi", {})
        p.log_shift_, p.mean_, p.std_, p.vocab_ = d["log_shift"], d["mean"], d["std"], d["vocab"]
        p.continuous_names_ = d["continuous_names"]
        p.categorical_names_ = d["categorical_names"]
        p.fitted_ = True
        return p
