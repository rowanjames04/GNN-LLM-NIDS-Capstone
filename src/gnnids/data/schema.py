"""Resolve the config into concrete column roles, validated against the file.

Keeping this separate from the transform logic means the "what goes where"
decisions -- all of which trace to a measured Phase 0/1 finding -- are stated
in one place and checked against the real schema before anything is computed.
Silently transforming a column that does not exist is a class of bug that
surfaces much later as a shape mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import json
import pandas as pd

LABEL_BINARY = "Label"
LABEL_MULTICLASS = "Attack"


@dataclass
class FeatureSchema:
    """Which columns play which role in the pipeline."""

    identity: list[str] = field(default_factory=list)
    continuous: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    conditional: list[str] = field(default_factory=list)
    ports: list[str] = field(default_factory=list)
    log_transform: list[str] = field(default_factory=list)
    dropped: dict[str, list[str]] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        return [LABEL_BINARY, LABEL_MULTICLASS]

    def summary(self) -> dict:
        return {
            "n_continuous": len(self.continuous),
            "n_categorical": len(self.categorical),
            "n_conditional_indicators": len(self.conditional),
            "n_port_buckets": len(self.ports),
            "n_log_transformed": len(self.log_transform),
            "dropped": self.dropped,
            "continuous": self.continuous,
            "categorical": self.categorical,
        }


def _log_columns(cfg: dict, repo_root: Path, available: set[str]) -> list[str]:
    """Log-transform list, read from the Phase 1 EDA output by default.

    Deriving it from `results/metrics/eda.json` rather than hardcoding keeps the
    decision traceable: the list is whatever the stated rule (raw |skew| > 2 and
    log1p at least halves it) actually selected, and it moves if the data does.
    """
    spec = cfg["features"]["log_transform"]
    if isinstance(spec, list):
        return [c for c in spec if c in available]

    if spec != "from_eda":
        raise ValueError(f"log_transform must be a list or 'from_eda', got {spec!r}")

    eda_path = repo_root / cfg["eda_metrics"]
    if not eda_path.exists():
        raise SystemExit(
            f"{eda_path} not found. Run scripts/eda.py first, or set an explicit "
            f"log_transform list in the config."
        )
    eda = json.loads(eda_path.read_text())
    return [c for c, v in eda["skew"].items() if v["recommend_log"] and c in available]


def build_schema(cfg: dict, df: pd.DataFrame, repo_root: Path) -> FeatureSchema:
    f = cfg["features"]
    cols = set(df.columns)

    drops = {
        "shortcuts": list(f["drop_shortcuts"]),
        "uninformative": list(f["drop_uninformative"]),
    }
    dropped = set(drops["shortcuts"]) | set(drops["uninformative"])

    missing = (dropped | set(f["identity"])) - cols
    if missing:
        raise SystemExit(f"Config names columns absent from the data: {sorted(missing)}")

    identity = list(f["identity"])
    categorical = [c for c in f["categorical"] if c in cols and c not in dropped]
    conditional = [c for c in f["conditional"] if c in cols and c not in dropped]
    ports = [c for c in f["ports"] if c in cols and c not in dropped]

    # Everything left over is continuous. Deriving it by subtraction rather than
    # listing it means a new column in a future dataset version is picked up
    # automatically instead of being silently ignored.
    accounted = dropped | set(identity) | set(categorical) | {LABEL_BINARY, LABEL_MULTICLASS}
    continuous = [c for c in df.columns if c not in accounted]

    return FeatureSchema(
        identity=identity,
        continuous=continuous,
        categorical=categorical,
        conditional=conditional,
        ports=ports,
        log_transform=_log_columns(cfg, repo_root, set(continuous)),
        dropped=drops,
    )
