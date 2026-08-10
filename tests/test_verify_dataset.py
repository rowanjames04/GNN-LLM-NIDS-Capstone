"""Tests for the Phase 0 dataset verification logic.

The real dataset is not in git, so these run against a synthetic frame shaped
like NF-UNSW-NB15-v2: a handful of hosts, heavy class imbalance, and a rare
attack family. That is enough to check the analysis functions behave, and in
particular that the three verdicts the design depends on are computed the way
the notes claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_dataset import (  # noqa: E402
    analyse_graph_shape,
    analyse_labels,
    analyse_schema,
    analyse_time,
)

COLUMNS = {
    "src_ip": "IPV4_SRC_ADDR",
    "dst_ip": "IPV4_DST_ADDR",
    "src_port": "L4_SRC_PORT",
    "dst_port": "L4_DST_PORT",
    "label_binary": "Label",
    "label_multiclass": "Attack",
}


@pytest.fixture
def dense_frame() -> pd.DataFrame:
    """8 hosts, every ordered pair present -> a near-complete host graph."""
    rng = np.random.default_rng(0)
    hosts = [f"10.0.0.{i}" for i in range(8)]
    pairs = [(s, d) for s in hosts for d in hosts if s != d]
    src, dst = zip(*(pairs * 40))
    n = len(src)

    # ~4% attacks, mirroring the real prevalence, with one deliberately rare
    # family so the "under 1000 examples" warning has something to catch.
    attack = np.array(["Benign"] * n, dtype=object)
    idx = rng.choice(n, size=int(n * 0.04), replace=False)
    attack[idx[: len(idx) // 2]] = "Exploits"
    attack[idx[len(idx) // 2 : -3]] = "Reconnaissance"
    attack[idx[-3:]] = "Worms"

    return pd.DataFrame(
        {
            "IPV4_SRC_ADDR": list(src),
            "IPV4_DST_ADDR": list(dst),
            "L4_SRC_PORT": rng.integers(1024, 65535, n),
            "L4_DST_PORT": rng.integers(1, 1024, n),
            "IN_BYTES": rng.integers(40, 100_000, n),
            "PROTOCOL": rng.choice([6, 17], n),
            "CONSTANT_COL": 1,
            "Label": (attack != "Benign").astype(int),
            "Attack": attack,
        }
    )


def test_dense_graph_is_flagged_as_dense(dense_frame):
    """The premise behind D2: few hosts, near-complete connectivity."""
    result = analyse_graph_shape(dense_frame, COLUMNS)

    assert result["unique_hosts_total"] == 8
    assert result["unique_ordered_pairs"] == 8 * 7
    assert result["graph_density"] == pytest.approx(1.0)
    assert result["flows_per_pair"]["mean"] == pytest.approx(40.0)


def test_sparse_graph_is_distinguishable():
    """A sparse graph must produce a low density, or the D2 check is useless."""
    n_hosts = 500
    df = pd.DataFrame(
        {
            "IPV4_SRC_ADDR": [f"10.0.{i // 256}.{i % 256}" for i in range(n_hosts)],
            "IPV4_DST_ADDR": ["10.9.9.9"] * n_hosts,
            "Label": 0,
            "Attack": "Benign",
        }
    )
    result = analyse_graph_shape(df, COLUMNS)
    assert result["graph_density"] < 0.01


def test_missing_ip_columns_reports_error():
    df = pd.DataFrame({"foo": [1, 2, 3]})
    assert "error" in analyse_graph_shape(df, COLUMNS)


def test_time_verdict_none_when_no_ordering(dense_frame):
    """No timestamp and no monotonic column -> Granger is not viable."""
    result = analyse_time(dense_frame, ["Stime", "Ltime"])
    assert result["verdict"] == "none"


def test_time_verdict_absolute_when_named_column_present(dense_frame):
    df = dense_frame.copy()
    df["Stime"] = np.arange(len(df)) + 1_600_000_000
    result = analyse_time(df, ["Stime", "Ltime"])

    assert result["verdict"] == "absolute"
    assert result["named_candidates_found"]["Stime"]["monotonic_non_decreasing"]


def test_time_verdict_ordinal_when_only_monotonic_column(dense_frame):
    """A monotonic column under an unexpected name still gives us ordering."""
    df = dense_frame.copy()
    df["record_seq"] = np.arange(len(df))
    result = analyse_time(df, ["Stime"])

    assert result["verdict"] == "ordinal"
    assert "record_seq" in result["monotonic_numeric_columns"]


def test_label_analysis_reports_imbalance_and_rare_families(dense_frame):
    result = analyse_labels(dense_frame, COLUMNS)

    assert result["binary"]["attack_prevalence"] == pytest.approx(0.04, abs=0.005)
    assert result["binary"]["imbalance_ratio"] > 20
    assert result["multiclass"]["n_classes"] == 4
    assert "Worms" in result["multiclass"]["families_under_1000"]
    assert "Benign" not in result["multiclass"]["families_under_1000"]


def test_class_weights_favour_rare_classes(dense_frame):
    """Inverse-frequency weighting must up-weight the rare families."""
    weights = analyse_labels(dense_frame, COLUMNS)["multiclass"]["suggested_class_weights"]
    assert weights["Worms"] > weights["Exploits"] > weights["Benign"]


def test_schema_flags_constant_columns(dense_frame):
    result = analyse_schema(dense_frame)

    assert result["n_rows"] == len(dense_frame)
    assert "CONSTANT_COL" in result["constant_columns"]
    assert "IN_BYTES" not in result["constant_columns"]
