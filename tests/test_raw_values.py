"""Tests for the measured-value side file that evidence packs are built from.

An evidence pack states facts about a flow. Until 2026-10-03 those facts were
recovered by inverting the model's float16 inputs, which returned port 80 as
77.1452. They are now read from the measurements. The danger that replaces the
old one is misalignment: a raw file offset by one window would put another
flow's port in every pack, and nothing would look wrong.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pre = _load("preprocess")
SCHEMA = SimpleNamespace(continuous=["L4_DST_PORT", "IN_BYTES", "FLOW_DURATION_MILLISECONDS"])


def _frame(n=6):
    return pd.DataFrame({
        "IPV4_SRC_ADDR": [f"10.0.0.{i}" for i in range(n)],
        "IPV4_DST_ADDR": [f"10.0.1.{i}" for i in range(n)],
        "L4_DST_PORT": [80, 443, 53, 22, 445, 8080][:n],
        "PROTOCOL": [6] * n, "IN_BYTES": [44, 180, 60, 4750, 298, 0][:n],
        "FLOW_DURATION_MILLISECONDS": [0, 0, 12, 4294966, 3, 0][:n],
        pre.LABEL_BINARY: [0, 1, 0, 1, 1, 0][:n],
        pre.LABEL_MULTICLASS: ["Benign", "dos", "Benign", "ddos", "dos", "Benign"][:n],
    })


def test_raw_values_keep_the_measurement_exactly(tmp_path):
    df = _frame()
    pre.write_raw_values(df, SCHEMA, tmp_path)

    raw = np.load(tmp_path / "raw_values.npy", mmap_mode="r")
    side = json.loads((tmp_path / "raw_values.json").read_text())

    assert side["columns"] == SCHEMA.continuous and side["n_rows"] == 6
    assert raw[0, 0] == 80 and raw[3, 2] == 4294966        # not 77.1452, not rounded
    assert raw.dtype == np.float64


def test_a_frame_in_the_original_order_is_accepted(tmp_path):
    df = _frame()
    df[pre.META_COLUMNS].to_parquet(tmp_path / "meta.parquet", index=False)

    pre.write_raw_only(df, SCHEMA, tmp_path)

    assert (tmp_path / "raw_values.npy").exists()


def test_rows_in_a_different_order_are_refused_and_nothing_is_written(tmp_path):
    """Same rows, different order -- what a different seed or load would give.
    Every count matches; only a row-for-row check can see it."""
    df = _frame()
    df[pre.META_COLUMNS].to_parquet(tmp_path / "meta.parquet", index=False)
    shuffled = df.iloc[[1, 0, 2, 3, 4, 5]].reset_index(drop=True)

    with pytest.raises(SystemExit, match="does not match"):
        pre.write_raw_only(shuffled, SCHEMA, tmp_path)
    assert not (tmp_path / "raw_values.npy").exists()


def test_a_different_row_count_is_refused(tmp_path):
    _frame()[pre.META_COLUMNS].to_parquet(tmp_path / "meta.parquet", index=False)

    with pytest.raises(SystemExit, match="row count differs"):
        pre.write_raw_only(_frame(5), SCHEMA, tmp_path)


def test_packs_cannot_be_built_without_measured_values(tmp_path):
    """No fallback to the lossy inversion: the pack generator stops and says
    how to create the file."""
    ev = _load("make_evidence")

    with pytest.raises(SystemExit, match="--raw-only"):
        ev.load_raw_values(tmp_path)


def test_a_measured_value_is_an_integer_where_it_is_one():
    ev = _load("make_evidence")

    assert ev._measured(80.0) == 80 and isinstance(ev._measured(80.0), int)
    assert ev._measured(95.44221) == 95.4422
    assert ev._measured(float("nan")) == 0.0
