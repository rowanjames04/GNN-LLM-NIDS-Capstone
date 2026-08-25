"""Tests for leave-one-attack-out row selection (Phase 5, D4).

A leaked flow of the held-out family is the failure that matters here, and it is
**silent**: the run completes, the metrics look plausible, and the zero-day claim
is quietly false. So the properties are asserted rather than eyeballed.

The second class of failure is subtler and cost an hour of compute to find on
NF-ToN-IoT-v2: a family can be held out successfully and still leave too few
flows in the test split to measure recall on. `ransomware` leaves three.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnnids.data.splits import Split  # noqa: E402
from gnnids.eval.holdout import (  # noqa: E402
    assert_holdout_is_clean, family_row_index, holdout_plan,
)

FAMILIES = {"Benign": 0, "scanning": 1, "ddos": 2, "ransomware": 3}


def _dataset():
    """30 rows: benign and three attack families, interleaved across 3 splits."""
    pattern = [0, 1, 2, 0, 3, 0, 1, 2, 0, 0]
    ymc = np.array(pattern * 3, dtype=np.int64)
    splits = {
        "train": {"start": 0, "stop": 10},
        "val": {"start": 10, "stop": 20},
        "test": {"start": 20, "stop": 30},
    }
    return ymc, splits


def test_train_and_val_lose_every_flow_of_the_held_out_family():
    ymc, splits = _dataset()

    plan = holdout_plan(ymc, splits, FAMILIES, "scanning")

    for name in ("train", "val"):
        surviving = ymc[plan["splits"][name]["row_index"]]
        assert (surviving == FAMILIES["scanning"]).sum() == 0
        # and nothing else was removed
        assert plan["splits"][name]["removed"] == 2


def test_test_split_keeps_benign_plus_the_held_out_family_only():
    """Leaving the other families in would mix seen and unseen attacks in one
    recall figure."""
    ymc, splits = _dataset()

    plan = holdout_plan(ymc, splits, FAMILIES, "scanning")
    surviving = ymc[plan["splits"]["test"]["row_index"]]

    assert set(np.unique(surviving)) == {FAMILIES["Benign"], FAMILIES["scanning"]}
    assert plan["splits"]["test"]["n_held_out"] == 2


def test_rows_never_migrate_between_splits():
    """The tempting implementation -- filter globally, then re-split -- would
    change what is in the test set for every held-out family."""
    ymc, splits = _dataset()

    plan = holdout_plan(ymc, splits, FAMILIES, "ddos")

    for name, s in splits.items():
        idx = plan["splits"][name]["row_index"]
        assert idx.min() >= s["start"] and idx.max() < s["stop"]
    all_idx = np.concatenate([plan["splits"][n]["row_index"] for n in splits])
    assert len(all_idx) == len(np.unique(all_idx))


def test_leak_into_training_is_an_assertion_not_a_warning():
    ymc, splits = _dataset()
    plan = holdout_plan(ymc, splits, FAMILIES, "scanning")
    # Simulate the leak the assertion exists to catch.
    plan["splits"]["train"]["n_held_out"] = 1

    with pytest.raises(AssertionError, match="is not a zero-day run"):
        assert_holdout_is_clean(plan)


def test_a_family_absent_from_test_cannot_be_measured():
    """NF-ToN-IoT-v2's `ransomware` case: 2,554 flows in train, 3 in test."""
    ymc = np.array([0, 1, 3, 0, 1, 0] + [0, 1, 0, 0, 1, 0], dtype=np.int64)
    splits = {"train": {"start": 0, "stop": 6}, "test": {"start": 6, "stop": 12}}

    plan = holdout_plan(ymc, splits, FAMILIES, "ransomware")

    assert plan["splits"]["test"]["n_held_out"] == 0
    with pytest.raises(AssertionError, match="nothing to measure recall on"):
        assert_holdout_is_clean(plan)


def test_holding_out_the_benign_class_is_refused():
    ymc, splits = _dataset()

    with pytest.raises(ValueError, match="benign"):
        holdout_plan(ymc, splits, FAMILIES, "Benign")


def test_unknown_family_names_the_alternatives():
    ymc, splits = _dataset()

    with pytest.raises(KeyError, match="scanning"):
        holdout_plan(ymc, splits, FAMILIES, "worms")


def test_role_test_and_role_train_select_disjoint_attack_rows():
    """The held-out family goes to exactly one side of the protocol."""
    ymc, _ = _dataset()
    split = Split("s", 0, 30)

    train_rows = family_row_index(ymc, split, FAMILIES["ddos"], "train")
    test_rows = family_row_index(ymc, split, FAMILIES["ddos"], "test")

    assert (ymc[train_rows] == FAMILIES["ddos"]).sum() == 0
    assert (ymc[test_rows] == FAMILIES["ddos"]).sum() == 6
    assert set(np.unique(ymc[test_rows])) == {0, FAMILIES["ddos"]}
