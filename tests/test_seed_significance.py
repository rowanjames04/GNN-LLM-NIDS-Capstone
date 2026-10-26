"""The seed-level significance test and the bound it reports."""

import importlib.util
from math import comb
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "seed_significance", Path(__file__).resolve().parents[1] / "scripts" / "seed_significance.py")
sig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sig)


def test_complete_separation_of_ten_against_ten_reaches_the_exact_bound():
    r = sig.compare([float(i) for i in range(10, 20)], [float(i) for i in range(10)])

    assert r["ranges_separated"]
    assert r["p_two_sided"] == pytest.approx(2 / comb(20, 10))
    assert r["p_two_sided"] == pytest.approx(r["min_possible_p"])


def test_three_seeds_against_three_can_never_go_below_one_tenth():
    """Why the three-seed comparisons are reported as range overlap only."""
    r = sig.compare([4.0, 5.0, 6.0], [1.0, 2.0, 3.0])

    assert r["min_possible_p"] == pytest.approx(0.1)
    assert r["p_two_sided"] == pytest.approx(0.1)


def test_overlapping_seeds_are_not_reported_as_separated():
    r = sig.compare([1.0, 3.0, 5.0], [2.0, 4.0, 6.0])

    assert not r["ranges_separated"]
    assert r["p_two_sided"] > 0.1
