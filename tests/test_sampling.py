"""Tests for detection sampling (Phase 6a / 7b).

The failure this guards against is silent and total: a sample drawn in file
order is dominated by whichever family the detector flags most, and contains no
false positives at all. Every model in the comparison then gets scored on the
same skewed slice, and the study's sharpest question -- what a model does with a
detection that is wrong -- cannot be asked.

That is not a bug that surfaces as an error. It surfaces as a study that quietly
measured something narrower than it claimed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnnids.explain.sampling import Candidate, stratified_sample  # noqa: E402


def _pool(counts: dict[str, int]) -> list[Candidate]:
    out, w = [], 0
    for fam, n in counts.items():
        out += [Candidate(window_index=w, edge_index=i, row_index=i,
                          score=0.9, true_family=fam) for i in range(n)]
        w += 1
    return out


def test_a_dominant_family_does_not_swallow_the_sample():
    """The whole point: file order would return ~all scanning."""
    pool = _pool({"scanning": 5000, "xss": 300, "ddos": 200, "Benign": 50})

    plan = stratified_sample(pool, n_total=60)

    assert plan.taken["scanning"] < 30
    assert set(plan.taken) == {"scanning", "xss", "ddos", "Benign"}


def test_false_positives_are_sampled_for_deliberately():
    """They are rare by construction and will not appear by accident."""
    pool = _pool({"scanning": 5000, "Benign": 40})

    plan = stratified_sample(pool, n_total=100, false_positive_share=0.2)

    assert plan.false_positives_requested == 20
    assert plan.taken["Benign"] == 20


def test_a_thin_family_is_capped_at_what_exists_and_the_shortfall_recorded():
    """A recall or report quality computed over one detection is not a
    measurement, and the reader must be able to see the denominator."""
    pool = _pool({"scanning": 500, "ransomware": 2, "Benign": 20})

    plan = stratified_sample(pool, n_total=60)

    assert plan.taken["ransomware"] == 2
    assert plan.shortfalls["ransomware"] > 0
    assert plan.available["ransomware"] == 2


def test_unfilled_budget_is_redistributed_so_the_sample_is_not_short():
    """One thin family must not shrink the whole study."""
    pool = _pool({"scanning": 5000, "xss": 5000, "ransomware": 1, "Benign": 100})

    plan = stratified_sample(pool, n_total=80)

    assert len(plan.selected) == 80


def test_absent_false_positives_are_recorded_not_silently_ignored():
    """A detector producing no false positives is a finding about the detector.
    The study simply cannot ask its false-positive question there."""
    pool = _pool({"scanning": 500, "xss": 300})

    plan = stratified_sample(pool, n_total=40, false_positive_share=0.25)

    assert plan.taken.get("Benign", 0) == 0
    assert plan.shortfalls["Benign"] == 10


def test_sampling_is_deterministic_for_a_seed():
    """Every model sees the same packs; a sample that moved between runs would
    break the fully crossed design."""
    pool = _pool({"scanning": 500, "xss": 300, "Benign": 40})

    a = stratified_sample(pool, n_total=50, seed=7)
    b = stratified_sample(pool, n_total=50, seed=7)
    c = stratified_sample(pool, n_total=50, seed=8)

    key = lambda p: [(x.window_index, x.edge_index) for x in p.selected]
    assert key(a) == key(b)
    assert key(a) != key(c)


def test_selection_is_ordered_by_window():
    """Pack construction walks windows once; out-of-order selection would score
    and attribute the same window repeatedly."""
    pool = _pool({"scanning": 400, "xss": 400, "Benign": 40})

    plan = stratified_sample(pool, n_total=50)

    order = [(c.window_index, c.edge_index) for c in plan.selected]
    assert order == sorted(order)


def test_never_returns_more_than_asked():
    pool = _pool({"scanning": 5000, "xss": 5000, "Benign": 500})

    assert len(stratified_sample(pool, n_total=37).selected) == 37


def test_an_invalid_false_positive_share_is_refused():
    with pytest.raises(ValueError, match="false_positive_share"):
        stratified_sample(_pool({"scanning": 10}), n_total=5, false_positive_share=1.0)


def test_is_false_positive_reads_the_true_label():
    assert Candidate(0, 0, 0, 0.9, "Benign").is_false_positive
    assert not Candidate(0, 0, 0, 0.9, "scanning").is_false_positive
