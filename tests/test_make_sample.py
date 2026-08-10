"""Tests for the committable-sample builder.

The properties that matter are the ones that would silently ruin the sample:
rare attack families surviving at all, row order being preserved so temporal
splitting still works, and flows from one family staying contiguous rather than
being scattered across the file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_sample import MIN_ROWS_PER_FAMILY, stratified_contiguous_sample  # noqa: E402

LABEL = "Attack"


@pytest.fixture
def frame() -> pd.DataFrame:
    """100k rows, ~4% attacks, with one family rare enough to vanish under
    naive random sampling."""
    families = (
        ["Benign"] * 96_000
        + ["Exploits"] * 2_500
        + ["Reconnaissance"] * 1_400
        + ["Worms"] * 100
    )
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            LABEL: families,
            "Label": [0 if f == "Benign" else 1 for f in families],
            "IN_BYTES": rng.integers(40, 100_000, len(families)),
        }
    )


def test_hits_approximately_the_target_size(frame):
    sample = stratified_contiguous_sample(frame, LABEL, 10_000)
    assert 9_000 <= len(sample) <= 11_500


def test_rare_family_survives(frame):
    """Worms is 0.1% of the data. A naive 10k random sample would yield ~10
    rows; the floor must rescue it."""
    sample = stratified_contiguous_sample(frame, LABEL, 10_000)
    n_worms = (sample[LABEL] == "Worms").sum()

    assert n_worms >= min(MIN_ROWS_PER_FAMILY, 100)


def test_every_family_is_represented(frame):
    sample = stratified_contiguous_sample(frame, LABEL, 5_000)
    assert set(sample[LABEL].unique()) == set(frame[LABEL].unique())


def test_common_family_proportions_are_roughly_preserved(frame):
    sample = stratified_contiguous_sample(frame, LABEL, 10_000)
    benign_share = (sample[LABEL] == "Benign").mean()

    # Not exact -- the floor over-samples rare families at Benign's expense.
    assert 0.90 <= benign_share <= 0.97


def test_row_order_is_preserved(frame):
    """Temporal splitting (D5) depends on whatever ordering the file carries."""
    sample = stratified_contiguous_sample(frame, LABEL, 10_000)
    assert sample.index.is_monotonic_increasing


def test_rows_within_a_family_are_contiguous(frame):
    """Attack episodes must stay together, or the neighbourhood structure the
    GNN depends on is destroyed."""
    sample = stratified_contiguous_sample(frame, LABEL, 10_000)
    worms = sample[sample[LABEL] == "Worms"].index.to_numpy()

    assert np.all(np.diff(worms) == 1)


def test_target_larger_than_source_returns_everything(frame):
    sample = stratified_contiguous_sample(frame, LABEL, 500_000)
    assert len(sample) == len(frame)
