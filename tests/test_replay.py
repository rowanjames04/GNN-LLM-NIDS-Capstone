"""Tests for the Phase 8 replay engine.

The engine is a library rather than Streamlit code precisely so it can be
tested: the video showcase is half the assessed deliverable, and a failure the
night before it should be a rendering bug, not a logic one.

The statistic worth guarding is the aggregation. Averaging per-window ratios
weights a 10-flow window equally with a 10,000-flow one, which is how a demo
ends up displaying a false-positive rate no offline run can reproduce.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnnids.ui.replay import ReplayEngine, ReplayStats  # noqa: E402


class _Windows:
    """Stand-in dataset: a list of (scores, labels) pairs."""

    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)


def _engine(windows, threshold=0.5):
    ds = _Windows(windows)
    return ReplayEngine(ds, lambda i: ds.windows[i], threshold)


def test_alerts_are_the_flows_at_or_above_threshold():
    e = _engine([(np.array([0.1, 0.9, 0.5]), np.array([0, 1, 1]))])

    found = e.step(0)

    assert [d.edge_index for d in found] == [1, 2]
    assert e.stats.alerts == 2


def test_detections_are_ordered_by_score_not_file_order():
    """An analyst works a queue from the top; the demo should show that."""
    e = _engine([(np.array([0.95, 0.6, 0.99]), np.array([1, 1, 1]))])

    found = e.step(0)

    assert [d.edge_index for d in found] == [2, 0, 1]


def test_rates_are_pooled_over_flows_not_averaged_over_windows():
    """One tiny window and one large one. Averaging per-window precision would
    give 0.75; pooling gives the true 0.99."""
    tiny = (np.array([0.9, 0.9]), np.array([1, 0]))          # 1 of 2 correct
    large = (np.concatenate([np.full(198, 0.9), np.array([0.9, 0.9])]),
             np.ones(200, dtype=int))                        # 200 of 200
    e = _engine([tiny, large])

    for i in range(2):
        e.step(i)

    assert e.stats.alerts == 202
    assert e.stats.true_positives == 201
    assert e.stats.precision == 201 / 202
    assert round(e.stats.precision, 2) != 0.75


def test_trigger_rate_counts_windows_not_alerts():
    """D7's second metric: how often an analyst is interrupted at all."""
    quiet = (np.array([0.1, 0.2]), np.array([0, 0]))
    noisy = (np.array([0.9, 0.9]), np.array([1, 1]))
    e = _engine([quiet, noisy, quiet, noisy])

    for i in range(4):
        e.step(i)

    assert e.stats.trigger_rate == 0.5
    assert e.stats.windows_with_alerts == 2


def test_alerts_per_thousand_flows_is_reported():
    e = _engine([(np.concatenate([np.full(10, 0.9), np.zeros(990)]),
                  np.zeros(1000, dtype=int))])

    e.step(0)

    assert e.stats.alerts_per_1000_flows == 10.0


def test_a_window_with_no_alerts_still_counts_its_flows():
    """Otherwise the alert rate is computed only over windows that alerted,
    which flatters it without limit."""
    e = _engine([(np.array([0.1] * 100), np.zeros(100, dtype=int))])

    found = e.step(0)

    assert found == []
    assert e.stats.flows == 100 and e.stats.alerts == 0
    assert e.stats.alerts_per_1000_flows == 0.0


def test_empty_stats_return_none_not_zero():
    """Nothing measured is not the same as measured zero, and a demo showing
    'precision 0.0' before it has run would be a lie."""
    s = ReplayStats()

    assert s.precision is None and s.recall is None
    assert s.trigger_rate is None
    assert s.as_dict()["scoring_ms_median"] is None


def test_run_yields_every_window_in_order():
    e = _engine([(np.array([0.9]), np.array([1])) for _ in range(5)])

    seen = [i for i, _ in e.run()]

    assert seen == [0, 1, 2, 3, 4]
    assert e.stats.windows == 5


def test_run_respects_a_window_cap():
    e = _engine([(np.array([0.9]), np.array([1])) for _ in range(10)])

    seen = [i for i, _ in e.run(max_windows=3)]

    assert seen == [0, 1, 2]


def test_latency_is_recorded_per_detection():
    e = _engine([(np.array([0.9]), np.array([1]))])

    found = e.step(0)

    assert "scoring" in found[0].latency_ms
    assert found[0].latency_ms["scoring"] >= 0
