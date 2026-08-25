"""Replay: stream the held-out split in order and score it as it arrives (D7).

**The honest framing, which matters more than the code.** This is not a
real-time NIDS and the project does not claim to be one. Every reported number
comes from offline batch evaluation. Replay exists to drive the demo and to
measure two things batch evaluation cannot:

- **end-to-end latency** -- flow arrives, detection fires, evidence pack is
  built, report is written. A system that takes four minutes per alert is a
  different proposition from one that takes four seconds, and no batch metric
  shows that.
- **trigger efficiency** -- how many windows produce at least one alert, and how
  many alerts an analyst would actually receive per unit of traffic. PR-AUC does
  not answer "how noisy is this in a shift".

Deliberately a library, not Streamlit code. The UI is a thin view over this, so
the demo's logic is testable and deterministic while the app stays a rendering
concern -- and a broken demo the night before the showcase is then a *rendering*
bug, not a logic one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np


@dataclass
class Detection:
    """One flagged flow, with the timings the demo is there to expose."""

    window_index: int
    edge_index: int
    score: float
    true_label: int
    predicted_class: str | None = None
    latency_ms: dict = field(default_factory=dict)


@dataclass
class ReplayStats:
    """Running totals. Recomputed from scratch, never accumulated as ratios.

    Averaging a ratio across windows weights a 10-flow window equally with a
    10,000-flow one, which is how a demo ends up showing a false-positive rate
    that no offline run can reproduce.
    """

    windows: int = 0
    flows: int = 0
    alerts: int = 0
    true_positives: int = 0
    false_positives: int = 0
    attacks_seen: int = 0
    windows_with_alerts: int = 0
    scoring_ms: list = field(default_factory=list)

    @property
    def precision(self) -> float | None:
        return self.true_positives / self.alerts if self.alerts else None

    @property
    def recall(self) -> float | None:
        return (self.true_positives / self.attacks_seen
                if self.attacks_seen else None)

    @property
    def alerts_per_1000_flows(self) -> float | None:
        return round(self.alerts / self.flows * 1000, 2) if self.flows else None

    @property
    def trigger_rate(self) -> float | None:
        """Share of windows producing at least one alert -- D7's second metric."""
        return round(self.windows_with_alerts / self.windows, 4) if self.windows else None

    def as_dict(self) -> dict:
        return {
            "windows": self.windows, "flows": self.flows, "alerts": self.alerts,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "attacks_seen": self.attacks_seen,
            "precision": round(self.precision, 4) if self.precision is not None else None,
            "recall": round(self.recall, 4) if self.recall is not None else None,
            "alerts_per_1000_flows": self.alerts_per_1000_flows,
            "trigger_rate": self.trigger_rate,
            "scoring_ms_median": (round(float(np.median(self.scoring_ms)), 2)
                                  if self.scoring_ms else None),
            "scoring_ms_p95": (round(float(np.percentile(self.scoring_ms, 95)), 2)
                               if self.scoring_ms else None),
        }


class ReplayEngine:
    """Walk windows in order, score each, and emit detections as they occur.

    `score_window` is injected rather than built here so the engine is testable
    without a model and without a GPU -- the UI passes the real scorer, the
    tests pass a function.
    """

    def __init__(
        self,
        dataset,
        score_window: Callable[[int], tuple[np.ndarray, np.ndarray]],
        threshold: float,
        family_names: dict[int, str] | None = None,
    ) -> None:
        self.dataset = dataset
        self.score_window = score_window
        self.threshold = threshold
        self.family_names = family_names or {}
        self.stats = ReplayStats()

    def __len__(self) -> int:
        return len(self.dataset)

    def step(self, window_index: int) -> list[Detection]:
        """Score one window. Returns the flows that would have alerted."""
        t0 = time.perf_counter()
        scores, labels = self.score_window(window_index)
        scoring_ms = (time.perf_counter() - t0) * 1000

        flagged = np.flatnonzero(scores >= self.threshold)
        self.stats.windows += 1
        self.stats.flows += len(scores)
        self.stats.alerts += len(flagged)
        self.stats.attacks_seen += int((labels == 1).sum())
        self.stats.true_positives += int((labels[flagged] == 1).sum())
        self.stats.false_positives += int((labels[flagged] == 0).sum())
        self.stats.windows_with_alerts += 1 if len(flagged) else 0
        self.stats.scoring_ms.append(scoring_ms)

        # Highest-scoring first: an analyst works a queue from the top, so the
        # demo should show what they would see, not file order.
        order = flagged[np.argsort(-scores[flagged])]
        return [
            Detection(window_index=window_index, edge_index=int(e),
                      score=float(scores[e]), true_label=int(labels[e]),
                      latency_ms={"scoring": round(scoring_ms, 2)})
            for e in order
        ]

    def run(self, max_windows: int | None = None) -> Iterator[tuple[int, list[Detection]]]:
        n = len(self.dataset) if max_windows is None else min(max_windows, len(self.dataset))
        for i in range(n):
            yield i, self.step(i)
