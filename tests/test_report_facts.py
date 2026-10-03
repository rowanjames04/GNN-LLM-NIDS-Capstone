"""Tests for the report-facts export.

This file is what a draft's numbers are checked against, so its failure modes
are the ones that put a wrong number in the report: a gap that reads as a value,
a superseded result presented as final, and a spread reported where none was
measured.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "export_report_facts_under_test", REPO_ROOT / "scripts" / "export_report_facts.py")
rf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rf)


def test_a_missing_statistic_prints_as_absent_not_as_a_number():
    assert rf.pm(None) == "n/a"
    assert rf.pm({"mean": None, "std": 0.0, "n_seeds": 3}) == "n/a"
    assert rf.f(None) == "n/a"


def test_a_statistic_always_carries_its_seed_count():
    assert rf.pm({"mean": 0.98871, "std": 0.003, "n_seeds": 3}) == "0.9887 ± 0.0030 (n = 3)"


def test_a_section_with_no_results_says_so_and_asks_for_a_placeholder():
    lines = rf.missing("Zero-day recall.", "python scripts/train_zeroday.py")

    assert lines[0].startswith("**NOT YET MEASURED.**")
    assert "[NUM?]" in lines[1]


def test_the_export_builds_from_the_committed_results():
    text = rf.build()

    assert text.startswith("# Report facts")
    for heading in ("## 1. Datasets", "## 6. Zero-day", "## 7. Cross-dataset",
                    "## 9. Language-model comparison"):
        assert heading in text
    # One seed is "one seed", never a standard deviation of exactly zero.
    assert "± 0.0000 |" not in text
    assert "(one seed)" in text


def test_early_stopped_results_are_marked_superseded_wherever_they_appear(monkeypatch):
    """The topology section and the XGBoost comparison both draw on the graph
    model's runs; a stale run must be flagged in both."""
    run = {"at_4%": {"pr_auc": 0.98, "f1": 0.94, "fpr_at_95_recall": 0.002},
           "selection": "early_stopping", "epochs_run": 16}
    stat = {"mean": 0.98, "std": 0.001, "n_seeds": 1}
    agg = {"pr_auc": stat, "f1": stat, "fpr_at_95_recall": stat}
    gnn = {"generated_at": "2026-08-23", "config": {"eval": {"target_prevalence": 0.04},
                                                   "train": {"max_epochs": 40}},
           "results": {"full": {"runs": [run], "aggregate": agg},
                       "channel1_only": {"runs": [run], "aggregate": agg}}}
    base = {"results": {"xgboost::flow": {"schema": rf.BASELINE_SCHEMA,
                                          "aggregate": agg}}}
    monkeypatch.setattr(rf, "read", lambda rel: gnn if "gnn" in rel else base)

    assert "SUPERSEDED" in "\n".join(rf.sec_ablation(rf.PRIMARY, "3", "t"))
    assert "SUPERSEDED" in "\n".join(rf.sec_vs_xgboost())

    run["selection"] = "fixed_budget_best_val"
    assert "SUPERSEDED" not in "\n".join(rf.sec_ablation(rf.PRIMARY, "3", "t"))
    assert "SUPERSEDED" not in "\n".join(rf.sec_vs_xgboost())
