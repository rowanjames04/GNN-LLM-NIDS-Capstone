"""Tests for `gnnids.eval.metrics`, against answers worked out by hand.

Every headline number in the project passes through this module -- the decision
threshold, the false-positive rate at a recall, per-family recall, the mean and
spread over seeds -- and until 2026-10-03 it had no direct tests. A defect here
would not raise; it would produce a slightly wrong number in every table.

Each case is small enough to verify on paper, which is the point: the expected
value is derived in the comment, not copied from the function's output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnnids.eval.metrics import (  # noqa: E402
    aggregate_seeds, choose_threshold, evaluate, evaluate_multiclass, fpr_at_recall,
)

FAMILIES = {"Benign": 0, "dos": 1, "scanning": 2, "mitm": 3}


# ------------------------------------------------------------------ threshold

def test_f1_threshold_is_the_score_that_maximises_f1():
    # Sorted by score: y = 1, 1, 0, 1, 0, 0.
    #   cut after 2: P 2/2, R 2/3 -> F1 0.800
    #   cut after 4: P 3/4, R 3/3 -> F1 0.857   <- best
    y = np.array([1, 1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])

    assert choose_threshold(y, s, "f1") == pytest.approx(0.6)


def test_recall_threshold_is_the_highest_score_reaching_the_target():
    # Recall reaches 2/3 at the second flow and 3/3 only at the fourth.
    y = np.array([1, 1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])

    assert choose_threshold(y, s, "recall", target_recall=0.6) == pytest.approx(0.8)
    assert choose_threshold(y, s, "recall", target_recall=0.95) == pytest.approx(0.6)


def test_threshold_with_no_positives_does_not_crash():
    assert choose_threshold(np.zeros(4), np.linspace(0, 1, 4)) == 0.5


# ------------------------------------------------------------- FPR at recall

def test_fpr_at_recall_counts_the_benign_flows_ranked_above_the_cut():
    # To reach recall 3/3 the cut falls after the fourth flow, by which point
    # one of the three benign flows has been flagged: FPR = 1/3.
    y = np.array([1, 1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])

    assert fpr_at_recall(y, s, 1.0) == pytest.approx(1 / 3)
    assert fpr_at_recall(y, s, 0.6) == pytest.approx(0.0)


def test_fpr_at_recall_is_nan_when_a_class_is_absent():
    """NaN, not 0.0 -- a rate with no denominator is not a perfect rate."""
    assert np.isnan(fpr_at_recall(np.ones(3), np.array([0.1, 0.2, 0.3])))


# ------------------------------------------------------------------- evaluate

def test_evaluate_reports_counts_and_per_family_recall():
    #          benign benign  dos   dos  scanning
    y = np.array([0,     0,    1,    1,     1])
    s = np.array([0.1,   0.8,  0.9,  0.3,   0.7])
    fam = np.array([0,   0,    1,    1,     2])

    out = evaluate(y, s, threshold=0.5, families=fam, family_names=FAMILIES)

    # Flagged: benign(0.8), dos(0.9), scanning(0.7) -> TP 2, FP 1, FN 1.
    assert out["true_positives"] == 2 and out["false_positives"] == 1
    assert out["precision"] == pytest.approx(2 / 3, abs=1e-5)
    assert out["recall"] == pytest.approx(2 / 3, abs=1e-5)
    assert out["alerts_per_true_positive"] == 1.5
    assert out["per_family_recall"]["dos"] == pytest.approx(
        {"n": 2, "recall": 0.5, "recall_ci95": out["per_family_recall"]["dos"]["recall_ci95"]})
    assert out["per_family_recall"]["scanning"]["recall"] == 1.0
    assert "Benign" not in out["per_family_recall"]
    assert "mitm" not in out["per_family_recall"]        # absent, not zero


def test_pr_auc_is_none_not_zero_when_there_are_no_positives():
    out = evaluate(np.zeros(4, dtype=int), np.linspace(0, 1, 4), 0.5)

    assert out["pr_auc"] is None


# ------------------------------------------------------------ seed aggregation

def test_aggregate_is_the_population_mean_and_std_per_metric():
    agg = aggregate_seeds([{"pr_auc": 0.90, "f1": 0.5},
                           {"pr_auc": 0.94, "f1": 0.7}])

    assert agg["pr_auc"] == {"mean": 0.92, "std": 0.02, "n_seeds": 2}
    assert agg["f1"]["mean"] == 0.6


def test_aggregate_skips_a_seed_whose_metric_is_missing_and_says_so():
    """n_seeds is per metric, so a table cannot present 2 seeds as 3."""
    agg = aggregate_seeds([{"pr_auc": 0.9}, {"pr_auc": None}, {"pr_auc": 0.7}])

    assert agg["pr_auc"]["n_seeds"] == 2
    assert agg["pr_auc"]["mean"] == 0.8


# ------------------------------------------------------- the attack-family head

def test_multiclass_scores_overall_attack_only_and_flagged_views():
    # true:  benign benign  dos   dos   scanning scanning
    # pred:  benign  dos    dos  benign scanning   dos
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 1, 1, 0, 2, 1])
    flagged = np.array([False, True, True, True, True, False])

    out = evaluate_multiclass(y, p, FAMILIES, flagged=flagged)

    assert out["accuracy"] == pytest.approx(3 / 6)
    # Among the four true attacks: dos right, dos wrong, scanning right, wrong.
    assert out["attack_family_accuracy"] == pytest.approx(2 / 4)
    # Flagged true attacks: dos(right), dos(called benign), scanning(right).
    tp = out["on_true_positive_detections"]
    assert tp["n"] == 3
    assert tp["family_accuracy"] == pytest.approx(2 / 3, abs=1e-5)
    assert tp["called_benign"] == pytest.approx(1 / 3, abs=1e-5)
    # Of the four flagged flows, one is called benign by the family head.
    assert out["heads_disagree_on_flagged"] == pytest.approx(1 / 4)


def test_macro_f1_gives_a_rare_family_the_same_weight_as_a_common_one():
    """A head that never predicts the rare family must not score well because
    the common family is large."""
    y = np.array([1] * 98 + [3] * 2)             # 98 dos, 2 mitm
    p = np.array([1] * 100)                      # predicts dos for everything

    out = evaluate_multiclass(y, p, FAMILIES)

    assert out["accuracy"] == 0.98
    # dos F1 = 2*0.98*1/(1.98) = 0.9899; mitm F1 = 0 -> macro 0.4949.
    assert out["macro_f1"] == pytest.approx(0.4949, abs=1e-4)
    assert out["per_class"]["mitm"]["recall"] == 0.0


def test_macro_f1_ignores_families_absent_from_the_split():
    """Leave-one-attack-out test splits hold two classes. Averaging in eight
    absent ones would report a near-zero macro-F1 for a perfect head."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 0, 1, 1])

    assert evaluate_multiclass(y, p, FAMILIES)["macro_f1"] == 1.0


def test_confusion_names_what_each_family_was_mistaken_for():
    y = np.array([3, 3, 3, 1])
    p = np.array([2, 2, 1, 1])

    conf = evaluate_multiclass(y, p, FAMILIES)["confusion"]

    assert conf["mitm"] == {"dos": 1, "scanning": 2}
    assert conf["dos"] == {"dos": 1}
