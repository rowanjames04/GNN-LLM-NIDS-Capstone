"""Tests for the Phase 3 -> Phase 4 comparator path (C17).

C17 was not a crash. `scripts/train_gnn.py` hardcoded an XGBoost score from a
different training regime and printed it as "<- to beat" for weeks, and every
"the GNN beat the baseline" statement inherited it. Nothing failed; the numbers
were simply not comparable.

So these tests target the property that failure had: a comparator that is
present but *wrong* must be impossible. A baseline entry earns its place in the
table only by carrying the schema tag AND the prevalence the GNN reports at.
Anything else is excluded by name rather than quietly averaged in.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_train_gnn():
    """Import the script without executing a training run.

    torch is imported lazily inside the script's functions (torch and xgboost
    cannot share a process on macOS), so importing the module is cheap.
    """
    spec = importlib.util.spec_from_file_location(
        "train_gnn_under_test", REPO_ROOT / "scripts" / "train_gnn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _entry(pr_auc: float, f1: float, *, schema: str, prevalence) -> dict:
    def stat(v):
        return {"mean": v, "std": 0.001, "n_seeds": 3}
    return {
        "model": "xgboost", "feature_set": "flow", "n_seeds": 3,
        "schema": schema, "reported_at_prevalence": prevalence,
        "aggregate": {
            "pr_auc": stat(pr_auc), "f1": stat(f1),
            "fpr_at_95_recall": stat(0.003),
        },
    }


@pytest.fixture
def baselines_file(tmp_path, monkeypatch):
    """A baselines file in the location load_comparators reads from."""
    mod = _load_train_gnn()
    out = tmp_path / "results" / "metrics" / "baselines"
    out.mkdir(parents=True)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)

    def write(results: dict):
        (out / "baselines_TESTSET.json").write_text(json.dumps({"results": results}))
    return mod, write


def test_reads_matching_entries(baselines_file):
    mod, write = baselines_file
    write({"xgboost::flow": _entry(0.986, 0.9416,
                                   schema=mod.BASELINE_SCHEMA, prevalence=0.04)})

    got = mod.load_comparators("TESTSET", 0.04)

    assert got["entries"]["xgboost::flow"]["pr_auc"] == 0.986
    assert got["entries"]["xgboost::flow"]["f1"] == 0.9416
    assert got["excluded_not_on_this_reporting_path"] == []


def test_excludes_entries_from_the_old_reporting_path(baselines_file):
    """The pre-C17 shape has no schema tag. It must never be silently used."""
    mod, write = baselines_file
    stale = _entry(0.9996, 0.99268, schema=mod.BASELINE_SCHEMA, prevalence=0.04)
    del stale["schema"]                      # exactly how the old files look
    write({"xgboost::flow": stale})

    got = mod.load_comparators("TESTSET", 0.04)

    assert got["entries"] == {}
    assert got["excluded_not_on_this_reporting_path"] == ["xgboost::flow"]


def test_excludes_entries_reported_at_another_prevalence(baselines_file):
    """Right schema, wrong scale -- still not comparable."""
    mod, write = baselines_file
    write({"xgboost::flow": _entry(0.9996, 0.9927,
                                   schema=mod.BASELINE_SCHEMA, prevalence=0.10)})

    got = mod.load_comparators("TESTSET", 0.04)

    assert got["entries"] == {}
    assert got["excluded_not_on_this_reporting_path"] == ["xgboost::flow"]


def test_missing_baselines_file_is_absent_not_zero(baselines_file):
    """A missing comparator must not read as a baseline of 0.0 to beat."""
    mod, _ = baselines_file

    got = mod.load_comparators("NO_SUCH_DATASET", 0.04)

    assert got["entries"] == {}
    assert "error" in got


def test_verdict_compares_against_the_strongest_baseline(baselines_file, capsys):
    """Overclaiming happens by picking the weakest comparator, not by lying."""
    mod, write = baselines_file
    comparators = {
        "entries": {
            "xgboost::flow": {"pr_auc": 0.9860, "pr_auc_std": 0.0005, "f1": 0.9416,
                              "f1_std": 0.0004, "fpr_at_95_recall": 0.00284,
                              "n_seeds": 3},
            "logreg::flow": {"pr_auc": 0.9500, "pr_auc_std": 0.0005, "f1": 0.9000,
                             "f1_std": 0.0004, "fpr_at_95_recall": 0.004,
                             "n_seeds": 3},
        },
        "excluded_not_on_this_reporting_path": [],
    }
    def stat(v):
        return {"mean": v, "std": 0.003}
    ablations = {"full": {"aggregate": {"pr_auc": stat(0.9887), "f1": stat(0.9410),
                                        "fpr_at_95_recall": stat(0.00248)}}}

    mod.print_comparison(comparators, ablations, ("full",), 0.04)
    out = capsys.readouterr().out

    assert "vs strongest baseline (xgboost::flow)" in out
    assert "+0.0027" in out      # PR-AUC gain against the strongest, not logreg
    assert "-0.0006" in out      # and the F1 loss is stated, not hidden
