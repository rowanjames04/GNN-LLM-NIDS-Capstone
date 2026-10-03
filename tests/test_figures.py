"""Tests for the results figures.

A figure is the part of a result most likely to be copied into the report and
least likely to be re-checked, so the tests guard what a reader cannot see from
the image: that seeds were paired correctly, that a figure says how its model
was trained, and that meaningless smoke numbers cannot reach the figures folder.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "make_figures_under_test", REPO_ROOT / "scripts" / "make_figures.py")
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)


def _doc(full, c1, selection="fixed_budget_best_val", seeds=True):
    def runs(vals, order):
        out = []
        for seed, v in zip(order, vals):
            r = {"at_4%": {"pr_auc": v}, "selection": selection}
            if seeds:
                r["seed"] = seed
            out.append(r)
        return out
    return {"config": {"eval": {"target_prevalence": 0.04}},
            "results": {"full": {"runs": runs(full, [2, 0, 1])},
                        "channel1_only": {"runs": runs(c1, [0, 1, 2])}}}


def test_topology_gain_pairs_runs_by_seed_not_by_position():
    """An extended or resumed campaign merges runs out of order. Subtracting by
    position would pair seed 2 with seed 0."""
    #            seeds:  2     0     1                 0     1     2
    doc = _doc(full=[0.99, 0.97, 0.98], c1=[0.96, 0.96, 0.96])

    gains = mf.paired_gain(doc)

    assert gains == pytest.approx([0.03, 0.01, 0.02])     # seed 2, 0, 1


def test_results_without_recorded_seeds_fall_back_to_position():
    doc = _doc(full=[0.99, 0.97], c1=[0.96, 0.95], seeds=False)

    assert mf.paired_gain(doc) == pytest.approx([0.03, 0.02])


def test_a_caption_says_when_the_run_is_the_superseded_early_stopped_one():
    assert "superseded" in mf.how_trained(_doc([0.9], [0.8], selection="early_stopping"))
    assert mf.how_trained(_doc([0.9], [0.8])) == "fixed training budget"


def test_a_missing_results_file_is_a_skip_with_a_reason_not_a_crash():
    with pytest.raises(mf.Skip, match="does not exist yet"):
        mf.load(REPO_ROOT / "results" / "metrics" / "nope.json")


def test_smoke_numbers_cannot_be_drawn_into_the_figures_folder():
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "make_figures.py"), "--smoke"],
        capture_output=True, text=True, timeout=120)

    assert r.returncode != 0
    assert "meaningless" in r.stderr


def test_a_figure_is_drawn_from_the_committed_results(tmp_path):
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "make_figures.py"),
         "--only", "transfer", "--out-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=180)

    assert r.returncode == 0, r.stderr
    assert (tmp_path / "results_05_cross_dataset_transfer.png").stat().st_size > 10_000


def test_the_manifest_gives_every_drawn_figure_a_caption_and_its_sources():
    manifest = REPO_ROOT / "results" / "figures" / "results_manifest.json"
    if not manifest.exists():
        pytest.skip("figures not generated in this checkout")
    for key, fig in json.loads(manifest.read_text())["figures"].items():
        if fig["status"] == "drawn":
            assert fig["caption"] and fig["sources"], key
        else:
            assert fig["reason"], key
