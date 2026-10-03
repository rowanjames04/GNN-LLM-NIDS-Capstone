"""Tests for the evidence-pack aggregation behind the explainability results."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "summarise_evidence_under_test", REPO_ROOT / "scripts" / "summarise_evidence.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def _pack(i, true, predicted, topo, features, confidence=0.99):
    return {
        "detection_id": f"d{i}",
        "flow": {"true_label": true},
        "detection": {"predicted_class": predicted, "class_confidence": confidence},
        "attribution": {
            "channel_weights": {"attribute": 1 - topo, "topological": topo},
            "top_features": [{"name": n, "contribution": c,
                              "direction": "increases_suspicion" if c > 0
                              else "decreases_suspicion"} for n, c in features],
            "influential_neighbours": []},
    }


PACKS = [
    _pack(0, "scanning", "scanning", 0.2,
          [("L4_DST_PORT", 2.0), ("L4_DST_PORT_bucket=1", 1.0), ("PROTOCOL=6", -1.0)]),
    _pack(1, "scanning", "dos", 0.4, [("L4_DST_PORT", 1.0)]),
    _pack(2, "Benign", "password", 0.9, [("IN_BYTES", 3.0)], confidence=0.6),
]


def test_an_empty_group_is_none_not_a_row_of_zeros():
    assert ev.dist([]) is None


def test_a_feature_and_its_bucket_count_as_one_feature_per_pack():
    """L4_DST_PORT and L4_DST_PORT_bucket=1 are one thing to a reader. Counting
    the pack twice would make a feature appear in more packs than exist."""
    s = ev.summarise(PACKS)
    ranked = {r["feature"]: r for r in s["features_shown_to_the_model"]["ranked"]}

    assert ranked["L4_DST_PORT"]["packs"] == 2
    assert ranked["L4_DST_PORT"]["share_of_packs"] == round(2 / 3, 4)
    assert ranked["PROTOCOL"]["lowers_suspicion"] == 1


def test_false_positives_are_separated_from_true_attacks():
    s = ev.summarise(PACKS)
    t = s["topological_weight"]

    assert s["n_true_attacks"] == 2 and s["n_false_positives"] == 1
    assert t["true_attacks"]["mean"] == 0.3
    assert t["false_positives"]["mean"] == 0.9
    assert t["by_true_family"]["scanning"]["n"] == 2


def test_the_family_named_to_the_model_is_scored_on_true_attacks_only():
    """A false positive has no true attack family to be right about, so it is
    reported as what it was named, not averaged in as a miss."""
    m = ev.summarise(PACKS)["family_named_to_the_model"]

    assert m["correct_on_true_attacks"] == 0.5 and m["n_true_attacks"] == 2
    assert m["false_positives_named_as"] == {"password": 1}
    assert m["ambiguous_packs"] == 1


def test_occlusion_is_absent_for_packs_made_before_it_was_recorded():
    assert ev.summarise(PACKS, sampling={"checkpoint": "x"})["neighbour_occlusion"] is None


def test_occlusion_reports_the_largest_change_at_full_precision():
    """The finding is how small the number is; rounding it to 4 places would
    print 0.0 and erase it."""
    sampling = {"neighbour_occlusion": {
        "threshold": 0.01, "tested_per_detection_cap": 50,
        "by_detection": {
            "d0": {"neighbours_sharing_a_host": 1800, "neighbours_tested": 50,
                   "max_abs_influence": 0.00009},
            "d1": {"neighbours_sharing_a_host": 300, "neighbours_tested": 50,
                   "max_abs_influence": 0.00003},
            "d2": {"neighbours_sharing_a_host": 0, "neighbours_tested": 0,
                   "max_abs_influence": None}}}}

    o = ev.summarise(PACKS, sampling)["neighbour_occlusion"]

    assert o["max_abs_influence"]["n"] == 2            # the None is not a zero
    assert o["max_abs_influence"]["max"] == 0.00009
    assert o["neighbours_sharing_a_host"]["median"] == 300
