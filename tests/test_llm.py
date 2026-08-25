"""Tests for Phase 7a: prompts, adapters and the groundedness metric.

Groundedness is the study's most defensible number, so it gets the most tests.
Its failure modes are asymmetric and both are bad in different ways: scoring a
**fabricated IP** as grounded would make a dangerous model look safe, while
scoring a **rounded number** as ungrounded would punish a model for writing
readable prose and quietly select for worse reports.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnnids.llm.adapters import build_adapter  # noqa: E402
from gnnids.llm.base import StubAdapter  # noqa: E402
from gnnids.llm.groundedness import (  # noqa: E402
    check_no_jargon, check_uncertainty_conveyed, extract_entities, score_report,
)
from gnnids.llm.prompts import PROMPTS, build_prompt, render_facts  # noqa: E402


def _pack(**over):
    p = {
        "schema_version": "1.0", "detection_id": "abc-123",
        "flow": {"src_ip": "192.168.1.32", "dst_ip": "192.168.1.180",
                 "IN_BYTES": 48, "L4_DST_PORT": 1069, "true_label": "scanning"},
        "detection": {"score": 0.96004, "threshold": 0.86577, "flagged": True,
                      "predicted_class": "scanning", "class_confidence": 0.9995,
                      "class_distribution": {"scanning": 0.9995, "Benign": 0.0005}},
        "attribution": {"channel_weights": {"attribute": 0.66, "topological": 0.34},
                        "top_features": [
                            {"name": "PROTOCOL=6", "value": 1.0,
                             "contribution": 1.9, "direction": "increases_suspicion"},
                            {"name": "IN_BYTES", "value": 48,
                             "contribution": -1.2, "direction": "decreases_suspicion"}],
                        "influential_neighbours": []},
        "context": {"src_host": {"flow_count": 212, "distinct_dst_ips": 10,
                                 "distinct_dst_ports": 77, "total_bytes": 12436,
                                 "baseline_deviation": {"times_typical": 10.0}}},
        "causal_path": None,
        "guidance": {"attack_family": "scanning", "mitre_techniques": ["T1046"],
                     "description": "Systematic probing of hosts or ports.",
                     "standard_mitigations": ["Rate-limit the source"]},
        "provenance": {"checkpoint": "seed0"},
    }
    p.update(over)
    return p


# -------------------------------------------------------------- groundedness

def test_a_fabricated_address_is_caught_and_called_out_separately():
    """The failure with operational consequences: it sends an analyst to a host
    that was never in the evidence."""
    report = "Traffic from 10.99.99.99 was observed."

    got = score_report(_pack(), report)

    assert got["fabricated_addresses"] == ["10.99.99.99"]
    assert got["per_class"]["ips"]["groundedness"] == 0.0


def test_a_real_address_from_the_evidence_is_grounded():
    got = score_report(_pack(), "Traffic from 192.168.1.32 to 192.168.1.180.")

    assert got["fabricated_addresses"] == []
    assert got["groundedness"] == 1.0


def test_a_rounded_number_counts_as_grounded():
    """0.96004 restated as 0.96 is the report doing its job. Penalising it would
    select for reports that parrot full precision."""
    got = score_report(_pack(), "The detection score was 0.96.")

    assert got["n_ungrounded"] == 0


def test_thousands_separators_do_not_count_as_invention():
    got = score_report(_pack(), "The host transferred 12,436 bytes.")

    assert got["n_ungrounded"] == 0


def test_a_percentage_restatement_is_not_penalised():
    """A model writing "96%" for 0.96 is clearer, not less grounded."""
    got = score_report(_pack(), "Confidence was around 96%.")

    assert got["n_ungrounded"] == 0


def test_an_invented_mitre_technique_is_caught():
    got = score_report(_pack(), "This maps to T1046 and T9999.")

    assert got["per_class"]["mitre"]["ungrounded"] == ["T9999"]


def test_small_integers_used_as_prose_are_free():
    """"three ports" must not be flagged just because 3 is not in the evidence."""
    got = score_report(_pack(), "There are 3 things to check and 2 hosts involved.")

    assert got["n_ungrounded"] == 0


def test_a_report_with_no_technical_entities_scores_none_not_one():
    """No claims is not the same as all claims correct, and averaging a 1.0 in
    would reward a model for saying nothing."""
    got = score_report(_pack(), "Something happened. Investigate it.")

    assert got["groundedness"] is None
    assert got["n_entities_claimed"] == 0


def test_extract_entities_finds_each_class():
    got = extract_entities("Host 10.0.0.1 used T1046 and sent 4,498 bytes")

    assert got["ips"] == {"10.0.0.1"}
    assert got["mitre"] == {"T1046"}
    assert "4498" in got["numbers"]


# ------------------------------------------------------- behavioural checks

def test_uncertainty_check_returns_none_on_confident_detections():
    """Averaging the easy cases in would turn a sharp test into a flattering
    constant."""
    assert check_uncertainty_conveyed(_pack(), "It was scanning.") is None


def test_uncertainty_check_detects_a_hedge_on_an_ambiguous_detection():
    pack = _pack(detection={
        "score": 0.9, "threshold": 0.5, "flagged": True,
        "predicted_class": "scanning", "class_confidence": 0.4,
        "class_distribution": {"scanning": 0.4, "dos": 0.35, "ddos": 0.25}})

    hedged = check_uncertainty_conveyed(pack, "The category is uncertain.")
    flat = check_uncertainty_conveyed(pack, "This is a scan.")

    assert hedged["uncertainty_conveyed"] is True
    assert flat["uncertainty_conveyed"] is False


def test_jargon_from_the_models_own_knowledge_is_caught():
    """The pack cannot contain these terms -- validate_pack refuses them -- so a
    report containing them got them from the model itself."""
    got = check_no_jargon("The graph neural network flagged this.")

    assert got["clean"] is False
    assert "graph neural" in got["jargon_found"]


# ---------------------------------------------------------------- prompting

def test_the_prompt_states_ambiguity_in_words_not_just_numbers():
    """Rule 2 asks the model to convey uncertainty; that is only checkable if
    the ambiguity is in the input as prose."""
    pack = _pack(detection={
        "score": 0.9, "threshold": 0.5, "flagged": True,
        "predicted_class": "scanning", "class_confidence": 0.4,
        "class_distribution": {"scanning": 0.4, "dos": 0.35, "ddos": 0.25}})

    facts = render_facts(pack)

    assert "uncertain" in facts.lower()
    assert "dos" in facts


def test_one_hot_features_render_as_properties_not_quantities():
    """"PROTOCOL=6 measured 1.0" hands the model a number with no referent."""
    facts = render_facts(_pack())

    assert "The flow's PROTOCOL was 6" in facts
    assert "PROTOCOL=6 measured" not in facts


def test_an_unknown_prompt_version_is_refused_by_name():
    with pytest.raises(KeyError, match="v1"):
        build_prompt(_pack(), "v99")


def test_every_prompt_version_has_both_halves():
    for version, t in PROMPTS.items():
        assert t["system"].strip() and "{facts}" in t["user"], version


# ----------------------------------------------------------------- adapters

def test_the_stub_is_deterministic_and_free():
    a = StubAdapter()
    system, user = build_prompt(_pack())

    first, second = a.generate(system, user, "v1"), a.generate(system, user, "v1")

    assert first.text == second.text
    assert first.cost_usd == 0.0 and first.ok


def test_the_stub_invents_nothing():
    """The control arm has to score 1.0 on groundedness or it is not a floor,
    it is just another unreliable narrator."""
    pack = _pack()
    system, user = build_prompt(pack)

    r = StubAdapter().generate(system, user, "v1")

    assert score_report(pack, r.text)["n_ungrounded"] == 0


def test_building_a_paid_adapter_makes_no_network_call():
    """Construction must be free; only generate() costs money."""
    a = build_adapter("anthropic")

    assert a.provider == "anthropic" and a.model == "claude-opus-5"


def test_an_unknown_provider_names_the_alternatives():
    with pytest.raises(KeyError, match="anthropic"):
        build_adapter("gpt5-turbo-ultra")
