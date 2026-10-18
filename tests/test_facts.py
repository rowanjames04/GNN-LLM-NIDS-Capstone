"""Tests for the facts layer and the checks built on it (D49, 2026-10-17).

The facts layer exists because the round-1 audit found models doing arithmetic,
filling gaps and flipping directions. These tests pin the properties that stop
that: every number the model needs is computed and stated, codes are decoded,
reasons and context are separate, and the two new checks catch the audit's
errors without inventing ones that are not there.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnids.explain.units import feature_names  # noqa: E402
from gnnids.llm import facts  # noqa: E402
from gnnids.llm.fidelity import check_direction_fidelity  # noqa: E402
from gnnids.llm.groundedness import check_borderline_conveyed, score_report  # noqa: E402
from gnnids.llm.prompts import build_prompt  # noqa: E402


def _pack(score=0.95, threshold=0.88, conf=0.99, dist=None, top=None):
    return {
        "schema_version": "2.0", "detection_id": "d-1",
        "flow": {"src_ip": "192.168.1.30", "dst_ip": "192.168.1.46",
                 "L4_SRC_PORT": 47744, "L4_DST_PORT": 445, "PROTOCOL": 6,
                 "L7_PROTO": 0.0, "IN_BYTES": 96, "IN_PKTS": 2, "OUT_BYTES": 0,
                 "OUT_PKTS": 0, "CLIENT_TCP_FLAGS": 2, "SERVER_TCP_FLAGS": 20,
                 "SHORTEST_FLOW_PKT": 48, "LONGEST_FLOW_PKT": 48},
        "detection": {"score": score, "threshold": threshold, "flagged": True,
                      "predicted_class": "scanning", "class_confidence": conf,
                      "class_distribution": dist or {"scanning": conf, "Benign": 1 - conf}},
        "attribution": {
            "channel_weights": {"attribute": 0.66, "topological": 0.34},
            "top_features": top or [
                {"name": "PROTOCOL=6", "value": 1.0, "direction": "increases_suspicion"},
                {"name": "TCP_WIN_MAX_IN", "value": 4096, "direction": "decreases_suspicion"},
                {"name": "L4_DST_PORT_bucket=well_known", "value": 1.0,
                 "direction": "increases_suspicion"},
            ],
            "influential_neighbours": [],
            "neighbour_note": "None of the 50 nearby flows tested changed this score."},
        "context": {"window_flow_count": 10000, "window_host_count": 21,
                    "src_host": {"flow_count": 4371, "distinct_dst_ips": 11,
                                 "distinct_dst_ports": 3647,
                                 "baseline_deviation": {"times_typical": 5.5}},
                    "dst_host": {"flow_count": 1100, "distinct_dst_ips": 1,
                                 "distinct_dst_ports": 1100,
                                 "baseline_deviation": {"times_typical": 0.4}}},
        "causal_path": None,
        "data_quality": {"withheld_columns": ["FLOW_DURATION_MILLISECONDS"],
                         "failed_checks": [], "notes": ["How long the flow lasted is not known."]},
        "guidance": {"attack_family": "scanning", "description": "Probing.",
                     "mitre_techniques": ["T1046"], "standard_mitigations": ["Rate-limit"]},
        "provenance": {},
    }


# --- decoding ----------------------------------------------------------------

def test_codes_are_decoded_so_the_model_never_has_to():
    assert facts.tcp_flags(2) == "SYN"
    assert facts.tcp_flags(18) == "SYN+ACK"
    assert facts.tcp_flags(0) == "no TCP flags"
    assert facts.protocol(6) == "TCP" and facts.protocol(17) == "UDP"
    assert facts.icmp(2048) == "ICMP type 8, code 0"
    assert facts.app_protocol(7.178) == "HTTP"
    assert facts.app_protocol(0) == "not identified"
    assert "not decoded" in facts.app_protocol(131)          # unknown: said, not guessed
    assert facts.port(445) == "445 (SMB)" and facts.port(59378) == "59378"


def test_one_hot_names_say_what_slot_zero_means():
    """C34: slot 0 of a port bucket is the well-known range, not "other"."""
    meta = {"continuous_names": ["IN_BYTES"],
            "categorical_names": ["PROTOCOL", "L4_DST_PORT_bucket", "DNS_QUERY_TYPE_present"],
            "cardinalities": [2, 3, 2], "vocab": {"PROTOCOL": {"6": 1}}}
    names = feature_names(meta)
    assert "L4_DST_PORT_bucket=well_known" in names
    assert "L4_DST_PORT_bucket=ephemeral" in names
    assert "DNS_QUERY_TYPE_present=absent" in names and "DNS_QUERY_TYPE_present=present" in names
    assert "PROTOCOL=other" in names and "PROTOCOL=6" in names


# --- what the facts say -----------------------------------------------------

def test_the_margin_and_confidence_are_computed_and_worded():
    m = facts.margin({"score": 0.95, "threshold": 0.90})
    assert m["gap"] == 0.05 and m["band"] == "moderate"
    assert facts.margin({"score": 0.905, "threshold": 0.90})["band"] == "narrow"
    assert facts.margin({"score": 0.99, "threshold": 0.90})["band"] == "wide"
    text = facts.render(_pack(score=0.885, threshold=0.88))
    assert "narrow margin" in text and "borderline detection" in text


def test_a_plausible_second_category_makes_the_detection_borderline():
    p = _pack(conf=0.7, dist={"scanning": 0.7, "dos": 0.25, "Benign": 0.05})
    text = facts.render(p)
    assert facts.borderline(p)
    assert "dos (0.25)" in text and "uncertain" in text


def test_reasons_carry_their_direction_and_context_is_labelled_as_context():
    text = facts.render(_pack())
    why = text.split("WHY THE DETECTOR FLAGGED IT")[1].split("BACKGROUND CONTEXT")[0]
    ctx = text.split("BACKGROUND CONTEXT")[1]
    assert "Raised the score: the transport protocol, TCP." in why
    assert "Lowered the score: the largest TCP window the source advertised, 4,096 bytes." in why
    assert "a well-known port (below 1,024)" in why
    assert "4,371" not in why and "4,371" in ctx                 # host activity is context
    assert "not a reason for the score" in text


def test_the_destination_host_is_described_as_receiving():
    """B15: its counts are inbound."""
    text = facts.render(_pack())
    assert "The destination host received 1,100 flows" in text
    assert "from 1 distinct address," in text


def test_both_directions_of_traffic_are_stated_with_averages_computed():
    text = facts.render(_pack())
    assert "From source to destination: 96 bytes in 2 packets, an average of 48 bytes per packet." in text
    assert "From destination back to source: 0 bytes in 0 packets." in text
    assert "The source sent these TCP flags: SYN." in text
    assert "Every packet was 48 bytes." in text


def test_what_is_not_known_is_stated_and_the_duration_never_appears():
    text = facts.render(_pack())
    assert "WHAT IS NOT KNOWN" in text and "not known" in text
    assert "Duration (ms)" not in text and "4294" not in text


def test_the_facts_carry_no_model_vocabulary():
    from gnnids.explain.evidence import BANNED_TERMS
    text = facts.render(_pack()).lower()
    assert not [t for t in BANNED_TERMS if t in text]


def test_v3_and_v4_use_the_facts_and_forbid_arithmetic():
    for v in ("v3", "v4"):
        system, user = build_prompt(_pack(), v)
        assert "calculate" in system
        assert "WHY THE DETECTOR FLAGGED IT" in user
    _, old = build_prompt(_pack(), "v1")
    assert "WHY THE DETECTOR FLAGGED IT" not in old       # v1 keeps its own renderer


def test_numbers_the_facts_layer_computed_count_as_supplied():
    p = _pack()
    s = score_report(p, "It cleared the threshold by 0.07, an average of 48 bytes per packet.")
    assert s["n_ungrounded"] == 0
    s = score_report(p, "It cleared the threshold by 0.31.")
    assert s["per_class"]["numbers"]["ungrounded"] == ["0.31"]


# --- the direction check ------------------------------------------------------

def test_a_flipped_reason_inside_a_list_is_caught():
    """The audit's 010d73db: a feature that lowered the score listed among the
    contributing factors."""
    r = "Contributing factors included the TCP protocol (6) and a large inbound TCP window of 4,096."
    c = check_direction_fidelity(_pack(), r)
    assert c["n_errors"] == 1 and c["errors"][0]["feature"] == "TCP_WIN_MAX_IN"


def test_a_correct_report_scores_no_errors():
    r = ("The transport protocol, TCP, raised the score, while the largest TCP window "
         "the source advertised lowered it.")
    c = check_direction_fidelity(_pack(), r)
    assert c["n_checked"] == 2 and c["n_errors"] == 0 and c["direction_faithful"]


def test_a_clause_with_both_directions_or_only_context_is_skipped():
    mixed = "The protocol 6 raised and lowered things."
    assert check_direction_fidelity(_pack(), mixed)["n_checked"] == 0
    context = "The source opened 4,371 connections, which contributed to the alert."
    assert check_direction_fidelity(_pack(), context)["n_checked"] == 0


def test_an_unqualified_tcp_window_is_read_only_when_unambiguous():
    both = _pack(top=[
        {"name": "TCP_WIN_MAX_IN", "value": 4096, "direction": "decreases_suspicion"},
        {"name": "TCP_WIN_MAX_OUT", "value": 0, "direction": "increases_suspicion"}])
    assert check_direction_fidelity(both, "A large TCP window raised the score.")["n_checked"] == 0
    assert check_direction_fidelity(_pack(), "A large TCP window raised the score.")["n_errors"] == 1


# --- the borderline check ------------------------------------------------------

def test_a_borderline_detection_must_be_reported_as_one():
    p = _pack(score=0.885, threshold=0.88)
    assert not check_borderline_conveyed(p, "A scan was detected.")["borderline_conveyed"]
    assert check_borderline_conveyed(p, "The score only just crossed the threshold.")["borderline_conveyed"]
    assert check_borderline_conveyed(_pack(), "A scan was detected.") is None   # clear detection


# --- the pack's data-quality section -------------------------------------------

def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_withheld_timing_and_failed_checks_become_stated_notes():
    ev = _load("make_evidence")
    vals = {"IN_PKTS": np.array([3.0]), "OUT_PKTS": np.array([0.0]),
            "IN_BYTES": np.array([30.0]), "OUT_BYTES": np.array([0.0]),
            "FLOW_DURATION_MILLISECONDS": np.array([0.0])}
    q = ev.data_quality_notes(vals, ["FLOW_DURATION_MILLISECONDS", "DURATION_IN"])
    assert q["notes"][0] == ev.TIMING_NOTE
    assert "in_bytes_below_20_per_packet" in q["failed_checks"]
    assert any("fewer than an IP header" in n for n in q["notes"])


def test_a_collapsed_generation_is_a_failure_not_a_fabrication():
    """C37."""
    from gnnids.llm.groundedness import is_degenerate
    from gnnids.llm.scoring import summarise

    junk = ". . .. .. 0.000.0.0.0.0.0.0 . . . .. ..0e. The . . . " * 20
    assert is_degenerate(junk)
    assert not is_degenerate("The detector flagged the flow from 192.168.1.30. It is a scan.")
    rows = [{"ok": True, "degenerate": True, "scores": {"groundedness": 0.0,
             "fabricated_addresses": ["0.0.0.0"], "report_words": 9}, "jargon": {"clean": True},
             "latency_seconds": 1.0, "cost_usd": None, "reference_cost_usd": 0.0},
            {"ok": True, "degenerate": False, "scores": {"groundedness": 1.0,
             "fabricated_addresses": [], "report_words": 200}, "jargon": {"clean": True},
             "latency_seconds": 1.0, "cost_usd": None, "reference_cost_usd": 0.0}]
    s = summarise(rows)
    assert s["n_degenerate"] == 1 and s["groundedness_mean"] == 1.0
    assert s["reports_with_fabricated_addresses"] == 0
