"""Tests for the fidelity checks, the shared scorer, and re-scoring.

Groundedness cannot see a report that invents nothing and still misleads: one
that names a different attack from the one the detector predicted, or ignores
the features the detector's decision rested on. These checks exist for those
two cases, so the tests are built around them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnids.llm.fidelity import (  # noqa: E402
    base_column, check_attribution_fidelity, check_classification_fidelity,
    cites_feature, families_named,
)
from gnnids.llm.scoring import score_all, summarise  # noqa: E402

FAMILIES = ["Benign", "ddos", "dos", "scanning", "xss", "password"]


def _pack(predicted="scanning", dist=None, features=None):
    dist = dist or {f: (0.99 if f == predicted else 0.002) for f in FAMILIES}
    feats = features if features is not None else [
        {"name": "L4_SRC_PORT", "value": 2592, "direction": "decreases_suspicion"},
        {"name": "L7_PROTO=0.0", "value": 1.0, "direction": "increases_suspicion"},
        {"name": "L4_SRC_PORT_bucket=1", "value": 1.0, "direction": "increases_suspicion"},
        {"name": "PROTOCOL=6", "value": 1.0, "direction": "increases_suspicion"},
        {"name": "IN_BYTES", "value": 48, "direction": "decreases_suspicion"},
    ]
    return {"detection_id": "d1",
            "flow": {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "true_label": predicted},
            "detection": {"score": 0.97, "threshold": 0.86, "predicted_class": predicted,
                          "class_confidence": dist[predicted], "class_distribution": dist},
            "attribution": {"top_features": feats}}


# ---------------------------------------------------- classification fidelity

def test_a_report_naming_the_predicted_category_is_faithful():
    out = check_classification_fidelity(
        _pack(), "The activity is consistent with network scanning.")

    assert out["names_predicted"] and out["faithful"]
    assert out["unsupported_families"] == []


def test_a_report_that_swaps_the_category_is_a_substitution():
    """The detector said scanning; the report says brute force instead. Every
    entity in it may be grounded -- this is the failure groundedness misses."""
    out = check_classification_fidelity(
        _pack(), "This is a brute-force login attempt against the host.")

    assert not out["names_predicted"]
    assert out["unsupported_families"] == ["password"]
    assert out["substituted"] and not out["faithful"]


def test_naming_an_alternative_the_evidence_offered_is_not_a_substitution():
    """The prompt itself lists runner-up categories above 5%. Repeating one is
    relaying the evidence, not overriding the detector."""
    dist = {"Benign": 0.0, "ddos": 0.55, "dos": 0.40, "scanning": 0.05,
            "xss": 0.0, "password": 0.0}
    out = check_classification_fidelity(
        _pack("ddos", dist),
        "Most likely a distributed denial of service, though a single-source "
        "denial of service cannot be ruled out.")

    assert out["faithful"]
    assert set(out["families_named"]) == {"ddos", "dos"}


def test_a_ddos_report_is_not_also_scored_as_naming_dos():
    """'distributed denial of service' contains 'denial of service'."""
    assert families_named("A distributed denial-of-service attack (DDoS).",
                          FAMILIES) == {"ddos"}
    assert families_named("A DoS attack.", FAMILIES) == {"dos"}


def test_a_family_name_inside_another_word_is_not_a_match():
    assert families_named("The windows host responded.", ["dos"]) == set()


def test_a_report_naming_no_category_is_neither_faithful_nor_a_substitution():
    out = check_classification_fidelity(_pack(), "A flow was flagged for review.")

    assert not out["faithful"] and not out["substituted"]


# ------------------------------------------------------- attribution fidelity

def test_encoding_suffixes_collapse_to_one_feature():
    assert base_column("L4_SRC_PORT_bucket=1") == "L4_SRC_PORT"
    assert base_column("DNS_QUERY_TYPE_present=other") == "DNS_QUERY_TYPE"
    assert base_column("PROTOCOL=6") == "PROTOCOL"


def test_a_feature_is_cited_by_name_or_by_its_words_in_one_sentence():
    assert cites_feature("L4_SRC_PORT measured 2592.", "L4_SRC_PORT")
    assert cites_feature("The source port lowered the assessment.", "L4_SRC_PORT")
    assert cites_feature("Only 48 bytes were received.", "IN_BYTES")
    assert cites_feature("The application-layer protocol was unusual.", "L7_PROTO")


def test_words_split_across_sentences_do_not_count():
    """'source' in one sentence and 'port' in another is not a citation."""
    assert not cites_feature(
        "The source host is internal. The destination port was 443.", "L4_SRC_PORT")


def test_coverage_counts_distinct_features_the_model_was_shown():
    report = ("The source port and the transport protocol raised suspicion. "
              "Reconnaissance of this kind usually precedes an attack.")

    out = check_attribution_fidelity(_pack(), report)

    # Four distinct columns: L4_SRC_PORT (twice, once as a bucket), L7_PROTO,
    # PROTOCOL, IN_BYTES. "protocol" alone cites PROTOCOL but not L7_PROTO.
    assert out["n_attributed"] == 4
    assert out["features_cited"] == ["L4_SRC_PORT", "PROTOCOL"]
    assert out["coverage"] == 0.5 and not out["cites_none"]


def test_a_generic_report_that_ignores_the_evidence_cites_none():
    """Pattern-matched to a textbook description: plausible, and uses nothing
    the attribution found."""
    out = check_attribution_fidelity(
        _pack(), "Scanning is the systematic probing of hosts to map services.")

    assert out["cites_none"] and out["coverage"] == 0.0


def test_no_attribution_in_the_pack_is_none_not_a_score():
    assert check_attribution_fidelity(_pack(features=[]), "anything") is None


# ------------------------------------------------------------ shared scoring

def test_a_failed_generation_scores_nothing_rather_than_zero():
    out = score_all(_pack(), "", ok=False)

    assert out["scores"] == {} and out["classification"] is None


def test_summary_reports_fidelity_rates_and_latency_percentiles():
    pack = _pack()
    rows = []
    for text, latency in (("A scanning sweep from the source port.", 1.0),
                          ("A brute-force attempt.", 3.0)):
        rows.append({"ok": True, "latency_seconds": latency, "cost_usd": 0.01,
                     "reference_cost_usd": 0.01, **score_all(pack, text)})

    s = summarise(rows)

    assert s["classification_faithful_rate"] == 0.5
    assert s["classification_substituted_rate"] == 0.5
    assert s["attribution_cites_none_rate"] == 0.5
    assert s["latency_seconds_p50"] == 2.0
    assert s["latency_seconds_p95"] == 2.9
    assert s["generation_seconds_total"] is None        # no server timing given


def test_generation_time_gives_tokens_per_second_for_self_hosted_rows():
    rows = [{"ok": True, "latency_seconds": 5.0, "cost_usd": None,
             "output_tokens": 300, "timing": {"generation_seconds": 4.0},
             **score_all(_pack(), "A scanning sweep.")},
            {"ok": True, "latency_seconds": 5.0, "cost_usd": None,
             "output_tokens": 100, "timing": {"generation_seconds": 4.0},
             **score_all(_pack(), "A scanning sweep.")}]

    s = summarise(rows)

    assert s["generation_seconds_total"] == 8.0
    assert s["output_tokens_per_second"] == 50.0


# ----------------------------------------------------------------- re-scoring

def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rescoring_adds_new_scores_and_keeps_run_metadata():
    """A report written before the fidelity checks existed gains them, and what
    described the run (not the scores) survives."""
    rescore = _load("rescore_reports")
    doc = {"summary": {"determinism": {"supported": True}, "n_sessions": 2},
           "reports": [{"detection_id": "d1", "ok": True, "latency_seconds": 1.0,
                        "cost_usd": 0.0, "report": "A scanning sweep.",
                        "scores": {"groundedness": 0.1}}]}

    out, missing = rescore.rescore(doc, {"d1": _pack()})

    assert missing == []
    assert out["reports"][0]["classification"]["faithful"] is True
    assert out["summary"]["determinism"] == {"supported": True}
    assert out["summary"]["n_sessions"] == 2
    assert out["summary"]["classification_faithful_rate"] == 1.0


def test_rescoring_against_regenerated_packs_is_refused():
    """A report scored against a pack it was not written from would parse and
    mean nothing."""
    rescore = _load("rescore_reports")
    doc = {"summary": {}, "reports": [{"detection_id": "gone", "ok": True,
                                       "report": "x", "scores": {"groundedness": 0.4}}]}

    out, missing = rescore.rescore(doc, {"d1": _pack()})

    assert missing == ["gone"]
    assert out["reports"][0]["scores"] == {"groundedness": 0.4}      # untouched


# ------------------------------------------------------------------- excerpts

def _run(model, scores: dict[str, float], labels=None):
    labels = labels or {}
    return {"provider": "p", "model": model, "prompt_version": "v1", "reports": [
        {"detection_id": d, "ok": True, "report": f"{model} on {d}",
         "true_label": labels.get(d, "scanning"), "uncertainty": None,
         "scores": {"groundedness": g, "fabricated_addresses": [],
                    "report_words": 3, "per_class": {}}}
        for d, g in scores.items()]}


def test_excerpts_are_chosen_by_rule_not_by_eye():
    ex = _load("export_llm_excerpts")
    labels = {"c": "Benign", "d": "Benign"}
    runs = [_run("big", {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}, labels),
            _run("small", {"a": 1.0, "b": 0.4, "c": 0.9, "d": 0.5}, labels)]

    picks = ex.choose(runs, 3)

    assert picks[0][0] == "b"                      # widest gap overall
    assert picks[1][0] == "d"                      # widest gap among false positives
    assert "false positive" in picks[1][1]
    assert len({p[0] for p in picks}) == len(picks)


def test_an_excerpt_heading_does_not_claim_a_difference_that_is_not_there():
    ex = _load("export_llm_excerpts")

    picks = ex.choose([_run("only", {"a": 1.0, "b": 1.0})], 1)

    assert "no difference between models" in picks[0][1]


def test_excerpts_quote_each_models_report_exactly():
    ex = _load("export_llm_excerpts")
    runs = [_run("big", {"a": 1.0}), _run("small", {"a": 0.5})]

    md = ex.render(runs, {}, [("a", "rule")], "v1")

    assert "> big on a" in md and "> small on a" in md
