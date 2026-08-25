"""Tests for Phase 6a: attribution, unit restoration and the evidence pack.

The pack is the frozen contract the whole LLM layer is written against, and its
failure mode is the one this project keeps meeting: a pack carrying a z-score or
a model-internal term does not crash Phase 7. It produces a report that reads
fine and a groundedness score that is quietly meaningless.

So the schema properties are asserted, and integrated gradients is checked
against its own completeness axiom rather than against a number someone typed in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnnids.explain.attribution import integrated_gradients, top_features  # noqa: E402
from gnnids.explain.evidence import (  # noqa: E402
    SCHEMA_VERSION, build_pack, config_hash, host_context, validate_pack,
)
from gnnids.explain.units import UnitRestorer, feature_names  # noqa: E402
from gnnids.models.dual_channel import DualChannelGNN  # noqa: E402


# --------------------------------------------------------------- attribution

def _tiny_model_and_graph(edge_dim=6, n_edges=5, n_nodes=4):
    torch.manual_seed(0)
    model = DualChannelGNN(edge_dim=edge_dim, hidden_dim=8, n_mlp_blocks=1,
                           n_gnn_layers=1, n_classes=3, dropout=0.0).eval()
    x = torch.zeros(n_nodes, 1)
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]])
    edge_attr = torch.randn(n_edges, edge_dim)
    return model, x, edge_index, edge_attr


def test_integrated_gradients_satisfies_completeness():
    """The axiom that makes IG trustworthy: attributions sum to the change in
    model output between the baseline and the input.

    Checked rather than assumed, because a wrong Riemann sum, a wrong baseline
    or a missing (x - x') factor all still return a plausible-looking array of
    numbers."""
    model, x, edge_index, edge_attr = _tiny_model_and_graph()
    targets = torch.arange(edge_attr.shape[0])
    baseline = torch.zeros_like(edge_attr)

    attr = integrated_gradients(model, x, edge_index, edge_attr, targets,
                                steps=256, baseline=baseline)

    with torch.no_grad():
        f_input, _, _ = model(x, edge_index, edge_attr)
        f_base, _, _ = model(x, edge_index, baseline)
    # Summed over all edges: each edge's logit depends on its neighbours' features
    # too, so completeness holds for the total rather than edge by edge.
    assert attr.sum().item() == pytest.approx(
        float((f_input - f_base).sum()), abs=0.05)


def test_integrated_gradients_rejects_a_mismatched_baseline():
    model, x, edge_index, edge_attr = _tiny_model_and_graph()
    with pytest.raises(ValueError, match="must match"):
        integrated_gradients(model, x, edge_index, edge_attr,
                             torch.arange(5), baseline=torch.zeros(2, 2))


def test_top_features_keeps_both_directions_and_orders_by_magnitude():
    """A report listing only incriminating evidence is a prosecution, not an
    assessment."""
    attr = np.array([0.1, -0.9, 0.4, -0.05])
    names = ["a", "b", "c", "d"]

    got = top_features(attr, names, np.array([1.0, 2.0, 3.0, 4.0]), k=3)

    assert [f["name"] for f in got] == ["b", "c", "a"]
    assert got[0]["direction"] == "decreases_suspicion"
    assert got[1]["direction"] == "increases_suspicion"


def test_top_features_refuses_a_name_count_mismatch():
    with pytest.raises(ValueError, match="names for"):
        top_features(np.zeros(4), ["a", "b"], np.zeros(4))


# --------------------------------------------------------------------- units

PIPELINE = {
    "continuous_names": ["IN_BYTES", "RATE"],
    "mean": {"IN_BYTES": 5.0, "RATE": 100.0},
    "std": {"IN_BYTES": 2.0, "RATE": 10.0},
    "log_shift": {"IN_BYTES": 0.0},
    "clip_hi": {"RATE": 130.0},
    "categorical_names": ["PROTOCOL"],
    "cardinalities": [3],
    "vocab": {"PROTOCOL": {"6": 1, "17": 2}},
}


def test_restore_inverts_the_log_and_standardisation():
    r = UnitRestorer(PIPELINE)
    # forward: log1p(x) -> (l - 5)/2 ; so z = 0 means log1p(x) = 5
    got = r.restore("IN_BYTES", 0.0)
    # rel=1e-2 because values >= 100 are deliberately rounded to whole units --
    # the pack reports "147 bytes", not "147.41328 bytes".
    assert got["value"] == pytest.approx(np.expm1(5.0), rel=1e-2)
    assert isinstance(got["value"], int)
    assert got["exact"] is True


def test_a_winsorised_value_is_flagged_as_a_lower_bound():
    """Clipping (D22) is lossy. Presenting a clipped value as exact would state
    a measurement that was never made."""
    r = UnitRestorer(PIPELINE)

    got = r.restore("RATE", 10.0)   # far above the 130 clip bound

    assert got["exact"] is False
    assert "lower bound" in got["note"]


def test_near_integer_counts_come_back_as_integers():
    """float16 storage plus log1p round-trip turns a packet count of 1 into
    1.0003, and "1.0003 packets" invites the model to reason about the fraction."""
    r = UnitRestorer({**PIPELINE, "log_shift": {}, "clip_hi": {},
                      "mean": {"IN_BYTES": 1.0003, "RATE": 100.0},
                      "std": {"IN_BYTES": 1.0, "RATE": 10.0}})

    assert r.restore("IN_BYTES", 0.0)["value"] == 1


def test_feature_names_cover_every_column_and_use_the_capture_values():
    names = feature_names(PIPELINE)

    assert len(names) == 2 + 3
    assert names[2:] == ["PROTOCOL=other", "PROTOCOL=6", "PROTOCOL=17"]


# ---------------------------------------------------------------------- pack

def _pack(**overrides):
    base = dict(
        flow={"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "IN_BYTES": 400},
        detection={"score": 0.9, "threshold": 0.5, "flagged": True,
                   "predicted_class": "scanning", "class_confidence": 0.8,
                   "class_distribution": {"scanning": 0.8, "Benign": 0.2}},
        attribution={"channel_weights": {"attribute": 0.6, "topological": 0.4},
                     "top_features": [], "influential_neighbours": []},
        context={"window_flow_count": 10},
        guidance={"attack_family": "scanning", "mitre_techniques": ["T1046"]},
        provenance={"checkpoint": "seed0", "dataset": "NF-ToN-IoT-v2"},
    )
    base.update(overrides)
    return build_pack(**base)


def test_a_well_formed_pack_validates():
    validate_pack(_pack())


def test_model_internal_vocabulary_is_refused():
    """The no-jargon rule is enforced on the INPUT. A model cannot mention an
    embedding it has never seen, which is stronger than telling it not to."""
    pack = _pack(context={"window_flow_count": 10, "note": "from the embedding"})

    with pytest.raises(ValueError, match="model-internal vocabulary"):
        validate_pack(pack)


def test_a_class_distribution_that_does_not_sum_to_one_is_refused():
    pack = _pack(detection={"score": 0.9, "threshold": 0.5, "flagged": True,
                            "predicted_class": "scanning", "class_confidence": 0.8,
                            "class_distribution": {"scanning": 0.8, "Benign": 0.9}})

    with pytest.raises(ValueError, match="sums to"):
        validate_pack(pack)


def test_an_empty_class_distribution_is_refused():
    """Handing over only the argmax manufactures confidence the model does not
    have."""
    pack = _pack(detection={"score": 0.9, "threshold": 0.5, "flagged": True,
                            "predicted_class": "scanning", "class_confidence": 0.8,
                            "class_distribution": {}})

    with pytest.raises(ValueError, match="class_distribution"):
        validate_pack(pack)


def test_a_pack_from_another_schema_version_is_refused():
    pack = _pack()
    pack["schema_version"] = "0.9"

    with pytest.raises(ValueError, match="schema"):
        validate_pack(pack)


def test_causal_path_is_present_and_null_rather_than_absent():
    """Nullable from day one, so Granger's absence (D14) is a data condition the
    prompt already handles rather than a late schema migration."""
    pack = _pack()

    assert "causal_path" in pack and pack["causal_path"] is None
    assert pack["schema_version"] == SCHEMA_VERSION


def test_host_context_compares_a_host_to_its_own_window_not_a_global_average():
    src = np.array([0, 0, 0, 0, 1, 2])
    dst = np.array([1, 2, 3, 4, 2, 3])
    ports = np.array([80, 81, 82, 83, 80, 80])
    byts = np.array([10, 10, 10, 10, 5, 5])

    got = host_context(src, dst, ports, byts, host=0)

    assert got["flow_count"] == 4
    assert got["distinct_dst_ips"] == 4
    assert got["baseline_deviation"]["times_typical"] == 4.0


def test_config_hash_is_stable_and_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
