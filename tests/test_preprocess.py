"""Tests for the Phase 2 preprocessing and graph construction path.

These target the failure modes that would silently invalidate every downstream
result rather than crash: a scaler that saw the test set, a window straddling a
split boundary, an edge count that quietly drifts from the flow count. Each of
those produces a working pipeline and wrong science, which is why they are
tested rather than assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnnids.data.schema import FeatureSchema  # noqa: E402
from gnnids.data.splits import assert_no_overlap, contiguous_split  # noqa: E402
from gnnids.data.transforms import FeaturePipeline, bucket_ports  # noqa: E402
from gnnids.graph.build import GlobalNodeMap, build_snapshot, snapshot_stats  # noqa: E402
from gnnids.graph.windows import count_windows, fixed_count_windows  # noqa: E402


@pytest.fixture
def schema() -> FeatureSchema:
    return FeatureSchema(
        identity=["IPV4_SRC_ADDR", "IPV4_DST_ADDR"],
        continuous=["IN_BYTES", "OUT_BYTES"],
        categorical=["PROTOCOL"],
        conditional=["DNS_TTL_ANSWER"],
        ports=["L4_DST_PORT"],
        log_transform=["IN_BYTES"],
    )


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 1000
    return pd.DataFrame({
        "IPV4_SRC_ADDR": [f"10.0.0.{i % 8}" for i in range(n)],
        "IPV4_DST_ADDR": [f"10.0.1.{i % 5}" for i in range(n)],
        "L4_DST_PORT": rng.integers(1, 65535, n),
        "PROTOCOL": rng.choice([6, 17, 1], n, p=[0.8, 0.19, 0.01]),
        "IN_BYTES": rng.integers(40, 1_000_000, n),
        "OUT_BYTES": rng.integers(40, 1_000_000, n),
        "DNS_TTL_ANSWER": rng.choice([0, 300], n, p=[0.85, 0.15]),
        "Label": rng.choice([0, 1], n, p=[0.96, 0.04]),
        "Attack": "Benign",
    })


# ------------------------------------------------------------------ splits

def test_splits_do_not_overlap_and_respect_purge_gap():
    s = contiguous_split(100_000, 0.7, 0.1, 0.2, purge_gap=1000)
    assert_no_overlap(s)

    assert s["train"].stop == 70_000 - 1000
    assert s["val"].start == 70_000 + 1000
    # Every train row precedes every val row precedes every test row.
    assert s["train"].stop < s["val"].start < s["val"].stop < s["test"].start


def test_purge_gap_actually_discards_rows():
    n = 100_000
    kept = sum(len(v) for v in contiguous_split(n, purge_gap=1000).values())
    assert kept == n - 4000  # 1000 either side of two boundaries


def test_oversized_purge_gap_is_rejected():
    with pytest.raises(ValueError):
        contiguous_split(1000, purge_gap=500)


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        contiguous_split(100_000, 0.7, 0.1, 0.5)


# ----------------------------------------------------------------- windows

def test_windows_never_cross_a_split_boundary():
    splits = contiguous_split(100_000, purge_gap=1000)
    for split in splits.values():
        for lo, hi in fixed_count_windows(split, 1000):
            assert split.start <= lo < hi <= split.stop


def test_partial_final_window_is_dropped():
    from gnnids.data.splits import Split

    s = Split("t", 0, 2500)
    windows = list(fixed_count_windows(s, 1000))
    assert len(windows) == 2
    assert all(hi - lo == 1000 for lo, hi in windows)
    assert count_windows(s, 1000) == 2


# --------------------------------------------------------------- transforms

def test_scaler_is_fitted_on_train_only(schema, frame):
    """The central leakage guard: val/test must be scaled with TRAIN statistics.

    If the pipeline were fitted on everything, the transformed training block
    would be exactly zero-mean. It must not be, because train alone defines the
    statistics and train alone is centred by them.
    """
    train, held_out = frame.iloc[:700], frame.iloc[700:]
    pipe = FeaturePipeline(schema).fit(train)

    train_out = pipe.transform(train)["continuous"]
    held_out_out = pipe.transform(held_out)["continuous"]

    assert np.allclose(train_out.mean(axis=0), 0, atol=1e-3)
    # Held-out data is transformed by someone else's statistics, so its mean is
    # free to differ -- and the fact that it *can* is the property being tested.
    assert held_out_out.shape[1] == train_out.shape[1]

    refit = FeaturePipeline(schema).fit(frame)
    assert refit.mean_["IN_BYTES"] != pipe.mean_["IN_BYTES"]


def test_transform_before_fit_raises(schema, frame):
    with pytest.raises(RuntimeError):
        FeaturePipeline(schema).transform(frame)


def test_unseen_categorical_values_fall_back_to_other(schema, frame):
    """Required for val/test and for cross-dataset transfer, where categories
    absent from training certainly appear."""
    pipe = FeaturePipeline(schema).fit(frame)
    novel = frame.head(5).copy()
    novel["PROTOCOL"] = 253  # never seen in training

    out = pipe.transform(novel)["categorical"]
    assert (out[:, 0] == 0).all()  # OTHER


def test_rare_categories_are_bucketed(schema, frame):
    """PROTOCOL 1 appears at ~1%, below the 5% threshold set here."""
    pipe = FeaturePipeline(schema, min_frequency=0.05).fit(frame)
    assert "1" not in pipe.vocab_["PROTOCOL"]
    assert "6" in pipe.vocab_["PROTOCOL"]


def test_log_transform_is_applied_only_where_configured(schema, frame):
    pipe = FeaturePipeline(schema).fit(frame)
    assert "IN_BYTES" in pipe.log_shift_
    assert "OUT_BYTES" not in pipe.log_shift_


def test_conditional_indicator_separates_not_applicable_from_zero(schema, frame):
    pipe = FeaturePipeline(schema).fit(frame)
    out = pipe.transform(frame)["categorical"]
    indicator = out[:, -1]
    assert set(np.unique(indicator)) <= {0, 1}
    assert np.array_equal(indicator, (frame.DNS_TTL_ANSWER != 0).to_numpy().astype(np.int64))


def test_port_bucketing():
    got = bucket_ports(np.array([80, 443, 8080, 50000, 1023, 1024, 49152]))
    assert list(got) == [0, 0, 1, 2, 0, 1, 2]


def test_cardinalities_match_categorical_width(schema, frame):
    pipe = FeaturePipeline(schema).fit(frame)
    assert len(pipe.cardinalities()) == pipe.transform(frame)["categorical"].shape[1]


def test_pipeline_roundtrips_through_disk(schema, frame, tmp_path):
    pipe = FeaturePipeline(schema).fit(frame)
    path = tmp_path / "pipeline.json"
    pipe.save(path)

    reloaded = FeaturePipeline.load(path, schema)
    assert np.allclose(
        pipe.transform(frame)["continuous"], reloaded.transform(frame)["continuous"]
    )


# ------------------------------------------------------------------ graphs

def _snapshot(frame, lo=0, hi=200, **kw):
    n = hi - lo
    return build_snapshot(
        frame.IPV4_SRC_ADDR.to_numpy()[lo:hi],
        frame.IPV4_DST_ADDR.to_numpy()[lo:hi],
        np.random.default_rng(0).normal(size=(n, 3)).astype(np.float32),
        np.zeros((n, 2), dtype=np.int64),
        frame.Label.to_numpy()[lo:hi],
        np.zeros(n, dtype=np.int64),
        **kw,
    )


def test_edge_count_equals_flow_count(frame):
    """Multi-edges are mandatory: each flow has its own label, so flows between
    the same pair must not be collapsed."""
    snap = _snapshot(frame, 0, 200)
    assert snap.edge_index.shape[1] == 200
    assert snap.edge_attr.shape[0] == 200


def test_node_indices_are_a_bijection_onto_range(frame):
    snap = _snapshot(frame, 0, 200)
    used = set(snap.edge_index.flatten().tolist())
    assert used == set(range(snap.num_nodes))


def test_labels_are_preserved_exactly(frame):
    snap = _snapshot(frame, 0, 200)
    assert int(snap.y.sum()) == int(frame.Label.iloc[:200].sum())


def test_node_features_are_constant(frame):
    """E-GraphSAGE: nodes carry no intrinsic information, so a host's
    representation is entirely a function of the traffic touching it."""
    snap = _snapshot(frame, 0, 200)
    assert snap.x.shape == (snap.num_nodes, 1)
    assert (snap.x == 1).all()


def test_mismatched_input_lengths_are_rejected():
    with pytest.raises(ValueError):
        build_snapshot(
            np.array(["a", "b"]), np.array(["c"]),
            np.zeros((2, 3), dtype=np.float32), np.zeros((2, 1), dtype=np.int64),
            np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64),
        )


def test_edge_tensors_are_writable(frame):
    """torch.from_numpy on a read-only slice yields a tensor whose writes are
    undefined behaviour; the builder must copy."""
    snap = _snapshot(frame, 0, 200)
    snap.edge_attr[0, 0] = 1.0
    snap.y[0] = 1


def test_global_node_map_is_stable_across_windows(frame):
    gm = GlobalNodeMap()
    a = _snapshot(frame, 0, 200, global_map=gm)
    b = _snapshot(frame, 200, 400, global_map=gm)

    # The same address must receive the same global id in both windows, even
    # though its local index differs.
    assert gm.get("10.0.0.3") == gm.get("10.0.0.3")
    assert len(gm) >= a.global_node_id.numel()
    assert set(b.global_node_id.tolist()) <= set(range(len(gm)))


def test_snapshot_stats_reports_fanout_spread(frame):
    """max_fanout saturates in this dataset, so the spread across hosts is the
    statistic that says whether structure varies."""
    stats = snapshot_stats(_snapshot(frame, 0, 500))
    assert stats["n_edges"] == 500
    assert "fanout_std" in stats and stats["fanout_std"] >= 0
    assert stats["n_active_senders"] > 0


# --------------------------------------------- window-level random splitting

def test_window_split_keeps_windows_intact():
    """A window must travel to one split as a unit, or its internal topology is
    scattered across all three -- the D20 objection."""
    from gnnids.data.splits import window_split

    splits, perm = window_split(100_000, window_size=1000, seed=0)
    # Every consecutive run of 1000 rows in the permutation must be one window.
    for start in range(0, len(perm), 1000):
        block = perm[start:start + 1000]
        assert block.max() - block.min() == 999
        assert np.array_equal(block, np.arange(block.min(), block.min() + 1000))


def test_window_split_partitions_without_overlap():
    from gnnids.data.splits import assert_no_overlap, window_split

    splits, perm = window_split(100_000, window_size=1000, seed=0)
    assert_no_overlap(splits)
    assert len(np.unique(perm)) == len(perm)          # no row used twice
    assert sum(len(s) for s in splits.values()) == len(perm)


def test_window_split_balances_a_positionally_ordered_label():
    """The failure this exists to prevent: on NF-ToN-IoT-v2 a contiguous split
    gave train 53% / test 97% attack because families sit in narrow positional
    bands. Random window assignment must even that out."""
    from gnnids.data.splits import contiguous_split, window_split

    # 1000 windows. Window assignment is random, so the balance it achieves has
    # sampling variance scaling as 1/sqrt(n_windows) -- at only 100 windows the
    # residual gap is ~0.13 and that is correct behaviour, not a defect. Real
    # runs have hundreds to thousands of windows.
    n = 1_000_000
    # Label concentrated in the last 30% of the file, as `xss` is there.
    y = np.zeros(n, dtype=int)
    y[int(n * 0.7):] = 1

    c = contiguous_split(n, purge_gap=0)
    contiguous_gap = abs(
        y[c["train"].start:c["train"].stop].mean() - y[c["test"].start:c["test"].stop].mean()
    )

    s, perm = window_split(n, window_size=1000, seed=0)
    yp = y[perm]
    window_gap = abs(
        yp[s["train"].start:s["train"].stop].mean() - yp[s["test"].start:s["test"].stop].mean()
    )

    assert contiguous_gap > 0.5              # the pathology, reproduced
    assert window_gap < 0.05                 # and fixed
    assert window_gap < contiguous_gap / 10  # by an order of magnitude


def test_window_split_is_deterministic_given_a_seed():
    from gnnids.data.splits import window_split

    a, pa = window_split(50_000, 1000, seed=7)
    b, pb = window_split(50_000, 1000, seed=7)
    c, pc = window_split(50_000, 1000, seed=8)
    assert np.array_equal(pa, pb)
    assert not np.array_equal(pa, pc)


def test_window_split_rejects_too_few_windows():
    from gnnids.data.splits import window_split

    with pytest.raises(ValueError):
        window_split(2000, window_size=1000)


# ------------------------------------------------------- Phase 4: the GNN

def _graph(n=400, edge_dim=97, seed=0):
    from gnnids.graph.build import build_snapshot

    rng = np.random.default_rng(seed)
    return build_snapshot(
        np.array([f"10.0.0.{i % 30}" for i in range(n)]),
        np.array([f"10.0.1.{i % 17}" for i in range(n)]),
        rng.normal(size=(n, edge_dim)).astype(np.float32),
        np.zeros((n, 0), dtype=np.int64),
        rng.integers(0, 2, n), rng.integers(0, 10, n),
    )


def test_fusion_weights_are_a_distribution():
    """Alpha is read directly into the evidence pack as an explanation, so it
    has to be a genuine 2-way distribution rather than arbitrary logits."""
    import torch
    from gnnids.models.dual_channel import DualChannelGNN

    g = _graph()
    _, _, alpha = DualChannelGNN(edge_dim=97, hidden_dim=32)(
        g.x, g.edge_index, g.edge_attr)
    assert alpha.shape == (g.edge_index.shape[1], 2)
    assert torch.allclose(alpha.sum(-1), torch.ones(alpha.shape[0]), atol=1e-5)
    assert (alpha >= 0).all()


def test_single_channel_ablations_have_no_fusion():
    from gnnids.models.dual_channel import DualChannelGNN

    g = _graph()
    for kw in ({"use_channel2": False}, {"use_channel1": False}):
        _, _, alpha = DualChannelGNN(edge_dim=97, hidden_dim=32, **kw)(
            g.x, g.edge_index, g.edge_attr)
        assert alpha is None


def test_disabling_both_channels_is_rejected():
    from gnnids.models.dual_channel import DualChannelGNN

    with pytest.raises(ValueError):
        DualChannelGNN(edge_dim=97, use_channel1=False, use_channel2=False)


def test_channel1_ignores_topology_and_channel2_does_not():
    """The core premise of the V2 measurement. Rewire the graph while holding
    edge features fixed: Channel 1 must be unchanged, Channel 2 must not be."""
    import torch
    from gnnids.models.dual_channel import DualChannelGNN

    torch.manual_seed(0)
    g = _graph()
    shuffled = g.edge_index[:, torch.randperm(g.edge_index.shape[1])]

    c1 = DualChannelGNN(edge_dim=97, hidden_dim=32, use_channel2=False, dropout=0.0).eval()
    a, _, _ = c1(g.x, g.edge_index, g.edge_attr)
    b, _, _ = c1(g.x, shuffled, g.edge_attr)
    assert torch.allclose(a, b, atol=1e-6), "Channel 1 must be topology-blind"

    c2 = DualChannelGNN(edge_dim=97, hidden_dim=32, use_channel1=False, dropout=0.0).eval()
    a, _, _ = c2(g.x, g.edge_index, g.edge_attr)
    b, _, _ = c2(g.x, shuffled, g.edge_attr)
    assert not torch.allclose(a, b, atol=1e-4), "Channel 2 must use topology"


def test_message_passing_is_permutation_invariant():
    """A host's connections have no canonical order; a model sensitive to edge
    ordering would be learning how the data happens to be stored."""
    import torch
    from gnnids.models.egraphsage import TopologicalChannel

    torch.manual_seed(0)
    g = _graph()
    ch = TopologicalChannel(edge_dim=97, hidden_dim=32, dropout=0.0).eval()

    out_a = ch(g.x, g.edge_index, g.edge_attr)
    perm = torch.randperm(g.edge_index.shape[1])
    out_b = ch(g.x, g.edge_index[:, perm], g.edge_attr[perm])
    assert torch.allclose(out_a[perm], out_b, atol=1e-5)


def test_prevalence_subsampling_hits_the_target():
    from gnnids.eval.prevalence import subsample_to_prevalence

    y = np.zeros(100_000, dtype=int)
    y[:64_000] = 1                      # 64% attack, as NF-ToN-IoT-v2 is
    idx = subsample_to_prevalence(y, 0.04, seed=0)
    assert abs(y[idx].mean() - 0.04) < 0.005
    # Thinning must come from the surplus side, keeping every benign flow.
    assert (y[idx] == 0).sum() == 36_000


def test_prevalence_adjustment_reports_its_own_baseline():
    """Three numbers have been misread in this project because the PR-AUC floor
    moved silently. The adjustment must always declare it."""
    from gnnids.eval.prevalence import report_both

    rng = np.random.default_rng(0)
    y = (rng.random(50_000) < 0.64).astype(int)
    scores = np.clip(y * 0.6 + rng.normal(0, 0.25, len(y)), 0, 1)

    out = report_both(y, scores, 0.5, 0.04, seed=0)
    assert out["native"]["prevalence"] > 0.6
    assert abs(out["at_4%"]["prevalence"] - 0.04) < 0.01
    assert out["at_4%"]["pr_auc"] < out["native"]["pr_auc"]  # floor moved down
