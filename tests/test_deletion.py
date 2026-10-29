"""The deletion test of attributions (D56), on a model whose answer is known."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnnids.explain.deletion import deletion_check, logit_after_reset  # noqa: E402


class Linear:
    """logit of each edge = w . its own features; no message passing."""

    def __init__(self, w):
        self.w = torch.as_tensor(w, dtype=torch.float32)

    def __call__(self, x, edge_index, edge_attr):
        return edge_attr @ self.w, None, None


def window(n_edges=4, n_features=6):
    attr = torch.ones(n_edges, n_features)
    return torch.zeros(2, 1), torch.zeros(2, n_edges, dtype=torch.long), attr


def test_reset_sets_only_the_target_flows_features_to_the_baseline():
    model = Linear([1, 1, 1, 1, 1, 1])
    x, ei, attr = window()

    assert logit_after_reset(model, x, ei, attr, 2, [0, 1]) == pytest.approx(4.0)
    assert logit_after_reset(model, x, ei, attr, 1, []) == pytest.approx(6.0)
    assert torch.all(attr == 1), "the input window must not be modified"


def test_removing_the_real_reasons_costs_more_than_removing_random_features():
    model = Linear([5, 4, 0.1, 0.1, -2, 0.1])
    x, ei, attr = window()

    r = deletion_check(model, x, ei, attr, 0, raising=[0, 1], lowering=[4],
                       exclude=[0, 1, 4], n_random=10, rng=np.random.default_rng(0))

    assert r["drop_without_raising"] == pytest.approx(9.0)
    assert r["drop_without_random"] == pytest.approx(0.2)    # two of the 0.1 features
    assert r["rise_without_lowering"] == pytest.approx(2.0)


def test_random_draws_never_include_a_stated_reason():
    model = Linear([100, 100, 1, 1, 1, 1])
    x, ei, attr = window()

    r = deletion_check(model, x, ei, attr, 0, raising=[0], lowering=[],
                       exclude=[0, 1], n_random=50, rng=np.random.default_rng(1))

    assert max(r["random_logits"]) == pytest.approx(203.0)
    assert min(r["random_logits"]) == pytest.approx(203.0)
