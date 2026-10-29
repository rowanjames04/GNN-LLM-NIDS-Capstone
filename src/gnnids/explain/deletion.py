"""Are the attributions an evidence pack reports actually the reasons? (D56)

Integrated gradients is checked in the tests for completeness (the attributions
add up to the change in score), but completeness says nothing about whether the
features it ranks highest are the ones the model depends on. The standard check
is a **deletion test**: remove the features an explanation names and see how far
the score falls, against removing the same number of features at random. If the
named features are the real reasons, removing them should cost far more than
removing random ones.

"Remove" means setting the feature to the integrated-gradients baseline, zero in
standardised units, i.e. the value of an average flow. That keeps the test on
the same reference point the attributions were measured from. Only the target
flow's own features change; its neighbours are left as they were, because the
pack's attributions are over the flow's own features (occlusion covers the
neighbours).

Scores are compared as logits, the quantity the attributions are measured in.
A confident detection sits on the flat part of the sigmoid, where a large change
in the logit barely moves the probability, so probabilities would hide the
effect. Whether the flow would still be flagged is reported separately.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def logit_after_reset(model, x, edge_index, edge_attr, target: int,
                      features: np.ndarray | list[int]) -> float:
    """The target flow's binary logit with `features` of that flow set to zero."""
    attr = edge_attr.clone()
    if len(features):
        attr[target, torch.as_tensor(np.asarray(features), device=attr.device)] = 0
    logit, _, _ = model(x, edge_index, attr)
    return float(logit[target])


def deletion_check(model, x, edge_index, edge_attr, target: int,
                   raising: list[int], lowering: list[int], exclude: list[int],
                   n_random: int, rng: np.random.Generator) -> dict:
    """Logit change when the stated reasons are removed, against random features.

    `raising` and `lowering` are the feature indices the pack says raised and
    lowered the score; `exclude` is every feature the pack names, so the random
    draws never include a stated reason. The random comparison removes as many
    features as `raising` holds, `n_random` times.
    """
    base = logit_after_reset(model, x, edge_index, edge_attr, target, [])
    pool = np.setdiff1d(np.arange(edge_attr.shape[1]), np.asarray(exclude, dtype=int))
    k = len(raising)
    random_logits = [
        logit_after_reset(model, x, edge_index, edge_attr, target,
                          rng.choice(pool, size=k, replace=False))
        for _ in range(n_random)
    ] if k else []
    without_raising = logit_after_reset(model, x, edge_index, edge_attr, target, raising)
    without_lowering = logit_after_reset(model, x, edge_index, edge_attr, target, lowering)
    return {
        "base_logit": base,
        "n_raising": k,
        "n_lowering": len(lowering),
        "drop_without_raising": base - without_raising,
        "drop_without_random": (base - float(np.mean(random_logits))) if k else None,
        "rise_without_lowering": without_lowering - base,
        "logit_without_raising": without_raising,
        "random_logits": random_logits,
    }
