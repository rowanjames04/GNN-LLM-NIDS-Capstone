"""Tier 1 explainability: why did the model flag *this* flow? (Phase 6a, D6)

Three complementary answers, because each is blind to what the others see:

1. **Integrated gradients** over the flow's own features -- which measured
   quantities pushed the score up, and by how much.
2. **Fusion weights**, read straight off the model -- how much of the decision
   came from the flow itself versus from its neighbourhood. This is the
   per-detection form of the V2 claim and it costs nothing; the model already
   computes it.
3. **Neighbour occlusion** -- remove one neighbouring edge, re-score, and see
   how far the score moves. Answers "which *other* flows made this look bad",
   which neither of the first two can.

Integrated gradients is implemented here rather than taken from `captum`. Two
reasons, and the second is the real one: captum would need a custom forward
wrapper anyway to attribute over `edge_attr` while holding `edge_index` fixed,
so the dependency buys ~15 lines; and this is a method the report has to explain
and defend, so it is better as readable code than as an opaque call. See
[[Integrated Gradients]].
"""

from __future__ import annotations

import numpy as np
import torch


def integrated_gradients(
    model,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    targets: torch.Tensor,
    steps: int = 32,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attribution of each edge's binary score to each of its input features.

    Integrated gradients answers "how much did feature $i$ contribute?" by
    integrating the model's gradient along a straight path from a **baseline**
    input to the actual one:

    $$\\mathrm{IG}_i(x) = (x_i - x'_i)\\int_0^1
      \\frac{\\partial F(x' + \\alpha(x - x'))}{\\partial x_i}\\,d\\alpha$$

    approximated here by a Riemann sum over `steps` points.

    **Why not a plain gradient.** A raw gradient says how the score responds to
    an infinitesimal nudge *at the current input*, which is close to meaningless
    for a saturated network -- a feature can be the entire reason for a
    detection and still have near-zero local gradient, because the model is
    already certain. Integrating along the path recovers the contribution that
    the saturated gradient hides.

    **The baseline is zero, which is the dataset mean.** Features are
    standardised upstream, so the zero vector is "a perfectly average flow", and
    attributions read as "relative to a typical flow on this network" -- which
    is the sentence the incident report wants. A baseline of all-zeros in *raw*
    units would instead mean a flow with no bytes and no duration, which does
    not exist and is not a useful reference point.

    Returns `[n_edges, n_features]`, same shape as `edge_attr`.
    """
    if baseline is None:
        baseline = torch.zeros_like(edge_attr)
    if baseline.shape != edge_attr.shape:
        raise ValueError(
            f"baseline {tuple(baseline.shape)} must match edge_attr "
            f"{tuple(edge_attr.shape)}")

    model.eval()
    delta = edge_attr - baseline
    total = torch.zeros_like(edge_attr)

    # Midpoint rule: alpha at step centres rather than edges. Endpoint sampling
    # includes alpha=0, where a saturated model contributes nothing but still
    # costs a full forward and backward pass.
    for step in range(steps):
        alpha = (step + 0.5) / steps
        point = (baseline + alpha * delta).detach().requires_grad_(True)
        logit, _, _ = model(x, edge_index, point)
        # One backward for the whole window: each edge's binary logit depends
        # on its own features and on its neighbours', and summing the selected
        # logits gives each edge its own gradient in a single pass.
        selected = logit[targets].sum()
        grad, = torch.autograd.grad(selected, point)
        total += grad

    return (delta * total / steps).detach()


def top_features(
    attributions: np.ndarray,
    feature_names: list[str],
    values: np.ndarray,
    k: int = 8,
) -> list[dict]:
    """The `k` features that moved this detection most, largest magnitude first.

    Both directions are kept. A feature that argued *against* the detection is
    informative -- it is what an analyst would raise as the counter-argument,
    and a report that only ever lists incriminating evidence reads as a
    prosecution rather than an assessment.
    """
    if len(feature_names) != attributions.shape[-1]:
        raise ValueError(
            f"{len(feature_names)} names for {attributions.shape[-1]} features")

    order = np.argsort(-np.abs(attributions))[:k]
    return [
        {
            "name": feature_names[i],
            "value": _clean(values[i]),
            "contribution": round(float(attributions[i]), 5),
            "direction": ("increases_suspicion" if attributions[i] > 0
                          else "decreases_suspicion"),
        }
        for i in order
    ]


@torch.no_grad()
def neighbour_influence(
    model,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    target: int,
    candidates: np.ndarray,
    max_neighbours: int = 5,
) -> list[dict]:
    """How far the target's score moves when each neighbouring flow is removed.

    Occlusion rather than `GNNExplainer`, deliberately. GNNExplainer learns a
    soft mask over edges by optimisation, which means a hyperparameter, a
    convergence question and a stochastic result for every single detection --
    inside a pack that has to be produced thousands of times and be identical on
    re-run. Occlusion is exact, deterministic and needs no tuning: the number it
    reports is literally "the score changes by this much without that flow",
    which is also the sentence an analyst wants.

    Its cost is one forward pass per candidate, so candidates are capped.
    """
    model.eval()
    base_logit, _, _ = model(x, edge_index, edge_attr)
    base = float(torch.sigmoid(base_logit[target]))

    out = []
    for edge in candidates[:max_neighbours]:
        if int(edge) == int(target):
            continue
        keep = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
        keep[int(edge)] = False
        logit, _, _ = model(x, edge_index[:, keep], edge_attr[keep])
        # Removing an earlier edge shifts every later index down by one.
        moved = int(target) - (1 if int(edge) < int(target) else 0)
        out.append({
            "edge": int(edge),
            "score_without": round(float(torch.sigmoid(logit[moved])), 5),
            "importance": round(base - float(torch.sigmoid(logit[moved])), 5),
        })
    out.sort(key=lambda d: -abs(d["importance"]))
    return out


def _clean(v) -> float | int:
    v = float(v)
    if not np.isfinite(v):
        return 0.0
    return round(v, 5)
