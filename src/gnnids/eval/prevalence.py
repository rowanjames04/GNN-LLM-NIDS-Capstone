"""Prevalence-adjusted evaluation (D23).

Two requirements that look irreconcilable:

- **D20** forbids downsampling attack flows, because it destroys the topology
  NF-ToN-IoT-v2 was adopted for. A scanner reaching 5,161 peers keeps ~155 of
  them at 3% sampling, and that fan-out variance is exactly the signal a GNN
  exists to find.
- **Realism** demands a low attack rate. At this dataset's native 63.99%,
  PR-AUC has a floor of 0.64 and every model scores ~0.999 -- numbers that say
  nothing about a network where traffic is overwhelmingly benign.

They resolve because they act at different stages. Train and run inference on
the **full graph at native prevalence**, so every edge is scored with its
complete neighbourhood. *Then* subsample the resulting **scores** and compute
metrics. Subsampling after inference cannot damage topology, because the
predictions have already used it.

Every metric returned carries its own baseline, because three numbers have now
been misread in this project purely because the PR-AUC floor moved: the audit
tool at 64% prevalence, the first NF-ToN-IoT-v2 run at 97%, and the Phase 3
results at 63%.
"""

from __future__ import annotations

import numpy as np


def subsample_to_prevalence(
    y: np.ndarray, target: float, seed: int = 42, keep_all_positives: bool = False
) -> np.ndarray:
    """Indices of a subsample with the requested attack prevalence.

    By default the *majority* side is kept whole and the minority side thinned,
    or vice versa, whichever direction the adjustment requires. Dropping from
    the side that has surplus preserves the most information.
    """
    if not 0 < target < 1:
        raise ValueError(f"target prevalence must be in (0, 1), got {target}")

    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(pos) or not len(neg):
        return np.arange(len(y))

    current = len(pos) / len(y)
    if keep_all_positives or current < target:
        # Too few positives for the target: thin the negatives.
        n_neg = min(len(neg), int(len(pos) * (1 - target) / target))
        chosen = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
    else:
        # Too many positives: thin them, keeping every benign flow. This is the
        # usual direction here, since the dataset is 64% attack.
        n_pos = min(len(pos), max(1, int(len(neg) * target / (1 - target))))
        chosen = np.concatenate([rng.choice(pos, n_pos, replace=False), neg])
    return np.sort(chosen)


def evaluate_at_prevalence(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    target_prevalence: float | None = 0.04,
    families: np.ndarray | None = None,
    family_names: dict[str, int] | None = None,
    seed: int = 42,
) -> dict:
    """Metrics at a chosen prevalence, computed from already-made predictions."""
    from .metrics import evaluate

    if target_prevalence is None:
        out = evaluate(y, scores, threshold, families, family_names)
        out["prevalence_adjustment"] = "none (native)"
        return out

    idx = subsample_to_prevalence(y, target_prevalence, seed)
    out = evaluate(
        y[idx], scores[idx], threshold,
        families[idx] if families is not None else None, family_names,
    )
    out["prevalence_adjustment"] = {
        "target": target_prevalence,
        "achieved": round(float(y[idx].mean()), 5),
        "native": round(float(y.mean()), 5),
        "n_before": int(len(y)),
        "n_after": int(len(idx)),
        "note": "scores subsampled after inference; the graph was never altered",
    }
    return out


def report_both(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    target_prevalence: float = 0.04,
    families: np.ndarray | None = None,
    family_names: dict[str, int] | None = None,
    seed: int = 42,
) -> dict:
    """Native and standardised prevalence side by side.

    Native is for comparability with published work on the same dataset;
    standardised is the operationally meaningful figure and the only one
    comparable across datasets whose base rates differ 16-fold.
    """
    return {
        "native": evaluate_at_prevalence(
            y, scores, threshold, None, families, family_names, seed),
        f"at_{target_prevalence:.0%}": evaluate_at_prevalence(
            y, scores, threshold, target_prevalence, families, family_names, seed),
    }
