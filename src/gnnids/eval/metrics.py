"""The metric suite. Shared by baselines (Phase 3) and the GNN (Phase 4), so
the comparison between them is exact rather than approximately similar.

Two decisions are baked in here, both from measured findings:

**Accuracy is never reported.** At 96% benign, predicting "benign" always scores
96%. PR-AUC is the headline instead: ROC-AUC divides false positives by the vast
benign majority, so thousands of false alarms barely move it. See the worked
example in the Obsidian note "Why PR-AUC Not ROC-AUC".

**Thresholds are chosen on validation, never on test and never on train.**
Phase 2 measured a 2.8x base-rate shift between train (2.62%) and test (7.33%),
caused by a 717k-flow stretch of the dataset that contains no attacks at all.
Validation sits at 6.71%, close enough to test for a threshold to transfer;
train would not (D16).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def choose_threshold(
    y_val: np.ndarray, scores_val: np.ndarray, mode: str = "f1", target_recall: float = 0.95
) -> float:
    """Pick a decision threshold on the validation split.

    'f1' maximises F1 on the attack class. 'recall' picks the highest threshold
    that still reaches `target_recall`, which is the operationally meaningful
    framing: a SOC decides how much of the attack traffic it must catch, then
    lives with whatever false-positive rate that costs.
    """
    order = np.argsort(-scores_val)
    s, y = scores_val[order], y_val[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    total_pos = y.sum()
    if total_pos == 0:
        return 0.5

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / total_pos

    if mode == "recall":
        ok = np.flatnonzero(recall >= target_recall)
        return float(s[ok[0]]) if len(ok) else float(s[-1])

    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(s[int(np.argmax(f1))])


def fpr_at_recall(y: np.ndarray, scores: np.ndarray, target_recall: float = 0.95) -> float:
    """False positive rate at a fixed recall — the number an analyst feels.

    Answers "to catch 95% of attacks, how much benign traffic must I wade
    through?" far more directly than any threshold-free summary.
    """
    order = np.argsort(-scores)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    total_pos, total_neg = y.sum(), len(y) - y.sum()
    if total_pos == 0 or total_neg == 0:
        return float("nan")

    hit = np.flatnonzero(tp / total_pos >= target_recall)
    return float(fp[hit[0]] / total_neg) if len(hit) else 1.0


def evaluate(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    families: np.ndarray | None = None,
    family_names: dict[str, int] | None = None,
) -> dict:
    """Full metric set for one model on one split.

    `families` (the multi-class label per row) drives the per-family recall
    breakdown. Aggregates hide total failure on the rare families -- Worms has
    164 examples in the whole dataset -- so they are always broken out.
    """
    pred = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )

    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())

    out = {
        "n": int(len(y)),
        "n_positive": n_pos,
        "prevalence": round(n_pos / len(y), 6) if len(y) else 0.0,
        "threshold": round(float(threshold), 6),
        # Headline. Baseline for a random model is the prevalence itself, so
        # this metric encodes the difficulty of the problem in its own floor.
        "pr_auc": round(float(average_precision_score(y, scores)), 5) if n_pos else None,
        # Reported alongside only for comparability with literature that uses
        # it. Never the headline -- see the module docstring.
        "roc_auc": round(float(roc_auc_score(y, scores)), 5) if n_pos and n_neg else None,
        "precision": round(float(precision), 5),
        "recall": round(float(recall), 5),
        "f1": round(float(f1), 5),
        "fpr_at_95_recall": round(fpr_at_recall(y, scores, 0.95), 6),
        "fpr_at_99_recall": round(fpr_at_recall(y, scores, 0.99), 6),
        "false_positives": fp,
        "true_positives": tp,
        # How many alerts an analyst opens per real attack found.
        "alerts_per_true_positive": round((tp + fp) / tp, 3) if tp else None,
    }

    if families is not None and family_names is not None:
        inv = {v: k for k, v in family_names.items()}
        per_family = {}
        for code, name in sorted(inv.items()):
            if name == "Benign":
                continue
            mask = families == code
            n = int(mask.sum())
            if n == 0:
                continue
            per_family[name] = {
                "n": n,
                "recall": round(float(pred[mask].mean()), 5),
                # Rare families give very noisy recall -- Worms has 164 examples
                # in the entire dataset -- so the interval is reported with it
                # rather than left for the reader to infer.
                "recall_ci95": round(float(1.96 * np.sqrt(
                    max(pred[mask].mean() * (1 - pred[mask].mean()), 1e-12) / n)), 5),
            }
        out["per_family_recall"] = per_family

    return out


def aggregate_seeds(runs: list[dict]) -> dict:
    """Mean +/- std across seeds for every scalar metric.

    Headline numbers are reported as mean +/- std over >=3 seeds; a single run
    on an imbalanced problem can move several points on chance alone.
    """
    if not runs:
        return {}
    keys = [k for k, v in runs[0].items() if isinstance(v, (int, float)) and v is not None]
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if r.get(k) is not None]
        if vals:
            out[k] = {
                "mean": round(float(np.mean(vals)), 5),
                "std": round(float(np.std(vals)), 5),
                "n_seeds": len(vals),
            }
    return out
