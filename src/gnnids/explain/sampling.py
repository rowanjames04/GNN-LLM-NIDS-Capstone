"""Choosing WHICH detections become evidence packs (Phase 6a / 7b).

The comparative study is a fully crossed design: every model sees every pack.
That makes the choice of packs a design decision rather than a convenience, and
two properties are required of it.

**Stratification across attack families.** A sample drawn in file order is
dominated by whichever families happen to be frequent -- on NF-ToN-IoT-v2 the
first detections above threshold are almost entirely `scanning`. A model that
writes good scanning reports and poor ransomware reports would then look
uniformly good.

**A deliberate false-positive arm.** The single sharpest question in the study is
what a model does with a detection that is *wrong*: given evidence for a benign
flow the detector flagged, does it manufacture a threat narrative because the
prompt implies one exists, or does it convey that the evidence is thin? A false
positive is a flow with `score >= threshold` and a benign true label. Those are
rare by construction -- a good detector produces few -- so they **must be
sampled for deliberately**. They will not appear in useful numbers by accident,
and a study without them cannot ask its most interesting question.

Both properties are about the *true* label, which the pack never shows the
model. Stratification is a property of the experiment, not an input to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Candidate:
    """One flagged flow, before the expensive attribution work is done."""

    window_index: int
    edge_index: int
    row_index: int
    score: float
    true_family: str

    @property
    def is_false_positive(self) -> bool:
        return self.true_family == "Benign"


@dataclass
class SamplePlan:
    """What was asked for, what was available, and what was taken."""

    selected: list = field(default_factory=list)
    per_family_requested: int = 0
    false_positives_requested: int = 0
    available: dict = field(default_factory=dict)
    taken: dict = field(default_factory=dict)
    shortfalls: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_selected": len(self.selected),
            "per_family_requested": self.per_family_requested,
            "false_positives_requested": self.false_positives_requested,
            "false_positives_taken": self.taken.get("Benign", 0),
            "available_by_true_family": self.available,
            "taken_by_true_family": self.taken,
            # Reported, never silently absorbed: a family the detector rarely
            # flags is under-represented in the study, and a reader comparing
            # per-family report quality has to know the denominator.
            "shortfalls": self.shortfalls,
        }


def stratified_sample(
    candidates: list[Candidate],
    n_total: int,
    false_positive_share: float = 0.15,
    seed: int = 42,
    benign_name: str = "Benign",
) -> SamplePlan:
    """Pick detections spread across true families, with a false-positive quota.

    `false_positive_share` reserves that fraction of the budget for flows the
    detector got wrong. The remainder is divided as evenly as possible across
    the attack families actually present, and any budget a thin family cannot
    fill is redistributed to families that can -- so the sample reaches
    `n_total` where the data allows rather than silently returning short.
    """
    if not 0.0 <= false_positive_share < 1.0:
        raise ValueError(f"false_positive_share must be in [0, 1), got {false_positive_share}")
    rng = np.random.default_rng(seed)

    by_family: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_family.setdefault(c.true_family, []).append(c)

    plan = SamplePlan(
        available={k: len(v) for k, v in sorted(by_family.items())},
        false_positives_requested=int(round(n_total * false_positive_share)),
    )

    attack_families = sorted(k for k in by_family if k != benign_name)
    n_fp = plan.false_positives_requested
    n_attack = n_total - n_fp
    plan.per_family_requested = n_attack // max(len(attack_families), 1)

    quotas = {f: plan.per_family_requested for f in attack_families}
    if benign_name in by_family:
        quotas[benign_name] = n_fp
    elif n_fp:
        # No false positives exist at all. That is a finding about the detector,
        # not a reason to fail: the study simply cannot ask the question here.
        plan.shortfalls[benign_name] = n_fp

    # First pass: take what each stratum can supply, record what it cannot.
    for family, quota in sorted(quotas.items()):
        pool = by_family.get(family, [])
        take = min(quota, len(pool))
        if take < quota:
            plan.shortfalls[family] = quota - take
        if take:
            idx = rng.choice(len(pool), take, replace=False)
            plan.selected.extend(pool[i] for i in idx)
            plan.taken[family] = take

    # Second pass: redistribute unfilled budget to families with headroom, so a
    # thin family does not shrink the whole study.
    remaining = n_total - len(plan.selected)
    if remaining > 0:
        chosen = {(c.window_index, c.edge_index) for c in plan.selected}
        spare = [c for c in candidates if (c.window_index, c.edge_index) not in chosen]
        if spare:
            idx = rng.choice(len(spare), min(remaining, len(spare)), replace=False)
            for i in idx:
                plan.selected.append(spare[i])
                plan.taken[spare[i].true_family] = plan.taken.get(spare[i].true_family, 0) + 1

    # Window order keeps downstream pack construction sequential, so each window
    # is scored and attributed once rather than revisited per detection.
    plan.selected.sort(key=lambda c: (c.window_index, c.edge_index))
    return plan
