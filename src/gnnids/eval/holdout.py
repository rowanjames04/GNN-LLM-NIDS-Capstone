"""Leave-one-attack-out row selection -- the zero-day protocol (D4).

The claim this project exists to support is that the model detects attacks it
has **never seen**. Most papers that claim it measure it on a random split of a
dataset containing every attack family, which is not a zero-day result: every
family was in training. See [[Zero-Day Evaluation Protocol]], Protocol 2.

Doing it honestly means, for each family `a`:

    train, val   every flow labelled `a` is REMOVED
    test         every attack family EXCEPT `a` is removed; benign traffic stays

so the model is trained in a world where `a` did not happen and then asked about
a world containing only benign traffic and `a`.

Two design choices worth defending, because both had a tempting alternative.

**Removal, not relabelling.** A held-out family's flows are dropped, not marked
benign. Relabelling keeps the graph's shape intact, which is superficially
attractive given how much of this project is about protecting topology (D20).
But it does not simulate absence -- it actively teaches the model that attack
traffic is benign, which is worse than never seeing it and would understate
transfer. D20's rule protects the *prevalence adjustment* from damaging topology;
holding out a family is a different operation, and the counterfactual it encodes
is "these flows did not occur", which means they are not there.

**Filtering happens inside each split, never across them.** Rows are selected
within `[split.start, split.stop)` and windows are recounted within that range,
so no row can migrate between train, val and test. This matters more than it
looks: the obvious implementation -- filter the whole dataset, then re-split --
would recompute split boundaries over a different row population and silently
change what is in the test set for every held-out family.
"""

from __future__ import annotations

import numpy as np

from ..data.splits import Split


def family_row_index(
    y_multiclass: np.ndarray,
    split: Split,
    held_out: int,
    role: str,
    benign_class: int = 0,
) -> np.ndarray:
    """Absolute row indices this split contributes under leave-one-attack-out.

    `role` is "train", "val" or "test". Train and val drop the held-out family;
    test keeps benign traffic plus the held-out family and nothing else.
    """
    if role not in {"train", "val", "test"}:
        raise ValueError(f"role must be train, val or test, got {role!r}")

    rows = np.arange(split.start, split.stop)
    fam = y_multiclass[split.start:split.stop]

    if role == "test":
        # Benign + the held-out family. Leaving the other eight families in
        # would mean "recall" mixed seen and unseen attacks in one number.
        keep = (fam == benign_class) | (fam == held_out)
    else:
        keep = fam != held_out
    return rows[keep]


def holdout_plan(
    y_multiclass: np.ndarray,
    splits: dict,
    families: dict,
    held_out_name: str,
    benign_class: int = 0,
) -> dict:
    """Row indices for all three splits, plus the counts needed to judge them.

    The counts are returned rather than logged because a family can be held out
    and still leave a test split with too few positives to measure. On
    NF-ToN-IoT-v2 `ransomware` (3,425 flows) and `mitm` (7,723) are thin enough
    that their held-out recall carries very wide error bars, and a run that
    silently produced a recall over 40 positives would be quoted as though it
    were the 3.8M-flow `scanning` number.
    """
    if held_out_name not in families:
        raise KeyError(
            f"{held_out_name!r} is not a known family; have {sorted(families)}")
    held_out = families[held_out_name]
    if held_out == benign_class:
        raise ValueError(
            f"{held_out_name!r} is the benign class -- holding it out would "
            f"remove the negative class, not an attack family")

    plan = {"held_out": held_out_name, "held_out_class": held_out, "splits": {}}
    for name, s in splits.items():
        split = Split(name, s["start"], s["stop"])
        role = name if name in {"train", "val", "test"} else "test"
        idx = family_row_index(y_multiclass, split, held_out, role, benign_class)
        fam = y_multiclass[idx]
        plan["splits"][name] = {
            "row_index": idx,
            "n_rows": int(len(idx)),
            "n_rows_before": len(split),
            "n_held_out": int((fam == held_out).sum()),
            "n_benign": int((fam == benign_class).sum()),
            "removed": int(len(split) - len(idx)),
        }
    return plan


def assert_holdout_is_clean(plan: dict) -> None:
    """The check that makes the zero-day claim believable rather than asserted.

    A leave-one-attack-out run that quietly leaked one flow of the held-out
    family into training is indistinguishable, from the outside, from one that
    did not -- the metrics look plausible either way. So it is asserted, not
    inspected. This is the same reasoning that caught C4-C8: a check that
    *could disagree*.
    """
    for name in ("train", "val"):
        info = plan["splits"].get(name)
        if info is None:
            continue
        if info["n_held_out"] != 0:
            raise AssertionError(
                f"{info['n_held_out']} flows of held-out family "
                f"{plan['held_out']!r} survived in {name} -- this is not a "
                f"zero-day run")

    test = plan["splits"].get("test")
    if test is not None:
        if test["n_held_out"] == 0:
            raise AssertionError(
                f"test split contains no {plan['held_out']!r} flows; there is "
                f"nothing to measure recall on")
        if test["n_held_out"] + test["n_benign"] != test["n_rows"]:
            raise AssertionError(
                "test split contains families other than the held-out one")
