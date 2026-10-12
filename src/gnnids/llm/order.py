"""A pack order in which every prefix is a fair sample (D46).

`make_evidence.py` writes packs in the order it found them, which is close to
grouped by attack family: the first 21 are almost all scanning. That is harmless
when all 200 are used and wrong the moment a run covers fewer -- a pilot of five,
or a study that runs out of free credit after 80. Both scripts that read packs
therefore put them in this order first, so "the first N" is always spread across
families and false positives in proportion to the full sample.
"""

from __future__ import annotations


def balanced_order(packs: list[dict]) -> list[dict]:
    """Interleave the true-label groups evenly. Deterministic: no seed.

    Each pack gets the position it would hold if its group were spread uniformly
    over the whole list, `(rank + 0.5) / group size`, and the list is sorted by
    that. A group of 30 in a list of 200 then supplies about 3 of every 20.
    """
    groups: dict[str, list[dict]] = {}
    for p in packs:
        groups.setdefault(str(p.get("flow", {}).get("true_label")), []).append(p)
    keyed = []
    for label in sorted(groups):
        members = groups[label]
        for rank, p in enumerate(members):
            keyed.append(((rank + 0.5) / len(members), label, rank, p))
    keyed.sort(key=lambda k: k[:3])
    return [k[3] for k in keyed]
