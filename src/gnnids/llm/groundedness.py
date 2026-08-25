"""Automated groundedness: is every technical claim traceable to the evidence?

The design in one sentence: **the LLM sees only the evidence pack, so any
technical entity in its output that is absent from the pack was invented.**

That makes hallucination measurable rather than a matter of reading reports and
forming an impression, which is what most comparative LLM work in this space
does. It is the single most defensible metric in
[[LLM Comparative Study Design]] and it exists only because of the
[[Evidence Pack Schema]] constraint.

**What this measures, and what it does not.** It catches *fabricated specifics* —
an IP that was never in the evidence, a port the flow never touched, a MITRE
technique not in the guidance, a number with no source. It does not catch a
fluent, wrong *interpretation* built entirely from real facts. That is a
genuine limit and belongs in the write-up beside the numbers: high groundedness
means "invented nothing", not "reasoned correctly".

**Why entity extraction and not an LLM judge.** A judge model is itself a
confound — it has its own biases, it costs money per evaluation, and it makes
the headline metric depend on a model rather than on arithmetic. Set difference
over extracted entities is deterministic, free, and reproducible from the
committed artefacts alone.
"""

from __future__ import annotations

import json
import re

IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
MITRE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
# Numbers with optional thousands separators and decimals. Percentages are
# excluded deliberately: a model restating 0.96 as "96%" is a faithful
# reformatting, not an invention, and counting it as ungrounded would punish
# the clearer report.
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?!\s*%)(?![\w.])")

# Numbers a report may use freely: section counts, "one", ordinary small
# integers used as prose. Requiring 0-10 to appear in the evidence would flag
# "three ports" written about a list of three things.
FREE_NUMBERS = {str(i) for i in range(11)}


def extract_entities(text: str) -> dict[str, set[str]]:
    """Technical specifics a claim could be built from."""
    return {
        "ips": set(IPV4.findall(text)),
        "mitre": set(MITRE.findall(text)),
        "numbers": {_normalise(n) for n in NUMBER.findall(text)} - FREE_NUMBERS,
    }


def _normalise(n: str) -> str:
    """`4,498` and `4498` and `4498.0` are the same number.

    Without this the metric would punish a model for writing readable prose,
    which is the opposite of what the study wants to reward.
    """
    n = n.replace(",", "")
    try:
        f = float(n)
    except ValueError:
        return n
    return str(int(f)) if f == int(f) else str(f)


def _evidence_entities(pack: dict) -> dict[str, set[str]]:
    """Everything the pack legitimately supplies.

    Taken from the serialised pack rather than from the rendered prompt, so a
    model is credited for a fact that was available to it even if the template
    happened not to surface that field.
    """
    blob = json.dumps(pack)
    found = extract_entities(blob)
    # Rounded forms count as grounded: a score of 0.96004 restated as 0.96 is
    # the report doing its job.
    rounded = set()
    for n in found["numbers"]:
        try:
            f = float(n)
        except ValueError:
            continue
        for places in (0, 1, 2, 3):
            rounded.add(_normalise(f"{round(f, places):.{places}f}"))
    found["numbers"] |= rounded
    return found


def score_report(pack: dict, report: str) -> dict:
    """Groundedness for one generated report.

    Reported per entity class as well as overall, because the classes are not
    equally serious. **A fabricated IP address is a different kind of failure
    from a rounded number** -- it would send an analyst to the wrong host -- so
    an aggregate that hides it is not good enough.
    """
    evidence = _evidence_entities(pack)
    claimed = extract_entities(report)

    per_class, total, grounded_total = {}, 0, 0
    for kind in ("ips", "mitre", "numbers"):
        ungrounded = sorted(claimed[kind] - evidence[kind])
        n = len(claimed[kind])
        per_class[kind] = {
            "claimed": n,
            "ungrounded": ungrounded,
            "n_ungrounded": len(ungrounded),
            "groundedness": round((n - len(ungrounded)) / n, 4) if n else None,
        }
        total += n
        grounded_total += n - len(ungrounded)

    return {
        "groundedness": round(grounded_total / total, 4) if total else None,
        "n_entities_claimed": total,
        "n_ungrounded": total - grounded_total,
        # Called out separately because it is the failure with operational
        # consequences, not just a scoring one.
        "fabricated_addresses": per_class["ips"]["ungrounded"],
        "per_class": per_class,
        "report_words": len(report.split()),
    }


def check_uncertainty_conveyed(pack: dict, report: str,
                               ambiguous_below: float = 0.9) -> dict | None:
    """When the model was unsure, did the report say so?

    Only meaningful on genuinely ambiguous detections, so it returns None
    otherwise rather than a passing score -- averaging in the easy cases would
    turn a sharp behavioural test into a flattering constant.
    """
    confidence = pack["detection"].get("class_confidence")
    if confidence is None or confidence >= ambiguous_below:
        return None

    hedges = ("uncertain", "unclear", "may ", "might", "possibly", "could be",
              "not certain", "ambiguous", "likely", "appears", "suggests",
              "consistent with", "cannot be confirmed")
    lower = report.lower()
    return {
        "class_confidence": confidence,
        "hedges_found": sorted({h.strip() for h in hedges if h in lower}),
        "uncertainty_conveyed": any(h in lower for h in hedges),
    }


def check_no_jargon(report: str) -> dict:
    """The output side of the no-jargon rule.

    The input side is enforced by `validate_pack`; this catches a model that
    supplied the vocabulary from its own knowledge of how such systems work --
    which it can do without seeing a single one of these terms in the prompt.
    """
    banned = ("neural network", "gnn", "graph neural", "embedding",
              "model was trained", "machine learning model", "logit",
              "softmax", "activation", "training data", "the algorithm")
    lower = report.lower()
    found = sorted({t for t in banned if t in lower})
    return {"jargon_found": found, "clean": not found}
