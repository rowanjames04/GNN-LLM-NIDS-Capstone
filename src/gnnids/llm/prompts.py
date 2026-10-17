"""Versioned prompt templates, and the pack-to-prose rendering (7a).

**Versioned, because the study measures prompts as well as models.** The
prompt-sensitivity ablation in [[LLM Comparative Study Design]] compares outputs
across template versions, so every generated report records the version that
produced it. An unversioned prompt edited mid-study silently invalidates every
comparison made before the edit.

The system prompt does the constraining; the user message carries only facts.
That split matters: instructions in the user turn are easier for a model to
treat as content to summarise, and here the user turn *is* content.
"""

from __future__ import annotations

SYSTEM_V1 = """You are a security analyst writing an incident note for a colleague.

You will be given the complete evidence for one network detection. Write a short \
report with these sections:

## What happened
## Why it was flagged
## What to do next

Rules, in order of importance:

1. Use ONLY facts from the evidence provided. Do not add context, threat-actor \
names, or technical detail that is not in the evidence. If something is not in \
the evidence, it is not known.
2. Where the evidence expresses uncertainty, say so. If the detection's class \
distribution is spread across several categories, the report must convey that \
the classification is uncertain rather than state the top category as fact.
3. Write for a competent colleague who does not work on machine learning. Do not \
describe how the detector works internally. Refer to what was measured about the \
traffic, not to the detector's mechanism.
4. Be brief. Under 250 words.
5. If a value is marked as a lower bound, describe it as "at least" that value."""

USER_V1 = """Evidence for detection {detection_id}:

{facts}

Write the incident note."""


# ---------------------------------------------------------------------------
# v2 -- the prompt-sensitivity ablation arm.
#
# A useful ablation needs a *contrast*, not a paraphrase. v2 differs on one
# axis that plausibly changes behaviour on this task: v1 constrains heavily and
# enumerates the rules; v2 states the goal and the single hard constraint, and
# otherwise trusts the model.
#
# The question it asks is whether the elaborate scaffolding in v1 is doing the
# work, or whether the *evidence pack* is -- if groundedness is unchanged
# between v1 and v2, then the anti-fabrication property comes from constraining
# the input, which is the study's central architectural claim. If groundedness
# collapses under v2, the property was coming from the prompt and is far more
# fragile than the design assumes.
#
# Either result is publishable. That is what makes it worth running.

SYSTEM_V2 = """You are a security analyst. You will be given the complete evidence for one network detection, and nothing else.

Write a short incident note for a colleague who is technically competent but does not work on machine learning.

There is one hard constraint: **every technical detail you state must come from the evidence provided.** If it is not in the evidence, it is not known, and you should say so rather than fill the gap.

Beyond that, use your judgement about what matters and how to say it."""

USER_V2 = """Evidence for detection {detection_id}:

{facts}

Write the incident note."""


# ---------------------------------------------------------------------------
# v3 and v4 -- the facts layer (D49, 2026-10-17).
#
# Same contrast as v1/v2 (prescriptive against permissive), but the user turn
# is now `facts.render(pack)`: finished sentences in labelled sections, with
# every number computed, every code decoded and every gap stated. The prompts
# add the rule that makes that worth doing: no arithmetic. v1 and v2 are kept,
# with their original renderer, so the archived round-1 reports still score
# against exactly the text their models saw.

SYSTEM_V3 = """You are a security analyst writing an incident note for a colleague.

You will be given the complete findings for one network detection, already worked \
out and written in plain sentences. Your job is to turn them into a clear, short \
note. Write these sections:

## What happened
## Why it was flagged
## What to do next

Rules, in order of importance:

1. Use ONLY the findings provided. Do not add context, threat-actor names, or \
technical detail that is not in them. If something is not in the findings, it is \
not known.
2. Do not calculate anything. Every number you write must appear in the findings \
exactly as written. Do not convert units, total, average, round differently or \
compare numbers yourself; the findings already state every comparison that matters.
3. Only the items under "WHY THE DETECTOR FLAGGED IT" are reasons for the score. \
Keep the direction each one is given (pushed the score up, or down). The \
"BACKGROUND CONTEXT" items describe the hosts; you may mention them, but never \
as reasons for the score.
4. Convey the confidence the findings state, in their words. If the findings call \
the detection borderline or the category uncertain, the note must say so.
5. Where the findings say something is not recorded or not known, say it is not \
known. Never estimate it.
6. Write for a competent colleague who does not work on machine learning. Do not \
describe how the detector works internally.
7. Be brief. Under 250 words."""

SYSTEM_V4 = """You are a security analyst. You will be given the complete findings for one network detection, already worked out in plain sentences, and nothing else.

Write a short incident note for a colleague who is technically competent but does not work on machine learning.

There are two hard constraints: **every technical detail you state must come from the findings**, and **you must not calculate anything** -- every number you write must appear in the findings exactly as written. If something is not in the findings, it is not known.

Beyond that, use your judgement about what matters and how to say it."""

USER_V3 = """Findings for detection {detection_id}:

{facts}

Write the incident note."""


def _render_facts_v3(pack: dict) -> str:
    from .facts import render
    return render(pack)


PROMPTS = {
    "v1": {"system": SYSTEM_V1, "user": USER_V1, "render": "render_facts"},
    "v2": {"system": SYSTEM_V2, "user": USER_V2, "render": "render_facts"},
    "v3": {"system": SYSTEM_V3, "user": USER_V3, "render": "facts"},
    "v4": {"system": SYSTEM_V4, "user": USER_V3, "render": "facts"},
}
DEFAULT_VERSION = "v3"

# What separates each version, recorded so the ablation can be described
# without re-reading the templates.
PROMPT_NOTES = {
    "v1": "prescriptive: numbered rules, mandated three-section structure, "
          "explicit word limit, explicit uncertainty and lower-bound instructions",
    "v2": "permissive: states the goal and the single grounding constraint, "
          "leaves structure and length to the model",
    "v3": "v1's prescriptive rules over the facts layer (D49), plus no arithmetic, "
          "reasons vs context, and stated confidence in the findings' own words",
    "v4": "v2's permissive framing over the facts layer (D49), plus no arithmetic",
}


def render_facts(pack: dict) -> str:
    """The evidence pack as flat, checkable bullet points.

    Flattened rather than passed as raw JSON for one reason: the groundedness
    checker set-differences entities in the *output* against entities in the
    *input*, so every fact the model is allowed to state has to be present as a
    literal string here. JSON with nested keys would put values in the prompt
    that the checker's extractor would have to re-derive, and the two extractors
    drifting apart would make the metric quietly wrong.
    """
    f, d = pack["flow"], pack["detection"]
    out = [
        f"- Source address: {f.get('src_ip')}",
        f"- Destination address: {f.get('dst_ip')}",
        f"- Detection score: {d['score']} (alert threshold {d['threshold']})",
        f"- Most likely category: {d['predicted_class']} "
        f"(confidence {d['class_confidence']})",
    ]

    # Uncertainty, stated as a fact rather than left for the model to infer from
    # the distribution. Rule 2 of the system prompt is only checkable if the
    # ambiguity is in the input in words.
    ranked = sorted(d["class_distribution"].items(), key=lambda kv: -kv[1])
    runners = [f"{k} ({v})" for k, v in ranked[1:3] if v >= 0.05]
    if runners:
        out.append(f"- The classification is uncertain. Other plausible "
                   f"categories: {', '.join(runners)}")
    else:
        out.append("- No other category was plausible.")

    for key, label in (("IN_BYTES", "Bytes in"), ("OUT_BYTES", "Bytes out"),
                       ("IN_PKTS", "Packets in"), ("L4_DST_PORT", "Destination port"),
                       ("FLOW_DURATION_MILLISECONDS", "Duration (ms)")):
        if key in f:
            out.append(f"- {label}: {f[key]}")
    for name in f.get("_values_at_measurement_ceiling", []):
        out.append(f"- {name} is a lower bound; the true value is at least the "
                   f"figure given.")

    a = pack.get("attribution", {})
    for feat in a.get("top_features", [])[:5]:
        direction = ("raised" if feat["direction"] == "increases_suspicion"
                     else "lowered")
        # A one-hot indicator is a property, not a quantity. Rendering
        # "PROTOCOL=6 measured 1.0" hands the model a number with no referent
        # and invites it to reason about the magnitude of a flag.
        if "=" in feat["name"]:
            column, value = feat["name"].split("=", 1)
            out.append(f"- The flow's {column} was {value}, which {direction} "
                       f"the assessment")
        else:
            out.append(f"- {feat['name']} measured {feat['value']}, which "
                       f"{direction} the assessment")
    for n in a.get("influential_neighbours", [])[:3]:
        out.append(f"- A related flow {n['src_ip']} to {n['dst_ip']} also "
                   f"affected the assessment")
    if a.get("neighbour_note"):
        out.append(f"- {a['neighbour_note']}")

    c = pack.get("context", {})
    for side, label in (("src_host", "The source host"), ("dst_host", "The destination host")):
        h = c.get(side)
        if not h:
            continue
        times = h.get("baseline_deviation", {}).get("times_typical")
        out.append(
            f"- {label} made {h['flow_count']} connections in this period, to "
            f"{h['distinct_dst_ips']} distinct addresses across "
            f"{h['distinct_dst_ports']} ports"
            + (f", which is {times} times the typical number of addresses for a "
               f"host here" if times else ""))

    g = pack.get("guidance", {})
    if g.get("description"):
        out.append(f"- Reference description of {g.get('attack_family')}: "
                   f"{g['description']}")
    for t in g.get("mitre_techniques", []):
        out.append(f"- Associated technique identifier: {t}")
    for m in g.get("standard_mitigations", []):
        out.append(f"- Standard mitigation: {m}")

    return "\n".join(out)


def build_prompt(pack: dict, version: str = DEFAULT_VERSION) -> tuple[str, str]:
    """(system, user) for one pack."""
    if version not in PROMPTS:
        raise KeyError(f"unknown prompt version {version!r}; have {sorted(PROMPTS)}")
    t = PROMPTS[version]
    facts = _render_facts_v3(pack) if t["render"] == "facts" else render_facts(pack)
    return t["system"], t["user"].format(detection_id=pack["detection_id"], facts=facts)
