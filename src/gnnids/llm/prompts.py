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


PROMPTS = {
    "v1": {"system": SYSTEM_V1, "user": USER_V1},
}
DEFAULT_VERSION = "v1"


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
    return t["system"], t["user"].format(
        detection_id=pack["detection_id"], facts=render_facts(pack))
