"""The evidence pack: the frozen contract between the GNN and the LLM (Phase 6a).

When the GNN flags a flow, the system assembles one JSON document holding
everything known about that detection. **The LLM sees only this document.**

That constraint is the entire anti-hallucination design. If a fact is not in the
pack, the model has no legitimate source for it, so any technical detail in the
output that is absent from the input is provably invented -- which is what makes
the automated groundedness metric possible at all. See [[Evidence Pack Schema]].

Three properties are enforced here rather than left to convention:

**No z-scores.** Every measured number is inverted back to its real unit first
(`gnnids.explain.units`). A model handed `IN_BYTES: 1.83` will pick a unit and
state it confidently.

**No GNN vocabulary.** No embeddings, no adjacency matrices, no layer
activations, no logits. The proposal bans jargon from the *output*; keeping it
out of the *input* is the enforceable version -- a model cannot mention an
embedding it has never seen. `validate_pack` fails on the banned terms.

**The full class distribution, not just the argmax.** A model that is
0.4/0.35/0.25 across three families is genuinely uncertain and the report should
say so. Handing over only the top class manufactures confidence, and confident
wrong attribution is worse than acknowledged uncertainty in a security context.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import numpy as np

# Bumped only when a field changes meaning. Phase 7's prompts and the
# groundedness checker both key off this, so a silent change would invalidate a
# comparison across models rather than breaking anything visibly.
SCHEMA_VERSION = "1.0"

# Terms that must never reach the LLM. The check is on the serialised pack, so
# it catches them wherever they appear -- including inside a feature name that
# someone adds later without thinking about this file.
BANNED_TERMS = (
    "embedding", "adjacency", "activation", "logit", "hidden_dim",
    "softmax", "tensor", "gradient", "neural", "graph neural", "gnn",
)


def build_pack(
    *,
    flow: dict,
    detection: dict,
    attribution: dict,
    context: dict,
    guidance: dict,
    provenance: dict,
    causal_path=None,
    detection_id: str | None = None,
) -> dict:
    """Assemble one pack. Field order follows [[Evidence Pack Schema]]."""
    return {
        "schema_version": SCHEMA_VERSION,
        "detection_id": detection_id or str(uuid.uuid4()),
        "flow": flow,
        "detection": detection,
        "attribution": attribution,
        "context": context,
        # Nullable from day one. Granger causality (Tier 2, D14) may never
        # materialise, and building the field in as optional means its absence
        # is a data condition the prompt already handles rather than a schema
        # migration late in the project.
        "causal_path": causal_path,
        "guidance": guidance,
        "provenance": provenance,
    }


def host_context(
    src_codes: np.ndarray,
    dst_codes: np.ndarray,
    ports: np.ndarray,
    byte_counts: np.ndarray,
    host: int,
) -> dict:
    """What this host did inside the window, and how unusual that is for it.

    `baseline_deviation` compares the host to **its own** behaviour in the
    window rather than to a global average. "This host contacted 200 distinct
    destinations, against a typical 3" is actionable; "200 is above average" is
    not, and it is also the sentence a model turns into a good paragraph without
    extra prompting.
    """
    as_src = src_codes == host
    n_flows = int(as_src.sum())
    peers = np.unique(dst_codes[as_src])
    dports = np.unique(ports[as_src])

    # Per-host fan-out across the window, so "typical" means typical *here*,
    # not typical on some other network.
    all_hosts, counts = np.unique(src_codes, return_counts=True)
    fanouts = np.array([len(np.unique(dst_codes[src_codes == h]))
                        for h in all_hosts]) if len(all_hosts) else np.array([1])
    median_fanout = float(np.median(fanouts)) if len(fanouts) else 1.0

    return {
        "flow_count": n_flows,
        "distinct_dst_ips": int(len(peers)),
        "distinct_dst_ports": int(len(dports)),
        "total_bytes": int(byte_counts[as_src].sum()) if n_flows else 0,
        "baseline_deviation": {
            "typical_distinct_dst_ips_for_hosts_in_this_window": round(median_fanout, 1),
            "times_typical": (round(len(peers) / median_fanout, 1)
                              if median_fanout > 0 else None),
        },
    }


def validate_pack(pack: dict) -> None:
    """Fail loudly on a pack that would poison the groundedness measurement.

    Called on every pack rather than on a sample. A pack carrying a banned term
    or a raw z-score does not break Phase 7 -- it produces a report that reads
    fine and a groundedness score that is quietly meaningless, which is the
    failure mode this project keeps finding (C4-C8).
    """
    required = {"schema_version", "detection_id", "flow", "detection",
                "attribution", "context", "causal_path", "guidance", "provenance"}
    missing = required - set(pack)
    if missing:
        raise ValueError(f"pack is missing {sorted(missing)}")

    if pack["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"pack is schema {pack['schema_version']}, this code writes "
            f"{SCHEMA_VERSION}")

    blob = json.dumps(pack).lower()
    found = sorted({t for t in BANNED_TERMS if t in blob})
    if found:
        raise ValueError(
            f"pack contains model-internal vocabulary {found}; the LLM must "
            f"never see it, or the no-jargon requirement becomes a prompt "
            f"instruction rather than a property of the input")

    dist = pack["detection"].get("class_distribution")
    if not isinstance(dist, dict) or not dist:
        raise ValueError("class_distribution must be present and non-empty")
    total = sum(dist.values())
    if not 0.98 <= total <= 1.02:
        raise ValueError(f"class_distribution sums to {total:.4f}, not 1")


def config_hash(cfg: dict) -> str:
    """Short stable digest, so a report traces to the config that produced it."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def load_guidance(path: Path) -> dict:
    """Static per-family guidance (D10). Retrieved, never generated."""
    return json.loads(Path(path).read_text())["families"]
