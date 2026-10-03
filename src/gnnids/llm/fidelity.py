"""Did the report relay the detector's conclusions, or substitute its own?

Groundedness asks whether a report *invented* anything. These two checks ask a
different question, and [[LLM Comparative Study Design]] lists both as metric
families the study needs:

- **Classification fidelity** -- does the report name the attack category the
  detector predicted? A model that quietly "corrects" the detector to a category
  it finds more plausible is a specific and interesting failure: every entity in
  such a report can be grounded, so groundedness scores it 1.0.
- **Attribution fidelity** -- does the report discuss the features the
  attribution ranked highest, or does it pattern-match to a generic description
  of the attack and ignore the actual evidence? That report looks perfect, which
  is what makes it the dangerous one.

**Both checks are lexical, and both say so.** They look for names and phrases;
they do not parse meaning. Two consequences belong beside every number they
produce:

1. They are blind to negation. "This is not a DDoS" names DDoS. The
   `unsupported_families` list is kept so a reader can inspect what was named
   rather than trust a boolean.
2. Attribution matching is generous: a feature counts as cited if its column
   name appears, or if its content words occur together in one sentence. A model
   that restates a flow fact ("bytes in: 48") is credited with citing `IN_BYTES`
   whether or not it used it as evidence. The measure is therefore an upper
   bound on engagement with the attribution, and its useful signal is at the low
   end: a report that cites *none* of the attributed features.

No language model is involved, for the reason given in `groundedness.py` (D31).
"""

from __future__ import annotations

import re

# How a report might name each attack family. Keyed by lower-cased family name;
# a family with no entry is matched by its own name only, so a new dataset
# degrades to exact-name matching rather than failing.
FAMILY_ALIASES: dict[str, tuple[str, ...]] = {
    "scanning": ("scanning", "scan", "reconnaissance", "port sweep", "probing"),
    "ddos": ("ddos", "distributed denial of service",
             "distributed denial-of-service"),
    "dos": ("dos", "denial of service", "denial-of-service"),
    "xss": ("xss", "cross-site scripting", "cross site scripting"),
    "injection": ("injection",),
    "password": ("password", "brute force", "brute-force", "credential"),
    "backdoor": ("backdoor",),
    "mitm": ("mitm", "man-in-the-middle", "man in the middle"),
    "ransomware": ("ransomware",),
    "benign": ("benign",),
}

# A category the evidence itself offers as an alternative may be named freely.
PLAUSIBLE_ABOVE = 0.05


def families_named(report: str, families: list[str]) -> set[str]:
    """Which of `families` the report names, longest alias first.

    Longest first, with matched text blanked out, because the aliases nest:
    "distributed denial of service" contains "denial of service", and a report
    about a DDoS must not be scored as also naming a DoS.
    """
    text = report.lower()
    pairs = []
    for fam in families:
        for alias in FAMILY_ALIASES.get(fam.lower(), (fam.lower(),)):
            pairs.append((alias, fam))
    named: set[str] = set()
    for alias, fam in sorted(pairs, key=lambda p: -len(p[0])):
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])")
        if pattern.search(text):
            named.add(fam)
            text = pattern.sub(" " * len(alias), text)
    return named


def check_classification_fidelity(pack: dict, report: str) -> dict:
    """Does the report name the predicted category, and only supported ones?"""
    d = pack["detection"]
    predicted = d["predicted_class"]
    dist = d.get("class_distribution", {})
    families = sorted(dist) or [predicted]
    named = families_named(report, families)

    plausible = {f for f, p in dist.items() if p >= PLAUSIBLE_ABOVE} | {predicted}
    unsupported = sorted(named - plausible)
    names_predicted = predicted in named
    return {
        "predicted_class": predicted,
        "families_named": sorted(named),
        "names_predicted": names_predicted,
        # Named, and neither the prediction nor an alternative the evidence
        # offered. Lexical: inspect before calling it a substitution.
        "unsupported_families": unsupported,
        # The failure the study is looking for: the detector said one thing and
        # the report says another instead.
        "substituted": bool(unsupported) and not names_predicted,
        "faithful": names_predicted and not unsupported,
    }


# ------------------------------------------------------------ attribution

# How each token of a NetFlow column name may appear in prose. Tokens mapped to
# an empty tuple carry no meaning a writer would reproduce and are ignored.
_TOKEN_WORDS: dict[str, tuple[str, ...]] = {
    "l4": (), "num": (), "to": (), "up": (), "ipv4": (),
    "src": ("source", "src", "originating"),
    "dst": ("destination", "dst", "dest"),
    "in": ("in", "inbound", "incoming", "received"),
    "out": ("out", "outbound", "outgoing", "sent"),
    "pkts": ("packets", "packet", "pkts"),
    "pkt": ("packets", "packet", "pkt"),
    "proto": ("protocol", "proto"),
    "protocol": ("protocol", "proto"),
    "l7": ("application", "l7", "layer"),
    "avg": ("average", "avg", "mean"),
    "max": ("max", "maximum", "largest"),
    "min": ("min", "minimum", "smallest"),
    "win": ("window", "win"),
    "len": ("length", "len", "size"),
    "ret": ("return", "ret", "response"),
    "flags": ("flags", "flag"),
    "bytes": ("bytes", "byte"),
    "second": ("second", "sec"),
    "milliseconds": ("milliseconds", "ms", "millisecond", "duration"),
}
# Suffixes that are encoding details of one underlying column.
_SUFFIXES = ("_bucket", "_present")


def base_column(feature_name: str) -> str:
    """`L4_SRC_PORT_bucket=1` and `L4_SRC_PORT` are the same thing to a reader."""
    col = feature_name.split("=", 1)[0]
    for suffix in _SUFFIXES:
        if col.endswith(suffix):
            col = col[: -len(suffix)]
    return col


def _token_sets(column: str) -> list[tuple[str, ...]]:
    sets = []
    for token in column.lower().split("_"):
        words = _TOKEN_WORDS.get(token, (token,))
        if words:
            sets.append(words)
    return sets


def cites_feature(report: str, column: str) -> bool:
    """True if the report uses the column's name, or its words in one sentence."""
    lower = report.lower()
    if column.lower() in lower or column.lower().replace("_", " ") in lower:
        return True
    need = _token_sets(column)
    if not need:
        return False
    for sentence in re.split(r"[.!?\n]+", lower):
        words = set(re.findall(r"[a-z0-9]+", sentence))
        if all(words & set(options) for options in need):
            return True
    return False


def check_attribution_fidelity(pack: dict, report: str, top_k: int = 5) -> dict | None:
    """Of the attributed features the model was shown, how many does it cite?

    `top_k` matches the number `render_facts` puts in the prompt, so the report
    is only measured against features it actually saw. Returns None when the
    pack carries no attribution, rather than a perfect or a zero score.
    """
    feats = pack.get("attribution", {}).get("top_features", [])[:top_k]
    columns = list(dict.fromkeys(base_column(f["name"]) for f in feats))
    if not columns:
        return None
    cited = [c for c in columns if cites_feature(report, c)]
    return {
        "n_attributed": len(columns),
        "features_cited": cited,
        "features_missed": [c for c in columns if c not in cited],
        "coverage": round(len(cited) / len(columns), 4),
        "cites_none": not cited,
    }
