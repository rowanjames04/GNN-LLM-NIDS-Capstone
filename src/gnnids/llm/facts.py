"""The facts layer: every statement the LLM may make, computed and worded by code (D49).

Rowan's rule, 2026-10-17: *all the maths and technical heavy lifting is done
before the LLM; the LLM is a natural-language translator.* The round-1 audit
showed what happens otherwise. Handed raw values, models converted units wrongly
("4,294,131 ms (over 47 hours)"), filled in a packet count the pack never gave,
read a feature that lowered the score as one that raised it, and treated host
activity as a reason for the detection. Every one of those reports scored as
grounded, because each number was copied from somewhere.

So this module does, in code, everything the model was being left to do:

- **Units and arithmetic.** Sizes, averages, margins and shares are computed
  here and stated with their result. The prompt forbids further calculation.
- **Decoding.** Protocol numbers, TCP flag bitmasks, ICMP type/code, DNS query
  types, FTP return codes and well-known ports are translated to names.
- **Gaps.** A value that was not recorded, or is unreliable, is said to be so.
  The model is never left to notice an absence.
- **Judgement that is really arithmetic.** How far the score cleared the
  threshold and how confident the category is are put into words by fixed
  rules, so "borderline" means the same thing in every report.
- **Roles.** Reasons for the score and background context are in separate,
  labelled sections, and each reason carries its direction in words.

The output is a list of sections of plain sentences. `render(pack)` joins them
for the prompt, and the groundedness check reads the same text, so a number the
code computed counts as supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------- decoders

PROTOCOLS = {1: "ICMP", 2: "IGMP", 6: "TCP", 17: "UDP", 47: "GRE", 58: "ICMPv6"}

TCP_FLAG_BITS = ((1, "FIN"), (2, "SYN"), (4, "RST"), (8, "PSH"),
                 (16, "ACK"), (32, "URG"), (64, "ECE"), (128, "CWR"))

# nDPI master-protocol numbers, only those this project is sure of. Anything
# else is stated as an undecoded code rather than guessed.
APP_PROTOCOLS = {0: None, 1: "FTP (control)", 5: "DNS", 7: "HTTP", 91: "TLS", 92: "SSH"}

DNS_TYPES = {1: "A", 2: "NS", 5: "CNAME", 6: "SOA", 12: "PTR", 15: "MX",
             16: "TXT", 28: "AAAA", 33: "SRV", 255: "ANY"}

FTP_CODES = {220: "service ready", 221: "closing connection", 226: "transfer complete",
             230: "login successful", 331: "user name accepted, password required",
             421: "service not available", 425: "cannot open data connection",
             500: "command not recognised", 530: "not logged in",
             550: "file unavailable"}

SERVICES = {20: "FTP data", 21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
            67: "DHCP", 68: "DHCP", 69: "TFTP", 80: "HTTP", 110: "POP3", 123: "NTP",
            135: "Microsoft RPC", 137: "NetBIOS", 138: "NetBIOS", 139: "NetBIOS/SMB",
            143: "IMAP", 161: "SNMP", 389: "LDAP", 443: "HTTPS", 445: "SMB",
            502: "Modbus", 1433: "Microsoft SQL Server", 1883: "MQTT", 3306: "MySQL",
            3389: "Remote Desktop", 5432: "PostgreSQL", 5900: "VNC", 8080: "HTTP (alternate)",
            8443: "HTTPS (alternate)", 8883: "MQTT over TLS"}

# Plain-English names for every model input. Direction words follow NetFlow's
# convention: IN is the source-to-destination direction of the flow.
LABELS = {
    "L4_SRC_PORT": "the source port",
    "L4_DST_PORT": "the destination port",
    "IN_BYTES": "the bytes sent from source to destination",
    "OUT_BYTES": "the bytes sent from destination back to source",
    "IN_PKTS": "the packets sent from source to destination",
    "OUT_PKTS": "the packets sent from destination back to source",
    "LONGEST_FLOW_PKT": "the longest packet in the flow",
    "SHORTEST_FLOW_PKT": "the shortest packet in the flow",
    "MAX_IP_PKT_LEN": "the largest IP packet in the flow",
    "RETRANSMITTED_IN_BYTES": "the bytes the source retransmitted",
    "RETRANSMITTED_IN_PKTS": "the packets the source retransmitted",
    "RETRANSMITTED_OUT_BYTES": "the bytes the destination retransmitted",
    "RETRANSMITTED_OUT_PKTS": "the packets the destination retransmitted",
    "NUM_PKTS_UP_TO_128_BYTES": "the number of packets of 128 bytes or fewer",
    "NUM_PKTS_128_TO_256_BYTES": "the number of packets of 128 to 256 bytes",
    "NUM_PKTS_256_TO_512_BYTES": "the number of packets of 256 to 512 bytes",
    "NUM_PKTS_512_TO_1024_BYTES": "the number of packets of 512 to 1,024 bytes",
    "NUM_PKTS_1024_TO_1514_BYTES": "the number of packets of 1,024 to 1,514 bytes",
    "TCP_WIN_MAX_IN": "the largest TCP window the source advertised",
    "TCP_WIN_MAX_OUT": "the largest TCP window the destination advertised",
    "ICMP_TYPE": "the ICMP message type",
    "DNS_QUERY_TYPE": "the DNS query type",
    "DNS_TTL_ANSWER": "the DNS answer's time-to-live",
    "FTP_COMMAND_RET_CODE": "the FTP server's reply code",
    "PROTOCOL": "the transport protocol",
    "L7_PROTO": "the application protocol",
    "TCP_FLAGS": "the TCP flags seen in the flow",
    "CLIENT_TCP_FLAGS": "the TCP flags the source sent",
    "SERVER_TCP_FLAGS": "the TCP flags the destination sent",
    "L4_SRC_PORT_bucket": "the source port range",
    "L4_DST_PORT_bucket": "the destination port range",
}
UNITS = {"IN_BYTES": "bytes", "OUT_BYTES": "bytes", "IN_PKTS": "packets",
         "OUT_PKTS": "packets", "LONGEST_FLOW_PKT": "bytes", "SHORTEST_FLOW_PKT": "bytes",
         "MAX_IP_PKT_LEN": "bytes", "RETRANSMITTED_IN_BYTES": "bytes",
         "RETRANSMITTED_OUT_BYTES": "bytes", "RETRANSMITTED_IN_PKTS": "packets",
         "RETRANSMITTED_OUT_PKTS": "packets", "TCP_WIN_MAX_IN": "bytes",
         "TCP_WIN_MAX_OUT": "bytes", "DNS_TTL_ANSWER": "seconds"}
ALL_COLUMNS = tuple(LABELS) + ("ICMP_TYPE", "DNS_QUERY_TYPE")
# Words that refer to the background context, not to an attributed feature. A
# sentence using them is about host activity, so the direction check reads it
# as naming "CONTEXT" and skips it rather than pinning it on a port feature.
CONTEXT_WORDS = ("distinct destination ports", "distinct ports", "distinct addresses",
                 "distinct destinations", "distinct sources", "connections",
                 "flows in this period", "typical host", "connection count")
PORT_RANGES = {"well_known": "a well-known port (below 1,024)",
               "registered": "a registered port (1,024 to 49,151)",
               "ephemeral": "an ephemeral port (49,152 and above)"}

# Thresholds for the words used about confidence. Fixed here, stated in the
# report's method section, and identical for every model.
MARGIN_BANDS = ((0.10, "narrow"), (0.50, "moderate"), (1.01, "wide"))
CONFIDENCE_BANDS = ((0.60, "low"), (0.90, "moderate"), (1.01, "high"))
PLAUSIBLE_ALTERNATIVE = 0.05


def num(v) -> str:
    """A measured number as text: integers with thousands separators."""
    if v is None:
        return "not recorded"
    f = float(v)
    if f == int(f) and abs(f) < 1e15:
        return f"{int(f):,}"
    # Up to five decimals, which is how the pack records scores; never
    # rounded further, so the number stated is the number measured.
    return f"{f:,.5f}".rstrip("0").rstrip(".")


def tcp_flags(v) -> str:
    v = int(v)
    if v == 0:
        return "no TCP flags"
    names = [n for bit, n in TCP_FLAG_BITS if v & bit]
    return "+".join(names)


def protocol(v) -> str:
    v = int(float(v))
    return PROTOCOLS.get(v, f"IP protocol number {v}")


def app_protocol(v) -> str:
    master = int(float(v))
    name = APP_PROTOCOLS.get(master, "?")
    if name is None:
        return "not identified"
    if name == "?":
        return f"application protocol code {master} (not decoded here)"
    return name


def port(v) -> str:
    # No thousands separator: "port 59378", as ports are written.
    p = int(float(v))
    svc = SERVICES.get(p)
    return f"{p} ({svc})" if svc else f"{p}"


def icmp(v) -> str:
    v = int(float(v))
    return f"ICMP type {v // 256}, code {v % 256}"


def decode_value(column: str, value) -> str:
    """A categorical or coded value, as a phrase."""
    if value in (None, "other"):
        return "a value too rare to have its own category"
    if column in ("TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"):
        return tcp_flags(value)
    if column == "PROTOCOL":
        return protocol(value)
    if column == "L7_PROTO":
        return app_protocol(value)
    if column in ("L4_SRC_PORT", "L4_DST_PORT"):
        return port(value)
    if column.endswith("_bucket"):
        return PORT_RANGES.get(str(value), str(value))
    if column == "ICMP_TYPE":
        return icmp(value)
    if column == "DNS_QUERY_TYPE":
        name = DNS_TYPES.get(int(float(value)))
        return f"{int(float(value))} ({name})" if name else num(value)
    if column == "FTP_COMMAND_RET_CODE":
        name = FTP_CODES.get(int(float(value)))
        return f"{int(float(value))} ({name})" if name else num(value)
    unit = UNITS.get(column)
    return f"{num(value)} {unit}" if unit else num(value)


def times_phrase(times, what: str) -> str:
    """"4.7 times as many distinct addresses as a typical host here", or "about
    as many" when the ratio is close to one."""
    t = float(times)
    if 0.9 <= t <= 1.1:
        return f"about as many {what} as a typical host here"
    if t < 1:
        return f"{num(times)} times as many {what} as a typical host here (fewer than typical)"
    return f"{num(times)} times as many {what} as a typical host here"


def plural(n, word: str, many: str | None = None) -> str:
    return word if n == 1 else (many or word + "s")


def band(value: float, bands) -> str:
    for upper, word in bands:
        if value < upper:
            return word
    return bands[-1][1]


# ---------------------------------------------------------------- statements

@dataclass
class Section:
    title: str
    lines: list[str] = field(default_factory=list)


@dataclass
class Reason:
    """One attributed feature, worded, with its direction and the words a
    report would use to refer to it (for the direction check)."""
    column: str
    sentence: str
    direction: str          # "raised" or "lowered"
    aliases: tuple[str, ...]


def margin(detection: dict) -> dict:
    """How far the score cleared the threshold, as numbers and a word.

    The share is of the room above the threshold, (score - threshold) /
    (1 - threshold): a score of 0.95 against 0.90 has used half the room.
    """
    s, t = float(detection["score"]), float(detection["threshold"])
    gap = s - t
    room = 1.0 - t
    share = gap / room if room > 0 else 1.0
    return {"gap": round(gap, 4), "share_of_room": round(share, 2),
            "band": band(max(share, 0.0), MARGIN_BANDS)}


def confidence(detection: dict) -> dict:
    c = float(detection["class_confidence"])
    ranked = sorted(detection["class_distribution"].items(), key=lambda kv: -kv[1])
    alternatives = [(k, v) for k, v in ranked[1:] if v >= PLAUSIBLE_ALTERNATIVE][:3]
    return {"value": c, "band": band(c, CONFIDENCE_BANDS), "alternatives": alternatives}


def borderline(pack: dict) -> bool:
    """A detection the report should hedge: a narrow margin, a category held
    with low confidence, or a plausible alternative category."""
    m, c = margin(pack["detection"]), confidence(pack["detection"])
    return m["band"] == "narrow" or c["band"] == "low" or bool(c["alternatives"])


def _split_name(name: str) -> tuple[str, str | None]:
    """`CLIENT_TCP_FLAGS=2` -> ("CLIENT_TCP_FLAGS", "2"); `IN_BYTES` -> (..., None)."""
    if "=" in name:
        col, val = name.split("=", 1)
        return col, val
    return name, None


def _alias(column: str) -> tuple[str, ...]:
    """Words a report would use for this feature. Kept specific, so that a
    sentence about bytes in general is not taken to be about this feature."""
    base = LABELS.get(column, column.lower().replace("_", " "))
    words = {base.removeprefix("the ")}
    extra = {
        "L4_DST_PORT": ("destination port",), "L4_SRC_PORT": ("source port",),
        "L4_DST_PORT_bucket": ("destination port",), "L4_SRC_PORT_bucket": ("source port",),
        # Direction-specific only: a bare "TCP window" is ambiguous between the
        # two and is read as the pseudo-column TCP_WINDOW (see fidelity).
        "TCP_WIN_MAX_IN": ("tcp_win_max_in", "inbound tcp window", "incoming tcp window",
                           "source's tcp window", "window the source advertised",
                           "tcp window maximum in", "max inbound tcp window"),
        "TCP_WIN_MAX_OUT": ("tcp_win_max_out", "outbound tcp window", "outgoing tcp window",
                            "destination's tcp window", "window the destination advertised",
                            "tcp window maximum out", "maximum outgoing tcp window",
                            "maximum outbound tcp window", "max outbound tcp window"),
        "CLIENT_TCP_FLAGS": ("client tcp flag", "client_tcp_flags", "flags the source sent"),
        "SERVER_TCP_FLAGS": ("server tcp flag", "server_tcp_flags", "flags the destination sent"),
        "TCP_FLAGS": ("tcp flag",),
        "PROTOCOL": ("transport protocol", "tcp protocol", "udp protocol", "icmp protocol",
                     "protocol 6", "protocol 17", "protocol 1",
                     "protocol (tcp)", "protocol (udp)", "protocol=6", "protocol=17"),
        "L7_PROTO": ("application protocol", "l7 protocol", "l7_proto", "layer-7 protocol",
                     "layer 7 protocol"),
        "SHORTEST_FLOW_PKT": ("shortest packet",), "LONGEST_FLOW_PKT": ("longest packet",),
        "IN_BYTES": ("bytes sent from source", "inbound bytes", "bytes in"),
        "OUT_BYTES": ("bytes sent from destination", "outbound bytes", "bytes out"),
        "IN_PKTS": ("packets sent from source", "packets in"),
        "OUT_PKTS": ("packets sent from destination", "packets out"),
    }.get(column, ())
    return tuple(sorted({w.lower() for w in words | set(extra)}))


def mentioned_columns(sentence: str) -> set[str]:
    """Which features a sentence refers to, by their words, longest match first.

    Longest first so that "l7 protocol" is read as the application protocol and
    not also as the transport protocol, and "destination port range" not also as
    the destination port. Each matched span is blanked before shorter aliases
    are tried.
    """
    s = sentence.lower()
    found = set()
    pairs = sorted([(a, col) for col in ALL_COLUMNS for a in _alias(col)]
                   + [(a, "CONTEXT") for a in CONTEXT_WORDS]
                   + [(a, "TCP_WINDOW") for a in ("tcp window", "window size")],
                   key=lambda p: -len(p[0]))
    for alias, col in pairs:
        if alias in s:
            found.add(col.removesuffix("_bucket") if col.endswith("_bucket") else col)
            s = s.replace(alias, " " * len(alias))
    return found


def reasons(pack: dict, top_k: int = 5) -> list[Reason]:
    out = []
    flow = pack["flow"]
    for f in pack.get("attribution", {}).get("top_features", [])[:top_k]:
        col, val = _split_name(f["name"])
        direction = "raised" if f["direction"] == "increases_suspicion" else "lowered"
        label = LABELS.get(col, col)
        if col.endswith("_present"):
            base = col.removesuffix("_present")
            present = val not in (None, "absent", "other", "0")
            what = (f"{LABELS.get(base, base)} was recorded for this flow" if present
                    else f"{LABELS.get(base, base)} was not recorded for this flow")
            sentence = f"{direction.capitalize()} the score: {what}."
        else:
            value = val if val is not None else flow.get(col, f.get("value"))
            said = decode_value(col, value)
            proto = flow.get("PROTOCOL")
            if (col.endswith("TCP_FLAGS") and proto is not None
                    and int(float(proto)) != 6 and said == "no TCP flags"):
                said = "none, as this is not a TCP flow"
            sentence = f"{direction.capitalize()} the score: {label}, {said}."
        out.append(Reason(col.removesuffix("_present"), sentence, direction, _alias(col)))
    return out


def sections(pack: dict) -> list[Section]:
    f, d = pack["flow"], pack["detection"]
    a, c = pack.get("attribution", {}), pack.get("context", {})
    g, q = pack.get("guidance", {}), pack.get("data_quality", {}) or {}
    m, conf = margin(d), confidence(d)
    out = []

    # -- the detection -------------------------------------------------------
    s = Section("THE DETECTION")
    s.lines.append(f"The detector flagged the flow from {f.get('src_ip')} (the source) "
                   f"to {f.get('dst_ip')} (the destination).")
    s.lines.append(
        f"Its score was {num(d['score'])} against an alert threshold of "
        f"{num(d['threshold'])}: it cleared the threshold by {num(m['gap'])}, a "
        f"{m['band']} margin.")
    s.lines.append(
        f"The most likely attack category is {d['predicted_class']}, with "
        f"{conf['band']} confidence ({num(conf['value'])}).")
    if conf["alternatives"]:
        alts = ", ".join(f"{k} ({num(v)})" for k, v in conf["alternatives"])
        s.lines.append(f"Other categories were also plausible: {alts}. The category "
                       f"is therefore uncertain.")
    else:
        s.lines.append("No other category was plausible.")
    if borderline(pack):
        why = []
        if m["band"] == "narrow":
            why.append("the score only narrowly cleared the threshold")
        if conf["band"] == "low":
            why.append("the category is held with low confidence")
        if conf["alternatives"]:
            why.append("more than one category is plausible")
        s.lines.append(f"This is a borderline detection: {'; '.join(why)}. "
                       f"It should be reported as such.")
    else:
        s.lines.append(f"This is a clear detection: a {m['band']} margin and a "
                       f"category held with {conf['band']} confidence.")
    out.append(s)

    # -- the flow --------------------------------------------------------------
    s = Section("THE FLOW, AS MEASURED")
    proto = f.get("PROTOCOL")
    if proto is not None:
        s.lines.append(f"Transport protocol: {protocol(proto)}.")
    if f.get("L7_PROTO") is not None:
        s.lines.append(f"Application protocol: {app_protocol(f['L7_PROTO'])}.")
    if f.get("L4_SRC_PORT") is not None and f.get("L4_DST_PORT") is not None:
        s.lines.append(f"Source port {port(f['L4_SRC_PORT'])}; destination port "
                       f"{port(f['L4_DST_PORT'])}.")
    for d_name, b, p in (("source to destination", "IN_BYTES", "IN_PKTS"),
                         ("destination back to source", "OUT_BYTES", "OUT_PKTS")):
        if f.get(b) is None and f.get(p) is None:
            continue
        line = (f"From {d_name}: {num(f.get(b))} bytes in {num(f.get(p))} "
                f"packet{'s' if f.get(p) != 1 else ''}")
        if f.get(p):
            line += f", an average of {num(round(float(f[b]) / float(f[p])))} bytes per packet"
        s.lines.append(line + ".")
    for col, who in (("CLIENT_TCP_FLAGS", "The source"), ("SERVER_TCP_FLAGS", "The destination")):
        if f.get(col) is not None and proto is not None and int(float(proto)) == 6:
            s.lines.append(f"{who} sent these TCP flags: {tcp_flags(f[col])}.")
    if f.get("LONGEST_FLOW_PKT") is not None and f.get("SHORTEST_FLOW_PKT") is not None:
        lo_, hi_ = f["SHORTEST_FLOW_PKT"], f["LONGEST_FLOW_PKT"]
        s.lines.append(f"Every packet was {num(lo_)} bytes." if lo_ == hi_ else
                       f"Packet sizes ranged from {num(lo_)} to {num(hi_)} bytes.")
    for col in ("ICMP_TYPE", "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE"):
        if f.get(col):
            s.lines.append(f"{LABELS[col][0].upper()}{LABELS[col][1:]}: "
                           f"{decode_value(col, f[col])}.")
    for name in f.get("_values_at_measurement_ceiling", []):
        s.lines.append(f"{LABELS.get(name, name)} is a lower bound: the true value "
                       f"is at least the figure given.")
    out.append(s)

    # -- reasons ---------------------------------------------------------------
    s = Section("WHY THE DETECTOR FLAGGED IT (these are the reasons for the score)")
    rs = reasons(pack)
    # Raising reasons first, each its own line, its direction leading.
    s.lines.extend(r.sentence for r in rs if r.direction == "raised")
    s.lines.extend(r.sentence for r in rs if r.direction == "lowered")
    w = a.get("channel_weights")
    if w:
        own = round(100 * float(w["attribute"]))
        s.lines.append(f"About {own}% of this decision rested on the flow's own "
                       f"measurements and {100 - own}% on the activity of the hosts "
                       f"around it.")
    for n in a.get("influential_neighbours", [])[:3]:
        s.lines.append(f"A related flow, {n['src_ip']} to {n['dst_ip']}, also affected "
                       f"the score: {n['summary']}.")
    if a.get("neighbour_note"):
        s.lines.append(a["neighbour_note"])
    out.append(s)

    # -- context ----------------------------------------------------------------
    s = Section("BACKGROUND CONTEXT (what the two hosts were doing in the same period; "
                "context, not a reason for the score)")
    src, dst = c.get("src_host"), c.get("dst_host")
    if src:
        times = src.get("baseline_deviation", {}).get("times_typical")
        s.lines.append(
            f"The source host started {num(src['flow_count'])} "
            f"{plural(src['flow_count'], 'flow')} in this period, to "
            f"{num(src['distinct_dst_ips'])} distinct "
            f"{plural(src['distinct_dst_ips'], 'address', 'addresses')} on "
            f"{num(src['distinct_dst_ports'])} distinct "
            f"{plural(src['distinct_dst_ports'], 'port')}"
            + (f": {times_phrase(times, 'distinct addresses')}." if times else "."))
    if dst:
        times = dst.get("baseline_deviation", {}).get("times_typical")
        # The destination's counts are inbound (B15): flows it received and the
        # distinct hosts they came from.
        s.lines.append(
            f"The destination host received {num(dst['flow_count'])} "
            f"{plural(dst['flow_count'], 'flow')} in this period, from "
            f"{num(dst['distinct_dst_ips'])} distinct "
            f"{plural(dst['distinct_dst_ips'], 'address', 'addresses')}, on "
            f"{num(dst['distinct_dst_ports'])} distinct "
            f"{plural(dst['distinct_dst_ports'], 'port')} of its own"
            + (f": {times_phrase(times, 'distinct sources')}." if times else "."))
    if c.get("window_flow_count"):
        s.lines.append(f"The period covers {num(c['window_flow_count'])} flows among "
                       f"{num(c.get('window_host_count'))} hosts.")
    out.append(s)

    # -- data quality -------------------------------------------------------------
    s = Section("WHAT IS NOT KNOWN")
    for note in q.get("notes", []):
        s.lines.append(note)
    if not s.lines:
        s.lines.append("Nothing beyond what is stated above is known about this flow.")
    out.append(s)

    # -- guidance -------------------------------------------------------------------
    fam = g.get("attack_family")
    s = Section(f"REFERENCE GUIDANCE FOR {str(fam).upper()} (a fixed lookup by "
                f"category, not specific to this flow)")
    if g.get("description"):
        s.lines.append(f"Description: {g['description']}")
    if g.get("mitre_techniques"):
        s.lines.append("Associated MITRE ATT&CK technique identifiers: "
                       + ", ".join(g["mitre_techniques"]) + ".")
    for mit in g.get("standard_mitigations", []):
        s.lines.append(f"Standard mitigation: {mit}.")
    out.append(s)
    return out


def render(pack: dict) -> str:
    """The facts, as the prompt shows them: titled sections of plain sentences."""
    blocks = []
    for s in sections(pack):
        if s.lines:
            blocks.append(s.title + "\n" + "\n".join(f"- {x}" for x in s.lines))
    return "\n\n".join(blocks)
