"""Physical-validity checks on measured NetFlow values (A8, D47).

One definition, used in two places: the data-quality audit reports how often
each check fails, and preprocessing uses the same checks to decide which values
cannot be trusted. Keeping them in one module means the audit and the model can
never disagree about what "corrupt" means.

Every function takes the measured values as a dict of float64 arrays keyed by
the NetFlow column name (`raw_values.npy`, D41), so nothing here depends on the
model's scaled inputs.
"""

from __future__ import annotations

import numpy as np

WRAP_32_MS = 2 ** 32 / 1000.0           # 4,294,967.296 ms

# Columns that are computed from the flow duration by the exporter. When the
# duration is wrapped they are computed from a wrong denominator (and the
# 1.9e219 bytes/s seen on NF-ToN-IoT-v2 is the other end of the same fault: a
# near-zero denominator).
DURATION_DERIVED = (
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
)
DURATION_COLUMNS = ("FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT")


def wrapped_duration(col: dict[str, np.ndarray], tolerance_ms: float = 2.0) -> np.ndarray:
    """FLOW_DURATION_MILLISECONDS is a 32-bit wrap of a small negative interval.

    The exporter computes last-seen minus first-seen in unsigned 32-bit
    microseconds and divides by 1000. When the two timestamps arrive out of
    order the difference is negative and wraps to just under 2^32 us, so the
    recorded duration is 2^32/1000 minus the true gap. The signature measured on
    both datasets is that the duration plus DURATION_OUT (or DURATION_IN) equals
    2^32/1000 to within a millisecond or two; a long genuine flow does not
    satisfy it.
    """
    fd = col["FLOW_DURATION_MILLISECONDS"]
    sig = np.zeros(len(fd), dtype=bool)
    for other in ("DURATION_OUT", "DURATION_IN"):
        if other in col:
            sig |= np.abs(fd + col[other] - WRAP_32_MS) <= tolerance_ms
    # Anything in the last 30 s below the wrap point with no matching partner is
    # still not a plausible flow length for these captures; flag it too.
    sig |= (fd > WRAP_32_MS - 30_000) & (fd <= WRAP_32_MS + 1)
    return sig


def validity_flags(col: dict[str, np.ndarray], protocol: np.ndarray | None = None
                   ) -> dict[str, np.ndarray]:
    """Boolean masks, True where a physical relation is violated.

    Only relations the NetFlow definitions guarantee are tested. A flag here is
    a statement that the recorded value cannot describe a real flow, not that
    the flow is unusual.
    """
    n = len(next(iter(col.values())))
    f: dict[str, np.ndarray] = {}
    g = col.get

    if "FLOW_DURATION_MILLISECONDS" in col:
        f["duration_wrapped"] = wrapped_duration(col)
    for c in DURATION_COLUMNS[1:]:
        if c in col and "FLOW_DURATION_MILLISECONDS" in col:
            # A per-direction duration longer than the flow (allowing 1 ms).
            f[f"{c.lower()}_exceeds_flow"] = (
                (col[c] > col["FLOW_DURATION_MILLISECONDS"] + 1)
                & ~f["duration_wrapped"])

    if g("IN_PKTS") is not None and g("OUT_PKTS") is not None:
        total = col["IN_PKTS"] + col["OUT_PKTS"]
        buckets = [c for c in col if c.startswith("NUM_PKTS_")]
        if buckets:
            # The size buckets partition every packet, so they sum to the total.
            # A small tolerance absorbs the exporter's handling of the boundary.
            s = np.sum([col[c] for c in buckets], axis=0)
            f["size_buckets_not_packet_total"] = np.abs(s - total) > np.maximum(1, 0.01 * total)
        for d, b in (("IN", "IN_BYTES"), ("OUT", "OUT_BYTES")):
            pk = col[f"{d}_PKTS"]
            if b in col:
                f[f"{d.lower()}_bytes_without_packets"] = (col[b] > 0) & (pk == 0)
                # An IPv4 header alone is 20 bytes.
                f[f"{d.lower()}_bytes_below_20_per_packet"] = (pk > 0) & (col[b] < 20 * pk)
            rb, rp = f"RETRANSMITTED_{d}_BYTES", f"RETRANSMITTED_{d}_PKTS"
            if rp in col:
                f[f"{d.lower()}_retransmitted_exceeds_total"] = col[rp] > pk
            if rb in col and b in col:
                f[f"{d.lower()}_retransmitted_bytes_exceed_total"] = col[rb] > col[b]

    if g("LONGEST_FLOW_PKT") is not None and g("SHORTEST_FLOW_PKT") is not None:
        f["longest_below_shortest"] = col["LONGEST_FLOW_PKT"] < col["SHORTEST_FLOW_PKT"]
    if g("MAX_IP_PKT_LEN") is not None:
        f["ip_packet_over_65535"] = col["MAX_IP_PKT_LEN"] > 65535
    for c in ("L4_SRC_PORT", "L4_DST_PORT"):
        if c in col:
            f[f"{c.lower()}_out_of_range"] = (col[c] < 0) | (col[c] > 65535)
    for c in ("TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"):
        if c in col:
            # The TCP header field is 16 bits; scaling is not applied by NetFlow.
            f[f"{c.lower()}_over_65535"] = col[c] > 65535
    for c in DURATION_DERIVED:
        if c in col:
            f[f"{c.lower()}_non_finite_or_negative"] = ~np.isfinite(col[c]) | (col[c] < 0)

    return {k: v.astype(bool) for k, v in f.items() if len(v) == n}
