"""Hand-crafted per-host aggregates, computed within each window.

These exist to make the Phase 3 comparison decisive rather than merely
favourable. The obvious baseline -- an MLP on flow features -- has no access to
context at all, so a GNN beating it proves only that *some* context helps, not
that *learned message passing* was needed to get it.

These features are what a single round of message passing would aggregate,
written out by hand: how many distinct peers this host is talking to, how many
ports, how much traffic, how diverse its protocol mix. Phase 0 measured that
fan-out alone predicts an attacking host at AUC 0.749, so this is not a straw
man -- it is the strongest cheap alternative to the GNN.

The three-way reading:

    MLP(flow)                 what a row-wise classifier can do
    MLP(flow + these)         what context is worth, hand-crafted
    GNN                       what learned message passing adds beyond that

If the second and third are close, the GNN is an expensive `groupby` and the
report should say so. Commitment V2 in the Obsidian note "Vision Integrity"
turns on this number.

Computed per window so that a host's profile is local and time-bounded, exactly
as it is inside a graph snapshot. Computing them over the whole dataset would
leak future behaviour into past rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SRC, DST, DPORT, PROTO = "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_DST_PORT", "PROTOCOL"

FEATURE_NAMES = [
    "src_fanout",           # distinct destination hosts -- the AUC 0.749 signal
    "src_distinct_ports",   # distinct destination ports; scanning drives this up
    "src_flow_count",
    "src_distinct_proto",
    "src_bytes_total",
    "src_mean_duration",
    "dst_fanin",            # distinct sources contacting this host; DoS drives it
    "dst_flow_count",
    "src_ports_per_peer",   # ports touched per host reached -- scan vs. session
]


def windowed_host_features(
    df: pd.DataFrame, window_size: int, byte_col: str = "IN_BYTES",
    duration_col: str = "FLOW_DURATION_MILLISECONDS",
) -> np.ndarray:
    """Per-flow array of its endpoints' behaviour within the flow's own window.

    Returns raw (unscaled) values in FEATURE_NAMES order. Scaling is the
    caller's job, so that it can be fitted on the training split only.
    """
    n = len(df)
    win = np.arange(n) // window_size

    work = pd.DataFrame({
        "win": win,
        "src": df[SRC].to_numpy(),
        "dst": df[DST].to_numpy(),
        "port": df[DPORT].to_numpy(),
        "proto": df[PROTO].to_numpy(),
        "bytes": df[byte_col].to_numpy(),
        "dur": df[duration_col].to_numpy(),
    })

    src_agg = work.groupby(["win", "src"], observed=True, sort=False).agg(
        src_fanout=("dst", "nunique"),
        src_distinct_ports=("port", "nunique"),
        src_flow_count=("dst", "size"),
        src_distinct_proto=("proto", "nunique"),
        src_bytes_total=("bytes", "sum"),
        src_mean_duration=("dur", "mean"),
    ).reset_index()

    dst_agg = work.groupby(["win", "dst"], observed=True, sort=False).agg(
        dst_fanin=("src", "nunique"),
        dst_flow_count=("src", "size"),
    ).reset_index()

    merged = (
        work[["win", "src", "dst"]]
        .merge(src_agg, on=["win", "src"], how="left")
        .merge(dst_agg, on=["win", "dst"], how="left")
    )

    # Ports per peer separates a scan from a normal session: contacting three
    # hosts on one port each looks very different from contacting three hosts on
    # two hundred ports, even though fan-out is identical.
    merged["src_ports_per_peer"] = (
        merged["src_distinct_ports"] / merged["src_fanout"].clip(lower=1)
    )

    return merged[FEATURE_NAMES].to_numpy(dtype=np.float32)


def standardise(
    x: np.ndarray, fit_slice: slice, log_first: bool = True
) -> tuple[np.ndarray, dict]:
    """Log-compress then z-score, with statistics from `fit_slice` only.

    Counts here are as heavy-tailed as the raw NetFlow volumes -- a scanning
    host's flow count is orders of magnitude above a normal one -- so the same
    log treatment applies, and the same train-only rule (D5) applies to the
    statistics.
    """
    x = np.log1p(np.maximum(x, 0)) if log_first else x.copy()
    fit = x[fit_slice]
    mean = fit.mean(axis=0)
    std = fit.std(axis=0)
    std[std < 1e-12] = 1.0
    return ((x - mean) / std).astype(np.float32), {
        "mean": mean.tolist(), "std": std.tolist(), "log_first": log_first,
    }
