"""The neighbourhood of one flagged flow, as a small graph for the demo.

The project's claim is that a flow is easier to judge in the light of what its
two hosts are doing. This module makes that neighbourhood visible: the flagged
flow, its two endpoints, and every other host either of them talked to in the
same window.

**It draws the same neighbourhood the model used.** "Touching" here is the same
definition `make_evidence.py` uses when it occludes neighbouring flows: any flow
that shares a host with the flagged one. So the picture is of the model's input,
not an illustration beside it.

A busy host talks to thousands of others, which no diagram can show. Flows are
therefore pooled per host pair and only the busiest neighbours are drawn; the
counts of what was left out are returned and shown, because a diagram that
silently drops most of a neighbourhood misrepresents how dense it is.

Pure functions over arrays, with no Streamlit and no torch, so the logic is
tested without a browser. `figure` is the only part that touches plotly.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

ORANGE, BLUE, NEUTRAL, EDGE = "#eb6834", "#2a78d6", "#8d8c86", "#b9b8b2"


def window_bounds(row_index: int, split_start: int, split_stop: int,
                  window: int) -> tuple[int, int]:
    """The window of `window` consecutive rows that contains `row_index`.

    Windows are laid end to end from the start of the split, which is how
    `SnapshotDataset` builds them, so the window is recoverable from the row
    alone. (A pack's `window_index` is not enough: it indexes an evenly spaced
    *subsample* of windows, not the full list.)
    """
    if not split_start <= row_index < split_stop:
        raise ValueError(f"row {row_index} is outside the split "
                         f"[{split_start}, {split_stop})")
    lo = split_start + ((row_index - split_start) // window) * window
    return lo, min(lo + window, split_stop)


def ego_graph(src: np.ndarray, dst: np.ndarray, target: int,
              max_hosts: int = 28) -> dict:
    """Hosts and pooled flows around the flow at position `target`.

    `src` and `dst` are the host identifiers of every flow in one window. The
    result keeps the two endpoints of the flagged flow and up to `max_hosts`
    of the hosts they exchanged the most flows with.
    """
    src, dst = np.asarray(src), np.asarray(dst)
    a, b = src[target], dst[target]
    touching = (src == a) | (dst == a) | (src == b) | (dst == b)
    touching[target] = False

    pairs = Counter(zip(src[touching].tolist(), dst[touching].tolist()))
    volume: Counter = Counter()
    for (s, d), n in pairs.items():
        for host in (s, d):
            if host not in (a, b):
                volume[host] += n

    shown = {h for h, _ in volume.most_common(max_hosts)}
    side: dict = {}
    for (s, d) in pairs:
        for host, other in ((s, d), (d, s)):
            if host in shown and other in (a, b):
                which = "source" if other == a else "destination"
                side[host] = which if side.get(host, which) == which else "both"

    nodes = [{"id": a, "role": "source"}]
    if b != a:
        nodes.append({"id": b, "role": "destination"})
    nodes += [{"id": h, "role": "neighbour", "side": side.get(h, "both"),
               "flows": volume[h]} for h in sorted(shown, key=lambda h: -volume[h])]

    keep = shown | {a, b}
    edges = [{"from": s, "to": d, "flows": n}
             for (s, d), n in pairs.items() if s in keep and d in keep]
    return {
        "nodes": nodes, "edges": edges, "flagged": {"from": a, "to": b},
        "neighbours_sharing_a_host": int(touching.sum()),
        "hosts_in_neighbourhood": len(volume),
        "hosts_shown": len(shown),
        "flows_between_shown_hosts": sum(e["flows"] for e in edges),
        "window_flows": int(len(src)),
    }


def layout(graph: dict) -> dict:
    """Positions: the two endpoints in the middle, their neighbours around them.

    Hosts that talk only to the source fan out on the left, those that talk only
    to the destination on the right, and hosts connected to both sit between.
    Deterministic, so the picture is the same on every rerun.
    """
    nodes = graph["nodes"]
    pos = {nodes[0]["id"]: (-1.0, 0.0)}
    if len(nodes) > 1 and nodes[1]["role"] == "destination":
        pos[nodes[1]["id"]] = (1.0, 0.0)

    groups: dict[str, list] = {"source": [], "destination": [], "both": []}
    for n in nodes:
        if n["role"] == "neighbour":
            groups[n["side"]].append(n["id"])

    def fan(ids: list, centre: float, start: float, sweep: float) -> None:
        for i, host in enumerate(ids):
            t = start + sweep * ((i + 0.5) / len(ids))
            # Alternate two radii so labels on neighbouring spokes do not touch.
            r = 1.25 if i % 2 == 0 else 1.75
            pos[host] = (centre + r * math.cos(t), r * math.sin(t))

    fan(groups["source"], -1.0, math.radians(100), math.radians(160))
    fan(groups["destination"], 1.0, math.radians(-80), math.radians(160))
    for i, host in enumerate(groups["both"]):
        n = len(groups["both"])
        y = 0.0 if n == 1 else 1.6 - 3.2 * i / (n - 1)
        pos[host] = (0.0, y if abs(y) > 0.25 else (0.45 if i % 2 == 0 else -0.45))
    return pos


def figure(graph: dict, names: dict | None = None):
    """A plotly figure of the neighbourhood. `names` maps host id -> label."""
    import plotly.graph_objects as go

    names = names or {}
    label = lambda h: str(names.get(h, h))                          # noqa: E731
    pos = layout(graph)
    fig = go.Figure()

    heaviest = max((e["flows"] for e in graph["edges"]), default=1)
    for e in graph["edges"]:
        (x0, y0), (x1, y1) = pos[e["from"]], pos[e["to"]]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
            line={"color": EDGE,
                  "width": 0.8 + 3.2 * math.log1p(e["flows"]) / math.log1p(heaviest)}))
    # Hover targets at edge midpoints: a 1px line is not something to aim at.
    fig.add_trace(go.Scatter(
        x=[(pos[e["from"]][0] + pos[e["to"]][0]) / 2 for e in graph["edges"]],
        y=[(pos[e["from"]][1] + pos[e["to"]][1]) / 2 for e in graph["edges"]],
        mode="markers", marker={"size": 14, "color": "rgba(0,0,0,0)"},
        showlegend=False, hovertemplate="%{text}<extra></extra>",
        text=[f"{label(e['from'])} to {label(e['to'])}<br>{e['flows']:,} flows "
              f"in this window" for e in graph["edges"]]))

    f = graph["flagged"]
    (x0, y0), (x1, y1) = pos[f["from"]], pos[f["to"]]
    fig.add_trace(go.Scatter(
        x=[x0, x1], y=[y0, y1], mode="lines", name="The flagged flow",
        line={"color": ORANGE, "width": 4}, hoverinfo="skip"))

    for role, colour, size, name in (
            ("neighbour", NEUTRAL, 11, "Other hosts they exchanged flows with"),
            ("source", BLUE, 20, "The flagged flow's two hosts"),
            ("destination", BLUE, 20, None)):
        group = [n for n in graph["nodes"] if n["role"] == role]
        if not group:
            continue
        focus = role != "neighbour"
        fig.add_trace(go.Scatter(
            x=[pos[n["id"]][0] for n in group], y=[pos[n["id"]][1] for n in group],
            mode="markers+text" if focus else "markers",
            name=name, showlegend=name is not None,
            marker={"size": size, "color": colour,
                    "line": {"width": 2, "color": "rgba(255,255,255,0.9)"}},
            text=[f"{label(n['id'])}<br>({role})" for n in group] if focus else None,
            textposition="bottom center",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=[label(n["id"]) + (f"<br>{n['flows']:,} flows with the "
                                          f"flagged hosts" if not focus else
                                          f"<br>{role} of the flagged flow")
                        for n in group]))

    fig.update_layout(
        height=430, margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"visible": False}, yaxis={"visible": False, "scaleanchor": "x"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
        hoverlabel={"align": "left"})
    return fig
