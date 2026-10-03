"""Tests for the demo's neighbourhood view.

The view claims to show the neighbourhood the model used. The tests hold it to
that: the same definition of a neighbouring flow as the evidence packs, and no
silent dropping of hosts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnnids.ui.topology import ego_graph, figure, layout, window_bounds  # noqa: E402

#          flow:   0    1    2    3    4    5    6
SRC = np.array([10,  10,  10,  20,  30,  40,  10])
DST = np.array([20,  30,  30,  50,  20,  50,  20])
# The flagged flow is 0: host 10 -> host 20.


def test_the_window_is_recovered_from_the_row_alone():
    assert window_bounds(4_768_870, 4_720_000, 5_910_000, 10_000) == (4_760_000, 4_770_000)
    assert window_bounds(4_720_000, 4_720_000, 5_910_000, 10_000) == (4_720_000, 4_730_000)


def test_a_row_outside_the_split_is_an_error_not_a_wrong_window():
    with pytest.raises(ValueError):
        window_bounds(10, 100, 200, 10)


def test_neighbours_are_the_flows_sharing_a_host_with_the_flagged_one():
    """Same definition as the occlusion in make_evidence.py. Flow 5 (40 -> 50)
    touches neither endpoint and is not a neighbour; the flagged flow is not
    its own neighbour."""
    g = ego_graph(SRC, DST, target=0)

    assert g["neighbours_sharing_a_host"] == 5         # flows 1, 2, 3, 4, 6
    assert g["window_flows"] == 7
    assert {n["id"] for n in g["nodes"]} == {10, 20, 30, 50}


def test_flows_between_the_same_pair_are_pooled():
    g = ego_graph(SRC, DST, target=0)
    pooled = {(e["from"], e["to"]): e["flows"] for e in g["edges"]}

    assert pooled[(10, 30)] == 2                       # flows 1 and 2
    assert pooled[(10, 20)] == 1                       # flow 6, not the flagged one


def test_each_neighbour_is_placed_by_which_endpoint_it_talks_to():
    g = ego_graph(SRC, DST, target=0)
    side = {n["id"]: n.get("side") for n in g["nodes"] if n["role"] == "neighbour"}

    assert side[30] == "both"            # 10 -> 30 and 30 -> 20
    assert side[50] == "destination"     # only 20 -> 50


def test_a_busy_neighbourhood_is_capped_and_the_cap_is_reported():
    """Dropping hosts silently would make a dense neighbourhood look sparse."""
    src = np.array([1] * 60)
    dst = np.arange(100, 160)
    g = ego_graph(src, dst, target=0, max_hosts=10)

    assert g["hosts_shown"] == 10
    assert g["hosts_in_neighbourhood"] == 59
    assert g["neighbours_sharing_a_host"] == 59


def test_the_busiest_neighbours_are_the_ones_kept():
    src = np.array([1, 1, 1, 1, 1, 1])
    dst = np.array([2, 7, 7, 7, 8, 9])
    g = ego_graph(src, dst, target=0, max_hosts=1)

    assert [n["id"] for n in g["nodes"] if n["role"] == "neighbour"] == [7]


def test_layout_is_deterministic_and_places_every_host():
    g = ego_graph(SRC, DST, target=0)

    assert layout(g) == layout(g)
    assert set(layout(g)) == {n["id"] for n in g["nodes"]}
    assert layout(g)[10][0] < layout(g)[20][0]         # source left of destination


def test_the_figure_names_hosts_by_address_and_marks_the_flagged_flow():
    g = ego_graph(SRC, DST, target=0)

    fig = figure(g, {10: "192.168.1.30", 20: "192.168.1.195"})

    names = [t.name for t in fig.data if t.name]
    assert "The flagged flow" in names
    labels = " ".join(str(x) for t in fig.data if t.text is not None for x in t.text)
    assert "192.168.1.30" in labels and "192.168.1.195" in labels
