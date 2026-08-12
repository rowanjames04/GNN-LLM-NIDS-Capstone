"""Measure whether the graph actually carries structure a GNN can exploit.

Commitment V2 in the Obsidian note "Vision Integrity" is that the GNN detects
intrusions using topology a flat model cannot see. The aggregate figures from
verify_dataset.py put that at risk: 44 hosts, every one with a median AND
maximum of 10 peers. If the adjacency is effectively fixed, message passing
degenerates into a fixed linear mixing of per-host feature aggregates -- which
is reachable by an afternoon of feature engineering, and the GNN adds nothing.

But "the adjacency is fixed" was an inference from aggregate numbers, not a
measurement. This script measures it, and tests whether a different choice of
node fixes it.

Four questions:

  D1. Does the per-window adjacency actually vary? Aggregate density is over
      all 2.4M flows; within one window only a subset of pairs is active, and
      that subset may vary a great deal. Measured by Jaccard similarity between
      consecutive windows' edge sets -- low similarity means real variation.

  D2. How many distinct (dst_ip, dst_port) services exist? Determines whether
      the host->service reformulation is viable at all.

  D3. Does per-host fan-out vary between windows? Fixed degree means no signal;
      a host at degree 40 in one window and 3 in another is exactly what
      message passing exists to exploit.

  D4. THE DECIDING TEST. Do windows containing attacks differ *structurally*
      from benign-only windows? If structure carries no signal about attacks,
      no amount of graph machinery will help. Run under both node schemes so
      the comparison is like-for-like.

Writes results/metrics/topology_diagnostics.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "interim" / "NF-UNSW-NB15-v2.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "metrics" / "topology_diagnostics.json"

SRC, DST, DPORT = "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_DST_PORT"


def summarise(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {}
    return {
        "mean": round(float(np.mean(values)), 3),
        "std": round(float(np.std(values)), 3),
        "min": round(float(np.min(values)), 3),
        "median": round(float(np.median(values)), 3),
        "max": round(float(np.max(values)), 3),
        # Coefficient of variation is the headline: it is scale-free, so it
        # answers "does this vary?" independently of how big it is.
        "cv": round(float(np.std(values) / np.mean(values)), 3) if np.mean(values) else None,
    }


def window_slices(n: int, size: int):
    for start in range(0, n - size + 1, size):
        yield start, start + size


def analyse_windows(df: pd.DataFrame, size: int, scheme: str) -> dict:
    """Per-window structural statistics under one node scheme.

    scheme 'ip'      -> nodes are IP addresses on both sides (current design)
    scheme 'service' -> source nodes are hosts, destination nodes are
                        (ip, port) services. Bipartite host->service.
    """
    src = df[SRC].to_numpy()
    dst = (
        df[DST].to_numpy()
        if scheme == "ip"
        else (df[DST].astype(str) + ":" + df[DPORT].astype(str)).to_numpy()
    )
    is_attack = df["Label"].to_numpy().astype(bool)

    n_nodes, n_edges, densities, max_deg, deg_cv = [], [], [], [], []
    jaccards, prev_edges = [], None
    attack_flags, fanout_max = [], []

    for lo, hi in window_slices(len(df), size):
        s, d = src[lo:hi], dst[lo:hi]
        edges = set(zip(s.tolist(), d.tolist()))

        nodes = len(set(s.tolist()) | set(d.tolist()))
        n_nodes.append(nodes)
        n_edges.append(len(edges))
        possible = nodes * (nodes - 1)
        densities.append(len(edges) / possible if possible else 0.0)

        # Out-degree = distinct peers per source host in this window.
        out_deg = pd.Series(d).groupby(pd.Series(s)).nunique().to_numpy()
        max_deg.append(int(out_deg.max()))
        fanout_max.append(int(out_deg.max()))
        deg_cv.append(float(out_deg.std() / out_deg.mean()) if out_deg.mean() else 0.0)

        if prev_edges is not None:
            union = len(edges | prev_edges)
            jaccards.append(len(edges & prev_edges) / union if union else 1.0)
        prev_edges = edges

        attack_flags.append(bool(is_attack[lo:hi].any()))

    attack_flags = np.array(attack_flags)
    fanout_max = np.array(fanout_max)
    n_edges_arr = np.array(n_edges)

    result = {
        "window_size": size,
        "n_windows": len(n_nodes),
        "nodes_per_window": summarise(np.array(n_nodes)),
        "edges_per_window": summarise(n_edges_arr),
        "density_per_window": summarise(np.array(densities)),
        "max_out_degree_per_window": summarise(np.array(max_deg)),
        "out_degree_cv_within_window": summarise(np.array(deg_cv)),
        "jaccard_consecutive_windows": summarise(np.array(jaccards)),
        "windows_containing_attack": int(attack_flags.sum()),
    }

    # D4: do attack-bearing windows look structurally different?
    if attack_flags.any() and (~attack_flags).any():
        result["structural_separation"] = {
            "max_fanout_attack_windows": summarise(fanout_max[attack_flags]),
            "max_fanout_benign_windows": summarise(fanout_max[~attack_flags]),
            "edges_attack_windows": summarise(n_edges_arr[attack_flags]),
            "edges_benign_windows": summarise(n_edges_arr[~attack_flags]),
        }
    else:
        result["structural_separation"] = (
            "not computable: all windows contain attacks"
            if attack_flags.all()
            else "not computable: no window contains an attack"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--windows", type=int, nargs="+", default=[1_000, 10_000, 50_000])
    args = parser.parse_args()

    print(f"Loading {args.input.name} ...")
    df = pd.read_parquet(args.input, columns=[SRC, DST, DPORT, "Label", "Attack"])
    print(f"  {len(df):,} flows\n")

    # D2 -- is the service reformulation even viable?
    services = (df[DST].astype(str) + ":" + df[DPORT].astype(str)).nunique()
    node_counts = {
        "unique_src_ips": int(df[SRC].nunique()),
        "unique_dst_ips": int(df[DST].nunique()),
        "unique_dst_ports": int(df[DPORT].nunique()),
        "unique_services_dst_ip_port": int(services),
    }
    print("=== D2: node counts under each scheme ===")
    print(f"  IP scheme      : {node_counts['unique_dst_ips']} destination nodes")
    print(f"  Service scheme : {node_counts['unique_services_dst_ip_port']:,} "
          f"destination nodes  ({node_counts['unique_dst_ports']:,} distinct ports)")
    factor = services / max(node_counts["unique_dst_ips"], 1)
    print(f"  => service reformulation multiplies destination nodes by {factor:,.0f}x")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(args.input.relative_to(REPO_ROOT)),
        "node_counts": node_counts,
        "schemes": {},
    }

    for scheme in ("ip", "service"):
        report["schemes"][scheme] = {}
        print(f"\n{'=' * 66}\n  SCHEME: {scheme}\n{'=' * 66}")
        for size in args.windows:
            r = analyse_windows(df, size, scheme)
            report["schemes"][scheme][str(size)] = r

            print(f"\n  window = {size:,} flows  ({r['n_windows']} windows)")
            print(f"    nodes/window      {r['nodes_per_window']['mean']:>10,.1f} "
                  f"(cv {r['nodes_per_window']['cv']})")
            print(f"    edges/window      {r['edges_per_window']['mean']:>10,.1f} "
                  f"(cv {r['edges_per_window']['cv']})")
            print(f"    density/window    {r['density_per_window']['mean']:>10.4f} "
                  f"(cv {r['density_per_window']['cv']})")
            print(f"    max fan-out       {r['max_out_degree_per_window']['mean']:>10,.1f} "
                  f"(cv {r['max_out_degree_per_window']['cv']}, "
                  f"range {r['max_out_degree_per_window']['min']:.0f}"
                  f"-{r['max_out_degree_per_window']['max']:.0f})")
            j = r["jaccard_consecutive_windows"]
            if j:
                print(f"    Jaccard vs prev   {j['mean']:>10.3f}  "
                      f"(1.0 = identical adjacency, 0 = totally different)")
            sep = r["structural_separation"]
            if isinstance(sep, dict):
                a = sep["max_fanout_attack_windows"]["mean"]
                b = sep["max_fanout_benign_windows"]["mean"]
                print(f"    fan-out attack vs benign windows: {a:,.1f} vs {b:,.1f}")
            else:
                print(f"    structural separation: {sep}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nWritten to {args.output.relative_to(REPO_ROOT)}\n")


if __name__ == "__main__":
    main()
