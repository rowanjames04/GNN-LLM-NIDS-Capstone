"""Aggregate the evidence packs into the explainability results (Section 6.7).

One evidence pack shows how one detection was explained. The report needs the
distribution: how decisions divide between a flow's own features and its
neighbourhood, which features carry the detections, and how much any single
neighbouring flow matters. Read off one pack, each of those is an anecdote.

Everything here is computed from files already on disk -- the packs and their
sampling record -- so it costs nothing and is reproducible from the commit.

**Read the denominators.** The packs are a stratified sample of *flagged* flows
(D33), not a random sample of traffic, and the per-family rows rest on however
many packs that family has. Both are printed beside every figure.

Usage:
    python scripts/summarise_evidence.py
    python scripts/summarise_evidence.py --packs results/evidence/smoke_evidence_NF-ToN-IoT-v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnids.llm.fidelity import base_column  # noqa: E402

TOP_SHOWN_TO_MODEL = 5      # render_facts puts this many features in the prompt
AMBIGUOUS_BELOW = 0.9       # same line as check_uncertainty_conveyed


def dist(values: list[float]) -> dict | None:
    """Five-number summary plus mean. None for an empty group, never zeros."""
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {
        "n": int(len(a)), "mean": round(float(a.mean()), 4),
        "std": round(float(a.std()), 4), "min": round(float(a.min()), 4),
        "p25": round(float(np.percentile(a, 25)), 4),
        "median": round(float(np.median(a)), 4),
        "p75": round(float(np.percentile(a, 75)), 4),
        "max": round(float(a.max()), 4),
    }


def summarise(packs: list[dict], sampling: dict | None = None,
              benign: str = "Benign") -> dict:
    by_family: dict[str, list[dict]] = defaultdict(list)
    for p in packs:
        by_family[p["flow"].get("true_label", "unknown")].append(p)
    false_pos = by_family.get(benign, [])
    true_pos = [p for p in packs if p["flow"].get("true_label") != benign]

    def topo(group):
        return [p["attribution"]["channel_weights"]["topological"]
                for p in group if p["attribution"].get("channel_weights")]

    # ---- which features carry the detections
    shown, direction, magnitude = Counter(), defaultdict(Counter), defaultdict(list)
    per_family_features: dict[str, Counter] = defaultdict(Counter)
    for p in packs:
        seen = set()
        for f in p["attribution"].get("top_features", [])[:TOP_SHOWN_TO_MODEL]:
            col = base_column(f["name"])
            magnitude[col].append(abs(f["contribution"]))
            direction[col][f["direction"]] += 1
            if col not in seen:               # count a pack once per column
                shown[col] += 1
                per_family_features[p["flow"].get("true_label", "unknown")][col] += 1
                seen.add(col)
    features = [{
        "feature": col, "packs": n, "share_of_packs": round(n / len(packs), 4),
        "mean_abs_contribution": round(float(np.mean(magnitude[col])), 4),
        "raises_suspicion": direction[col]["increases_suspicion"],
        "lowers_suspicion": direction[col]["decreases_suspicion"],
    } for col, n in shown.most_common()]

    # ---- the family named to the language model
    def named_right(group):
        return [p["detection"]["predicted_class"] == p["flow"].get("true_label")
                for p in group]
    right = named_right(true_pos)
    fp_called = Counter(p["detection"]["predicted_class"] for p in false_pos)

    out = {
        "n_packs": len(packs),
        "n_true_attacks": len(true_pos),
        "n_false_positives": len(false_pos),
        "packs_by_true_family": {k: len(v) for k, v in sorted(by_family.items())},
        "topological_weight": {
            "all": dist(topo(packs)),
            "true_attacks": dist(topo(true_pos)),
            "false_positives": dist(topo(false_pos)),
            "by_true_family": {k: dist(topo(v)) for k, v in sorted(by_family.items())},
            "share_of_packs_above_half": (
                round(float(np.mean([t > 0.5 for t in topo(packs)])), 4)
                if topo(packs) else None),
        },
        "features_shown_to_the_model": {
            "top_k": TOP_SHOWN_TO_MODEL,
            "ranked": features,
            "top3_by_true_family": {
                fam: [c for c, _ in counts.most_common(3)]
                for fam, counts in sorted(per_family_features.items())},
        },
        "family_named_to_the_model": {
            "correct_on_true_attacks": (
                round(float(np.mean(right)), 4) if right else None),
            "n_true_attacks": len(right),
            "false_positives_named_as": dict(fp_called.most_common()),
            "ambiguous_packs": sum(
                1 for p in packs
                if p["detection"]["class_confidence"] < AMBIGUOUS_BELOW),
        },
    }

    # ---- neighbour occlusion, recorded since 2026-10-03
    occ = (sampling or {}).get("neighbour_occlusion")
    if occ:
        diag = [occ["by_detection"][p["detection_id"]] for p in packs
                if p["detection_id"] in occ["by_detection"]]
        influences = [d["max_abs_influence"] for d in diag
                      if d["max_abs_influence"] is not None]
        out["neighbour_occlusion"] = {
            "threshold": occ["threshold"],
            "tested_per_detection_cap": occ["tested_per_detection_cap"],
            "packs_with_an_influential_neighbour": sum(
                1 for p in packs if p["attribution"].get("influential_neighbours")),
            "neighbours_sharing_a_host": dist(
                [d["neighbours_sharing_a_host"] for d in diag]),
            "neighbours_tested": dist([d["neighbours_tested"] for d in diag]),
            # The largest change any single tested neighbour made, per pack.
            # Reported at full precision: the finding is how small it is.
            "max_abs_influence": {
                "n": len(influences),
                "median": float(np.median(influences)) if influences else None,
                "p95": float(np.percentile(influences, 95)) if influences else None,
                "max": float(max(influences)) if influences else None,
            },
        }
    else:
        out["neighbour_occlusion"] = None      # packs predate the diagnostics
    return out


def show(s: dict) -> None:
    def row(label, d):
        if d is None:
            print(f"  {label:<22} {'(none)':>5}")
            return
        print(f"  {label:<22} {d['n']:>5} {d['mean']:>8.4f} {d['median']:>8.4f} "
              f"{d['p25']:>8.4f} {d['p75']:>8.4f} {d['min']:>8.4f} {d['max']:>8.4f}")

    print("=" * 84)
    print(f"  EXPLAINABILITY -- {s['n_packs']} evidence packs "
          f"({s['n_true_attacks']} true attacks, {s['n_false_positives']} false positives)")
    print("=" * 84)
    print("\n  Share of each decision assigned to the neighbourhood (topological weight)")
    print(f"  {'group':<22} {'n':>5} {'mean':>8} {'median':>8} {'p25':>8} {'p75':>8} "
          f"{'min':>8} {'max':>8}")
    t = s["topological_weight"]
    row("all packs", t["all"]); row("true attacks", t["true_attacks"])
    row("false positives", t["false_positives"])
    for fam, d in t["by_true_family"].items():
        row(f"  {fam}", d)
    print(f"\n  packs where the neighbourhood carries more than half: "
          f"{t['share_of_packs_above_half']}")

    f = s["features_shown_to_the_model"]
    print(f"\n  Features among the top {f['top_k']} attributed (the ones the model is shown)")
    print(f"  {'feature':<30} {'packs':>6} {'share':>7} {'mean |contrib|':>15} "
          f"{'raises':>7} {'lowers':>7}")
    for r in f["ranked"][:12]:
        print(f"  {r['feature']:<30} {r['packs']:>6} {r['share_of_packs']:>7.2f} "
              f"{r['mean_abs_contribution']:>15.4f} {r['raises_suspicion']:>7} "
              f"{r['lowers_suspicion']:>7}")

    m = s["family_named_to_the_model"]
    print(f"\n  Attack family named in the pack is the true one: "
          f"{m['correct_on_true_attacks']} of {m['n_true_attacks']} true attacks")
    print(f"  false positives were named as: {m['false_positives_named_as']}")
    print(f"  ambiguous packs (confidence < {AMBIGUOUS_BELOW}): {m['ambiguous_packs']}")

    o = s["neighbour_occlusion"]
    if o is None:
        print("\n  Neighbour occlusion: not recorded -- these packs were made before "
              "2026-10-03.\n  Regenerate them (scripts/make_evidence.py) to measure it.")
    else:
        mi = o["max_abs_influence"]
        print(f"\n  Neighbour occlusion (up to {o['tested_per_detection_cap']} tested "
              f"per detection, threshold {o['threshold']})")
        print(f"  neighbours sharing a host, median: "
              f"{o['neighbours_sharing_a_host']['median']:.0f}   tested, median: "
              f"{o['neighbours_tested']['median']:.0f}")
        print(f"  largest single-neighbour change in score: median {mi['median']:.2e}, "
              f"p95 {mi['p95']:.2e}, max {mi['max']:.2e}")
        print(f"  packs with any neighbour above the threshold: "
              f"{o['packs_with_an_influential_neighbour']} of {s['n_packs']}")
    print("\n  These packs are a stratified sample of FLAGGED flows, not of traffic."
          "\n  Attribution explains the model's decision, not why traffic is malicious.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packs", type=Path,
                    default=REPO_ROOT / "results/evidence/evidence_NF-ToN-IoT-v2.json")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "results/metrics/evidence")
    args = ap.parse_args()

    if not args.packs.exists():
        raise SystemExit(f"no evidence packs at {args.packs}. "
                         f"Run scripts/make_evidence.py first.")
    packs = json.loads(args.packs.read_text())
    side = args.packs.with_name(args.packs.stem + "_sampling.json")
    sampling = json.loads(side.read_text()) if side.exists() else None

    summary = summarise(packs, sampling)
    show(summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.packs.stem}_summary.json"
    out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "packs_source": args.packs.name,
        "checkpoint": (sampling or {}).get("checkpoint"),
        "threshold": (sampling or {}).get("threshold"),
        "summary": summary,
    }, indent=2))
    shown = out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out
    print(f"written -> {shown}\n")


if __name__ == "__main__":
    main()
