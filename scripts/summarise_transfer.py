"""Collate every cross-dataset transfer result into one table (Phase 9).

Reads `results/metrics/transfer/*.json` -- both the GNN arm (`transfer_eval.py`)
and the tree arm (`transfer_baselines.py`) -- and prints the comparison the
Discussion section needs.

Written because the headline claim rests on **six numbers produced by two
different scripts across four runs**, and transcribing those by hand into a
report is exactly how C17 happened. Every figure quoted about transfer should
come out of here.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSFER = REPO_ROOT / "results" / "metrics" / "transfer"


def rows() -> list[dict]:
    """One row per (model, source, target), pooling per-seed files.

    The GNN arm writes one file per run and overwrites a shared name, so seeds
    are kept as `..._seed{N}.json` and must be pooled here rather than listed
    separately -- otherwise the same pair appears three times and a reader
    averages it by eye, wrongly.
    """
    groups: dict[tuple, list[dict]] = {}
    for f in sorted(TRANSFER.glob("transfer_*.json")):
        if f.name.startswith("smoke_"):
            continue
        d = json.loads(f.read_text())
        groups.setdefault((d.get("model", "gnn::full"),
                           d["source_dataset"], d["target_dataset"]), []).append(d)

    out = []
    for (model, src, tgt), docs in groups.items():
        d = docs[0]
        prev = d.get("reported_at_prevalence", 0.04)
        key = f"at_{prev:.0%}"
        oov = d.get("out_of_vocabulary", {}).get("_worst_oov_rate")
        vals = []
        for doc in docs:
            if "runs" in doc:                             # tree arm: seeds inside
                vals += [r["source_threshold"][key] for r in doc["runs"]]
            else:                                         # GNN arm: one per file
                vals.append(doc["results"]["source_threshold"][key])
        pr = [v["pr_auc"] for v in vals if isinstance(v.get("pr_auc"), (int, float))]
        f1 = [v["f1"] for v in vals if isinstance(v.get("f1"), (int, float))]
        rc = [v["recall"] for v in vals if isinstance(v.get("recall"), (int, float))]
        out.append({
            "files": len(docs), "model": model, "source": src, "target": tgt,
            "control": src == tgt, "n": len(pr), "oov": oov, "floor": prev,
            "pr_auc": st.mean(pr) if pr else None,
            "pr_std": st.stdev(pr) if len(pr) > 1 else 0.0,
            "f1": st.mean(f1) if f1 else None,
            "recall": st.mean(rc) if rc else None,
        })
    return out


def main() -> None:
    data = rows()
    if not data:
        raise SystemExit(f"no transfer results in {TRANSFER}")

    short = lambda s: s.replace("NF-", "").replace("-v2", "")
    print("=" * 100)
    print("  PHASE 9 -- CROSS-DATASET TRANSFER, all runs")
    print("=" * 100)
    print(f"  {'model':<12} {'source -> target':<26} {'n':>2} {'PR-AUC':>16} "
          f"{'F1':>8} {'recall':>8} {'OOV':>7}  note")
    for r in sorted(data, key=lambda x: (not x["control"], x["model"], x["source"])):
        pair = f"{short(r['source'])} -> {short(r['target'])}"
        note = []
        if r["control"]:
            note.append("CONTROL")
        if r["oov"] is not None and r["oov"] >= 0.5:
            note.append("HIGH OOV -- degraded input, not just a new network")
        if r["pr_auc"] is not None and not r["control"]:
            if r["pr_auc"] < r["floor"]:
                note.append("BELOW FLOOR")
            elif r["pr_auc"] < r["floor"] * 1.5:
                note.append("at floor")
        pr = (f"{r['pr_auc']:.4f} +/-{r['pr_std']:.4f}" if r["pr_auc"] is not None
              else "n/a")
        print(f"  {r['model']:<12} {pair:<26} {r['n']:>2} {pr:>16} "
              f"{r['f1']:>8.4f} {r['recall']:>8.4f} "
              f"{(r['oov'] if r['oov'] is not None else 0):>6.1%}  {'; '.join(note)}")

    floor = data[0]["floor"]
    print(f"\n  PR-AUC floor at {floor:.0%} prevalence: {floor:.2f}")

    controls = [r for r in data if r["control"] and r["pr_auc"]]
    xfers = [r for r in data if not r["control"] and r["pr_auc"]]
    if controls and xfers:
        print(f"\n  in-dataset controls:  {min(r['pr_auc'] for r in controls):.4f} "
              f"- {max(r['pr_auc'] for r in controls):.4f}")
        print(f"  cross-dataset:        {min(r['pr_auc'] for r in xfers):.4f} "
              f"- {max(r['pr_auc'] for r in xfers):.4f}")
        both_collapse = all(r["pr_auc"] < floor * 3 for r in xfers)
        models = {r["model"] for r in xfers}
        print()
        if both_collapse and len(models) > 1:
            print("  VERDICT: every model collapses to the floor across datasets while")
            print("  every control is healthy. The finding is about the DATASETS, not")
            print("  about any one architecture.")
        elif len(models) > 1:
            print("  VERDICT: models differ across datasets -- check whether the")
            print("  ordering is consistent across BOTH directions before concluding")
            print("  anything about the architecture.")
    print()


if __name__ == "__main__":
    main()
