"""Physical-validity and per-family shortcut audit of the preprocessed rows (A8).

`audit_dataset.py` asks one question of each column: does it separate attack
from benign on its own? That missed the wrapped flow duration, because the
artefact is tied to attack *families* (67% of NF-ToN-IoT-v2 ddos, 3% of benign)
and its binary AUC is an unremarkable 0.58. This script asks the questions
that would have caught it, of every column, before any retraining:

  1. **Counter wraps.** Values at or near 2^32, 2^32/1000 and 2^31, and the
     specific duration signature FLOW_DURATION + DURATION_OUT = 2^32/1000.
  2. **Physical consistency.** Relations NetFlow guarantees for a real flow:
     packets by size bucket sum to the packet total, the longest packet is no
     shorter than the shortest, bytes are at least 20 per packet, a direction
     with bytes has packets, retransmissions do not exceed the total, ports and
     TCP windows fit their fields, per-direction durations fit the flow.
  3. **Per-family shortcuts.** One-vs-rest AUC of every column, and of every
     validity flag, against every family. A flag that marks one family is a
     recording fingerprint the model can learn.
  4. **Duplicates and contradictions.** Identical flows across train and test
     (leakage), and identical flows carrying different labels (label noise).

It reads `raw_values.npy` (measured values, D41) and `meta.parquet`, so it
audits exactly the rows the model trains on. Read-only: it writes one JSON to
results/metrics/ and changes nothing else.

Usage:
    python scripts/audit_data_quality.py                    # both datasets
    python scripts/audit_data_quality.py --dataset toniot
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnids.data.validity import validity_flags  # noqa: E402

DATASETS = {
    "toniot": ("NF-ToN-IoT-v2", REPO_ROOT / "data/processed/toniot"),
    "unsw": ("NF-UNSW-NB15-v2", REPO_ROOT / "data/processed"),
}
WRAP_32 = 2 ** 32
WRAP_32_MS = WRAP_32 / 1000.0          # 4,294,967.296
SHORTCUT_AUC = 0.95


def auc(score: np.ndarray, y: np.ndarray) -> float | None:
    """Rank AUC with ties averaged. None when a class is absent."""
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return None
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def subsample(n: int, k: int, seed: int = 0) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


def audit(key: str, max_auc_rows: int) -> dict:
    name, base = DATASETS[key]
    info = json.loads((base / "raw_values.json").read_text())
    cols = info["columns"]
    X = np.load(base / "raw_values.npy", mmap_mode="r")
    meta = pd.read_parquet(base / "meta.parquet", columns=["IPV4_SRC_ADDR", "IPV4_DST_ADDR",
                                                           "PROTOCOL", "Attack"])
    fam = meta["Attack"].astype(str).to_numpy()
    splits = json.loads((base / "splits.json").read_text())
    n = len(fam)
    split_of = np.full(n, "purged", dtype=object)
    for s in ("train", "val", "test"):
        split_of[splits[s]["start"]:splits[s]["stop"]] = s
    col = {c: np.asarray(X[:, i], dtype="float64") for i, c in enumerate(cols)}
    proto = meta["PROTOCOL"].to_numpy()
    families = sorted(set(fam))
    print(f"\n=== {name}: {n:,} rows, {len(cols)} measured columns, {len(families)} labels")

    out: dict = {"dataset": name, "n_rows": n}

    # ---- 1. counter wraps, every column ----------------------------------
    wraps = {}
    for c, v in col.items():
        near_ms = np.abs(v - WRAP_32_MS) < 30_000          # within 30 s of the ms wrap
        hits = {
            "ge_2pow31": int((v >= 2 ** 31).sum()),
            "near_2pow32": int((np.abs(v - WRAP_32) < 1e6).sum()),
            "near_2pow32_over_1000": int(near_ms.sum()),
            "negative": int((v < 0).sum()),
            "non_finite": int((~np.isfinite(v)).sum()),
        }
        if any(hits.values()):
            wraps[c] = hits
    out["wrap_signatures"] = wraps
    for c, h in wraps.items():
        print(f"  {c:<30} " + "  ".join(f"{k}={v:,}" for k, v in h.items() if v))

    # ---- 2. physical consistency, via the shared validity flags ----------
    flags = validity_flags(col, proto)
    out["validity_flags"] = {}
    for f, m in flags.items():
        by_fam = {fm: round(float(m[fam == fm].mean()), 4) for fm in families}
        by_split = {s: round(float(m[split_of == s].mean()), 4) for s in ("train", "val", "test")}
        out["validity_flags"][f] = {"n": int(m.sum()), "share": round(float(m.mean()), 5),
                                    "share_by_family": by_fam, "share_by_split": by_split}
        if m.any():
            top = sorted(by_fam.items(), key=lambda kv: -kv[1])[:3]
            print(f"  FLAG {f:<34} {int(m.sum()):>9,} ({m.mean():.2%})  most in: {top}")

    # ---- 3. per-family one-vs-rest AUC: columns and flags -----------------
    idx = subsample(n, max_auc_rows)
    fam_s = fam[idx]
    per_family: dict = {}
    flagged = []
    sources = {**{c: v[idx] for c, v in col.items()},
               **{f"flag:{f}": m[idx].astype(float) for f, m in flags.items() if m.any()}}
    for fm in families:
        y = fam_s == fm
        row = {}
        for c, v in sources.items():
            a = auc(np.nan_to_num(v, nan=-1.0), y)
            if a is None:
                continue
            a = max(a, 1 - a)
            row[c] = round(a, 4)
            if a >= SHORTCUT_AUC:
                flagged.append((fm, c, round(a, 4)))
        per_family[fm] = dict(sorted(row.items(), key=lambda kv: -kv[1])[:8])
    out["per_family_top_auc"] = per_family
    out["per_family_shortcuts"] = [{"family": f, "feature": c, "abs_auc": a}
                                   for f, c, a in sorted(flagged, key=lambda t: -t[2])]
    print(f"  per-family |AUC| >= {SHORTCUT_AUC}: {len(flagged)}")
    for f, c, a in sorted(flagged, key=lambda t: -t[2])[:25]:
        print(f"    {f:<16} {c:<34} {a}")

    # ---- 3b. label sanity and documented ToN-IoT labelling errors --------
    # Raskovalov et al. (arXiv 2212.13994) report that ToN-IoT labels traffic
    # leaving the testbed, and ordinary router/DNS traffic to 192.168.1.1, as
    # attack. A graph model sees addresses as nodes, so if "talks to an outside
    # address" or "talks to the router" decides the label, that is a shortcut
    # through the topology rather than through any feature.
    full_meta = pd.read_parquet(base / "meta.parquet", columns=["Label", "Attack"])
    binary = full_meta["Label"].to_numpy()
    disagree = (binary == 1) != (fam != "Benign")
    src = meta["IPV4_SRC_ADDR"].astype(str).to_numpy()
    dst = meta["IPV4_DST_ADDR"].astype(str).to_numpy()
    prefixes = ("192.168.", "10.", "172.16.", "59.166.", "149.171.")
    private = lambda a: pd.Series(a).str.startswith(prefixes).to_numpy()
    involves_external = ~private(src) | ~private(dst)
    involves_router = (src == "192.168.1.1") | (dst == "192.168.1.1")
    def by_label(mask):
        return {"share_of_rows": round(float(mask.mean()), 4),
                "attack_rate_when_true": round(float(binary[mask].mean()), 4) if mask.any() else None,
                "attack_rate_when_false": round(float(binary[~mask].mean()), 4) if (~mask).any() else None,
                "share_by_family": {fm: round(float(mask[fam == fm].mean()), 4) for fm in families}}
    out["labels"] = {
        "binary_disagrees_with_family": int(disagree.sum()),
        "involves_external_address": by_label(involves_external),
        "involves_192_168_1_1": by_label(involves_router),
    }
    print(f"  labels: binary/family disagreements {int(disagree.sum()):,}")
    for k in ("involves_external_address", "involves_192_168_1_1"):
        v = out["labels"][k]
        print(f"  {k:<28} {v['share_of_rows']:.2%} of rows; attack rate "
              f"{v['attack_rate_when_true']} when true vs {v['attack_rate_when_false']} when false")

    # Field-width bounds for the categorical codes, where present (D47 keeps
    # them in raw_values.npy from the 2026-10-17 preprocessing on).
    bounds = {}
    for c, hi in (("TCP_FLAGS", 255), ("CLIENT_TCP_FLAGS", 255), ("SERVER_TCP_FLAGS", 255),
                  ("PROTOCOL", 255), ("ICMP_TYPE", 65535), ("MIN_TTL", 255), ("MAX_TTL", 255)):
        if c in col:
            bounds[c] = int(((col[c] < 0) | (col[c] > hi)).sum())
    out["field_bounds_violations"] = bounds
    if bounds:
        print(f"  field-bound violations: {bounds}")

    # ---- 4. duplicates across splits, and contradictory labels -----------
    feats = np.column_stack([col[c] for c in cols])
    key_df = pd.DataFrame(feats)
    key_df["src"], key_df["dst"], key_df["proto"] = (meta["IPV4_SRC_ADDR"].to_numpy(),
                                                     meta["IPV4_DST_ADDR"].to_numpy(), proto)
    h = pd.util.hash_pandas_object(key_df, index=False).to_numpy()
    h_nohost = pd.util.hash_pandas_object(pd.DataFrame(feats).assign(p=proto), index=False).to_numpy()
    dup = {}
    for label, hv in (("with_hosts", h), ("features_only", h_nohost)):
        s = pd.DataFrame({"h": hv, "split": split_of, "fam": fam})
        tr = set(s.loc[s.split == "train", "h"])
        te = s[s.split == "test"]
        cross = te["h"].isin(tr)
        lab = s.groupby("h")["fam"].nunique()
        contra = s["h"].map(lab) > 1
        dup[label] = {
            "duplicate_rows_total": int(s["h"].duplicated().sum()),
            "test_rows_also_in_train": int(cross.sum()),
            "test_rows_also_in_train_share": round(float(cross.mean()), 4),
            "test_rows_also_in_train_by_family": {k: round(float(v), 4) for k, v in
                                                  te.assign(c=cross).groupby("fam")["c"].mean().items()},
            "rows_whose_identical_twin_has_another_label": int(contra.sum()),
            "contradictory_by_family": {k: int(v) for k, v in
                                        s[contra].groupby("fam").size().items()},
        }
        print(f"  duplicates [{label}]: test rows identical to a train row "
              f"{cross.sum():,} ({cross.mean():.1%}); rows with a twin of another label "
              f"{contra.sum():,}")
    out["duplicates"] = dup
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), nargs="+", default=sorted(DATASETS))
    ap.add_argument("--max-auc-rows", type=int, default=400_000,
                    help="rows sampled for the per-family AUCs (ranks are expensive)")
    args = ap.parse_args()
    for key in args.dataset:
        res = audit(key, args.max_auc_rows)
        res["generated_at"] = datetime.now(timezone.utc).isoformat()
        res["shortcut_threshold"] = SHORTCUT_AUC
        path = REPO_ROOT / "results/metrics" / f"data_quality_{res['dataset']}.json"
        path.write_text(json.dumps(res, indent=2))
        print(f"  written -> {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
