"""Audit any NF-v2 dataset for shortcuts and saturation, before trusting it.

Written for NF-ToN-IoT-v2 but deliberately dataset-agnostic: Phase 9's
cross-dataset transfer needs the same audit run on every target, and running it
*before* building on a dataset is the lesson of Phases 1 and 3.

Two questions, and both change what the project should do:

  1. **Does this dataset have provenance shortcuts?**
     NF-UNSW-NB15-v2 has three features that separate attack from benign at
     |AUC| ~0.99 on their own, because they encode which generator produced the
     flow. If a transfer target has its own fingerprint, then a model that
     transfers well might simply be finding a different artefact, and the
     transfer result would mean nothing.

  2. **Is this dataset saturated?**
     On NF-UNSW-NB15-v2 a linear model reaches PR-AUC 0.9927, leaving no
     headroom in which topology could demonstrate anything. If a target is NOT
     saturated, it is a better primary benchmark than the one we started on --
     and that is worth knowing before Phase 4 rather than after.

Memory-safe by construction. NF-ToN-IoT-v2 is ~17M flows; loading it whole would
need roughly 6 GB and this machine has 16 GB total. Columns are read one at a
time for the audit, and the saturation test runs on a stratified subsample.

Usage:
    python scripts/audit_dataset.py --input data/raw/NF-ToN-IoT-v2.csv
    python scripts/audit_dataset.py --input data/interim/NF-ToN-IoT-v2.parquet
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import csv as pacsv

REPO_ROOT = Path(__file__).resolve().parents[1]

LABEL_BINARY, LABEL_MULTICLASS = "Label", "Attack"
IP_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
SHORTCUT_THRESHOLD = 0.90
# Known shortcuts in NF-UNSW-NB15-v2, checked here for comparison. If the same
# columns are shortcuts on a second, independently captured network, that is a
# much stronger claim: the feature set itself leaks, not just one dataset.
KNOWN_SHORTCUTS = ["MIN_TTL", "MAX_TTL", "MIN_IP_PKT_LEN"]


def to_parquet_streaming(csv_path: Path, out_path: Path, block_mb: int = 64) -> Path:
    """Convert CSV to Parquet in batches, never holding the whole file.

    pyarrow's one-shot read_csv would need several times the file size in RAM.
    The streaming reader keeps peak usage at roughly one block.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Converting {csv_path.name} -> {out_path.name} (streamed) ...")
    reader = pacsv.open_csv(
        csv_path, read_options=pacsv.ReadOptions(block_size=block_mb * 1024 * 1024)
    )
    writer = None
    rows = 0
    try:
        for batch in reader:
            if writer is None:
                writer = pq.ParquetWriter(out_path, batch.schema, compression="snappy")
            writer.write_batch(batch)
            rows += batch.num_rows
            print(f"\r  {rows:,} rows", end="")
    finally:
        if writer is not None:
            writer.close()
        reader.close()
    print(f"\n  {out_path.stat().st_size / 1024**2:,.0f} MB\n")
    return out_path


def read_column(pf: pq.ParquetFile, name: str) -> np.ndarray:
    return pf.read(columns=[name]).column(0).to_numpy(zero_copy_only=False)


def single_feature_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC without sklearn, so one column is all that is in memory."""
    x = np.asarray(x, dtype="float64")
    finite = np.isfinite(x)
    if not finite.all():
        x = np.where(finite, x, np.nanmedian(x[finite]) if finite.any() else 0.0)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype="float64")
    ranks[order] = np.arange(1, len(x) + 1)
    # Average ranks within ties, or heavily discrete columns score wrongly.
    xs = x[order]
    start = 0
    for i in range(1, len(xs) + 1):
        if i == len(xs) or xs[i] != xs[start]:
            if i - start > 1:
                ranks[order[start:i]] = (start + i + 1) / 2
            start = i
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def stratified_indices(
    y: np.ndarray, n: int, seed: int, target_prevalence: float | None = None
) -> np.ndarray:
    """Subsample, optionally forcing a chosen attack prevalence.

    PR-AUC's floor *is* the prevalence, so a raw PR-AUC cannot be compared
    between datasets with different class balance. NF-ToN-IoT-v2 is 64% attack,
    which puts its random baseline at 0.84 -- a model scoring 0.9989 there is
    doing far less than one scoring 0.9954 on NF-UNSW-NB15-v2 at 7% attack.

    Passing a target prevalence resamples to a common balance so the saturation
    verdicts are comparable across datasets. Row order is preserved so the
    contiguous split still approximates a temporal one (D5).
    """
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if target_prevalence is None:
        n_pos = min(len(pos), max(1, int(n * len(pos) / len(y))))
        n_neg = min(len(neg), n - n_pos)
    else:
        n_pos = min(len(pos), max(1, int(n * target_prevalence)))
        n_neg = min(len(neg), int(n_pos * (1 - target_prevalence) / target_prevalence))
    return np.sort(np.concatenate([
        rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)
    ]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--sample", type=int, default=600_000,
                    help="rows for the saturation test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-prevalence", type=float, default=0.04,
                    help="resample to this attack rate so datasets are comparable")
    args = ap.parse_args()

    name = args.name or args.input.stem
    path = args.input
    if path.suffix.lower() == ".csv":
        path = to_parquet_streaming(
            path, REPO_ROOT / "data" / "interim" / f"{args.input.stem}.parquet")

    pf = pq.ParquetFile(path)
    cols = [f.name for f in pf.schema_arrow]
    n_rows = pf.metadata.num_rows
    print(f"{'=' * 70}\n  {name}\n{'=' * 70}")
    print(f"\n{n_rows:,} rows x {len(cols)} columns\n")

    # ---- schema -----------------------------------------------------------
    have_ips = [c for c in IP_COLS if c in cols]
    print("Schema")
    print(f"  IP address columns: {have_ips if have_ips else 'ABSENT'}")
    if len(have_ips) < 2:
        print("  => WARNING: no graph can be built from this mirror. Usable for")
        print("     the shortcut audit, NOT for cross-dataset graph transfer.")
    missing_known = [c for c in KNOWN_SHORTCUTS if c not in cols]
    if missing_known:
        print(f"  known-shortcut columns absent here: {missing_known}")

    y = read_column(pf, LABEL_BINARY).astype(np.int64)
    prevalence = float(y.mean())
    print(f"\nClass balance\n  attack prevalence: {prevalence:.4%} "
          f"(1 per {(1 - prevalence) / max(prevalence, 1e-9):.1f} benign)")

    families = {}
    if LABEL_MULTICLASS in cols:
        fam = read_column(pf, LABEL_MULTICLASS)
        vals, counts = np.unique(fam.astype(str), return_counts=True)
        families = {str(v): int(c) for v, c in sorted(zip(vals, counts), key=lambda t: -t[1])}
        for k, v in families.items():
            print(f"    {k:<22} {v:>12,}  ({v / n_rows:.4%})")
        del fam

    # ---- shortcut audit ---------------------------------------------------
    print(f"\nShortcut audit (single-feature |AUC| > {SHORTCUT_THRESHOLD})")
    skip = set(IP_COLS + [LABEL_BINARY, LABEL_MULTICLASS])
    scores = {}
    for c in cols:
        if c in skip:
            continue
        x = read_column(pf, c)
        if not np.issubdtype(x.dtype, np.number):
            continue
        auc = single_feature_auc(x, y)
        del x
        if np.isfinite(auc):
            scores[c] = {"auc": round(auc, 4), "abs_auc": round(max(auc, 1 - auc), 4)}

    ranked = sorted(scores.items(), key=lambda kv: -kv[1]["abs_auc"])
    shortcuts = [c for c, v in ranked if v["abs_auc"] > SHORTCUT_THRESHOLD]
    for c, v in ranked[:12]:
        flag = "  <-- SHORTCUT" if v["abs_auc"] > SHORTCUT_THRESHOLD else ""
        print(f"    {c:<32} |AUC|={v['abs_auc']:.4f}{flag}")
    print(f"\n  {len(shortcuts)} shortcut feature(s): {shortcuts if shortcuts else 'NONE'}")
    for c in KNOWN_SHORTCUTS:
        if c in scores:
            print(f"    (NF-UNSW-NB15-v2 shortcut {c}: |AUC| {scores[c]['abs_auc']:.4f} here)")

    # ---- saturation test --------------------------------------------------
    print(f"\nSaturation test (subsample of {args.sample:,}, resampled to "
          f"{args.target_prevalence:.0%} attack for cross-dataset comparability)")
    feat_cols = [c for c, v in scores.items()]
    idx = stratified_indices(y, min(args.sample, n_rows), args.seed, args.target_prevalence)
    X = np.empty((len(idx), len(feat_cols)), dtype=np.float32)
    for j, c in enumerate(feat_cols):
        col = np.asarray(read_column(pf, c)[idx], dtype="float64")
        # Several NetFlow throughput columns exceed float32 range; clipping
        # before the cast avoids silent inf that would corrupt the model.
        np.clip(col, -3e38, 3e38, out=col)
        X[:, j] = np.nan_to_num(col.astype(np.float32))
        del col
    ys = y[idx]
    print(f"  actual prevalence in subsample: {ys.mean():.4%}")

    # Contiguous split on the subsample, preserving order (D5).
    cut = int(0.7 * len(idx))
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler

    # log1p before scaling. NetFlow volume features are heavily right-tailed
    # (Phase 1 measured 26 of 41 warranting it on NF-UNSW-NB15-v2), and without
    # it a linear model is crippled by a handful of enormous flows. Omitting
    # this made the tool report NF-UNSW-NB15-v2 as unsaturated at 0.798, against
    # the 0.993 Phase 3 measured with the transform in place.
    Xl = np.log1p(np.maximum(X, 0))
    sc = StandardScaler().fit(Xl[:cut])
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(sc.transform(Xl[:cut]), ys[:cut])
    ap_lr = average_precision_score(ys[cut:], lr.predict_proba(sc.transform(Xl[cut:]))[:, 1])
    del Xl

    from sklearn.ensemble import HistGradientBoostingClassifier
    gb = HistGradientBoostingClassifier(max_iter=150, random_state=args.seed)
    gb.fit(X[:cut], ys[:cut])
    ap_gb = average_precision_score(ys[cut:], gb.predict_proba(X[cut:])[:, 1])

    base = float(ys[cut:].mean())
    from sklearn.metrics import roc_auc_score
    auc_lr = roc_auc_score(ys[cut:], lr.predict_proba(sc.transform(np.log1p(
        np.maximum(X[cut:], 0))))[:, 1])
    auc_gb = roc_auc_score(ys[cut:], gb.predict_proba(X[cut:])[:, 1])
    print(f"  {'':30} {'PR-AUC':>9} {'ROC-AUC':>9}")
    print(f"  {'random baseline':30} {base:>9.4f} {0.5:>9.4f}")
    print(f"  {'logistic regression':30} {ap_lr:>9.4f} {auc_lr:>9.4f}")
    print(f"  {'gradient boosting':30} {ap_gb:>9.4f} {auc_gb:>9.4f}")
    # Judged on the better of the two: saturation means the problem is solvable
    # without sophistication, and a tree model reaching 0.99 says that just as
    # firmly as a linear one does.
    best = max(ap_lr, ap_gb)
    saturated = best > 0.98
    print(f"\n  => {'SATURATED' if saturated else 'NOT saturated'}: best simple model "
          f"reaches {best:.4f}")
    print("     " + ("No headroom for topology here either -- same problem as "
                     "NF-UNSW-NB15-v2." if saturated
                     else "There is headroom. This may be a better primary benchmark "
                          "than NF-UNSW-NB15-v2."))

    out = REPO_ROOT / "results" / "metrics" / f"audit_{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": name,
        "input_file": str(path),
        "n_rows": int(n_rows), "n_columns": len(cols), "columns": cols,
        "has_ip_columns": len(have_ips) == 2,
        "attack_prevalence": round(prevalence, 6),
        "families": families,
        "shortcut_threshold": SHORTCUT_THRESHOLD,
        "shortcut_features": shortcuts,
        "per_feature_auc": scores,
        "saturation": {
            "n_sampled": int(len(idx)),
            "target_prevalence": args.target_prevalence,
            "actual_prevalence": round(float(ys.mean()), 5),
            "baseline_pr_auc": round(base, 5),
            "logreg_roc_auc": round(float(auc_lr), 5),
            "gbdt_roc_auc": round(float(auc_gb), 5),
            "logreg_pr_auc": round(float(ap_lr), 5),
            "gbdt_pr_auc": round(float(ap_gb), 5),
            "best_simple_pr_auc": round(float(best), 5),
            "saturated": bool(saturated),
        },
    }, indent=2))
    print(f"\nwritten -> {out.relative_to(REPO_ROOT)}\n")


if __name__ == "__main__":
    main()
