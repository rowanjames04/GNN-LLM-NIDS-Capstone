"""Phase 1 exploratory analysis: understand the features well enough to defend
every preprocessing choice.

Phase 0 already settled class balance, host-graph shape and the absence of
timestamps. What remains is the feature level, and specifically the predictions
written into the Obsidian note "Feature Dictionary - NF-UNSW-NB15-v2". Those
were reasoned from the official column descriptions, not measured. This script
tests them, so the preprocessing decisions in Phase 2 rest on evidence:

  H1  Volume and throughput features are heavily right-skewed -> log transform.
  H2  Throughput fields are derived from bytes/duration and therefore largely
      redundant with the volume and timing groups.
  H3  ICMP_TYPE and ICMP_IPV4_TYPE are near-duplicates by definition.
  H4  DNS_QUERY_ID is a transaction identifier: high cardinality, no signal.
  H5  TCP flag fields are bitmasks, so decomposing them into individual bits
      exposes signal that treating them as integers hides.
  H6  Application-layer fields are protocol-conditional, so their zeros mean
      "not applicable" rather than "measured zero".

Writes results/metrics/eda.json and figures to results/figures/.

Usage:
    python scripts/eda.py
    python scripts/eda.py --mi-sample 400000
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "interim" / "NF-UNSW-NB15-v2.parquet"
METRICS_OUT = REPO_ROOT / "results" / "metrics" / "eda.json"
FIG_DIR = REPO_ROOT / "results" / "figures"


def load_frame(path: Path, max_rows: int | None) -> pd.DataFrame:
    """Load the dataset, capped at max_rows by reading whole row groups.

    NF-ToN-IoT-v2 is 16.9M rows and would need roughly 6 GB in pandas, which a
    16 GB machine cannot spare alongside model fitting. Parquet stores data in
    row groups, so taking every k-th group gives a sample spread across the
    whole file at a fraction of the memory -- and, unlike random row sampling,
    each retained group is a contiguous block, so local structure survives
    (the same principle as D20).
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    if max_rows is None or pf.metadata.num_rows <= max_rows:
        return pd.read_parquet(path)

    n_groups = pf.metadata.num_row_groups
    rows_per_group = pf.metadata.num_rows / max(n_groups, 1)
    keep_n = max(1, int(max_rows / max(rows_per_group, 1)))
    step = max(1, n_groups // keep_n)
    groups = list(range(0, n_groups, step))[:keep_n]
    print(f"  capping at ~{max_rows:,} rows: {len(groups)} of {n_groups} row groups")
    return pf.read_row_groups(groups).to_pandas()

IDENTITY = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
LABELS = ["Label", "Attack"]
TCP_FLAG_COLS = ["TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"]
APP_LAYER = [
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE",
]
# TCP flag bit meanings, low bit first (RFC 793 order as nProbe reports them).
TCP_BITS = ["FIN", "SYN", "RST", "PSH", "ACK", "URG", "ECE", "CWR"]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})
BENIGN_C, ATTACK_C = "#4C72B0", "#C44E52"


# ---------------------------------------------------------------- H1: skew

def analyse_skew(df: pd.DataFrame, features: list[str]) -> dict:
    """Which features are skewed enough to warrant a log transform?

    Rule: recommend log1p when raw |skew| > 2 and the transform at least halves
    it. Stating the rule explicitly matters more than the exact threshold --
    the point is that the decision is made on evidence rather than habit.
    """
    out = {}
    for col in features:
        x = df[col].to_numpy(dtype="float64")
        # Clip before measuring. NF-ToN-IoT-v2 carries throughput values up to
        # 1.9e219, and squaring those overflows float64 inside the skew
        # computation, yielding NaN -- which would silently mean "no log
        # transform needed" for exactly the columns that need it most.
        finite = x[np.isfinite(x)]
        if not len(finite):
            continue
        hi = float(np.quantile(finite, 0.99999))
        x = np.clip(np.nan_to_num(x, nan=0.0, posinf=hi, neginf=0.0), None, hi)
        if np.all(x == x[0]):
            continue
        raw = float(pd.Series(x).skew())
        # Shift to non-negative before log1p; several columns are counts but a
        # few could carry negatives, and log of a negative is undefined.
        shifted = x - min(0.0, float(np.nanmin(x)))
        logged = float(pd.Series(np.log1p(shifted)).skew())
        recommend = abs(raw) > 2 and abs(logged) < abs(raw) / 2
        out[col] = {
            "skew_raw": round(raw, 3),
            "skew_log1p": round(logged, 3),
            "recommend_log": bool(recommend),
        }
    return out


# ------------------------------------------- class separation + information

def cohens_d(df: pd.DataFrame, features: list[str]) -> dict:
    """Standardised mean difference between attack and benign, per feature.

    Complements mutual information: MI catches any dependence including
    non-monotonic, while Cohen's d says which direction and how far apart the
    means are -- which is what the LLM evidence pack will need to verbalise.
    """
    y = df["Label"].to_numpy(dtype=bool)
    out = {}
    for col in features:
        x = df[col].to_numpy(dtype="float64")
        a, b = x[y], x[~y]
        va, vb = a.var(ddof=1), b.var(ddof=1)
        pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(x) - 2))
        if pooled == 0 or not np.isfinite(pooled):
            continue
        out[col] = {
            "cohens_d": round(float((a.mean() - b.mean()) / pooled), 4),
            "mean_attack": round(float(a.mean()), 3),
            "mean_benign": round(float(b.mean()), 3),
        }
    return out


def mutual_information(df: pd.DataFrame, features: list[str], n: int, seed: int) -> dict:
    """MI between each feature and the binary label, on a stratified subsample.

    This is the information-theoretic feature selection the proposal promised
    as FSNID. Subsampled because MI estimation is O(n log n) per feature with a
    kNN estimator and 2.4M rows across 40+ features is needlessly slow for a
    ranking that is stable well before then.
    """
    rng = np.random.default_rng(seed)
    y_full = df["Label"].to_numpy()
    atk = np.flatnonzero(y_full == 1)
    ben = np.flatnonzero(y_full == 0)
    # Keep the true class ratio so the MI estimate reflects deployment.
    n_atk = min(len(atk), int(n * len(atk) / len(y_full)))
    n_ben = min(len(ben), n - n_atk)
    idx = np.sort(np.concatenate([
        rng.choice(atk, n_atk, replace=False),
        rng.choice(ben, n_ben, replace=False),
    ]))

    sub = df.iloc[idx]
    mi = mutual_info_classif(
        sub[features].to_numpy(dtype="float64"),
        sub["Label"].to_numpy(),
        discrete_features=False,
        random_state=seed,
    )
    return {
        "n_sampled": int(len(idx)),
        "attack_fraction": round(float(n_atk / len(idx)), 5),
        "scores": {c: round(float(v), 5) for c, v in zip(features, mi)},
    }


# ------------------------------------------------------- H2/H3: redundancy

def analyse_redundancy(df: pd.DataFrame, features: list[str], thresh: float) -> dict:
    corr = df[features].corr(method="pearson")
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= thresh:
                pairs.append({"a": cols[i], "b": cols[j], "r": round(float(r), 4)})
    pairs.sort(key=lambda p: -abs(p["r"]))

    # H2 directly: is throughput just bytes / duration?
    h2 = {}
    dur_s = df["FLOW_DURATION_MILLISECONDS"].to_numpy(dtype="float64") / 1000.0
    for thr_col, byte_col in (
        ("SRC_TO_DST_AVG_THROUGHPUT", "IN_BYTES"),
        ("DST_TO_SRC_AVG_THROUGHPUT", "OUT_BYTES"),
    ):
        with np.errstate(divide="ignore", invalid="ignore"):
            implied = np.where(dur_s > 0, df[byte_col].to_numpy(dtype="float64") / dur_s, np.nan)
        ok = np.isfinite(implied)
        h2[thr_col] = {
            "corr_with_bytes_over_duration": round(
                float(np.corrcoef(implied[ok], df[thr_col].to_numpy()[ok])[0, 1]), 4
            ),
            "n_compared": int(ok.sum()),
        }

    return {"threshold": thresh, "redundant_pairs": pairs, "h2_throughput_derived": h2}


# ------------------------------------------------------------ H5: TCP bits

def analyse_tcp_flags(df: pd.DataFrame, mi_sample: int, seed: int) -> dict:
    """Do the flag bitmasks carry more signal decomposed than as integers?

    Treating a bitmask as a continuous number is a category error -- flag value
    18 (SYN+ACK) is not "twice" value 9. This measures the cost of that error.
    """
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(df), min(mi_sample, len(df)), replace=False))
    sub = df.iloc[idx]
    y = sub["Label"].to_numpy()

    out = {}
    for col in TCP_FLAG_COLS:
        raw = sub[col].to_numpy(dtype="int64")
        mi_raw = float(mutual_info_classif(
            raw.reshape(-1, 1), y, discrete_features=True, random_state=seed
        )[0])
        bits = np.stack([(raw >> b) & 1 for b in range(len(TCP_BITS))], axis=1)
        mi_bits = mutual_info_classif(bits, y, discrete_features=True, random_state=seed)
        out[col] = {
            "n_distinct_values": int(pd.Series(raw).nunique()),
            "mi_as_integer": round(mi_raw, 5),
            "mi_summed_over_bits": round(float(mi_bits.sum()), 5),
            "mi_per_bit": {
                name: round(float(v), 5) for name, v in zip(TCP_BITS, mi_bits)
            },
            "decomposition_gain": round(float(mi_bits.sum() - mi_raw), 5),
        }
    return out


# ------------------------------------------------------- H7: shortcut audit

def analyse_shortcuts(df: pd.DataFrame, features: list[str], thresh: float) -> dict:
    """Find features that separate the classes almost perfectly on their own.

    A single raw measurement achieving near-perfect AUC is almost never a
    learnable attack behaviour -- it is a artefact of how the dataset was
    generated. UNSW-NB15's attack traffic came from a different generator, on a
    different network path, than its background traffic, so any feature that
    encodes *provenance* rather than *behaviour* will look devastatingly
    predictive and will not transfer to any other network.

    This matters far beyond feature selection. A shortcut feature would:
      - make the MLP-vs-GNN comparison meaningless, since both models would
        simply learn the shortcut (commitment V2);
      - silently invalidate leave-one-attack-out, because a held-out attack
        family still carries the attacker's fingerprint and would be "detected"
        without any generalisation at all (commitment V3).

    AUC is reported two-sided: a feature that is perfectly anti-correlated is
    just as much of a shortcut as one that is perfectly correlated.
    """
    from sklearn.metrics import roc_auc_score

    y = df["Label"].to_numpy()
    scores = {}
    for col in features:
        auc = float(roc_auc_score(y, df[col].to_numpy(dtype="float64")))
        scores[col] = {"auc": round(auc, 4), "abs_auc": round(max(auc, 1 - auc), 4)}

    shortcuts = sorted(
        [c for c, v in scores.items() if v["abs_auc"] > thresh],
        key=lambda c: -scores[c]["abs_auc"],
    )

    # Does every attack family carry the shortcut? If so, LOAO is corrupted.
    family_coverage = {}
    for col in shortcuts:
        vals = df.loc[df.Label == 1, col].value_counts()
        dominant = set(vals.head(2).index)
        family_coverage[col] = {
            "dominant_attack_values": [float(v) for v in dominant],
            "per_family_share": {
                str(fam): round(float(g[col].isin(dominant).mean()), 5)
                for fam, g in df[df.Label == 1].groupby("Attack", observed=True)
            },
            "benign_share": round(
                float(df.loc[df.Label == 0, col].isin(dominant).mean()), 5
            ),
        }

    return {
        "threshold": thresh,
        "per_feature_auc": scores,
        "shortcut_features": shortcuts,
        "family_coverage": family_coverage,
    }


# ------------------------------------------------- H4/H6: sparsity + IDs

def analyse_app_layer(df: pd.DataFrame) -> dict:
    out = {}
    n = len(df)
    for col in APP_LAYER:
        x = df[col]
        zero_frac = float((x == 0).mean())
        nz = x[x != 0]
        out[col] = {
            "zero_fraction": round(zero_frac, 5),
            "n_distinct": int(x.nunique()),
            "n_distinct_nonzero": int(nz.nunique()),
            # If a field is meaningful only for its own protocol, attack
            # prevalence among its non-zero rows should differ markedly from
            # the 3.98% base rate.
            "attack_rate_when_nonzero": (
                round(float(df.loc[x != 0, "Label"].mean()), 5) if len(nz) else None
            ),
        }
    return out


# ---------------------------------------------------------------- figures

def make_figures(df, skew, dvals, mi, redundancy, tcp, shortcuts, prefix="") -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    # 1. class balance
    counts = df["Attack"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = [BENIGN_C if k == "Benign" else ATTACK_C for k in counts.index]
    ax.bar(range(len(counts)), counts.values, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=40, ha="right")
    ax.set_ylabel("flows (log scale)")
    ax.set_title("Class distribution — NF-UNSW-NB15-v2")
    for i, v in enumerate(counts.values):
        ax.text(i, v * 1.25, f"{v:,}", ha="center", fontsize=7)
    p = FIG_DIR / f"{prefix}01_class_distribution.png"; fig.savefig(p); plt.close(fig)
    written.append(p.name)

    # 2. skew before/after log
    items = sorted(skew.items(), key=lambda kv: -abs(kv[1]["skew_raw"]))[:20]
    names = [k for k, _ in items]
    fig, ax = plt.subplots(figsize=(7, 6))
    ypos = np.arange(len(names))
    ax.barh(ypos - 0.2, [v["skew_raw"] for _, v in items], 0.4,
            label="raw", color=ATTACK_C)
    ax.barh(ypos + 0.2, [v["skew_log1p"] for _, v in items], 0.4,
            label="after log1p", color=BENIGN_C)
    ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("skewness"); ax.legend()
    ax.set_title("Log transform flattens the heaviest tails (top 20 by |skew|)")
    p = FIG_DIR / f"{prefix}02_feature_skew.png"; fig.savefig(p); plt.close(fig)
    written.append(p.name)

    # 3. mutual information ranking
    top = sorted(mi["scores"].items(), key=lambda kv: -kv[1])[:25]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([k for k, _ in top], [v for _, v in top], color=BENIGN_C)
    ax.invert_yaxis(); ax.set_xlabel("mutual information with Label (nats)")
    ax.set_title("Feature informativeness (top 25)")
    ax.tick_params(axis="y", labelsize=7)
    p = FIG_DIR / f"{prefix}03_mutual_information.png"; fig.savefig(p); plt.close(fig)
    written.append(p.name)

    # 4. class separation
    items = sorted(dvals.items(), key=lambda kv: -abs(kv[1]["cohens_d"]))[:20]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    vals = [v["cohens_d"] for _, v in items]
    ax.barh([k for k, _ in items], vals,
            color=[ATTACK_C if v > 0 else BENIGN_C for v in vals])
    ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Cohen's d   (positive = higher in attacks)")
    ax.set_title("Attack vs benign separation (top 20 by |d|)")
    ax.tick_params(axis="y", labelsize=7)
    p = FIG_DIR / f"{prefix}04_class_separation.png"; fig.savefig(p); plt.close(fig)
    written.append(p.name)

    # 5. TCP flag decomposition gain
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    cols = list(tcp.keys())
    x = np.arange(len(cols))
    ax.bar(x - 0.2, [tcp[c]["mi_as_integer"] for c in cols], 0.4,
           label="as integer", color=ATTACK_C)
    ax.bar(x + 0.2, [tcp[c]["mi_summed_over_bits"] for c in cols], 0.4,
           label="summed over bits", color=BENIGN_C)
    ax.set_xticks(x); ax.set_xticklabels(cols, fontsize=7)
    ax.set_ylabel("mutual information (nats)"); ax.legend()
    ax.set_title("Decomposing TCP flag bitmasks into bits")
    p = FIG_DIR / f"{prefix}05_tcp_flag_decomposition.png"; fig.savefig(p); plt.close(fig)
    written.append(p.name)

    # 6. shortcut audit -- the figure that changes what we train on
    aucs = sorted(shortcuts["per_feature_auc"].items(),
                  key=lambda kv: -kv[1]["abs_auc"])[:20]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    names = [k for k, _ in aucs]
    vals = [v["abs_auc"] for _, v in aucs]
    cols = [ATTACK_C if v > shortcuts["threshold"] else BENIGN_C for v in vals]
    ax.barh(names, vals, color=cols)
    ax.axvline(shortcuts["threshold"], color="k", ls="--", lw=1,
               label=f"shortcut threshold ({shortcuts['threshold']})")
    ax.set_xlim(0.5, 1.0); ax.invert_yaxis()
    ax.set_xlabel("single-feature |AUC|  (max of AUC, 1−AUC)")
    ax.set_title("Shortcut audit: features that separate classes alone")
    ax.tick_params(axis="y", labelsize=7); ax.legend(loc="lower right")
    p = FIG_DIR / f"{prefix}06_shortcut_audit.png"; fig.savefig(p); plt.close(fig)
    written.append(p.name)

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=METRICS_OUT)
    ap.add_argument("--fig-prefix", default="",
                    help="prefix for figure filenames, to keep datasets apart")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap rows by reading a subset of parquet row groups")
    ap.add_argument("--mi-sample", type=int, default=200_000)
    ap.add_argument("--corr-threshold", type=float, default=0.95)
    ap.add_argument("--shortcut-threshold", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading {args.input.name} ...")
    df = load_frame(args.input, args.max_rows)
    features = [c for c in df.columns if c not in IDENTITY + LABELS]
    print(f"  {len(df):,} flows, {len(features)} candidate features\n")

    print("H1  skew / log-transform ...")
    skew = analyse_skew(df, features)
    print("    class separation (Cohen's d) ...")
    dvals = cohens_d(df, features)
    print(f"    mutual information (subsample {args.mi_sample:,}) ...")
    mi = mutual_information(df, features, args.mi_sample, args.seed)
    print("H2/H3  redundancy ...")
    redundancy = analyse_redundancy(df, features, args.corr_threshold)
    print("H5  TCP flag decomposition ...")
    tcp = analyse_tcp_flags(df, args.mi_sample, args.seed)
    print("H4/H6  application-layer sparsity ...")
    app = analyse_app_layer(df)
    print("H7  shortcut audit ...")
    shortcuts = analyse_shortcuts(df, features, args.shortcut_threshold)

    print("figures ...")
    figs = make_figures(df, skew, dvals, mi, redundancy, tcp, shortcuts,
                        prefix=args.fig_prefix)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(Path(args.input).resolve().relative_to(REPO_ROOT)),
        "n_rows": int(len(df)),
        "n_features": len(features),
        "params": {
            "mi_sample": args.mi_sample,
            "corr_threshold": args.corr_threshold,
            "seed": args.seed,
        },
        "skew": skew,
        "class_separation": dvals,
        "mutual_information": mi,
        "redundancy": redundancy,
        "tcp_flags": tcp,
        "app_layer": app,
        "shortcuts": shortcuts,
        "figures": figs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    # ---- console summary ----
    log_cols = [c for c, v in skew.items() if v["recommend_log"]]
    print(f"\n{'=' * 68}")
    print(f"H1  log transform recommended for {len(log_cols)}/{len(features)} features")
    print(f"H2  throughput vs bytes/duration correlation:")
    for k, v in redundancy["h2_throughput_derived"].items():
        print(f"      {k:<28} r = {v['corr_with_bytes_over_duration']}")
    print(f"H3/H2  {len(redundancy['redundant_pairs'])} feature pairs with "
          f"|r| >= {args.corr_threshold}")
    for p in redundancy["redundant_pairs"][:8]:
        print(f"      {p['a']:<30} ~ {p['b']:<30} r={p['r']}")
    print(f"H4  DNS_QUERY_ID: {app['DNS_QUERY_ID']['n_distinct']:,} distinct values, "
          f"MI = {mi['scores'].get('DNS_QUERY_ID')}")
    print(f"H5  TCP flag decomposition gain:")
    for c, v in tcp.items():
        print(f"      {c:<20} integer {v['mi_as_integer']:.4f} -> "
              f"bits {v['mi_summed_over_bits']:.4f}  (+{v['decomposition_gain']:.4f})")
    print(f"H6  application-layer zero fractions:")
    for c, v in app.items():
        print(f"      {c:<24} {v['zero_fraction']:.1%} zero   "
              f"attack rate when non-zero: {v['attack_rate_when_nonzero']}")

    print(f"\nH7  SHORTCUT AUDIT -- {len(shortcuts['shortcut_features'])} feature(s) "
          f"with |AUC| > {args.shortcut_threshold} alone:")
    for c in shortcuts["shortcut_features"]:
        v = shortcuts["per_feature_auc"][c]
        cov = shortcuts["family_coverage"][c]
        shares = list(cov["per_family_share"].values())
        print(f"      {c:<24} |AUC|={v['abs_auc']:.4f}   "
              f"attack families carrying it: {min(shares):.1%}-{max(shares):.1%}, "
              f"benign {cov['benign_share']:.1%}")
    if shortcuts["shortcut_features"]:
        print("      => These must be excluded from training. See Obsidian:")
        print("         '01 - Dataset/Feature Shortcut Audit'.")

    print(f"\ntop 10 by mutual information:")
    for c, v in sorted(mi["scores"].items(), key=lambda kv: -kv[1])[:10]:
        print(f"      {c:<32} {v:.4f}")

    print(f"\n{len(figs)} figures -> results/figures/")
    print(f"metrics -> {args.output.relative_to(REPO_ROOT)}\n")


if __name__ == "__main__":
    main()
