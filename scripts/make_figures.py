"""Results figures for the report, drawn from the committed results files.

Every figure is built from a file under `results/`, never from numbers typed in
here. A figure whose data does not exist yet is skipped with the reason, so this
runs today and again, unchanged, after the Stage C campaign.

Beside the images it writes `results/figures/results_manifest.json`: for each
figure, a suggested caption, the files it was drawn from and when they were
generated. A figure in the report can then be traced to the run behind it, and
a stale one is visible as stale.

**Design rules these follow** (the same in every figure, so they read as a set):

- **Every seed is drawn**, with the mean as a tick. With three seeds a bar and an
  error bar would hide the only thing there is to see.
- **Dot plots, not bars, for PR-AUC and F1.** These live between 0.94 and 1.00;
  a bar from zero would make every model look identical, and a bar from 0.94
  would lie about magnitude. A dot makes no claim about the distance to zero.
- **Colour means one thing per figure**, drawn from a palette checked for
  colour-blind separation, and never carries identity alone: every series is
  also named in a label or the legend.
- **No titles in the image.** The caption belongs to the report; the manifest
  holds a suggested one.

Usage:
    python scripts/make_figures.py
    python scripts/make_figures.py --only ablation transfer
    python scripts/make_figures.py --smoke --out-dir /tmp/figs   # layout check only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
PRIMARY, SECONDARY = "NF-ToN-IoT-v2", "NF-UNSW-NB15-v2"
BASELINE_SCHEMA = "dual-prevalence-v1"

# Palette: three categorical slots that pass colour-blind separation for every
# pairing (validated 2026-10-03), plus neutrals. Text is always ink, never a
# series colour.
SURFACE, INK, INK_2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#8d8c86"
SEQUENTIAL = ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

WIDTH = 6.3          # inches: the text width of an A4 page with normal margins

VARIANTS = {
    "channel1_only": "Flow features\nonly (channel 1)",
    "channel2_only": "Neighbourhood\nonly (channel 2)",
    "full": "Both channels\n(full model)",
}


class Skip(Exception):
    """This figure cannot be drawn yet; the message says what is missing."""


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "legend.frameon": False,
        "figure.dpi": 110, "savefig.dpi": 300,
    })


# ------------------------------------------------------------------ data

def load(path: Path) -> dict:
    if not path.exists():
        raise Skip(f"{path.relative_to(REPO_ROOT)} does not exist yet")
    return json.loads(path.read_text())


def gnn_results(dataset: str, smoke: bool) -> tuple[dict, Path]:
    path = RESULTS / "metrics" / "gnn" / f"{'smoke_' if smoke else ''}gnn_{dataset}.json"
    return load(path), path


def seed_values(doc: dict, variant: str, metric: str) -> list[float]:
    key = f"at_{doc['config']['eval']['target_prevalence']:.0%}"
    runs = doc["results"].get(variant, {}).get("runs", [])
    return [r[key][metric] for r in runs if r[key].get(metric) is not None]


def how_trained(doc: dict) -> str:
    runs = next(iter(doc["results"].values()))["runs"]
    selection = runs[0].get("selection")
    if selection == "fixed_budget_best_val":
        return "fixed training budget"
    return "early stopping (superseded by the fixed-budget re-run)"


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ marks

def seed_dots(ax, pos: float, values: list[float], colour: str, *,
              horizontal: bool = False, hollow: bool = False, label_mean: bool = True,
              fmt: str = ".4f") -> None:
    """Each seed as a dot, the mean as an ink tick, the mean's value beside it."""
    if not values:
        return
    offsets = np.linspace(-0.09, 0.09, len(values)) if len(values) > 1 else [0.0]
    mean = float(np.mean(values))
    for off, v in zip(offsets, values):
        xy = (v, pos + off) if horizontal else (pos + off, v)
        ax.plot(*xy, "o", ms=6.5, mfc=SURFACE if hollow else colour, mec=colour if hollow
                else SURFACE, mew=1.3 if hollow else 1.1, zorder=3)
    if horizontal:
        ax.plot([mean, mean], [pos - 0.24, pos + 0.24], color=INK, lw=1.6, zorder=4,
                solid_capstyle="round")
        if label_mean:
            ax.annotate(format(mean, fmt), (mean, pos + 0.24), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=INK)
    else:
        ax.plot([pos - 0.2, pos + 0.2], [mean, mean], color=INK, lw=1.6, zorder=4,
                solid_capstyle="round")
        if label_mean:
            ax.annotate(format(mean, fmt), (pos + 0.2, mean), xytext=(4, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=8, color=INK)


def pad(ax, values: list[float], axis: str = "y", frac: float = 0.18) -> None:
    lo, hi = min(values), max(values)
    span = (hi - lo) or abs(hi) * 0.01 or 0.01
    (ax.set_ylim if axis == "y" else ax.set_xlim)(lo - frac * span, hi + frac * span)


# ------------------------------------------------------------------ figures
# Each returns (figure, caption, [source paths]).

def fig_ablation(smoke: bool):
    doc, path = gnn_results(PRIMARY, smoke)
    present = [v for v in VARIANTS if seed_values(doc, v, "pr_auc")]
    if len(present) < 2:
        raise Skip("fewer than two model variants in the results file")
    fig, ax = plt.subplots(figsize=(WIDTH * 0.8, 3.0))
    everything = []
    for i, v in enumerate(present):
        vals = seed_values(doc, v, "pr_auc")
        seed_dots(ax, i, vals, BLUE)
        everything += vals
    ax.set_xticks(range(len(present)), [VARIANTS[v] for v in present])
    ax.set_xlim(-0.6, len(present) - 0.4 + 0.25)
    pad(ax, everything)
    prev = doc["config"]["eval"]["target_prevalence"]
    ax.set_ylabel(f"PR-AUC at {prev:.0%} prevalence\n(a random model scores {prev:.2f})")
    ax.grid(axis="x", visible=False)
    n = len(seed_values(doc, present[0], "pr_auc"))
    caption = (f"Topology ablation on {PRIMARY}. Each dot is one seed (n = {n} per "
               f"variant); the tick is the mean. The variants share architecture, "
               f"optimiser, depth and width, so the gap between the first and third "
               f"is the contribution of message passing. Trained with "
               f"{how_trained(doc)}.")
    return fig, caption, [path]


def baselines(dataset: str) -> tuple[dict, Path]:
    path = RESULTS / "metrics" / "baselines" / f"baselines_{dataset}.json"
    doc = load(path)
    keep = {k: v for k, v in doc["results"].items()
            if v.get("schema") == BASELINE_SCHEMA}
    return keep, path


def fig_gnn_vs_xgboost(smoke: bool):
    doc, path = gnn_results(PRIMARY, smoke)
    base, bpath = baselines(PRIMARY)
    prev = doc["config"]["eval"]["target_prevalence"]
    key = f"at_{prev:.0%}"
    rows = []
    for name, label in (("xgboost::flow", "XGBoost\nflow features"),
                        ("xgboost::flow+host", "XGBoost\nflow + host features")):
        if name in base and base[name].get("reported_at_prevalence") == prev:
            rows.append((label, NEUTRAL, {m: [r[key][m] for r in base[name]["runs"]]
                                          for m in ("pr_auc", "f1")}))
    if not rows or not seed_values(doc, "full", "pr_auc"):
        raise Skip("needs the full model and an XGBoost baseline at the same prevalence")
    rows.append(("Graph model\n(full)", BLUE,
                 {m: seed_values(doc, "full", m) for m in ("pr_auc", "f1")}))

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), sharey=True)
    for ax, metric, xlabel in zip(axes, ("pr_auc", "f1"),
                                  (f"PR-AUC at {prev:.0%} prevalence  (ranking)",
                                   "F1 at the chosen threshold  (operating point)")):
        vals = []
        for i, (_, colour, data) in enumerate(rows):
            seed_dots(ax, i, data[metric], colour, horizontal=True)
            vals += data[metric]
        pad(ax, vals, "x", 0.22)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(rows) - 0.3)
    axes[0].set_yticks(range(len(rows)), [r[0] for r in rows])
    fig.subplots_adjust(wspace=0.08)
    caption = (f"The graph model against XGBoost on {PRIMARY}, both reported at "
               f"{prev:.0%} prevalence. Left: ranking quality. Right: the operating "
               f"point reached at the validation-chosen threshold. Each dot is one "
               f"seed; the tick is the mean. The two panels have separate scales and "
               f"must be read together: the graph model ranks better and operates "
               f"worse. Graph model trained with {how_trained(doc)}.")
    return fig, caption, [path, bpath]


def paired_gain(doc: dict) -> list[float]:
    """full minus channel1_only, seed by seed where seeds are recorded."""
    key = f"at_{doc['config']['eval']['target_prevalence']:.0%}"
    full = doc["results"].get("full", {}).get("runs", [])
    c1 = doc["results"].get("channel1_only", {}).get("runs", [])
    if not full or not c1:
        return []
    if all("seed" in r for r in full + c1):
        by = {r["seed"]: r[key]["pr_auc"] for r in c1}
        return [r[key]["pr_auc"] - by[r["seed"]] for r in full if r["seed"] in by]
    return [f[key]["pr_auc"] - c[key]["pr_auc"] for f, c in zip(full, c1)]


def fig_topology_by_dataset(smoke: bool):
    sources, gains = [], {}
    for ds in (PRIMARY, SECONDARY):
        try:
            doc, path = gnn_results(ds, smoke)
        except Skip:
            continue
        g = paired_gain(doc)
        if g:
            gains[ds], _ = g, sources.append(path)
    if len(gains) < 2:
        raise Skip("needs the ablation on both datasets")
    fig, ax = plt.subplots(figsize=(WIDTH * 0.6, 2.9))
    ax.axhline(0, color=INK_2, lw=0.8, zorder=1)
    for i, (ds, g) in enumerate(gains.items()):
        seed_dots(ax, i, g, BLUE, fmt="+.4f")
    ax.set_xticks(range(len(gains)), list(gains))
    ax.set_xlim(-0.6, len(gains) - 0.15)
    pad(ax, [v for g in gains.values() for v in g] + [0.0])
    ax.set_ylabel("Gain in PR-AUC from adding\nthe neighbourhood channel")
    ax.grid(axis="x", visible=False)
    caption = ("Gain in PR-AUC from message passing (full model minus the "
               "flow-features-only variant, matched by seed) on each dataset. Each "
               "dot is one seed; the tick is the mean; the line marks no gain. The "
               f"gain is separated from zero on {PRIMARY} and not on {SECONDARY}, "
               "where a linear model already scores above 0.99.")
    return fig, caption, sources


def fig_transfer(smoke: bool):
    st = script("summarise_transfer")
    rows = [r for r in st.rows() if r["pr_auc"] is not None]
    if not rows:
        raise Skip("no transfer results")
    controls = {(r["model"], r["source"]): r for r in rows if r["control"]}
    cross = [r for r in rows if not r["control"]]
    if not cross:
        raise Skip("no cross-dataset rows")
    short = lambda s: s.replace("NF-", "").replace("-v2", "")       # noqa: E731
    model_name = {"gnn::full": "Graph model", "xgboost::flow": "XGBoost"}
    cross.sort(key=lambda r: (r["model"], r["source"]))

    fig, ax = plt.subplots(figsize=(WIDTH, 3.1))
    floor = cross[0]["floor"]
    ax.axhline(floor, color=INK_2, lw=0.8, zorder=1)
    ax.annotate(f"random\nmodel\n({floor:.2f})", (len(cross) - 0.35, floor),
                xytext=(5, 0), textcoords="offset points", ha="left", va="center",
                fontsize=8, color=INK_2, annotation_clip=False)
    labels, confounded = [], False
    for i, r in enumerate(cross):
        c = controls.get((r["model"], r["source"]))
        if c:
            ax.plot([i, i], [r["pr_auc"], c["pr_auc"]], color=GRID, lw=2, zorder=2,
                    solid_capstyle="round")
            ax.plot(i, c["pr_auc"], "o", ms=7.5, mfc=BLUE, mec=SURFACE, mew=1.2, zorder=3)
            # Three places: a control of 0.9954 must not print as "1.00".
            ax.annotate(f"{c['pr_auc']:.3f}", (i, c["pr_auc"]), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=8)
        if r["pr_std"]:
            ax.plot([i, i], [r["pr_auc"] - r["pr_std"], r["pr_auc"] + r["pr_std"]],
                    color=ORANGE, lw=1.6, zorder=3, solid_capstyle="round")
        ax.plot(i, r["pr_auc"], "o", ms=7.5, mfc=ORANGE, mec=SURFACE, mew=1.2, zorder=4)
        ax.annotate(f"{r['pr_auc']:.3f}", (i, r["pr_auc"]), xytext=(8, 4),
                    textcoords="offset points", va="bottom", fontsize=8)
        mark = ""
        if r["oov"] is not None and r["oov"] >= 0.5:
            mark, confounded = " †", True
        labels.append(f"{model_name.get(r['model'], r['model'])}\n"
                      f"{short(r['source'])} to\n{short(r['target'])}{mark}")
    ax.set_xticks(range(len(cross)), labels)
    ax.set_xlim(-0.5, len(cross) - 0.35)
    ax.set_ylim(-0.03, 1.06)
    ax.set_ylabel(f"PR-AUC at {floor:.0%} prevalence")
    ax.grid(axis="x", visible=False)
    ax.plot([], [], "o", ms=7.5, mfc=BLUE, mec=SURFACE, label="Tested on the dataset it was trained on")
    ax.plot([], [], "o", ms=7.5, mfc=ORANGE, mec=SURFACE, label="Tested on the other dataset")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
              handletextpad=0.2, columnspacing=1.6, borderaxespad=0.2)
    caption = ("Cross-dataset transfer with no retraining. For each model and "
               "direction, the blue dot is the in-dataset control and the orange dot "
               "is the same model scored on the other dataset (bar: one standard "
               "deviation over seeds, where more than one was run). Every control is "
               "healthy and every transfer falls to the level of a random model."
               + (" † More than half of this direction's categorical values fall "
                  "outside the source vocabulary, so it measures a degraded input as "
                  "well as a different network." if confounded else ""))
    sources = sorted((RESULTS / "metrics" / "transfer").glob("transfer_*.json"))
    return fig, caption, sources


def fig_zeroday(smoke: bool):
    path = (RESULTS / "metrics" / "zeroday"
            / f"{'smoke_' if smoke else ''}zeroday_{PRIMARY}.json")
    doc = load(path)
    if not doc.get("results"):
        raise Skip("the zero-day results file holds no families")
    key = f"at_{doc['reported_at_prevalence']:.0%}"
    thin_below = doc["config"]["thin_below_test_positives"]
    fams = sorted(doc["results"].items(), key=lambda kv: kv[1]["test_positives"])
    fig, ax = plt.subplots(figsize=(WIDTH, 0.42 * len(fams) + 1.0))
    labels, any_thin = [], False
    for i, (name, r) in enumerate(fams):
        vals = [run[key]["recall"] for run in r["runs"]
                if run[key].get("recall") is not None]
        thin = r["test_positives"] < thin_below
        any_thin |= thin
        seed_dots(ax, i, vals, BLUE, horizontal=True, hollow=thin, fmt=".2f")
        labels.append(f"{name}  ({r['test_positives']:,} flows)")
    ax.set_yticks(range(len(fams)), labels)
    ax.set_xlim(-0.03, 1.08)
    ax.set_ylim(-0.6, len(fams) - 0.25)
    ax.set_xlabel("Recall on the attack family held out of training")
    ax.grid(axis="y", visible=False)
    skipped = ", ".join(doc.get("skipped", {})) or "none"
    caption = (f"Leave-one-attack-out on {PRIMARY}: recall on each attack family when "
               f"that family was removed entirely from training and validation. Each "
               f"dot is one seed; the tick is the mean. Families are ordered by the "
               f"number of held-out flows in the test split, shown in brackets."
               + (f" Hollow dots mark families with fewer than {thin_below:,} test "
                  f"flows, whose recall is correspondingly uncertain." if any_thin else "")
               + f" Not measurable on this dataset: {skipped}.")
    return fig, caption, [path]


def evidence_summary(smoke: bool) -> tuple[dict, Path]:
    path = (RESULTS / "metrics" / "evidence"
            / f"{'smoke_' if smoke else ''}evidence_{PRIMARY}_summary.json")
    return load(path), path


def fig_channel_weights(smoke: bool):
    doc, path = evidence_summary(smoke)
    fams = {k: v for k, v in doc["summary"]["topological_weight"]["by_true_family"].items()
            if v}
    if not fams:
        raise Skip("no channel weights in the evidence summary")
    order = sorted(fams.items(), key=lambda kv: kv[1]["median"])
    fig, ax = plt.subplots(figsize=(WIDTH * 0.85, 0.36 * len(order) + 1.0))
    ax.axvline(0.5, color=INK_2, lw=0.8, zorder=1)
    labels = []
    for i, (name, d) in enumerate(order):
        fp = name == "Benign"
        colour = ORANGE if fp else BLUE
        ax.plot([d["p25"], d["p75"]], [i, i], color=colour, lw=2, zorder=2,
                solid_capstyle="round")
        ax.plot(d["median"], i, "o", ms=7, mfc=colour, mec=SURFACE, mew=1.2, zorder=3)
        labels.append(f"{'false positives' if fp else name}  (n = {d['n']})")
    ax.set_yticks(range(len(order)), labels)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Share of the decision assigned to the neighbourhood channel")
    ax.annotate("equal split", (0.5, len(order) - 0.4), xytext=(3, -2),
                textcoords="offset points", va="top", fontsize=8, color=INK_2)
    ax.grid(axis="y", visible=False)
    ax.plot([], [], "o-", ms=7, lw=2, color=BLUE, mec=SURFACE, label="True attacks, by family")
    ax.plot([], [], "o-", ms=7, lw=2, color=ORANGE, mec=SURFACE,
            label="Benign flows wrongly flagged")
    ax.legend(loc="lower right", handletextpad=0.4)
    s = doc["summary"]
    caption = (f"How the fusion layer divides each decision between a flow's own "
               f"features and its neighbourhood, over {s['n_packs']} flagged flows. "
               f"Dot: median; bar: interquartile range; n: evidence packs in the "
               f"group. The sample is stratified by family, so the groups are small "
               f"and the ordering is indicative.")
    return fig, caption, [path]


def fig_top_features(smoke: bool):
    doc, path = evidence_summary(smoke)
    f = doc["summary"]["features_shown_to_the_model"]
    ranked = f["ranked"][:10][::-1]
    if not ranked:
        raise Skip("no attributed features in the evidence summary")
    fig, ax = plt.subplots(figsize=(WIDTH * 0.8, 0.3 * len(ranked) + 0.9))
    ax.barh(range(len(ranked)), [r["share_of_packs"] for r in ranked], height=0.5,
            color=BLUE, zorder=2)
    ax.set_yticks(range(len(ranked)), [r["feature"] for r in ranked])
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Share of flagged flows with this feature among the top "
                  f"{f['top_k']} attributed")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    caption = (f"The features that carry the detections: for each, the share of "
               f"{doc['summary']['n_packs']} evidence packs in which integrated "
               f"gradients ranks it among the {f['top_k']} most influential. A column "
               f"and its encoded variants are counted once per pack.")
    return fig, caption, [path]


def fig_confusion(smoke: bool):
    doc, path = gnn_results(PRIMARY, smoke)
    runs = doc["results"].get("full", {}).get("runs", [])
    conf = (runs[0].get("multiclass") or {}).get("confusion") if runs else None
    if not conf:
        raise Skip("the results predate scoring of the attack-family head "
                   "(re-run train_gnn.py)")
    names = sorted(conf, key=lambda n: (n != "Benign", n))
    cols = sorted({p for row in conf.values() for p in row} | set(names),
                  key=lambda n: (n != "Benign", n))
    m = np.array([[conf[t].get(p, 0) for p in cols] for t in names], dtype=float)
    share = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(WIDTH * 0.8, WIDTH * 0.8 * len(names) / len(cols) + 0.9))
    im = ax.imshow(share, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right")
    ax.set_yticks(range(len(names)), [f"{n}  ({int(m[i].sum()):,})"
                                      for i, n in enumerate(names)])
    ax.set_xlabel("Family the model named")
    ax.set_ylabel("True family  (flows in the test split)")
    ax.grid(False)
    for i in range(len(names)):                     # label only what matters
        for j in range(len(cols)):
            if share[i, j] >= 0.05:
                ax.text(j, i, f"{share[i, j]:.2f}", ha="center", va="center",
                        fontsize=7.5, color=SURFACE if share[i, j] > 0.55 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    bar.outline.set_visible(False)
    bar.set_label("Share of the true family", color=INK_2)
    mc = runs[0]["multiclass"]
    caption = (f"The attack-family head on the {PRIMARY} test split, seed "
               f"{runs[0].get('seed', 0)}: each row shows how flows of one true family "
               f"were named. Cells below 0.05 are left unlabelled. Macro-F1 "
               f"{mc['macro_f1']:.3f}; on true-positive detections the named family "
               f"is correct for {mc['on_true_positive_detections']['family_accuracy']:.1%}.")
    return fig, caption, [path]


HOSTING = {"stub": "template, no model", "anthropic": "cloud", "gemini": "cloud",
           "openai": "cloud", "ollama": "self-hosted"}


def fig_llm(smoke: bool, rows: list[dict] | None = None):
    if rows is None:
        rows = script("summarise_llm_study").rows()
    rows = [r for r in rows if r["prompt"] == "v1" and r["groundedness"] is not None]
    models = [r for r in rows if r["provider"] != "stub"]
    if not models:
        raise Skip("no language-model runs yet -- only the template control exists")
    models.sort(key=lambda r: r["groundedness"])
    stub = next((r for r in rows if r["provider"] == "stub"), None)
    fig, ax = plt.subplots(figsize=(WIDTH, 0.42 * len(models) + 1.25))
    if stub:
        ax.axvline(stub["groundedness"], color=INK_2, lw=0.8, zorder=1)
        ax.annotate("template\ncontrol", (stub["groundedness"], -0.55),
                    xytext=(-3, 0), textcoords="offset points", ha="right",
                    va="bottom", fontsize=8, color=INK_2)
    xs = []
    for i, r in enumerate(models):
        fp = r.get("groundedness_on_fp")
        if fp is not None:
            ax.plot([r["groundedness"], fp], [i, i], color=GRID, lw=2, zorder=2)
            ax.plot(fp, i, "o", ms=7, mfc=ORANGE, mec=SURFACE, mew=1.2, zorder=3)
            xs.append(fp)
        ax.plot(r["groundedness"], i, "o", ms=7, mfc=BLUE, mec=SURFACE, mew=1.2, zorder=4)
        xs.append(r["groundedness"])
    if stub:
        xs.append(stub["groundedness"])
    ax.set_yticks(range(len(models)),
                  [f"{r['model']}\n{HOSTING.get(r['provider'], r['provider'])}, "
                   f"n = {r['n_ok']}" for r in models])
    lo = min(xs)
    ax.set_xlim(max(0.0, lo - 0.25 * (1 - lo) - 0.01), 1.0 + 0.02 * (1 - lo) + 0.004)
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_xlabel("Groundedness: share of technical entities in a report "
                  "that appear in its evidence")
    ax.grid(axis="y", visible=False)
    ax.plot([], [], "o", ms=7, mfc=BLUE, mec=SURFACE, label="All reports")
    ax.plot([], [], "o", ms=7, mfc=ORANGE, mec=SURFACE,
            label="Reports on false-positive detections")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
              handletextpad=0.2, columnspacing=1.6, borderaxespad=0.2)
    caption = ("Groundedness of each language model's reports on identical evidence "
               "packs and an identical prompt. Blue: all reports. Orange: the "
               "false-positive arm, where the detector was wrong. The line is the "
               "template control, which invents nothing by construction. n is the "
               "number of reports scored. Groundedness measures whether a report "
               "invented anything, not whether its reasoning is correct.")
    sources = sorted((RESULTS / "reports").glob("reports_*_v1.json"))
    return fig, caption, sources


FIGURES = {
    "ablation": ("results_01_topology_ablation", fig_ablation),
    "gnn_vs_xgboost": ("results_02_gnn_vs_xgboost", fig_gnn_vs_xgboost),
    "topology_by_dataset": ("results_03_topology_gain_by_dataset", fig_topology_by_dataset),
    "zeroday": ("results_04_zero_day_recall", fig_zeroday),
    "transfer": ("results_05_cross_dataset_transfer", fig_transfer),
    "channel_weights": ("results_06_channel_weights", fig_channel_weights),
    "top_features": ("results_07_top_features", fig_top_features),
    "confusion": ("results_08_attack_family_confusion", fig_confusion),
    "llm": ("results_09_llm_groundedness", fig_llm),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES), default=None)
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "figures")
    ap.add_argument("--smoke", action="store_true",
                    help="draw from smoke-run files to check layout; refuses to "
                         "write into results/figures")
    args = ap.parse_args()
    if args.smoke and args.out_dir.resolve() == (RESULTS / "figures").resolve():
        raise SystemExit("--smoke draws meaningless numbers; give it an --out-dir "
                         "outside results/figures")

    style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for key in args.only or FIGURES:
        stem, draw = FIGURES[key]
        try:
            fig, caption, sources = draw(args.smoke)
        except Skip as why:
            print(f"  skipped  {key:<20} {why}")
            manifest[key] = {"status": "skipped", "reason": str(why)}
            continue
        out = args.out_dir / f"{stem}.png"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        src = []
        for p in sources:
            p = Path(p)
            when = None
            if p.suffix == ".json":
                loaded = json.loads(p.read_text())
                when = loaded.get("generated_at") if isinstance(loaded, dict) else None
            src.append({"file": str(p.relative_to(REPO_ROOT)), "generated_at": when})
        manifest[key] = {"status": "drawn", "file": out.name, "caption": caption,
                         "sources": src}
        print(f"  drew     {key:<20} {out.name}")

    if not args.only:
        (args.out_dir / "results_manifest.json").write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "smoke": args.smoke, "figures": manifest}, indent=2))
    drawn = sum(1 for m in manifest.values() if m["status"] == "drawn")
    print(f"\n{drawn} of {len(manifest)} figures drawn -> {args.out_dir}\n")


if __name__ == "__main__":
    main()
