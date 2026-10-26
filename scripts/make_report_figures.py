"""Publication figures for the final report: Figures 1-8 and Equations 1-3.

Writes every figure the report text refers to by number, under the file name the
text uses, into one folder. The drawing functions keep the numbers of the first
draft (fig09_...), which had 20 figures; REPORT_NUMBER maps the eight the report
kept when it was shortened (2026-10-26) to their numbers in the report. The other
twelve are still drawn, under their draft numbers, when named with --only. It
exists separately from `make_figures.py` (the
working figures behind `results/report_facts.md`) because the report needs a
fixed numbering, a print style and the diagrams, which the working set has not.

**Every plotted number is read from a results file when this runs** --
`results/metrics/`, `results/reports/`, `results/evidence/` and
`results/superseded/` -- using the same derivations as
`scripts/export_report_facts.py`, so a figure always agrees with the facts file.
Nothing measured is typed in here. Three inputs are not results files and are
named where they are used: the Defect Register in the project notes (draft
Figure 7), the per-class counts of the preprocessed NF-ToN-IoT-v2 subset, which
only the processed data holds (draft Figure 8, cross-checked against the
zero-day results), and `configs/report_timeline.yaml` (report Figure 8).

Style, the same in every figure so they read as a set: white background, no
title inside the image (the caption belongs to the report), 6.5 in wide at
300 dpi, and one colour per role throughout --

    graph model (full)            blue          flow features only (channel 1)  orange
    neighbourhood only (ch. 2)    teal          XGBoost and other flat models   grey-purple
    floors, controls, references  grey dashed   bad / missed / removed          vermillion

Every PR-AUC axis states the random-model floor (the reporting prevalence).
Error bars are one standard deviation over seeds; seeds are drawn as dots.

Usage:
    python scripts/make_report_figures.py --out <dir>               # the report's set
    python scripts/make_report_figures.py --out <dir> --only 9 12 eq1  # draft numbers
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import re
import shutil
import statistics
import subprocess
import tempfile
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
METRICS = RESULTS / "metrics"
REPORTS = RESULTS / "reports"
ARCHIVE = RESULTS / "superseded" / "2026-10-10_duration_artefact"
TIMELINE = REPO_ROOT / "configs" / "report_timeline.yaml"
PRIMARY, SECONDARY = "NF-ToN-IoT-v2", "NF-UNSW-NB15-v2"
BASELINE_SCHEMA = "dual-prevalence-v1"
MAIN_PROMPT, ALT_PROMPT = "v3", "v4"
# The facts file states plain English runs at roughly 1.3-1.7 tokens a word
# (export_report_facts._hidden_output). A reference band, not a result.
PLAIN_ENGLISH_TOKENS_PER_WORD = (1.3, 1.7)

# ------------------------------------------------------------------ style
# Roles, not series: a colour means the same thing in every figure. Blue, orange
# and teal are the first three slots of the dataviz reference palette, which
# pass colour-blind separation for every pairing. The comparator grey-purple is
# a deliberately low-chroma neutral (validated against the three: CVD dE >= 9.2,
# normal-vision dE >= 15.3); vermillion is the palette's "critical" status step.
SURFACE, INK, INK_2, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e4e3df", "#c3c2b7"
FULL, CH1, CH2 = "#2a78d6", "#eb6834", "#1baf7a"
TREE, REF, BAD = "#9a8cba", "#898781", "#d03b3b"
DATA = "#6b7f99"          # descriptive data with no model role (class counts etc.)
YELLOW = "#eda100"        # diagrams only, as a pastel: the language-model stage
WIDTH = 6.5               # inches; the report's text width
FONT = ["Arial", "Helvetica Neue", "DejaVu Sans"]

VARIANTS = ("channel1_only", "channel2_only", "full")
VARIANT_NAME = {"full": "Graph model (full)",
                "channel1_only": "Flow features only (channel 1)",
                "channel2_only": "Neighbourhood only (channel 2)"}
VARIANT_COLOUR = {"full": FULL, "channel1_only": CH1, "channel2_only": CH2}
BASELINE_MODEL = {"xgboost": "XGBoost", "random_forest": "Random forest",
                  "logreg": "Logistic regression", "mlp": "MLP"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT,
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS,
        "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
        "grid.linestyle": "-", "axes.axisbelow": True, "axes.linewidth": 0.7,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "legend.frameon": False,
        "figure.dpi": 100, "savefig.dpi": 300,
        "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 6, "axes.unicode_minus": True,
        "mathtext.fontset": "dejavusans",
    })


def tint(hex_colour: str, amount: float) -> str:
    """The colour mixed with white: `amount` of the colour, the rest white."""
    rgb = [int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]
    mixed = [round(c * amount + 255 * (1 - amount)) for c in rgb]
    return "#" + "".join(f"{c:02x}" for c in mixed)


def new_fig(height: float, ncols: int = 1, **kw):
    return plt.subplots(1, ncols, figsize=(WIDTH, height), layout="constrained", **kw)


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)


def value_grid(ax, axis: str) -> None:
    """Light grid on the value axis only."""
    ax.grid(True, axis=axis, color=GRID, linewidth=0.6)


def panel_label(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=9, fontweight="bold", color=INK)


# ------------------------------------------------------------------ data

def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def script(name: str):
    """Import a sibling script as a module, as make_figures.py does."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gnn(dataset: str, root: Path = METRICS) -> dict:
    return load(root / "gnn" / f"gnn_{dataset}.json")


def at_key(doc: dict) -> str:
    return f"at_{doc['config']['eval']['target_prevalence']:.0%}"


def floor_of(doc: dict) -> float:
    return doc["config"]["eval"]["target_prevalence"]


def baselines(dataset: str, root: Path = METRICS) -> dict:
    """Baselines on the current reporting path only (the D26 schema)."""
    doc = load(root / "baselines" / f"baselines_{dataset}.json")
    return {k: v for k, v in doc["results"].items() if v.get("schema") == BASELINE_SCHEMA}


def baseline_name(key: str) -> str:
    model, features = key.split("::")
    return f"{BASELINE_MODEL.get(model, model)} ({features})"


def runs_metric(runs: list[dict], key: str, metric: str) -> list[float]:
    return [r[key][metric] for r in runs if isinstance(r[key].get(metric), (int, float))]


def agg(entry: dict, metric: str) -> tuple[float, float, int]:
    a = entry["aggregate"][metric]
    return a["mean"], a["std"], a["n_seeds"]


# ------------------------------------------------------------------ marks

def seed_strip(ax, pos: float, values: list[float], colour: str, mean: float, sd: float,
               *, horizontal: bool = False, label: str | None = None,
               hollow: bool = False, offset: float = 0.13) -> None:
    """Seeds as small dots on one side of the row, mean +/- sd on the other.

    The jitter is fixed (seeded) so the figure is identical on every run.
    """
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.045, 0.045, len(values))
    seed_pos = pos - offset + jitter
    mean_pos = pos + offset * 0.75
    face = SURFACE if hollow else colour
    if horizontal:
        ax.scatter(values, seed_pos, s=16, color=face, edgecolor=colour, linewidth=0.9,
                   alpha=0.9 if hollow else 0.6, zorder=3)
        ax.errorbar(mean, mean_pos, xerr=sd, fmt="none", ecolor=colour, elinewidth=1.5,
                    capsize=3, capthick=1.2, zorder=4)
        ax.plot(mean, mean_pos, "o", ms=7, mfc=face, mec=colour if hollow else SURFACE,
                mew=1.4, zorder=5)
        if label:
            ax.annotate(label, (mean + sd, mean_pos), xytext=(6, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=8, color=INK)
    else:
        ax.scatter(seed_pos, values, s=16, color=face, edgecolor=colour, linewidth=0.9,
                   alpha=0.9 if hollow else 0.6, zorder=3)
        ax.errorbar(mean_pos, mean, yerr=sd, fmt="none", ecolor=colour, elinewidth=1.5,
                    capsize=3, capthick=1.2, zorder=4)
        ax.plot(mean_pos, mean, "o", ms=7, mfc=face, mec=colour if hollow else SURFACE,
                mew=1.4, zorder=5)
        if label:
            ax.annotate(label, (mean_pos, mean), xytext=(9, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=8, color=INK)


def ref_line(ax, value: float, *, vertical: bool = False, text: str | None = None,
             where: float = 1.0, ha: str = "right", va: str = "bottom",
             colour: str = REF) -> None:
    """A dashed reference line (floor, control, threshold) with a muted label."""
    (ax.axvline if vertical else ax.axhline)(value, color=colour, lw=1.0, ls=(0, (4, 3)),
                                             zorder=1)
    if text:
        if vertical:
            ax.annotate(text, (value, where), xycoords=("data", "axes fraction"),
                        xytext=(4 if ha == "left" else -4, 0), textcoords="offset points",
                        ha=ha, va=va, fontsize=7.5, color=INK_2)
        else:
            ax.annotate(text, (where, value), xycoords=("axes fraction", "data"),
                        xytext=(0, 3 if va == "bottom" else -3), textcoords="offset points",
                        ha=ha, va=va, fontsize=7.5, color=INK_2)


def dumbbell(ax, y: float, before: float, after: float, colour: str) -> None:
    """Hollow grey = the comparison condition; filled role colour = the other."""
    ax.plot([before, after], [y, y], color=tint(colour, 0.45), lw=2.0, zorder=2,
            solid_capstyle="round")
    ax.plot(before, y, "o", ms=6.5, mfc=SURFACE, mec=INK_2, mew=1.2, zorder=3)
    ax.plot(after, y, "o", ms=7, mfc=colour, mec=SURFACE, mew=1.2, zorder=4)
    if abs(after - before) < 1e-9:
        ax.plot(after, y, "o", ms=10, mfc="none", mec=INK_2, mew=1.2, zorder=5)


def hollow_handle(label: str) -> Line2D:
    return Line2D([], [], ls="none", marker="o", ms=6.5, mfc=SURFACE, mec=INK_2, mew=1.2,
                  label=label)


def filled_handle(label: str, colour: str) -> Line2D:
    return Line2D([], [], ls="none", marker="o", ms=7, mfc=colour, mec=SURFACE, mew=1.2,
                  label=label)


def signed(v: float, places: int = 4) -> str:
    return f"{v:+.{places}f}".replace("-", "\u2212")


def compact(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 10_000:
        return f"{n / 1e3:.1f}k"
    return f"{n:,}"


# ------------------------------------------------------------------ diagrams

DOT_FONT = "Arial"


def render_dot(src: str, path: Path, engine: str = "dot", extra: tuple = ()) -> None:
    exe = shutil.which(engine) or f"/opt/homebrew/bin/{engine}"
    subprocess.run([exe, *extra, "-Tpng", "-Gdpi=300", "-o", str(path)],
                   input=src.encode(), check=True)


def html(title: str, *lines: str, size: float = 7.5) -> str:
    """A node label: bold title (newlines allowed), then smaller detail lines.

    Each line is its own table row: Graphviz sizes a row to its content, so a
    subscript cannot run into the line below (it does with plain <BR/> breaks).
    """
    rows = [f"<TR><TD><B>{t}</B></TD></TR>" for t in title.split("\n")]
    rows += [f'<TR><TD><FONT POINT-SIZE="{size}">{line}</FONT></TD></TR>' for line in lines]
    return f'<<TABLE BORDER="0" CELLSPACING="1" CELLPADDING="0">{"".join(rows)}</TABLE>>'


def sub(text: str, size: float = 6.5) -> str:
    """A subscript at a smaller size, so it does not collide with the next line."""
    return f'<FONT POINT-SIZE="{size}"><SUB>{text}</SUB></FONT>'


def box(name: str, label: str, fill: str, border: str, *, x: float | None = None,
        y: float | None = None, w: float | None = None, h: float | None = None,
        dashed: bool = False, extra: str = "") -> str:
    """One node. With x/y (inches) the node is pinned, for `neato -n2`."""
    attrs = [f"label={label}", f'fillcolor="{fill}"', f'color="{border}"',
             f'style="rounded,filled{",dashed" if dashed else ""}"']
    if x is not None:
        attrs.append(f'pos="{x * 72:.1f},{y * 72:.1f}"')
    if w is not None:
        attrs += [f"width={w}", f"height={h}", "fixedsize=true"]
    if extra:
        attrs.append(extra)
    return f"  {name} [{', '.join(attrs)}];"


def dot_header(rankdir: str = "LR", **graph) -> str:
    # size: never wider than the text width; a smaller diagram keeps its natural size
    g = {"bgcolor": "white", "pad": "0.08", "rankdir": rankdir, "fontname": DOT_FONT,
         "size": f"{WIDTH - 0.17},9", **graph}
    gattrs = " ".join(f'{k}="{v}"' for k, v in g.items())
    return ("digraph G {\n"
            f"  graph [{gattrs}];\n"
            f'  node [shape=box, fontname="{DOT_FONT}", fontsize=8.5, fontcolor="{INK}", '
            'penwidth=0.9, margin="0.07,0.05"];\n'
            f'  edge [color="{INK_2}", penwidth=0.9, arrowsize=0.6, fontname="{DOT_FONT}", '
            f'fontsize=7.5, fontcolor="{INK_2}"];\n')


# ------------------------------------------------------------------ figures

def fig01_pipeline(out: Path) -> None:
    """Pipeline overview: detection on the top row, explanation on the bottom row
    (read right to left), evaluation branching off the graph model between them."""
    pre = load(METRICS / f"preprocess_{PRIMARY}.json")
    n_artefact = len(pre["config"]["features"]["drop_artefacts"])
    n_shortcut = len(pre["schema"]["dropped"]["shortcuts"])
    window = pre["config"]["graph"]["window_size"]
    n_models = len(list(REPORTS.glob(f"reports_ollama_*_{MAIN_PROMPT}.json")))

    neutral, blue, warm = tint(DATA, 0.14), tint(FULL, 0.16), tint(YELLOW, 0.18)
    nb, bb, wb = DATA, FULL, "#b07a00"
    W, H = 0.98, 0.86
    cx = [0.52 + i * 1.09 for i in range(6)]
    top, mid, bot = 3.12, 1.98, 0.80
    band = "#f5f5f3"
    title = f'<<FONT POINT-SIZE="7.5" COLOR="{INK_2}"><B>{{}}</B></FONT>>'
    nodes = [
        # background bands, declared first so they are drawn underneath
        box("bg1", title.format("DETECTION"), band, band, x=3.245, y=top + 0.08, w=6.62,
            h=1.26, extra='labelloc="t"'),
        box("bg2", title.format("EXPLANATION"), band, band, x=3.245, y=bot + 0.08, w=6.62,
            h=1.26, extra='labelloc="t"'),
        box("csv", html("NetFlow v2\nCSV files", "two datasets"), neutral, nb,
            x=cx[0], y=top - 0.06, w=W, h=H),
        box("audit", html("Integrity\naudit", "checksums and", "data-quality checks"),
            neutral, nb, x=cx[1], y=top - 0.06, w=W, h=H),
        box("prep", html("Preprocessing", f"{n_artefact} artefact +", f"{n_shortcut} shortcut",
                         "columns removed"), neutral, nb, x=cx[2], y=top - 0.06, w=W, h=H),
        box("snap", html("Graph\nsnapshots", f"{window:,}-flow windows", "hosts = nodes,",
                         "flows = edges"), neutral, nb, x=cx[3], y=top - 0.06, w=W, h=H),
        box("gnn", html("Dual-channel\nGNN", "flow features +", "neighbourhood"),
            blue, bb, x=cx[4], y=top - 0.06, w=W, h=H),
        box("det", html("Detection", "score against", "threshold,", "attack family"),
            blue, bb, x=cx[5], y=top - 0.06, w=W, h=H),
        box("eval", html("Evaluation", "in-dataset ·", "leave-one-attack-out ·",
                         "cross-dataset transfer"),
            "white", MUTED, x=cx[4], y=mid, w=1.40, h=0.66, dashed=True),
        box("pack", html("Evidence\npack", "measured values,", "attributions, context"),
            blue, bb, x=cx[5], y=bot - 0.06, w=W, h=H),
        box("facts", html("Facts layer", "code computes", "and words every", "statement"),
            neutral, nb, x=cx[4], y=bot - 0.06, w=W, h=H),
        box("llm", html("Language\nmodel", f"{n_models} models, Ollama", "hosted service"),
            warm, wb, x=cx[3], y=bot - 0.06, w=W, h=H),
        box("note", html("Incident\nreport", "plain-language", "note"),
            warm, wb, x=cx[2], y=bot - 0.06, w=W, h=H),
        box("score", html("Automated\nscoring", "groundedness,", "fidelity, direction"),
            "white", MUTED, x=cx[1], y=bot - 0.06, w=W, h=H, dashed=True),
    ]
    edges = ["csv -> audit", "audit -> prep", "prep -> snap", "snap -> gnn", "gnn -> det",
             "det -> pack", "pack -> facts", "facts -> llm", "llm -> note", "note -> score",
             f'gnn -> eval [style=dashed, color="{MUTED}"]']
    src = (dot_header(splines="line", outputorder="nodesfirst") + "\n".join(nodes) + "\n"
           + "\n".join(f"  {e};" for e in edges) + "\n}\n")
    render_dot(src, out / "Figure 01 - Pipeline overview.png", "neato", ("-n2",))


def fig02_shortcut_audit(out: Path) -> None:
    path = METRICS / f"audit_{SECONDARY}-selfcheck.json"
    d = load(path)
    threshold, shortcuts = d["shortcut_threshold"], set(d["shortcut_features"])
    ranked = sorted(d["per_feature_auc"].items(), key=lambda kv: -kv[1]["abs_auc"])[:20]
    kept = tint(DATA, 0.55)
    fig, ax = new_fig(4.6)
    ys = np.arange(len(ranked))[::-1]
    for y, (name, v) in zip(ys, ranked):
        removed = name in shortcuts
        ax.barh(y, v["abs_auc"] - 0.5, left=0.5, height=0.66, zorder=2,
                color=BAD if removed else kept)
        if removed:
            ax.annotate(f"{v['abs_auc']:.3f}  removed", (v["abs_auc"], y), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK,
                        fontweight="bold")
    ax.set_yticks(ys, [n for n, _ in ranked], fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.5, 1.07)
    ax.set_ylim(-0.7, len(ranked) - 0.3)
    ref_line(ax, threshold, vertical=True,
             text=f"shortcut threshold,\n|AUC| = {threshold:.2f}", where=0.03, ha="left",
             va="bottom")
    ax.set_xlabel(f"Single-feature |AUC|, attack against benign, on {SECONDARY} "
                  "(0.5 = no signal)")
    value_grid(ax, "x")
    ax.legend(handles=[Patch(color=BAD, label="Shortcut feature, removed before training"),
                       Patch(color=kept, label="Other feature (top 20 of "
                             f"{len(d['per_feature_auc'])} shown)")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=8,
              borderaxespad=0.3)
    save(fig, out / "Figure 02 - Shortcut feature audit.png")


def fig03_duration_wrap(out: Path) -> None:
    d = load(METRICS / f"data_quality_{PRIMARY}.json")
    w = d["validity_flags"]["duration_wrapped"]
    fams = sorted(w["share_by_family"].items(), key=lambda kv: kv[1])
    fig, ax = new_fig(3.1)
    ys = np.arange(len(fams))
    ax.barh(ys, [100 * v for _, v in fams], height=0.62, color=DATA, zorder=2)
    for y, (_, v) in zip(ys, fams):
        ax.annotate(f"{100 * v:.1f}%", (100 * v, y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK, zorder=5,
                    bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none"))
    ax.set_yticks(ys, [k for k, _ in fams])
    ax.tick_params(axis="y", length=0)
    ref_line(ax, 100 * w["share"], vertical=True,
             text=f"all flows: {100 * w['share']:.2f}%", where=0.02, ha="left")
    ax.set_xlim(0, 100 * max(v for _, v in fams) * 1.12)
    ax.set_ylim(-0.6, len(fams) - 0.4)
    ax.set_xlabel(f"Flows whose FLOW_DURATION_MILLISECONDS is a 32-bit counter wrap (%)\n"
                  f"{PRIMARY}, {d['n_rows']:,} flows, by class")
    value_grid(ax, "x")
    save(fig, out / "Figure 03 - Duration wrap by family.png")


def fig04_dual_channel(out: Path) -> None:
    """The dual-channel model, top to bottom: two parallel channels merging."""
    neutral, orange, teal, blue = (tint(DATA, 0.14), tint(CH1, 0.18), tint(CH2, 0.18),
                                   tint(FULL, 0.16))
    window = load(METRICS / f"preprocess_{PRIMARY}.json")["config"]["graph"]["window_size"]
    layers = {r["n_gnn_layers"] for r in gnn(PRIMARY)["results"]["full"]["runs"]}
    if len(layers) != 1:
        raise SystemExit(f"Figure 4: the full-model runs disagree on depth: {layers}")
    depth = {1: "one", 2: "two", 3: "three"}.get(*layers, str(*layers))
    s = 8.5
    z1, z2 = f"z{sub('1')}", f"z{sub('2')}"
    a1, a2 = f"α{sub('1')}", f"α{sub('2')}"
    nodes = [
        box("x", html(f"Flow features x{sub('uv', 7.5)}", "one flow's measured", "features",
                      size=s), neutral, DATA),
        box("g", html("Window graph", f"{window:,} flows: hosts as", "nodes, flows as edges",
                      size=s), neutral, DATA),
        box("c1", html("Channel 1: the flow", "MLP with residual", "connections",
                       f"→ flow representation {z1}", size=s), orange, CH1),
        box("c2", html("Channel 2: the neighbourhood", f"{depth} E-GraphSAGE layers",
                       "message = neighbour state", "+ flow features",
                       "mean aggregation, both directions",
                       f"→ edge representation {z2}", "from its two endpoints", size=s),
            teal, CH2),
        box("fuse", html("Attention fusion", f"weights {a1}, {a2} per flow",
                         f"({a1} + {a2} = 1)", size=s), blue, FULL),
        box("z", html("Shared representation", f"z = {a1}{z1} + {a2}{z2}", size=s), blue, FULL),
        box("hb", html("Binary head", "attack score", size=s), blue, FULL),
        box("hf", html("Family head", "attack family", size=s), blue, FULL),
    ]
    edges = ["x -> c1", "x -> c2", "g -> c2", "c1 -> fuse", "c2 -> fuse", "fuse -> z",
             "z -> hb", "z -> hf"]
    src = (dot_header("TB", nodesep="0.35", ranksep="0.30").replace("fontsize=8.5,",
                                                                     "fontsize=9.5,")
           + "\n".join(nodes)
           + "\n  {rank=same; x; g;}\n  {rank=same; c1; c2;}\n  {rank=same; hb; hf;}\n"
           + "\n".join(f"  {e};" for e in edges) + "\n}\n")
    render_dot(src, out / "Figure 04 - Dual-channel model.png")


def fig05_facts_layer(out: Path) -> None:
    """Evidence pack to scored note. Top row left to right, bottom row right to left."""
    n_models = len(list(REPORTS.glob(f"reports_ollama_*_{MAIN_PROMPT}.json")))
    blue, warm, neutral = tint(FULL, 0.16), tint(YELLOW, 0.18), tint(DATA, 0.14)
    sections = ["The detection", "The flow, as measured", "Why the detector flagged it",
                "Background context", "What is not known", "Reference guidance"]
    rows = "".join(f'<TR><TD ALIGN="LEFT" BGCOLOR="white" CELLPADDING="2">'
                   f'<FONT POINT-SIZE="7.5">{i}. {s}</FONT></TD></TR>'
                   for i, s in enumerate(sections, 1))
    facts = (f'<<TABLE BORDER="0" CELLSPACING="2" CELLPADDING="1">'
             f'<TR><TD><B>Facts layer (code)</B></TD></TR>'
             f'<TR><TD><FONT POINT-SIZE="7.5">computes and words every</FONT></TD></TR>'
             f'<TR><TD><FONT POINT-SIZE="7.5">statement, in six sections:</FONT></TD></TR>'
             f"{rows}</TABLE>>")
    nodes = [
        box("pack", html("Evidence pack", "JSON from the GNN:", "detection", "flow as measured",
                         "attributions", "context", "data-quality notes"), blue, FULL),
        box("facts", facts, neutral, DATA),
        box("prompt", html("Prompt", "v3 prescriptive", "or v4 permissive"), warm, "#b07a00"),
        box("llm", html("Language model", f"{n_models} models, Ollama", "hosted service"),
            warm, "#b07a00"),
        box("note", html("Incident note", "plain language"), warm, "#b07a00"),
        box("score", html("Automated scoring", "note checked against the pack:",
                          "groundedness · invented addresses", "direction kept",
                          "borderline conveyed · class named"),
            "white", MUTED, dashed=True),
    ]
    edges = ["pack -> facts", "facts -> prompt",
             'prompt -> llm [label=<<I> no arithmetic<BR/> by the model</I>>]',
             "note -> llm [dir=back]", "score -> note [dir=back]"]
    src = (dot_header("TB", nodesep="0.32", ranksep="0.30")
           + "\n".join(nodes)
           + "\n  {rank=same; pack; facts; prompt;}\n  {rank=same; score; note; llm;}\n"
           + "\n".join(f"  {e};" for e in edges) + "\n}\n")
    render_dot(src, out / "Figure 05 - Facts layer.png")


def fig06_architecture(out: Path) -> None:
    neutral, blue, purple, teal, warm = (tint(DATA, 0.14), tint(FULL, 0.16), tint(TREE, 0.22),
                                         tint(CH2, 0.16), tint(YELLOW, 0.18))
    clis = ["download_data", "preprocess", "train_baselines", "train_gnn", "train_zeroday",
            "transfer_eval", "make_evidence", "run_llm_study", "export_report_facts"]
    missing = [c for c in clis if not (REPO_ROOT / "scripts" / f"{c}.py").exists()]
    if missing:
        raise SystemExit(f"Figure 6 names scripts that do not exist: {missing}")
    modules = {"data": "loading, schema, splits", "graph": "snapshots, windows",
               "models": "dual channel, E-GraphSAGE", "training": "training loop",
               "eval": "metrics, zero-day, transfer", "explain": "attribution, evidence",
               "llm": "facts, prompts, scoring", "ui": "replay, Streamlit demo"}
    missing = [m for m in modules if not (REPO_ROOT / "src" / "gnnids" / m).is_dir()]
    if missing:
        raise SystemExit(f"Figure 6 names modules that do not exist: {missing}")
    cli_cells = "".join(
        "<TR>" + "".join(f'<TD ALIGN="LEFT"><FONT FACE="Menlo" POINT-SIZE="7.5">{c}&nbsp;&nbsp;</FONT></TD>'
                         for c in clis[i:i + 3]) + "</TR>" for i in range(0, len(clis), 3))
    scripts_label = (f'<<TABLE BORDER="0" CELLSPACING="1" CELLPADDING="1">'
                     f'<TR><TD COLSPAN="3"><B>scripts/</B> '
                     f'<FONT POINT-SIZE="7.5">thin command-line entry points</FONT></TD></TR>'
                     f"{cli_cells}</TABLE>>")
    mod_nodes = [box(f"m_{m}", f'<<B>{m}</B><BR/><FONT POINT-SIZE="7">{d}</FONT>>', blue, FULL,
                     extra="width=1.35")
                 for m, d in modules.items()]
    outputs = ["metrics JSON", "checkpoints", "evidence packs", "reports",
               "figures", "report_facts.md"]
    nodes = [
        box("cfg", html("configs/", "YAML: every experiment"), warm, "#b07a00"),
        box("data", html("data/", "not in git; fetched and", "checksum-verified"), neutral,
            DATA, dashed=True),
        box("scripts", scripts_label, purple, TREE),
        box("results", html("results/", " · ".join(outputs[:3]), " · ".join(outputs[3:])),
            teal, CH2),
    ]
    mods = list(modules)
    src = (dot_header("TB", nodesep="0.18", ranksep="0.26", compound="true")
           + "\n".join(nodes) + "\n"
           + '  subgraph cluster_pkg {\n    label=<<B>src/gnnids/</B>  '
           '<FONT POINT-SIZE="7.5">the pipeline package</FONT>>; labeljust="l";\n'
           f'    style="rounded,filled"; fillcolor="{tint(FULL, 0.05)}"; color="{FULL}";'
           ' penwidth=0.9; fontsize=8.5;\n'
           + "\n".join("  " + n for n in mod_nodes) + "\n"
           + f"    {{rank=same; {'; '.join('m_' + m for m in mods[:4])};}}\n"
           + f"    {{rank=same; {'; '.join('m_' + m for m in mods[4:])};}}\n"
           + "".join(f"    m_{a} -> m_{b} [style=invis];\n" for a, b in zip(mods[:4], mods[4:]))
           + "  }\n"
           + "  {rank=same; cfg; data;}\n"
           + "  cfg -> scripts [label=\" read by\"];\n"
           + "  scripts -> m_graph [lhead=cluster_pkg, label=\" call\"];\n"
           + "  data -> m_models [lhead=cluster_pkg, label=\" read by the data module\"];\n"
           + "  m_llm -> results [ltail=cluster_pkg, label=\" write\"];\n"
           + "}\n")
    render_dot(src, out / "Figure 06 - System architecture.png")


def fig07_defects(out: Path, register: Path) -> None:
    """Silent and non-silent defects per kind, parsed from the Defect Register.

    A row counts as silent when its last cell starts with YES (bold or not) and
    is not marked "(latent)": the figure defines silent as "produced plausible
    numbers", and a latent defect was caught before it produced anything (C24,
    "would have been ..."). Latent rows are printed so the rule stays visible.
    The counts the register's own header states are compared and any difference
    is printed, not forced: the rows are authoritative.
    """
    if not register.exists():
        raise FileNotFoundError(f"Defect Register not found at {register}; pass "
                                f"--defect-register")
    text = register.read_text()
    kinds = {"A": "Data", "B": "Analysis", "C": "Code"}
    counts = {k: {"silent": [], "not": []} for k in kinds}
    latent = []
    section = None
    for line in text.splitlines():
        if line.startswith("## "):
            m = re.match(r"## ([ABC])\. ", line)
            section = m.group(1) if m else None
            continue
        if section is None or not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
        ident = re.match(rf"^[*~]*({section}\d+)", cells[0])
        if not ident:
            continue
        flag = re.sub(r"[*~ ]", "", cells[-1]).upper()
        silent = flag.startswith("YES") and "(LATENT)" not in flag
        if flag.startswith("YES") and not silent:
            latent.append(ident.group(1))
        counts[section]["silent" if silent else "not"].append(ident.group(1))
    n = {k: {c: len(v[c]) for c in v} for k, v in counts.items()}
    totals = {k: v["silent"] + v["not"] for k, v in n.items()}
    n_silent = sum(v["silent"] for v in n.values())
    print("    Figure 7 silent rows: "
          + "; ".join(f"{k} {', '.join(v['silent'])}" for k, v in counts.items())
          + (f"; marked YES but latent, counted as not silent: {', '.join(latent)}"
             if latent else ""))
    claim = re.search(r"\*\*(\d+) logged\*\* — (\d+) data, (\d+) analysis, (\d+) code, "
                      r"(\d+) of them silent", text)
    if claim:
        stated = dict(zip(("all", "A", "B", "C", "silent"), map(int, claim.groups())))
        found = {"all": sum(totals.values()), **totals, "silent": n_silent}
        diff = {k: (stated[k], found[k]) for k in stated if stated[k] != found[k]}
        if diff:
            print(f"    NOTE Figure 7: the register's header and its rows disagree "
                  f"(header, rows): {diff}. Plotted from the rows.")
    fig, ax = new_fig(1.95)
    ys = np.arange(len(kinds))[::-1]
    quiet = tint(DATA, 0.35)
    for y, (k, name) in zip(ys, kinds.items()):
        s, ns = n[k]["silent"], n[k]["not"]
        ax.barh(y, s, height=0.5, color=BAD, zorder=2)
        ax.barh(y, ns, left=s, height=0.5, color=quiet, zorder=2, edgecolor=SURFACE,
                linewidth=1.5)
        for left, width, colour in ((0, s, SURFACE), (s, ns, INK)):
            if width >= 3:
                ax.text(left + width / 2, y, str(width), ha="center", va="center",
                        fontsize=8, color=colour, fontweight="bold")
        ax.annotate(f"{s + ns} logged", (s + ns, y), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_yticks(ys, [f"{name} ({k})" for k, name in kinds.items()])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(totals.values()) * 1.2)
    ax.set_ylim(-0.5, len(kinds) - 0.5)
    ax.set_xlabel(f"Defects logged ({sum(totals.values())} in all, {n_silent} silent)")
    value_grid(ax, "x")
    ax.legend(handles=[Patch(color=BAD, label="Silent: produced plausible numbers"),
                       Patch(color=quiet, label="Not silent: crashed, was visible, "
                             "or was caught before it ran")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=8,
              borderaxespad=0.3)
    save(fig, out / "Figure 07 - Defects by type.png")


def toniot_subset_counts() -> dict[str, int]:
    """Flows per class in the preprocessed NF-ToN-IoT-v2 subset.

    No results file holds these (the audit counted the whole source file), so
    they are counted from the processed data and checked against the zero-day
    results, which record each family's train and test counts.
    """
    import pyarrow.parquet as pq
    pre = load(METRICS / f"preprocess_{PRIMARY}.json")
    meta = REPO_ROOT / pre["config"]["output"]["dir"] / "meta.parquet"
    labels = pq.read_table(meta, columns=["Attack"]).column("Attack").to_numpy(
        zero_copy_only=False)
    if len(labels) != pre["n_rows"]:
        raise SystemExit(f"{meta} has {len(labels):,} rows; preprocessing reported "
                         f"{pre['n_rows']:,}")
    names, counts = np.unique(labels, return_counts=True)
    sp = pre["splits"]
    zd = load(METRICS / "zeroday" / f"zeroday_{PRIMARY}.json")
    for fam, r in zd["results"].items():
        train = int((labels[sp["train"]["start"]:sp["train"]["stop"]] == fam).sum())
        test = int((labels[sp["test"]["start"]:sp["test"]["stop"]] == fam).sum())
        if (train, test) != (r["train_rows_removed"], r["test_positives"]):
            raise SystemExit(f"Figure 8: {fam} counts {train}/{test} disagree with the "
                             f"zero-day file {r['train_rows_removed']}/{r['test_positives']}")
    return {str(n): int(c) for n, c in zip(names, counts)}


def fig08_class_balance(out: Path) -> None:
    ton = toniot_subset_counts()
    audit = load(METRICS / f"audit_{SECONDARY}-selfcheck.json")
    unsw = audit["families"]
    unsw_used = load(METRICS / "preprocess.json")["n_rows"]
    if sum(unsw.values()) != unsw_used:
        raise SystemExit("Figure 8: the UNSW audit does not cover every flow used")
    fig, axes = new_fig(3.2, 2)
    every = [*ton.values(), *unsw.values()]
    base = 10 ** np.floor(np.log10(min(every)))          # one shared axis for both panels
    for ax, (ds, counts, n) in zip(axes, ((PRIMARY, ton, sum(ton.values())),
                                          (SECONDARY, unsw, unsw_used))):
        order = sorted(counts.items(), key=lambda kv: kv[1])
        ys = np.arange(len(order))
        ax.barh(ys, [c - base for _, c in order], left=base, height=0.62, zorder=2,
                color=[MUTED if k == "Benign" else DATA for k, _ in order])
        for y, (k, c) in zip(ys, order):
            text = f"{c:,} flows" if k == "ransomware" else compact(c)
            ax.annotate(text, (c, y), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=INK,
                        fontweight="bold" if k == "ransomware" else "normal")
        ax.set_xscale("log")
        ax.set_xlim(base, max(every) * 9)
        ax.set_yticks(ys, [k for k, _ in order])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlabel("Flows (log scale)")
        panel_label(ax, f"{ds}  ({n:,} flows)")
        value_grid(ax, "x")
    save(fig, out / "Figure 08 - Class balance.png")


def fig09_topology_ablation(out: Path) -> None:
    doc = gnn(PRIMARY)
    key, floor = at_key(doc), floor_of(doc)
    res = doc["results"]
    fig, ax = new_fig(3.3)
    for i, v in enumerate(VARIANTS):
        vals = runs_metric(res[v]["runs"], key, "pr_auc")
        mean, sd, n = agg(res[v], "pr_auc")
        seed_strip(ax, i, vals, VARIANT_COLOUR[v], mean, sd, label=f"{mean:.4f}\n± {sd:.4f}")
    c1 = runs_metric(res["channel1_only"]["runs"], key, "pr_auc")
    full = runs_metric(res["full"]["runs"], key, "pr_auc")
    if min(full) > max(c1):
        ax.axhspan(max(c1), min(full), color=tint(FULL, 0.08), zorder=0, lw=0)
        ax.annotate(f"no overlap: channel 1 best seed {max(c1):.4f},\n"
                    f"full model worst seed {min(full):.4f}",
                    (0.02, (max(c1) + min(full)) / 2), xycoords=("axes fraction", "data"),
                    va="center", fontsize=7.5, color=INK_2)
    labels = [f"{VARIANT_NAME[v].replace(' (', chr(10) + '(')}\n"
              f"n = {res[v]['aggregate']['pr_auc']['n_seeds']} seeds" for v in VARIANTS]
    ax.set_xticks(range(len(VARIANTS)), labels)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, len(VARIANTS) - 0.2)
    ax.set_ylabel(f"PR-AUC at {floor:.0%} prevalence\n(random-model floor {floor:.2f})")
    value_grid(ax, "y")
    save(fig, out / "Figure 09 - Topology ablation.png")


def _artefact_rows() -> list[tuple[str, str, float, float]]:
    """(name, colour, with the artefact, without) on PRIMARY, over the seeds both
    runs share, as in export_report_facts.sec_step_back."""
    matched = script("export_report_facts")._matched_pr_auc
    rows = []
    old, new = gnn(PRIMARY, ARCHIVE / "metrics")["results"], gnn(PRIMARY)["results"]
    for v in ("full", "channel1_only"):
        rows.append((VARIANT_NAME[v], VARIANT_COLOUR[v], *matched(old[v], new[v])[:2]))
    ob, nb = baselines(PRIMARY, ARCHIVE / "metrics"), baselines(PRIMARY)
    for k in sorted(ob):
        if k in nb:
            rows.append((baseline_name(k), TREE, *matched(ob[k], nb[k])[:2]))
    return rows


def fig10_artefact_effect(out: Path) -> None:
    rows = sorted(_artefact_rows(), key=lambda r: r[3] - r[2])   # largest drop on top
    floor = floor_of(gnn(PRIMARY))
    fig, ax = new_fig(3.6)
    for y, (name, colour, before, after) in enumerate(rows):
        dumbbell(ax, y, before, after, colour)
        ax.annotate(signed(after - before), (1.0, y), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8,
                    color=INK, annotation_clip=False)
    ax.annotate("change", (1.0, -0.55), xycoords=("axes fraction", "data"),
                xytext=(6, 0), textcoords="offset points", va="bottom", fontsize=7.5,
                color=INK_2, annotation_clip=False)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(rows) - 0.4, -0.75)
    ax.set_xlabel(f"PR-AUC at {floor:.0%} prevalence on {PRIMARY} "
                  f"(random-model floor {floor:.2f})")
    value_grid(ax, "x")
    ax.legend(handles=[hollow_handle("With the duration artefact (archived run)"),
                       filled_handle("Without it (current run)", INK_2)],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.3)
    save(fig, out / "Figure 10 - Artefact effect.png")


def _novel_rows() -> list[tuple[str, str, float, float]]:
    """(name, colour, all test flows, novel only), PRIMARY, means over seeds, as in
    export_report_facts.sec_step_back."""
    rows = []
    res = gnn(PRIMARY)["results"]
    for v in ("full", "channel1_only", "channel2_only"):
        runs = res[v]["runs"]
        rows.append((VARIANT_NAME[v], VARIANT_COLOUR[v],
                     statistics.mean(r["at_4%"]["pr_auc"] for r in runs),
                     statistics.mean(r["novel_only"]["at_4%"]["pr_auc"] for r in runs)))
    for k, v in sorted(baselines(PRIMARY).items()):
        runs = v["runs"]
        rows.append((baseline_name(k), TREE, statistics.mean(r["at_4%"]["pr_auc"] for r in runs),
                     statistics.mean(r["novel_only"]["at_4%"]["pr_auc"] for r in runs)))
    return rows


def fig11_novel_only(out: Path) -> None:
    rows = sorted(_novel_rows(), key=lambda r: -r[3])
    doc = gnn(PRIMARY)
    floor = floor_of(doc)
    share = doc["results"]["full"]["runs"][0]["novel_only"]["share_of_test"]
    fig, ax = new_fig(3.8)
    for y, (name, colour, allf, novel) in enumerate(rows):
        dumbbell(ax, y, allf, novel, colour)
        ax.annotate(signed(novel - allf), (1.0, y), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8,
                    color=INK, annotation_clip=False)
    ax.annotate("change", (1.0, -0.55), xycoords=("axes fraction", "data"),
                xytext=(6, 0), textcoords="offset points", va="bottom", fontsize=7.5,
                color=INK_2, annotation_clip=False)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(rows) - 0.4, -0.75)
    ax.set_xlabel(f"PR-AUC at {floor:.0%} prevalence on {PRIMARY} "
                  f"(random-model floor {floor:.2f})")
    value_grid(ax, "x")
    ax.legend(handles=[hollow_handle("All test flows (headline)"),
                       filled_handle(f"Novel test flows only ({share:.0%} of the test split)",
                                     INK_2)],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.3)
    save(fig, out / "Figure 11 - Novel-only scores.png")


def fig12_vs_xgboost(out: Path) -> None:
    doc = gnn(PRIMARY)
    key, floor = at_key(doc), floor_of(doc)
    base = baselines(PRIMARY)
    rows = [("Graph model (full)", FULL, doc["results"]["full"])]
    for k in ("xgboost::flow", "xgboost::flow+host"):
        if base[k]["reported_at_prevalence"] != floor:
            raise SystemExit(f"{k} is not reported at {floor:.0%}")
        rows.append((baseline_name(k), TREE, base[k]))
    fig, axes = new_fig(2.6, 2, sharey=True)
    for ax, metric, title, xlabel in (
            (axes[0], "pr_auc", "Ranking",
             f"PR-AUC at {floor:.0%} prevalence (floor {floor:.2f})"),
            (axes[1], "f1", "Operating point", "F1 at the validation-chosen threshold")):
        everything = []
        for i, (name, colour, entry) in enumerate(rows):
            y = len(rows) - 1 - i
            vals = runs_metric(entry["runs"], key, metric)
            mean, sd, _ = agg(entry, metric)
            seed_strip(ax, y, vals, colour, mean, sd, horizontal=True, label=f"{mean:.4f}")
            everything += vals + [mean - sd, mean + sd]
        lo, hi = min(everything), max(everything)
        ax.set_xlim(lo - 0.08 * (hi - lo), hi + 0.42 * (hi - lo))
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="y", length=0)
        panel_label(ax, title)
        value_grid(ax, "x")
    axes[0].set_yticks(range(len(rows))[::-1],
                       [f"{name}\nn = {entry['aggregate']['pr_auc']['n_seeds']} seeds"
                        for name, _, entry in rows])
    axes[0].tick_params(axis="y", length=0)
    axes[0].set_ylim(-0.55, len(rows) - 0.45)
    save(fig, out / "Figure 12 - Graph model vs XGBoost.png")


def _paired_gain(doc: dict) -> list[float]:
    """Full minus channel 1, seed by seed (same as make_figures.paired_gain)."""
    key = at_key(doc)
    c1 = {r["seed"]: r[key]["pr_auc"] for r in doc["results"]["channel1_only"]["runs"]}
    return [r[key]["pr_auc"] - c1[r["seed"]] for r in doc["results"]["full"]["runs"]
            if r["seed"] in c1]


def fig13_topology_by_dataset(out: Path) -> None:
    docs = {ds: gnn(ds) for ds in (PRIMARY, SECONDARY)}
    fig, (a, b) = new_fig(2.9, 2)
    a.axhline(0, color=AXIS, lw=0.9, zorder=1)
    for i, (ds, doc) in enumerate(docs.items()):
        g = _paired_gain(doc)
        res = doc["results"]
        mean = res["full"]["aggregate"]["pr_auc"]["mean"] - \
            res["channel1_only"]["aggregate"]["pr_auc"]["mean"]
        seed_strip(a, i, g, FULL, mean, statistics.pstdev(g),
                   label=f"{signed(mean)}\nn = {len(g)} seeds")
    a.set_xticks(range(2), list(docs))
    a.tick_params(axis="x", length=0)
    a.set_xlim(-0.5, 1.75)
    a.set_ylabel("PR-AUC gain, full model minus\nchannel 1, matched by seed")
    panel_label(a, "(a) Topology gain")
    value_grid(a, "y")

    shares = [docs[ds]["results"]["full"]["runs"][0]["channel_attribution"]
              ["topological_weight_mean"] for ds in docs]
    seed = docs[PRIMARY]["results"]["full"]["runs"][0]["seed"]
    b.bar(range(2), shares, width=0.36, color=CH2, zorder=2)
    for i, s in enumerate(shares):
        b.annotate(f"{s:.4f}", (i, s), xytext=(0, 3), textcoords="offset points",
                   ha="center", va="bottom", fontsize=8, color=INK)
    ref_line(b, 0.5, text="equal split", where=0.99, ha="right")
    b.set_xticks(range(2), list(docs))
    b.tick_params(axis="x", length=0)
    b.set_ylim(0, 1)
    b.set_xlim(-0.6, 1.6)
    b.set_ylabel(f"Mean share of each decision given\nto the neighbourhood channel\n"
                 f"(full model, seed {seed}, test split)")
    panel_label(b, "(b) Weight on the neighbourhood")
    value_grid(b, "y")
    save(fig, out / "Figure 13 - Topology gain by dataset.png")


def fig14_zero_day(out: Path) -> None:
    doc = load(METRICS / "zeroday" / f"zeroday_{PRIMARY}.json")
    key = f"at_{doc['reported_at_prevalence']:.0%}"
    floor = doc["reported_at_prevalence"]
    thin = doc["config"]["thin_below_test_positives"]
    fams = sorted(doc["results"].items(), key=lambda kv: kv[1]["aggregate"]["recall"]["mean"])
    fig, axes = new_fig(4.0, 2, sharey=True)
    for ax, metric in zip(axes, ("recall", "pr_auc")):
        for y, (name, r) in enumerate(fams):
            vals = runs_metric(r["runs"], key, metric)
            mean, sd, _ = agg(r, metric)
            seed_strip(ax, y, vals, FULL, mean, sd, horizontal=True, label=f"{mean:.4f}",
                       hollow=r["test_positives"] < thin)
        ax.set_xlim(-0.03, 1.27)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.tick_params(axis="y", length=0)
        value_grid(ax, "x")
    ref_line(axes[1], floor, vertical=True, text=f"floor {floor:.2f}", where=1.0, ha="left",
             va="top")
    axes[0].set_xlabel("Recall on the held-out family")
    axes[1].set_xlabel(f"PR-AUC at {floor:.0%} prevalence")
    panel_label(axes[0], "Recall")
    panel_label(axes[1], "PR-AUC")
    labels = [f"{name}\n(thin: {r['test_positives']:,} test flows)"
              if r["test_positives"] < thin else name for name, r in fams]
    axes[0].set_yticks(range(len(fams)), labels)
    axes[0].set_ylim(-0.6, len(fams) - 0.4)
    n = sorted({r["n_seeds"] for _, r in fams})
    fig.legend(handles=[filled_handle(f"Mean ± sd over {'/'.join(map(str, n))} seeds; "
                                      "small dots are seeds", FULL),
                        Line2D([], [], ls="none", marker="o", ms=6.5, mfc=SURFACE, mec=FULL,
                               mew=1.4, label=f"Thin family (< {thin:,} held-out test flows)")],
               loc="outside lower center", ncol=2, fontsize=8)
    save(fig, out / "Figure 14 - Zero-day recall by family.png")


def fig15_transfer(out: Path) -> None:
    rows = [r for r in script("summarise_transfer").rows() if r["pr_auc"] is not None]
    by = {(r["model"], r["source"], r["target"]): r for r in rows}
    floor = rows[0]["floor"]
    models = (("gnn::full", "Graph model (full)", FULL), ("xgboost::flow", "XGBoost (flow)", TREE))
    fig, axes = new_fig(3.4, 2, sharey=True)
    for ax, (src, tgt) in zip(axes, ((PRIMARY, SECONDARY), (SECONDARY, PRIMARY))):
        oov = by[("gnn::full", src, tgt)]["oov"]
        confounded = oov is not None and oov >= 0.5
        for i, (model, name, colour) in enumerate(models):
            for j, r in enumerate((by[(model, src, src)], by[(model, src, tgt)])):
                x = i + (j - 0.5) * 0.36
                transfer = j == 1
                hatched = confounded and transfer
                ax.bar(x, r["pr_auc"], width=0.34, zorder=2,
                       color=colour if transfer else tint(colour, 0.35),
                       hatch="////" if hatched else None,
                       edgecolor=tint(colour, 0.45) if hatched else "none", linewidth=0)
                if r["n"] > 1:
                    ax.errorbar(x, r["pr_auc"], yerr=r["pr_std"], fmt="none", ecolor=INK_2,
                                elinewidth=1.0, capsize=2.5, zorder=3)
                note = "\n(1 seed)" if r["n"] == 1 else ""
                top = r["pr_auc"] + (r["pr_std"] if r["n"] > 1 else 0)
                ax.annotate(f"{r['pr_auc']:.4f}{note}", (x, top), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=7.5, color=INK, zorder=5)
        ref_line(ax, floor)
        ax.set_xticks(range(len(models)), [m[1] for m in models])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.set_ylim(0, 1.14)
        oov_text = (f"Confounded: {oov:.1%} of values out of vocabulary" if confounded
                    else f"{oov:.1%} of values out of vocabulary")
        ax.set_title(f"Trained on {src}\ntested on {tgt}", loc="left", fontsize=8.5,
                     fontweight="bold", color=INK, pad=16)
        ax.annotate(oov_text, (0, 1), xycoords="axes fraction", xytext=(0, 4),
                    textcoords="offset points", fontsize=7.5, va="bottom",
                    color=INK if confounded else INK_2,
                    fontweight="bold" if confounded else "normal")
        value_grid(ax, "y")
    axes[0].set_ylabel(f"PR-AUC at {floor:.0%} prevalence")
    axes[1].tick_params(axis="y", length=0)
    handles = [Patch(facecolor=tint(INK_2, 0.3), label="In-dataset control (tested on the "
                     "training dataset)"),
               Patch(facecolor=INK_2, label="Tested on the other dataset, no retraining"),
               Patch(facecolor=SURFACE, edgecolor=INK_2, hatch="////",
                     label="Confounded direction"),
               Line2D([], [], color=REF, lw=1.0, ls=(0, (4, 3)),
                      label=f"Random-model floor ({floor:.2f})")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=7.5)
    save(fig, out / "Figure 15 - Cross-dataset transfer.png")


def fig16_neighbourhood_by_family(out: Path) -> None:
    doc = load(METRICS / "evidence" / f"evidence_{PRIMARY}_summary.json")
    t = doc["summary"]["topological_weight"]["by_true_family"]
    order = sorted(((k, v) for k, v in t.items() if v), key=lambda kv: kv[1]["median"])
    fig, ax = new_fig(3.3)
    ref_line(ax, 0.5, vertical=True, text="equal split", where=1.0, ha="left", va="top")
    labels = []
    for y, (name, d) in enumerate(order):
        fp = name == "Benign"
        colour = BAD if fp else CH2
        ax.plot([d["p25"], d["p75"]], [y, y], color=colour, lw=2.4, zorder=2,
                solid_capstyle="round", alpha=0.6)
        ax.plot(d["median"], y, "o", ms=7.5, mfc=colour, mec=SURFACE, mew=1.4, zorder=3)
        ax.annotate(f"{d['median']:.4f}", (d["p75"], y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=INK_2)
        labels.append(f"{'false positives (benign)' if fp else name}  (n = {d['n']})")
    ax.set_yticks(range(len(order)), labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel(f"Share of the decision given to the neighbourhood channel "
                  f"({doc['summary']['n_packs']} evidence packs)")
    value_grid(ax, "x")
    ax.legend(handles=[Line2D([], [], color=CH2, marker="o", ms=7, mec=SURFACE, lw=2.4,
                              label="True attacks: median and interquartile range"),
                       Line2D([], [], color=BAD, marker="o", ms=7, mec=SURFACE, lw=2.4,
                              label="False positives")],
              loc="lower right", fontsize=7.5, frameon=True, facecolor=SURFACE,
              edgecolor="none", framealpha=1.0)
    save(fig, out / "Figure 16 - Neighbourhood share by family.png")


def _llm_rows(prompt: str) -> tuple[list[dict], dict | None]:
    rows = [r for r in script("summarise_llm_study").rows() if r["prompt"] == prompt]
    models = [r for r in rows if r["provider"] != "stub"]
    for r in models:
        served = {((s.get("model") or {}).get("served_by") or "") for s in r["sessions"]}
        if served != {"ollama cloud"}:
            raise SystemExit(f"{r['model']} {prompt} was not served only by Ollama's hosted "
                             f"service ({served}); the figure labels would be wrong")
    stub = next((r for r in rows if r["provider"] == "stub"), None)
    return models, stub


def fig17_groundedness(out: Path) -> None:
    models, stub = _llm_rows(MAIN_PROMPT)
    models.sort(key=lambda r: (r["groundedness"], r["groundedness_on_fp"] or 0, r["model"]))
    lo = min(min(r["groundedness"], r["groundedness_on_fp"]) for r in models)
    left = np.floor((lo - 0.0005) * 1000) / 1000
    fig, ax = new_fig(3.6)
    ref_line(ax, stub["groundedness"], vertical=True,
             text=f"template control ({stub['groundedness']:.4f})", where=1.0, ha="right",
             va="bottom")
    for y, r in enumerate(models):
        ax.plot([left, 1.0], [y, y], color=GRID, lw=0.6, zorder=0)
        for value, dy, marker, colour, ms in ((r["groundedness"], 0.14, "o", FULL, 7),
                                              (r["groundedness_on_fp"], -0.14, "D", BAD, 5.5)):
            ax.plot(value, y + dy, marker, ms=ms, mfc=colour, mec=SURFACE, mew=1.0, zorder=3)
            if value < 1.0:
                ax.annotate(f"{value:.4f}", (value, y + dy), xytext=(-7, 0),
                            textcoords="offset points", ha="right", va="center",
                            fontsize=7.5, color=INK_2)
    ax.set_yticks(range(len(models)),
                  [f"{r['model']}  (n = {r['n_ok']}; {r['n_false_positive']} FP)"
                   for r in models])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.75, len(models) - 0.4)
    ax.set_xlim(left, 1.0 + (1.0 - left) * 0.03)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_xlabel(f"Groundedness on prompt {MAIN_PROMPT}: share of a report's technical entities\n"
                  f"that appear in its evidence pack (axis zoomed to {left:.3f}–1.000)")
    ax.set_ylabel("Language model\n(Ollama hosted service)")
    value_grid(ax, "x")
    ax.legend(handles=[filled_handle("All reports (n)", FULL),
                       Line2D([], [], ls="none", marker="D", ms=5.5, mfc=BAD, mec=SURFACE,
                              label="Reports on false-positive (FP) detections")],
              loc="upper left", fontsize=8, borderaxespad=0.6, frameon=True,
              facecolor=SURFACE, edgecolor="none", framealpha=1.0)
    save(fig, out / "Figure 17 - LLM groundedness.png")


def _paired_prompts() -> dict[str, dict]:
    """Each model against itself on packs answered under both prompts, as
    export_report_facts._prompt_sensitivity computes it."""
    runs = {}
    for f in sorted(REPORTS.glob(f"reports_ollama_*_v[{MAIN_PROMPT[1]}{ALT_PROMPT[1]}].json")):
        d = load(f)
        runs[(d["model"], d["prompt_version"])] = {
            r["detection_id"]: r for r in d["reports"] if r.get("ok") and not r.get("degenerate")}
    out = {}
    for model in sorted({m for m, _ in runs}):
        if (model, MAIN_PROMPT) not in runs or (model, ALT_PROMPT) not in runs:
            continue
        common = sorted(set(runs[(model, MAIN_PROMPT)]) & set(runs[(model, ALT_PROMPT)]))
        entry = {"n": len(common)}
        for v in (MAIN_PROMPT, ALT_PROMPT):
            rs = [runs[(model, v)][i] for i in common]
            direction = [float(r["direction"]["direction_faithful"]) for r in rs
                         if (r.get("direction") or {}).get("n_checked")]
            words = [r["scores"]["report_words"] for r in rs
                     if (r.get("scores") or {}).get("report_words") is not None]
            entry[v] = {"direction": statistics.mean(direction), "words": statistics.mean(words)}
        out[model] = entry
    return out


def fig18_prompt_sensitivity(out: Path) -> None:
    pairs = _paired_prompts()
    order = sorted(pairs, key=lambda m: pairs[m][ALT_PROMPT]["direction"]
                   - pairs[m][MAIN_PROMPT]["direction"], reverse=True)
    fig, axes = new_fig(3.5, 2, sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, metric, xlabel, fmt in (
            (axes[0], "direction", "Share of reports whose reasons keep their direction", ".3f"),
            (axes[1], "words", "Mean words per note", ".0f")):
        for y, m in enumerate(order):
            dumbbell(ax, y, pairs[m][ALT_PROMPT][metric], pairs[m][MAIN_PROMPT][metric], FULL)
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="y", length=0)
        value_grid(ax, "x")
    vals = [pairs[m][v]["direction"] for m in order for v in (MAIN_PROMPT, ALT_PROMPT)]
    axes[0].set_xlim(min(vals) - 0.01, 1.005)
    axes[0].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    words = [pairs[m][v]["words"] for m in order for v in (MAIN_PROMPT, ALT_PROMPT)]
    axes[1].set_xlim(0, max(words) * 1.08)
    panel_label(axes[0], "Direction kept")
    panel_label(axes[1], "Length")
    axes[0].set_yticks(range(len(order)), [f"{m}  (n = {pairs[m]['n']})" for m in order])
    axes[0].tick_params(axis="y", length=0)
    axes[0].set_ylim(len(order) - 0.4, -0.6)
    fig.legend(handles=[filled_handle(f"Prompt {MAIN_PROMPT} (prescriptive)", FULL),
                        hollow_handle(f"Prompt {ALT_PROMPT} (permissive)")],
               loc="outside upper left", ncol=2, fontsize=8)
    axes[0].set_ylabel("Language model, paired packs\n(Ollama hosted service)")
    save(fig, out / "Figure 18 - Prompt sensitivity.png")


def _output_per_word(prompt: str) -> dict[str, float]:
    """Median output tokens per word of the kept note, as
    export_report_facts._hidden_output computes it."""
    out = {}
    for f in sorted(REPORTS.glob(f"reports_ollama_*_{prompt}.json")):
        d = load(f)
        rs = [r for r in d["reports"] if r.get("ok") and not r.get("degenerate")
              and r.get("output_tokens")]
        if rs:
            out[d["model"]] = statistics.median(
                r["output_tokens"] / max(1, len(r["report"].split())) for r in rs)
    return out


def fig19_output_beyond_note(out: Path) -> None:
    ratio = _output_per_word(MAIN_PROMPT)
    order = sorted(ratio, key=lambda m: ratio[m])
    lo, hi = PLAIN_ENGLISH_TOKENS_PER_WORD
    fig, ax = new_fig(3.3)
    ax.axvspan(lo, hi, color=tint(REF, 0.3), zorder=0, lw=0)
    for edge in (lo, hi):
        ax.axvline(edge, color=REF, lw=0.9, ls=(0, (3, 2)), zorder=3)
    ax.annotate(f"plain English, no hidden reasoning (about {lo}–{hi} tokens a word)",
                (hi, len(order) - 0.15), xytext=(5, 0), textcoords="offset points",
                fontsize=7.5, color=INK_2, va="center", ha="left")
    ys = np.arange(len(order))
    ax.barh(ys, [ratio[m] for m in order], height=0.6, color=FULL, zorder=2)
    for y, m in zip(ys, order):
        ax.annotate(f"{ratio[m]:.1f}", (ratio[m], y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_yticks(ys, order)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(order) + 0.2)
    ax.set_xlim(0, max(ratio.values()) * 1.1)
    ax.set_xlabel(f"Output tokens per word of the kept note, prompt {MAIN_PROMPT} "
                  "(median over completed reports)")
    ax.set_ylabel("Language model\n(Ollama hosted service)")
    value_grid(ax, "x")
    save(fig, out / "Figure 19 - Output beyond the note.png")


def fig20_timeline(out: Path) -> None:
    """Planned blocks (proposal), actual work and milestones, one row each."""
    import textwrap
    t = yaml.safe_load(TIMELINE.read_text())
    w1 = t["week1_monday"]
    day = dt.timedelta(days=1)
    sections = [
        ("Planned (proposal)", [(p["label"], w1 + 7 * (p["weeks"][0] - 1) * day,
                                 w1 + 7 * p["weeks"][1] * day) for p in t["planned"]]),
        ("Actual", [(a["label"], a["start"], a["end"] + day) for a in t["actual"]]),
        ("Milestones", [(f"{m['label']}, {m['date'].day} {m['date']:%b}", m["date"], None)
                        for m in sorted(t["milestones"], key=lambda m: m["date"])]),
    ]
    rows = []
    for title, items in sections:
        rows.append((title.upper(), None, None, "header"))
        rows += [(label, start, end, title) for label, start, end in items]
    fig, ax = new_fig(0.205 * len(rows) + 0.9)
    num = mdates.date2num
    n = len(rows)
    plan_fill = tint(DATA, 0.22)
    for i, (label, start, end, kind) in enumerate(rows):
        y = n - 1 - i
        if kind == "Planned (proposal)":
            ax.barh(y, num(end) - num(start), left=num(start), height=0.62, zorder=2,
                    color=plan_fill, edgecolor=DATA, linewidth=0.6)
        elif kind == "Actual":
            ax.barh(y, num(end) - num(start), left=num(start), height=0.62, zorder=2,
                    color=DATA)
        elif kind == "Milestones":
            ax.axvline(num(start), color=tint(INK, 0.25), lw=0.7, ls=(0, (2, 2)), zorder=1)
            ax.plot(num(start), y, "D", ms=6, color=INK, mec=SURFACE, mew=0.8, zorder=4)
    labels = [textwrap.fill(r[0], 38) for r in rows]
    ax.set_yticks([n - 1 - i for i in range(n)], labels, fontsize=7.5)
    for i, tick in enumerate(ax.get_yticklabels()):
        if rows[i][3] == "header":
            tick.set_fontweight("bold")
            tick.set_color(INK_2)
            tick.set_fontsize(7)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, n - 0.4)
    ends = [num(r[2]) for r in rows if r[2] is not None]
    start, end = num(w1) - 1, max(ends) + 2
    ax.set_xlim(start, end)
    fortnights = [w1 + 14 * k * day for k in range(int((end - num(w1)) // 14) + 1)]
    ax.set_xticks([num(d) for d in fortnights], [f"{d.day} {d:%b}" for d in fortnights])
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.grid(True, axis="x", which="both", color=GRID, lw=0.6)
    top = ax.secondary_xaxis("top")
    weeks = [w1 + 7 * k * day for k in range(int((end - num(w1)) // 7) + 1)]
    top.set_xticks([num(w) + 3.5 for w in weeks], [str(k + 1) for k in range(len(weeks))],
                   fontsize=7)
    top.tick_params(length=0, colors=INK_2, labelcolor=INK_2)
    top.spines["top"].set_visible(False)
    top.set_xlabel("Teaching week", fontsize=7.5, color=INK_2)
    ax.set_xlabel(f"2026 (teaching week 1 begins Monday {w1.day} {w1:%B})")
    fig.legend(handles=[Patch(facecolor=plan_fill, edgecolor=DATA, label="Planned"),
                        Patch(facecolor=DATA, label="Actual"),
                        Line2D([], [], ls="none", marker="D", ms=6, color=INK,
                               label="Milestone")],
               loc="outside lower center", fontsize=7.5, ncol=3)
    save(fig, out / "Figure 20 - Project timeline.png")


# ------------------------------------------------------------------ equations

EQUATIONS = {
    "eq1": ("Equation 1.png",
            r"$h_v^{(k)} = W^{(k)} \cdot \left[\, h_v^{(k-1)} \;\Vert\; "
            r"\mathrm{mean}_{u \in \mathcal{N}(v)}\, \phi\!\left(h_u^{(k-1)},\, x_{uv}\right)"
            r" \right]$"),
    "eq2": ("Equation 2.png", r"$\mathrm{AP}_{\mathrm{random}} = \pi$"),
    "eq3": ("Equation 3.png",
            r"$G = \dfrac{|\,E_{\mathrm{report}} \cap E_{\mathrm{evidence}}\,|}"
            r"{|\,E_{\mathrm{report}}\,|}$"),
}


def equation(out: Path, key: str) -> None:
    name, tex = EQUATIONS[key]
    with plt.rc_context({"mathtext.fontset": "cm", "text.color": "black"}):
        fig = plt.figure(figsize=(6, 1))
        fig.text(0.5, 0.5, tex, fontsize=12, ha="center", va="center", color="black")
        fig.savefig(out / name, dpi=300, facecolor="white", bbox_inches="tight",
                    pad_inches=0.04)
        plt.close(fig)


# ------------------------------------------------------------------ main

FIGURES = {
    "1": fig01_pipeline, "2": fig02_shortcut_audit, "3": fig03_duration_wrap,
    "4": fig04_dual_channel, "5": fig05_facts_layer, "6": fig06_architecture,
    "7": fig07_defects, "8": fig08_class_balance, "9": fig09_topology_ablation,
    "10": fig10_artefact_effect, "11": fig11_novel_only, "12": fig12_vs_xgboost,
    "13": fig13_topology_by_dataset, "14": fig14_zero_day, "15": fig15_transfer,
    "16": fig16_neighbourhood_by_family, "17": fig17_groundedness,
    "18": fig18_prompt_sensitivity, "19": fig19_output_beyond_note, "20": fig20_timeline,
}
# Draft number -> number in the report, for the figures the report kept.
REPORT_NUMBER = {"1": 1, "3": 2, "4": 3, "9": 4, "13": 5, "16": 6, "18": 7, "20": 8}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="folder to write the PNGs into")
    ap.add_argument("--only", nargs="+", metavar="N",
                    help="draft figure numbers (1-20) and/or eq1 eq2 eq3; default: the "
                         "eight report figures and the three equations")
    ap.add_argument("--defect-register", type=Path, default=None,
                    help="the Defect Register note (Figure 7); default: "
                         "'00 - Project Management/Defect Register.md' two levels above --out")
    args = ap.parse_args()

    wanted = [str(w).lower().lstrip("0") or "0" for w in (args.only or [])]
    valid = set(FIGURES) | set(EQUATIONS)
    bad = [w for w in wanted if w not in valid]
    if bad:
        ap.error(f"unknown figure(s) {bad}; choose from 1-20, eq1, eq2, eq3")
    todo = wanted or [*REPORT_NUMBER, *EQUATIONS]
    register = args.defect_register or (args.out.resolve().parents[1] / "00 - Project Management"
                                        / "Defect Register.md")

    style()
    args.out.mkdir(parents=True, exist_ok=True)
    failed = []
    for key in todo:
        try:
            # Draw into a scratch folder, then move each file under its report number
            # (a cut figure keeps its draft number).
            with tempfile.TemporaryDirectory(dir=args.out) as tmp:
                tmp = Path(tmp)
                if key in EQUATIONS:
                    equation(tmp, key)
                elif key == "7":
                    fig07_defects(tmp, register)
                else:
                    FIGURES[key](tmp)
                for p in tmp.glob("*.png"):
                    name = p.name
                    if key in REPORT_NUMBER:
                        name = re.sub(r"^Figure \d+", f"Figure {REPORT_NUMBER[key]:02d}", name)
                    p.replace(args.out / name)
            print(f"  drew  {key}")
        except Exception as e:                                     # noqa: BLE001
            failed.append(key)
            print(f"  FAILED {key}: {type(e).__name__}: {e}")
    print(f"\n{len(todo) - len(failed)} of {len(todo)} written -> {args.out}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
