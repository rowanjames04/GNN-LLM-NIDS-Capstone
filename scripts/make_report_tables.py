"""The numeric tables of the final report, one APA-shaped Markdown file each.

Every value is read from a results file when this runs; none is typed in. The
derivations are the ones `scripts/export_report_facts.py` uses for
`results/report_facts.md` (its helpers are imported, not copied), so a table
and the facts file cannot disagree. Tables 1, 2, 4 and 17-24 are written by
hand and are not produced here.

Usage:
    python scripts/make_report_tables.py --out "<vault>/09 - Final Report/Tables"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Importing it also puts src/ on sys.path (needed for gnnids.llm.facts).
FACTS = _load_script("export_report_facts")
RESULTS = FACTS.RESULTS
PRIMARY, SECONDARY = FACTS.PRIMARY, FACTS.SECONDARY
f = FACTS.f

GNN_NAMES = {"full": "Graph model (full)",
             "channel1_only": "Flow features only (channel 1)",
             "channel2_only": "Neighbourhood only (channel 2)"}
GNN_ORDER = ("full", "channel1_only", "channel2_only")
ABLATION_ORDER = ("channel1_only", "channel2_only", "full")
BASELINE_NAMES = {"logreg": "Logistic regression", "mlp": "MLP",
                  "random_forest": "Random forest", "xgboost": "XGBoost"}
# Roster facts that are not in the results files (developer, weight release).
DEVELOPERS = (("gpt-oss", "OpenAI"), ("gemma", "Google"), ("nemotron", "NVIDIA"),
              ("deepseek", "DeepSeek"), ("glm", "Z.ai"), ("mistral", "Mistral AI"))
WEIGHTS_NOT_RELEASED = {"mistral-large-4"}
SPLIT_LABELS = {"window_random": "Whole windows, assigned at random",
                "contiguous": "Contiguous by position in the capture"}
THIN = "†"
WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


# ------------------------------------------------------------------ helpers

def words(n: int) -> str:
    """APA: numbers below 10 in words, larger ones as numerals."""
    return WORDS[n] if 0 <= n < 10 else f"{n:,}"


def bare(s: str) -> str:
    """Drop the trailing "(n = k)" of a facts-file statistic; n goes in a column."""
    return re.sub(r" \(n = \d+\)$", "", s)


def pm(stat: dict | None, places: int = 4) -> str:
    return bare(FACTS.pm(stat, places))


def runs_pm(entry: dict, metric: str, key: str = "at_4%", places: int = 4) -> str:
    """mean ± population sd from the per-seed runs, as the novel-only table and the facts
    file compute it. The stored aggregate is already rounded to five places,
    so rounding it again to four can differ in the last digit (0.00485 ->
    0.0049 where the runs give 0.004846 -> 0.0048)."""
    v = [r[key][metric] for r in entry["runs"] if isinstance(r[key].get(metric), (int, float))]
    if not v:
        return "n/a"
    return f"{statistics.mean(v):.{places}f} ± {statistics.pstdev(v):.{places}f}"


def model_name(key: str) -> str:
    """`xgboost::flow+host` -> `XGBoost (flow+host)`; `gnn::full` -> graph model."""
    model, _, feats = key.partition("::")
    if model == "gnn":
        return GNN_NAMES[feats]
    return f"{BASELINE_NAMES[model]} ({feats})"


def short(ds: str) -> str:
    return ds.replace("NF-", "").replace("-v2", "")


def rel(path: str) -> str:
    return f"results/{path}"


def floor_text(prev: float) -> str:
    return f"{prev:.0%} prevalence (PR-AUC of a random model: {prev:.2f})"


# Report table numbers. The functions keep the numbers the tables had in the first
# draft (table_03 ...); the report later dropped two front tables, so each is printed
# under its number in reading order. Hand-written tables are numbered in their files.
RENUMBER = {3: 1, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9, 12: 10, 13: 11, 14: 12,
            15: 13, 16: 14}
SOURCES_TABLE = 23          # the appendix table that lists every results file
SOURCES: dict[int, tuple[str, list[str]]] = {}   # report number -> (title, files)

# The results files each figure in Sections 5 and 6 reads, as make_report_figures.py
# reads them. Paths are relative to results/.
FIGURE_SOURCES = {
    8: ("Number of Flows in Each Class for Both Datasets",
        ["metrics/preprocess_NF-ToN-IoT-v2.json (and the processed data it points to)",
         "metrics/preprocess.json", "metrics/audit_NF-UNSW-NB15-v2-selfcheck.json"]),
    9: ("PR-AUC of Each Model Variant by Seed on NF-ToN-IoT-v2",
        ["metrics/gnn/gnn_NF-ToN-IoT-v2.json"]),
    10: ("Change in PR-AUC After Removing the Artefact Columns",
         ["superseded/2026-10-10_duration_artefact/metrics/gnn/gnn_NF-ToN-IoT-v2.json",
          "superseded/2026-10-10_duration_artefact/metrics/baselines/baselines_NF-ToN-IoT-v2.json",
          "metrics/gnn/gnn_NF-ToN-IoT-v2.json", "metrics/baselines/baselines_NF-ToN-IoT-v2.json"]),
    11: ("Drop in PR-AUC When Duplicated Test Flows Are Removed",
         ["metrics/gnn/gnn_NF-ToN-IoT-v2.json", "metrics/baselines/baselines_NF-ToN-IoT-v2.json"]),
    12: ("The Graph Model Against XGBoost on PR-AUC and F1",
         ["metrics/gnn/gnn_NF-ToN-IoT-v2.json", "metrics/baselines/baselines_NF-ToN-IoT-v2.json"]),
    13: ("Topology Gain and the Share of Each Decision Given to the Neighbourhood, by Dataset",
         ["metrics/gnn/gnn_NF-ToN-IoT-v2.json", "metrics/gnn/gnn_NF-UNSW-NB15-v2.json"]),
    14: ("Recall and PR-AUC on Each Attack Family When It Was Removed From Training",
         ["metrics/zeroday/zeroday_NF-ToN-IoT-v2.json"]),
    15: ("In-Dataset Controls Against Cross-Dataset Results",
         ["metrics/transfer/transfer_*.json (pooled by scripts/summarise_transfer.py)"]),
    16: ("Share of Each Decision Given to the Neighbourhood, by True Class",
         ["metrics/evidence/evidence_NF-ToN-IoT-v2_summary.json"]),
    17: ("Groundedness of Each Model on All Reports and on False Positives",
         ["reports/reports_*_v3.json (pooled by scripts/summarise_llm_study.py)"]),
    18: ("Direction Kept and Note Length Under Prompts v3 and v4",
         ["reports/reports_ollama_*_v3.json", "reports/reports_ollama_*_v4.json"]),
    19: ("Output Tokens Produced per Word of the Final Note, by Model",
         ["reports/reports_ollama_*_v3.json"]),
}


def apa(number: int, title: str, header: list[str], rows: list[list], note: str,
        sources: list[str]) -> str:
    number = RENUMBER.get(number, number)
    SOURCES[number] = (title, sources)
    def cell(v) -> str:
        s = str(v)
        if "\n" in s:
            raise ValueError(f"newline in a table cell: {s!r}")
        return s.replace("|", "\\|")
    lines = [f"**Table {number}**", "", f"*{title}*", "",
             "| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        if len(r) != len(header):
            raise ValueError(f"Table {number}: row {r} does not match header {header}")
        lines.append("| " + " | ".join(cell(c) for c in r) + " |")
    # The results files are listed in the sources appendix (sources_table), not here.
    lines += ["", f"*Note.* {note}"]
    return "\n".join(lines) + "\n"


def sources_table() -> tuple[str, str]:
    """Appendix table: the results file behind every table and figure in Sections 5
    and 6, so the notes under them stay short."""
    rows = [[f"Table {n}", t, "; ".join(f"`{x}`" for x in files)]       # already results/...
            for n, (t, files) in sorted(SOURCES.items())]
    rows += [[f"Figure {n}", t, "; ".join(f"`results/{x}`" for x in files)]
             for n, (t, files) in sorted(FIGURE_SOURCES.items())]
    head = [f"**Table {SOURCES_TABLE}**", "", "*Results Files Behind the Tables and Figures "
            "in Sections 5 and 6*", "", "| Item | Title | Results file |", "|---|---|---|"]
    body = ["| " + " | ".join(c.replace("|", "\\|") for c in r) + " |" for r in rows]
    note = ("*Note.* Paths are relative to the root of the code repository (Appendix VI). "
            "Each results file also records the configuration that produced it. A `*` "
            "matches every model or seed, and a file pooled by a script is read through "
            "that script.")
    return (f"Table {SOURCES_TABLE:02d} - Sources of results.md",
            "\n".join(head + body + ["", note]) + "\n")


def need(doc, what: str):
    if not doc:
        raise SystemExit(f"missing results file: {what}")
    return doc


def one_n(entries, key="n_seeds") -> int:
    ns = {e[key] for e in entries}
    if len(ns) != 1:
        raise SystemExit(f"expected one seed count, found {sorted(ns)}")
    return ns.pop()


# ------------------------------------------------------------------ tables

def table_03() -> tuple[str, str]:
    docs = [(PRIMARY, "metrics/preprocess_NF-ToN-IoT-v2.json"),
            (SECONDARY, "metrics/preprocess.json")]
    d = {ds: need(FACTS.read(p), p) for ds, p in docs}
    names = {"train": "Training", "val": "Validation", "test": "Test"}

    def row(label, fn):
        return [label] + [fn(d[ds]) for ds, _ in docs]
    rows = [
        row("Flows in the source file", lambda x: f"{x['load']['source_rows']:,}"),
        row("Flows used", lambda x: f"{x['n_rows']:,}"),
        row("Attack families", lambda x: f"{len(x['attack_families']) - 1} (plus Benign)"),
        row("Graph window (flows)", lambda x: f"{x['config']['graph']['window_size']:,}"),
        row("Split method", lambda x: SPLIT_LABELS[x["split_method"]]),
        row("Split proportions (train / val / test)",
            lambda x: " / ".join(f"{x['config']['split'][k]:.0%}"
                                 for k in ("train", "val", "test"))),
        row("Rows purged between splits", lambda x: f"{x.get('rows_purged', 0):,}"),
    ]
    for s in ("train", "val", "test"):
        rows.append(row(f"{names[s]} flows (attack rate)",
                        lambda x, s=s: f"{x['splits'][s]['n_rows']:,} "
                                       f"({x['split_attack_rates'][s]:.2%})"))
    capped = [ds for ds, _ in docs if d[ds]["load"].get("capped")]
    gnn = need(FACTS.read(f"metrics/gnn/gnn_{PRIMARY}.json"), "gnn results")
    prev = gnn["config"]["eval"]["target_prevalence"]
    note = ("Attack rate is the share of flows in a split labelled as an attack, at the "
            f"native prevalence of the capture (every model is scored at {prev:.0%} "
            "prevalence). "
            + (f"{', '.join(capped)} was capped for memory; every attack family is "
               f"represented. " if capped else "")
            + f"The {PRIMARY} capture is ordered by scenario rather than time, so whole graph "
            f"windows are assigned to splits at random; {SECONDARY} is split contiguously, "
            f"with rows purged at each boundary so that no window straddles two splits.")
    return "Table 03 - Datasets and splits.md", apa(
        3, "Datasets, Flows Used and Splits",
        ["Characteristic", PRIMARY, SECONDARY], rows, note, [rel(p) for _, p in docs])


def table_05() -> tuple[str, str]:
    files = sorted((RESULTS / "reports").glob("reports_ollama_*_v3.json"))
    if not files:
        raise SystemExit("no v3 language-model reports in results/reports")
    rows, n_models = [], 0
    for fp in files:
        d = json.loads(fp.read_text())
        sessions = (d.get("provenance") or {}).get("sessions") or []
        models = [s.get("model") or {} for s in sessions]
        for key in ("digest", "parameter_size", "quantization_level"):
            if len({m.get(key) for m in models}) != 1:
                raise SystemExit(f"{fp.name}: {key} changed between sessions")
        if any(m.get("served_by") != "ollama cloud" for m in models) or \
                any(s.get("gpus") for s in sessions):
            raise SystemExit(f"{fp.name}: not every session was on the hosted service")
        m = models[-1]
        tag = d["model"]
        dev = next(name for prefix, name in DEVELOPERS if tag.startswith(prefix))
        size = m.get("parameter_size")
        size = f"{int(size) / 1e9:,.1f}" if str(size).isdigit() else "not reported"
        weights = ("Announced, not released at time of study"
                   if tag in WEIGHTS_NOT_RELEASED else "Yes")
        rows.append([tag, dev, size, m.get("quantization_level") or "not reported",
                     weights, m["digest"][:12]])
        n_models += 1
    stub = need(FACTS.read("reports/reports_stub_template-v1_v3.json"), "stub v3 reports")
    rows.append([f"{stub['model']} (control)", "None (fixed template)", "n/a", "n/a", "n/a",
                 "n/a"])
    n_open = n_models - sum(1 for r in rows[:n_models] if r[0] in WEIGHTS_NOT_RELEASED)
    note = (f"All {words(n_models)} models were called through Ollama's hosted service, "
            f"which reports each model's parameter count (in billions), quantisation and "
            f"digest (the first 12 characters identify the exact build served) but not the "
            f"hardware that served it; \"not reported\" means the service returned no value. "
            f"{words(n_open).capitalize()} of the {words(n_models)} models have published "
            f"weights; "
            f"mistral-large-4's weights had been announced but not released when the study "
            f"ran. Quantisation is the number format the weights were served in: BF16 is "
            f"16-bit brain floating point (unquantised), FP8 is 8-bit floating point, and "
            f"MXFP4 and NVFP4 are 4-bit block-scaled floating-point formats. The template "
            f"control writes every report from a fixed template with no language model "
            f"(n/a: not applicable).")
    return "Table 05 - Language model roster.md", apa(
        5, "Language Models Compared, With the Template Control",
        ["Model", "Developer", "Parameters (B)", "Quantisation", "Published weights",
         "Digest"], rows, note,
        ["results/reports/reports_ollama_*_v3.json (provenance)",
         "results/reports/reports_stub_template-v1_v3.json"])


def table_06() -> tuple[str, str]:
    from gnnids.llm.facts import borderline

    samp_rel = f"evidence/evidence_{PRIMARY}_sampling.json"
    packs_rel = f"evidence/evidence_{PRIMARY}.json"
    s = need(FACTS.read(samp_rel), samp_rel)
    packs = need(json.loads((RESULTS / packs_rel).read_text()), packs_rel)
    taken, avail = s["taken_by_true_family"], s["available_by_true_family"]
    counts, border = {}, {}
    for pk in packs:
        lab = pk["flow"]["true_label"]
        counts[lab] = counts.get(lab, 0) + 1
        border[lab] = border.get(lab, 0) + int(borderline(pk))
    if counts != taken or len(packs) != s["n_selected"]:
        raise SystemExit(f"packs file and sampling file disagree: {counts} vs {taken}")
    fams = sorted(k for k in taken if k != "Benign")
    rows = [[fam, f"{avail[fam]:,}", taken[fam], border[fam]] for fam in fams]
    rows.append(["Benign (false positives)", f"{avail['Benign']:,}", taken["Benign"],
                 border["Benign"]])
    rows.append(["Total", f"{sum(avail.values()):,}", len(packs), sum(border.values())])
    per = s["per_family_requested"]
    extra = {k: v - per for k, v in taken.items() if k != "Benign" and v > per}
    pre = FACTS.read(f"metrics/preprocess_{PRIMARY}.json") or {}
    absent = sorted(set(pre.get("attack_families", {})) - set(avail))
    note = (f"Packs were drawn from flows in the {PRIMARY} test split that the graph model "
            f"(checkpoint {s['checkpoint']}) flagged at its threshold of {s['threshold']}. "
            f"Of the {s['n_selected']} packs, {s['false_positives_taken']} were reserved for "
            f"false positives (benign flows flagged as attacks) and the rest were divided "
            f"evenly over the {words(len(fams))} attack families present ({per} each)"
            + (f"; the {words(sum(extra.values()))} left over were drawn at random from the "
               f"remaining flagged flows and fell in {', '.join(sorted(extra))}"
               if extra else "")
            + "." + (f" No {', '.join(absent)} flow was among the flagged flows available."
                     if absent else "")
            + " Borderline: the facts layer's fixed rules (a narrow margin over the "
            "threshold, a low category confidence, or a second plausible category).")
    return "Table 06 - Evidence pack sample.md", apa(
        6, "Evidence Packs Sampled for the Language-Model Study, by True Class",
        ["True class", "Flagged flows available", "Packs", "Borderline packs"],
        rows, note, [rel(samp_rel), rel(packs_rel)])


def table_07() -> tuple[str, str]:
    p = f"metrics/baselines/baselines_{PRIMARY}.json"
    d = need(FACTS.read(p), p)
    on_path = {k: v for k, v in d["results"].items() if v.get("schema") == FACTS.BASELINE_SCHEMA}
    if len(on_path) != len(d["results"]):
        raise SystemExit(f"stale baselines in {p}: {set(d['results']) - set(on_path)}")
    prev = next(iter(on_path.values()))["reported_at_prevalence"]
    key = f"at_{prev:.0%}"
    rows = []
    for k, v in sorted(on_path.items()):
        model, _, feats = k.partition("::")
        rows.append([BASELINE_NAMES[model], feats, len(v["runs"]), runs_pm(v, "pr_auc", key),
                     runs_pm(v, "f1", key), runs_pm(v, "fpr_at_95_recall", key, 5)])
    n_host = len(d.get("host_feature_names") or [])
    note = (f"Mean ± standard deviation over n seeds, reported at {floor_text(prev)}. "
            f"FPR at 95% recall is the false-positive rate at the threshold that catches 95% "
            f"of attacks. \"flow\" is the flow's own features; \"flow+host\" adds {n_host} "
            f"per-host aggregates (such as a source's fan-out and flow count). None of these "
            f"models sees the graph. MLP = multilayer perceptron.")
    return "Table 07 - Baselines on NF-ToN-IoT-v2.md", apa(
        7, f"Baseline Models Without Topology on {PRIMARY}",
        ["Model", "Features", "n", "PR-AUC", "F1", "FPR at 95% recall"], rows, note, [rel(p)])


def _ablation(ds: str, number: int, fname: str, per_seed: bool) -> tuple[str, str]:
    p = f"metrics/gnn/gnn_{ds}.json"
    d = need(FACTS.read(p), p)
    prev = d["config"]["eval"]["target_prevalence"]
    key = f"at_{prev:.0%}"
    res = d["results"]
    rows = []
    for v in ABLATION_ORDER:
        if v not in res:
            continue
        a, runs = res[v]["aggregate"], res[v]["runs"]
        vals = [r[key]["pr_auc"] for r in runs]
        spread = (", ".join(f(x) for x in vals) if per_seed
                  else f"{min(vals):.4f}–{max(vals):.4f}")
        rows.append([GNN_NAMES[v], a["pr_auc"]["n_seeds"], pm(a.get("pr_auc")), spread,
                     pm(a.get("f1")), pm(a.get("fpr_at_95_recall"), 5)])
    full = [r[key]["pr_auc"] for r in res["full"]["runs"]]
    c1 = [r[key]["pr_auc"] for r in res["channel1_only"]["runs"]]
    gain = (res["full"]["aggregate"]["pr_auc"]["mean"]
            - res["channel1_only"]["aggregate"]["pr_auc"]["mean"])
    overlap = "do not overlap" if min(full) > max(c1) else "overlap"
    selection = res["full"]["runs"][0].get("selection")
    if selection != "fixed_budget_best_val":
        raise SystemExit(f"{p}: runs are not on the fixed training budget ({selection})")
    note = (f"Mean ± standard deviation over n seeds, at {floor_text(prev)}. The three "
            f"variants share architecture, optimiser, depth and width; only the channels "
            f"used differ. Fixed budget of {d['config']['train']['max_epochs']} epochs, "
            f"best-validation checkpoint kept. Topology gain (full minus channel 1, PR-AUC): "
            f"{gain:+.4f}; the seed ranges of the two {overlap}. FPR at 95% recall is the "
            f"false-positive rate at the threshold that catches 95% of attacks.")
    title = f"Topology Ablation on {ds}"
    head = ["Variant", "n", "PR-AUC", "PR-AUC per seed" if per_seed else "Seed range",
            "F1", "FPR at 95% recall"]
    return fname, apa(number, title, head, rows, note, [rel(p)])


def table_08() -> tuple[str, str]:
    return _ablation(PRIMARY, 8, "Table 08 - Topology ablation on NF-ToN-IoT-v2.md", False)


def table_11() -> tuple[str, str]:
    return _ablation(SECONDARY, 11, "Table 11 - Topology ablation on NF-UNSW-NB15-v2.md", True)


def table_09() -> tuple[str, str]:
    arch = FACTS.ARCHIVE
    pg_old, pg_new = f"{arch}/metrics/gnn/gnn_{PRIMARY}.json", f"metrics/gnn/gnn_{PRIMARY}.json"
    pb_old = f"{arch}/metrics/baselines/baselines_{PRIMARY}.json"
    pb_new = f"metrics/baselines/baselines_{PRIMARY}.json"
    new_doc = need(FACTS.read(pg_new), pg_new)
    old_g, new_g = need(FACTS.read(pg_old), pg_old)["results"], new_doc["results"]
    old_b, new_b = need(FACTS.read(pb_old), pb_old)["results"], need(FACTS.read(pb_new), pb_new)["results"]
    rows = []

    def add(name, o, n):
        a, b, k = FACTS._matched_pr_auc(o, n)
        rows.append([name, str(k), f"{a:.4f}", f"{b:.4f}", f"{b - a:+.4f}"])
    for v in GNN_ORDER:
        if v in old_g and v in new_g:
            add(GNN_NAMES[v], old_g[v], new_g[v])
    for k in sorted(old_b):
        if k in new_b and old_b[k].get("schema") == FACTS.BASELINE_SCHEMA:
            add(model_name(k), old_b[k], new_b[k])
    cols = need(FACTS.read(f"metrics/preprocess_{PRIMARY}.json"), "preprocess")[
        "config"]["features"]["drop_artefacts"]
    prev = new_doc["config"]["eval"]["target_prevalence"]
    note = (f"Mean PR-AUC over seeds at {floor_text(prev)}. \"With\" is the archived run, "
            f"whose inputs included the {words(len(cols))} duration and throughput columns later "
            f"removed as an artefact; \"without\" is the current run. The rows and their order "
            f"are identical, the training configuration is identical, and both columns are "
            f"averaged over the same seeds, so the feature list is the only difference. "
            f"Difference = without minus with, from the unrounded means. Seeds = the number "
            f"of seeds both runs share.")
    return "Table 09 - Effect of the duration artefact.md", apa(
        9, f"Effect of the Duration Artefact on PR-AUC, {PRIMARY}",
        ["Model", "Seeds", "PR-AUC with artefact", "PR-AUC without",
         "Difference"], rows, note,
        [rel(pg_old), rel(pb_old), rel(pg_new), rel(pb_new)])


def table_10() -> tuple[str, str]:
    pg, pb = f"metrics/gnn/gnn_{PRIMARY}.json", f"metrics/baselines/baselines_{PRIMARY}.json"
    gdoc = need(FACTS.read(pg), pg)
    g, b = gdoc["results"], need(FACTS.read(pb), pb)["results"]
    prev = gdoc["config"]["eval"]["target_prevalence"]
    key = f"at_{prev:.0%}"
    entries = [(GNN_NAMES[v], g[v]) for v in GNN_ORDER if v in g]
    entries += [(model_name(k), b[k]) for k in sorted(b)]
    rows = []
    for name, v in entries:
        runs = v.get("runs", [])
        allv = [r[key]["pr_auc"] for r in runs]
        nov = [(r.get("novel_only") or {}).get(key, {}).get("pr_auc") for r in runs]
        if any(x is None for x in nov):
            raise SystemExit(f"{name}: a run has no novel-only score")
        rows.append([name, len(runs), bare(FACTS._seed_stat(allv)),
                     bare(FACTS._seed_stat(nov)),
                     f"{statistics.mean(nov) - statistics.mean(allv):+.4f}"])
    pre = need(FACTS.read(f"metrics/preprocess_{PRIMARY}.json"), "preprocess")
    share = pre["share_identical_to_a_training_row"]["test"]
    note = (f"PR-AUC at {floor_text(prev)}, mean ± standard deviation over n seeds. \"Novel only\" drops the {share:.1%} of test flows whose "
            f"model input is identical to some training flow's. Change = novel only minus all "
            f"test flows, from the unrounded means.")
    return "Table 10 - Novel-only scores.md", apa(
        10, f"PR-AUC on All Test Flows and on Novel Test Flows Only, {PRIMARY}",
        ["Model", "n", "All test flows", "Novel only", "Change"], rows, note,
        [rel(pg), rel(pb), rel(f"metrics/preprocess_{PRIMARY}.json")])


def table_12() -> tuple[str, str]:
    p = f"metrics/zeroday/zeroday_{PRIMARY}.json"
    d = need(FACTS.read(p), p)
    thin = d["config"]["thin_below_test_positives"]
    prev = d["reported_at_prevalence"]
    rows = []
    items = sorted(d["results"].items(), key=lambda kv: -kv[1]["test_positives"])
    for name, r in items:
        a = r["aggregate"]
        named = ", ".join(f"{k} {v:.0%}" for k, v in
                          list(r.get("held_out_mistaken_for", {}).items())[:2]) or "n/a"
        mark = f" {THIN}" if r["test_positives"] < thin else ""
        rows.append([f"{name}{mark}", f"{r['test_positives']:,}", pm(a.get("recall")),
                     pm(a.get("pr_auc")), named])
    n = one_n([r for _, r in items])
    skipped = " ".join(f"{k.capitalize()} is not measurable: {why}."
                       for k, why in d.get("skipped", {}).items())
    note = (f"Graph model ({d['ablation']}). Each family was removed from training and "
            f"validation entirely; recall is on that family's flows in the test split, and "
            f"PR-AUC is at {floor_text(prev)}. Mean ± standard deviation over n = {n} seeds "
            f"per family. The last column gives the two categories the family head most often "
            f"named for the held-out flows. {THIN} Thin: fewer than {thin:,} held-out test "
            f"flows, so quote the recall with its count. {skipped}")
    return "Table 12 - Leave-one-attack-out results.md", apa(
        12, f"Leave-One-Attack-Out Results on {PRIMARY}",
        ["Held-out family", "Test flows", "Recall", "PR-AUC", "Family head named them as"],
        rows, note, [rel(p)])


def table_13() -> tuple[str, str]:
    rows_in = FACTS.script("summarise_transfer").rows()
    if not rows_in:
        raise SystemExit("no transfer results")
    rows, worst = [], {}
    for r in sorted(rows_in, key=lambda x: (not x["control"], x["model"], x["source"])):
        pr = ("n/a" if r["pr_auc"] is None else
              f"{r['pr_auc']:.4f}" if r["n"] < 2 else f"{r['pr_auc']:.4f} ± {r['pr_std']:.4f}")
        oov = f"{r['oov']:.1%}" if r["oov"] is not None else "n/a"
        if r["oov"] is not None and r["oov"] > 0.5:
            oov += f" {THIN}"
            worst[(r["source"], r["target"])] = None
        rows.append([model_name(r["model"]), f"{short(r['source'])} → {short(r['target'])}",
                     "Control" if r["control"] else "Transfer", r["n"], pr, oov])
    # Which categorical column drives the confounded direction, from the file itself.
    for f_ in sorted((RESULTS / "metrics" / "transfer").glob("transfer_*.json")):
        d = json.loads(f_.read_text())
        pair = (d["source_dataset"], d["target_dataset"])
        if pair in worst and worst[pair] is None:
            o = {k: v for k, v in d["out_of_vocabulary"].items() if isinstance(v, dict)}
            worst[pair] = max(o, key=lambda k: o[k]["oov_rate"])
    conf = "; ".join(f"{short(s)} → {short(t)}, where the worst column is {col}"
                     for (s, t), col in worst.items())
    floor = rows_in[0]["floor"]
    note = (f"No model was retrained on the target. PR-AUC at {floor_text(floor)}; mean ± "
            f"standard deviation over seeds, and a one-seed control has no standard "
            f"deviation. Out-of-vocabulary is the share of target flows whose value in the "
            f"worst-affected categorical column lies outside the source's vocabulary. "
            f"{THIN} Confounded ({conf}): more than half the values are out of vocabulary, "
            f"so the direction measures a degraded input as well as a different network; "
            f"rely on the other direction. ToN-IoT = {PRIMARY}; UNSW-NB15 = {SECONDARY}.")
    return "Table 13 - Cross-dataset transfer.md", apa(
        13, "Cross-Dataset Transfer Without Retraining",
        ["Model", "Trained → tested", "Kind", "Seeds", "PR-AUC", "Out-of-vocabulary"],
        rows, note, ["results/metrics/transfer/transfer_*.json (pooled by "
                     "scripts/summarise_transfer.py)"])


def table_14() -> tuple[str, str]:
    p = f"metrics/evidence/evidence_{PRIMARY}_summary.json"
    d = need(FACTS.read(p), p)
    s = d["summary"]
    feats = s["features_shown_to_the_model"]
    rows = [[r["feature"], r["packs"], f"{r['share_of_packs']:.1%}", r["raises_suspicion"],
             r["lowers_suspicion"]] for r in feats["ranked"][:10]]
    note = (f"Over {s['n_packs']} evidence packs ({s['n_true_attacks']} true attacks, "
            f"{s['n_false_positives']} false positives) from the graph model "
            f"({d.get('checkpoint')}); attributions by integrated gradients. A feature is "
            f"counted once per pack when it is among the {words(feats['top_k'])} largest "
            f"attributions. A feature's encoded variants (a port and its port bucket) count "
            f"separately in the last two columns, which can therefore exceed the pack count.")
    return "Table 14 - Most attributed features.md", apa(
        14, f"Features Most Often Among the {words(feats['top_k']).capitalize()} Largest "
            f"Attributions",
        ["Feature", "Packs", "Share of packs", "Raised suspicion", "Lowered suspicion"],
        rows, note, [rel(p)])


def table_15() -> tuple[str, str]:
    rows_in = [r for r in FACTS.script("summarise_llm_study").rows() if r["prompt"] == "v3"]
    real = sorted((r for r in rows_in if r["provider"] != "stub"), key=lambda r: r["model"])
    stub = [r for r in rows_in if r["provider"] == "stub"]
    if not real or len(stub) != 1:
        raise SystemExit("expected the v3 language models and one v3 template control")
    rows = []
    for r in real + stub:
        name = r["model"] if r["provider"] != "stub" else f"{r['model']} (control)"
        rows.append([name, r["n_ok"], r["n_failed"], f(r["groundedness"]),
                     f"{f(r['groundedness_on_fp'])} ({r['n_false_positive']})",
                     r["fabricated_addr_reports"], f(r["class_named"]),
                     f(r["direction_faithful"]), f(r["borderline_conveyed"]),
                     f(r["latency_p50"], 2)])
    samp = need(FACTS.read(f"evidence/evidence_{PRIMARY}_sampling.json"), "sampling")
    bl = {r["borderline_n"] for r in real}
    chk = [r["direction_checked"] for r in real]
    note = (f"Prompt v3 (prescriptive), {samp['n_selected']} evidence packs per model, "
            f"{samp['false_positives_taken']} of them false positives; every model was called "
            f"through Ollama's hosted service. Reports: completed and scored; failed: empty "
            f"after the output limit, or an error. Groundedness: mean share of the technical "
            f"entities in a report that appear in its evidence pack (1 = invented nothing); it "
            f"does not measure whether the reasoning is correct. FPs: the false-positive "
            f"packs (n = reports scored). Invented-address reports: reports naming an address "
            f"not in the pack. Direction kept: share of reports with a checkable reason "
            f"({min(chk)}–{max(chk)} per model) in which every reason kept its direction; "
            f"lexical, so a lower bound on errors. Borderline conveyed: share of the "
            f"{'/'.join(str(x) for x in sorted(bl))} borderline packs whose report said so. "
            f"Latency is the hosted service's response time, not model speed. The template "
            f"control makes no call and states no checkable reason (n/a).")
    return "Table 15 - Language model results on prompt v3.md", apa(
        15, "Language-Model Results on Prompt v3",
        ["Model", "Reports", "Failed", "Groundedness", "Groundedness on FPs (n)",
         "Invented-address reports", "Names predicted class", "Direction kept",
         "Borderline conveyed", "Latency p50 (s)"], rows, note,
        ["results/reports/reports_*_v3.json (pooled by scripts/summarise_llm_study.py)"])


# The same per-report readings `export_report_facts._prompt_sensitivity` averages.
PAIRED = {
    "groundedness": (lambda r: (r.get("scores") or {}).get("groundedness"), 4),
    "direction": (lambda r: None if not (r.get("direction") or {}).get("n_checked")
                  else float(r["direction"]["direction_faithful"]), 4),
    "uncertainty": (lambda r: None if not r.get("uncertainty")
                    else float(r["uncertainty"]["uncertainty_conveyed"]), 4),
    "words": (lambda r: (r.get("scores") or {}).get("report_words"), 0),
}


def paired_prompt_stats() -> list[dict]:
    runs = {}
    for fp in sorted((RESULTS / "reports").glob("reports_ollama_*_v[34].json")):
        d = json.loads(fp.read_text())
        runs[(d["model"], d["prompt_version"])] = {
            r["detection_id"]: r for r in d["reports"] if r.get("ok") and not r.get("degenerate")}
    out = []
    for model in sorted({m for m, _ in runs}):
        if (model, "v3") not in runs or (model, "v4") not in runs:
            continue
        common = sorted(set(runs[(model, "v3")]) & set(runs[(model, "v4")]))
        row = {"model": model, "packs": len(common)}
        for v in ("v3", "v4"):
            rs = [runs[(model, v)][i] for i in common]
            for key, (fn, _) in PAIRED.items():
                vals = [x for x in map(fn, rs) if x is not None]
                row[(key, v)] = (statistics.mean(vals) if vals else None, len(vals))
        out.append(row)
    return out


def table_16() -> tuple[str, str]:
    data = paired_prompt_stats()
    if not data:
        raise SystemExit("no model has reports on both v3 and v4")

    def cell(row, key):
        nd = PAIRED[key][1]
        a, b = (row[(key, v)][0] for v in ("v3", "v4"))
        g = lambda x: "n/a" if x is None else f"{x:.{nd}f}"           # noqa: E731
        return f"{g(a)} → {g(b)}"
    rows = [[r["model"], r["packs"], cell(r, "groundedness"), cell(r, "direction"),
             cell(r, "uncertainty"), cell(r, "words")] for r in data]
    unc = {}
    for r in data:
        unc.setdefault(r[("uncertainty", "v3")][1], []).append(r["model"])
    if any(r[("uncertainty", "v3")][1] != r[("uncertainty", "v4")][1] for r in data):
        raise SystemExit("ambiguous-pack counts differ between v3 and v4 on paired packs")
    common_n = max(unc, key=lambda k: len(unc[k]))
    others = "; ".join(f"{', '.join(ms)}: {words(k)}" for k, ms in sorted(unc.items()) if k != common_n)
    note = (f"Each model is compared with itself on the packs it answered under both prompts "
            f"(failed and degenerate reports left out on either side); cells read v3 → v4. "
            f"Prompt v3 is prescriptive and v4 permissive. The v4 run of deepseek-v4-pro:0813 "
            f"was capped at 40 packs after runaway generations, so its pairing is small. Direction kept is "
            f"over reports with a checkable reason; uncertainty conveyed is over the paired "
            f"packs whose attack category was ambiguous ({words(common_n)} per model"
            + (f"; {others}" if others else "") + "). Mean words is the length of the note.")
    return "Table 16 - Prompt v3 against v4.md", apa(
        16, "Prompt v3 Against Prompt v4, Each Model on Its Own Paired Packs",
        ["Model", "Paired packs", "Groundedness", "Direction kept", "Uncertainty conveyed",
         "Mean words"], rows, note,
        ["results/reports/reports_ollama_*_v3.json", "results/reports/reports_ollama_*_v4.json"])


TABLES = (table_03, table_05, table_06, table_07, table_08, table_09, table_10, table_11,
          table_12, table_13, table_14, table_15, table_16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="directory for the .md files")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for build in TABLES:
        name, text = build()
        old = int(re.match(r"Table (\d+)", name).group(1))
        name = re.sub(r"^Table \d+", f"Table {RENUMBER.get(old, old):02d}", name)
        (args.out / name).write_text(text)
        print(f"written -> {name}")
    name, text = sources_table()
    (args.out / name).write_text(text)
    print(f"written -> {name}")


if __name__ == "__main__":
    main()
