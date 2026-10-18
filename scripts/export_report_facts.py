"""Every number the report may quote, in one file, each with its source.

The report is drafted with a language model's help. A model handed ten JSON
files will mis-transcribe a number or fill a gap with a plausible one; a model
handed one file of stated facts, and told to use nothing else, cannot. This
writes that file: `results/report_facts.md`.

It is also the check the project has needed since C17. Every figure in it is
read from a committed results file at the moment of export -- none is typed in
-- and each section names the file and the time it was generated, so a number
in a draft can be traced back and a stale one is visible as stale.

**What it does not do.** It reports; it does not interpret. Sections whose
results do not exist yet say so, rather than being left out, so a gap in the
report is a visible gap and not a silent one.

Usage:
    python scripts/export_report_facts.py
    python scripts/export_report_facts.py --out /tmp/facts.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
PRIMARY, SECONDARY = "NF-ToN-IoT-v2", "NF-UNSW-NB15-v2"
BASELINE_SCHEMA = "dual-prevalence-v1"
VARIANTS = (("channel1_only", "flow features only (channel 1)"),
            ("channel2_only", "neighbourhood only (channel 2)"),
            ("full", "both channels (full model)"))


SPLITS = {"window_random": "whole 10,000-flow windows assigned to splits at random "
                           "(the capture is ordered by scenario, not by time)"}


def read(rel: str) -> dict | None:
    path = RESULTS / rel
    return json.loads(path.read_text()) if path.exists() else None


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def f(v, places: int = 4) -> str:
    return format(v, f".{places}f") if isinstance(v, (int, float)) else "n/a"


def pm(stat: dict | None, places: int = 4) -> str:
    """`0.9887 ± 0.0030 (n = 3)` from an aggregate entry."""
    if not stat or not isinstance(stat.get("mean"), (int, float)):
        return "n/a"
    return (f"{stat['mean']:.{places}f} ± {stat['std']:.{places}f} "
            f"(n = {stat['n_seeds']})")


def source(rel: str, doc: dict | None) -> str:
    when = (doc or {}).get("generated_at", "unknown")
    return f"*Source: `results/{rel}`, generated {str(when)[:19]} UTC.*"


def missing(what: str, how: str) -> list[str]:
    return [f"**NOT YET MEASURED.** {what} To produce it: `{how}`.",
            "Write `[NUM?]` wherever a number from this section is needed.", ""]


# ------------------------------------------------------------------ sections

def sec_datasets() -> list[str]:
    out = ["## 1. Datasets and preprocessing", ""]
    for ds, rel in ((PRIMARY, f"metrics/preprocess_{PRIMARY}.json"),
                    (SECONDARY, "metrics/preprocess.json")):
        d = read(rel)
        if not d:
            out += [f"### {ds}", ""] + missing("Preprocessing metrics.",
                                              "python scripts/preprocess.py")
            continue
        load, sch, sp = d["load"], d["schema"], d["splits"]
        role = "primary" if ds == PRIMARY else "secondary"
        out += [f"### {ds} ({role})", "", source(rel, d), "",
                f"- Flows used: **{d['n_rows']:,}** of {load['source_rows']:,} in the "
                f"source file" + (" (capped for memory; every attack family is "
                                  "represented)" if load.get("capped") else ""),
                f"- Attack families: {', '.join(k for k in d['attack_families'] if k != 'Benign')} "
                f"({len(d['attack_families']) - 1}, plus Benign)",
                f"- Features: {sch['n_continuous']} continuous "
                f"({sch['n_log_transformed']} log-transformed), "
                f"{sch['n_categorical']} categorical, "
                f"{sch['n_conditional_indicators']} presence indicators, "
                f"{sch['n_port_buckets']} port buckets",
                f"- Dropped as shortcuts: {', '.join(sch['dropped']['shortcuts'])}",
                f"- Dropped as uninformative: {', '.join(sch['dropped']['uninformative'])}",
                f"- Split method: {SPLITS.get(d.get('split_method'), 'contiguous by position in the capture')}; "
                f"rows purged between splits: {d.get('rows_purged', 0):,}",
                "", "| Split | Flows | Attack rate |", "|---|---|---|"]
        for name in ("train", "val", "test"):
            out.append(f"| {name} | {sp[name]['n_rows']:,} | "
                       f"{d['split_attack_rates'][name]:.2%} |")
        out.append("")
    return out


def sec_baselines() -> list[str]:
    out = ["## 2. Baselines (models that cannot see topology)", ""]
    for ds in (PRIMARY, SECONDARY):
        rel = f"metrics/baselines/baselines_{ds}.json"
        d = read(rel)
        if not d:
            continue
        on_path = {k: v for k, v in d["results"].items()
                   if v.get("schema") == BASELINE_SCHEMA}
        stale = sorted(set(d["results"]) - set(on_path))
        out += [f"### {ds}", "", source(rel, d), ""]
        if on_path:
            prev = next(iter(on_path.values()))["reported_at_prevalence"]
            out += [f"Reported at {prev:.0%} prevalence (PR-AUC of a random model: "
                    f"{prev:.2f}).", "",
                    "| Model :: features | PR-AUC | F1 | FPR at 95% recall |",
                    "|---|---|---|---|"]
            for k, v in sorted(on_path.items()):
                a = v["aggregate"]
                out.append(f"| {k} | {pm(a.get('pr_auc'))} | {pm(a.get('f1'))} | "
                           f"{pm(a.get('fpr_at_95_recall'), 5)} |")
            out.append("")
        if stale:
            out += [f"**Do not quote:** {', '.join(stale)}. These were run before the "
                    f"reporting path was fixed and are at native prevalence, not "
                    f"{0.04:.0%}. Re-run with `python scripts/train_baselines.py`.", ""]
    return out


def sec_ablation(ds: str, number: str, title: str) -> list[str]:
    rel = f"metrics/gnn/gnn_{ds}.json"
    d = read(rel)
    out = [f"## {number}. {title}", ""]
    if not d:
        return out + missing("The graph model on this dataset.",
                             "python scripts/train_gnn.py --seeds 3")
    prev = d["config"]["eval"]["target_prevalence"]
    key = f"at_{prev:.0%}"
    res = d["results"]
    first = next(iter(res.values()))["runs"][0]
    selection = first.get("selection", "early_stopping")
    out += [source(rel, d), "",
            f"Reported at {prev:.0%} prevalence (PR-AUC of a random model: {prev:.2f})."]
    if selection != "fixed_budget_best_val":
        out += ["", "> **SUPERSEDED.** These runs used early stopping. The final "
                "results use a fixed training budget (decision D39). Re-run "
                "`python scripts/train_gnn.py --seeds 3` and re-export before "
                "quoting anything in this section."]
    else:
        out += [f"Training: fixed budget of {d['config']['train']['max_epochs']} "
                f"epochs, best-validation checkpoint kept."]
    out += ["", "| Variant | PR-AUC | F1 | FPR at 95% recall | PR-AUC per seed | "
            "Best epoch per seed |", "|---|---|---|---|---|---|"]
    for v, label in VARIANTS:
        if v not in res:
            continue
        a, runs = res[v]["aggregate"], res[v]["runs"]
        per = ", ".join(f(r[key]["pr_auc"]) for r in runs)
        ep = ", ".join(str(r.get("best_epoch", r.get("epochs_run"))) for r in runs)
        out.append(f"| {label} | {pm(a.get('pr_auc'))} | {pm(a.get('f1'))} | "
                   f"{pm(a.get('fpr_at_95_recall'), 5)} | {per} | {ep} |")
    out.append("")
    if "full" in res and "channel1_only" in res:
        full = [r[key]["pr_auc"] for r in res["full"]["runs"]]
        c1 = [r[key]["pr_auc"] for r in res["channel1_only"]["runs"]]
        gain = (res["full"]["aggregate"]["pr_auc"]["mean"]
                - res["channel1_only"]["aggregate"]["pr_auc"]["mean"])
        separated = min(full) > max(c1)
        fa = res["full"]["aggregate"].get("fpr_at_95_recall", {}).get("mean")
        ca = res["channel1_only"]["aggregate"].get("fpr_at_95_recall", {}).get("mean")
        out += [f"- **Topology gain** (full minus flow-features-only, PR-AUC): "
                f"**{gain:+.4f}**",
                f"- Seed ranges: full {min(full):.4f}–{max(full):.4f}; "
                f"flow-features-only {min(c1):.4f}–{max(c1):.4f}. "
                f"**{'They do not overlap.' if separated else 'They overlap.'}**"]
        if fa and ca:
            out.append(f"- False-positive rate at 95% recall: {ca:.5f} → {fa:.5f} "
                       f"({(fa - ca) / ca:+.0%})")
        out.append(f"- Seeds per variant: {len(full)}. With so few, say whether the "
                   f"ranges overlap; do not use the word \"significant\".")
    alpha = res.get("full", {}).get("runs", [{}])[0].get("channel_attribution")
    if alpha:
        out.append(f"- Mean share of the decision assigned to the neighbourhood "
                   f"channel (full model, seed {res['full']['runs'][0].get('seed', 0)}, "
                   f"whole test split): {alpha['topological_weight_mean']:.4f}")
    out.append("")

    mc = res.get("full", {}).get("multiclass_aggregate")
    out += ["### Attack-family head (full model)", ""]
    if mc:
        out += ["Scored on the whole test split at native prevalence.", "",
                f"- Macro-F1 over families: {pm(mc.get('macro_f1'))}",
                f"- Accuracy over all flows: {pm(mc.get('accuracy'))}",
                f"- Correct family among true attacks: "
                f"{pm(mc.get('attack_family_accuracy'))}",
                f"- **Correct family on true-positive detections** (the flows that "
                f"become evidence packs): {pm(mc.get('family_accuracy_on_detections'))}",
                ""]
    else:
        out += missing("The attack-family head was not scored in these runs.",
                       "python scripts/train_gnn.py --seeds 3")
    return out


def sec_vs_xgboost() -> list[str]:
    out = ["## 4. The graph model against XGBoost", ""]
    g = read(f"metrics/gnn/gnn_{PRIMARY}.json")
    b = read(f"metrics/baselines/baselines_{PRIMARY}.json")
    if not g or not b or "full" not in g["results"]:
        return out + missing("Needs the full model and the XGBoost baselines.",
                             "python scripts/train_gnn.py --seeds 3")
    ga = g["results"]["full"]["aggregate"]
    if g["results"]["full"]["runs"][0].get("selection") != "fixed_budget_best_val":
        out += ["> **SUPERSEDED.** The graph-model figures here come from the "
                "early-stopped runs (see Section 3). Re-export after the re-run.", ""]
    rows = [("graph model (full)", ga)]
    for k in ("xgboost::flow", "xgboost::flow+host"):
        v = b["results"].get(k)
        if v and v.get("schema") == BASELINE_SCHEMA:
            rows.append((k, v["aggregate"]))
    out += ["Same dataset, same prevalence, same reporting path.", "",
            "| Model | PR-AUC (ranking) | F1 (operating point) |", "|---|---|---|"]
    for name, a in rows:
        out.append(f"| {name} | {pm(a.get('pr_auc'))} | {pm(a.get('f1'))} |")
    best_pr = max(rows[1:], key=lambda r: r[1]["pr_auc"]["mean"], default=None)
    best_f1 = max(rows[1:], key=lambda r: r[1]["f1"]["mean"], default=None)
    if best_pr and best_f1:
        d_pr = ga["pr_auc"]["mean"] - best_pr[1]["pr_auc"]["mean"]
        d_f1 = ga["f1"]["mean"] - best_f1[1]["f1"]["mean"]
        out += ["", f"- Against the best XGBoost by PR-AUC ({best_pr[0]}): "
                f"PR-AUC **{d_pr:+.4f}**",
                f"- Against the best XGBoost by F1 ({best_f1[0]}): F1 **{d_f1:+.4f}**",
                "- **Both halves must be stated together.** "
                + ("The graph model ranks better and operates worse."
                   if d_pr > 0 > d_f1 else
                   "State the sign of each difference as given above.")]
    return out + [""]


def sec_zeroday() -> list[str]:
    rel = f"metrics/zeroday/zeroday_{PRIMARY}.json"
    d = read(rel)
    out = ["## 6. Zero-day: leave-one-attack-out", ""]
    if not d or not d.get("results"):
        return out + missing(
            "Recall on attack families held out of training. This is the claim in "
            "the project's title and has never been run.",
            "python scripts/train_zeroday.py --seeds 3")
    key_prev = d["reported_at_prevalence"]
    thin = d["config"]["thin_below_test_positives"]
    out += [source(rel, d), "",
            f"Each family was removed from training and validation entirely; recall "
            f"is on that family in the test split, at {key_prev:.0%} prevalence. "
            f"Model variant: {d['ablation']}.", "",
            "| Held-out family | Test flows | Recall | PR-AUC | Seeds | Thin? | "
            "The family head calls it |", "|---|---|---|---|---|---|---|"]
    for name, r in sorted(d["results"].items(), key=lambda kv: -kv[1]["test_positives"]):
        a = r["aggregate"]
        wrong = ", ".join(f"{k} {v:.0%}" for k, v in
                          list(r.get("held_out_mistaken_for", {}).items())[:2]) or "n/a"
        out.append(f"| {name} | {r['test_positives']:,} | {pm(a.get('recall'))} | "
                   f"{pm(a.get('pr_auc'))} | {r['n_seeds']} | "
                   f"{'yes' if r['test_positives'] < thin else 'no'} | {wrong} |")
    out += ["", f"\"Thin\" means fewer than {thin:,} held-out flows in the test split: "
            f"quote that recall with its count.", ""]
    for name, why in d.get("skipped", {}).items():
        out.append(f"- **Not measurable: {name}** — {why}")
    if d.get("runs_failed"):
        out.append(f"- Runs that failed and are excluded: {d['runs_failed']}")
    out += ["", "Do not report the mean across families. Which families are caught "
            "and which are not is the finding.", ""]
    return out


def sec_transfer() -> list[str]:
    out = ["## 7. Cross-dataset transfer (no retraining)", ""]
    rows = script("summarise_transfer").rows()
    if not rows:
        return out + missing("Transfer results.", "python scripts/transfer_eval.py")
    short = lambda s: s.replace("NF-", "").replace("-v2", "")      # noqa: E731
    out += ["*Source: every `results/metrics/transfer/transfer_*.json`, pooled by "
            "`scripts/summarise_transfer.py`.*", "",
            f"PR-AUC of a random model at this prevalence: {rows[0]['floor']:.2f}.", "",
            "| Model | Trained on → tested on | Kind | Seeds | PR-AUC | F1 | "
            "Out-of-vocabulary |", "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (not x["control"], x["model"], x["source"])):
        # One seed has no spread. Printing "± 0.0000" would state a measured
        # standard deviation of zero that was never measured.
        pr = ("n/a" if r["pr_auc"] is None else
              f"{r['pr_auc']:.4f} (one seed)" if r["n"] < 2 else
              f"{r['pr_auc']:.4f} ± {r['pr_std']:.4f}")
        oov = f"{r['oov']:.1%}" if r["oov"] is not None else "n/a"
        out.append(f"| {r['model']} | {short(r['source'])} → {short(r['target'])} | "
                   f"{'in-dataset control' if r['control'] else '**cross-dataset**'} | "
                   f"{r['n']} | {pr} | {f(r['f1'])} | {oov} |")
    out += ["", "- A direction with out-of-vocabulary above 50% is confounded: the "
            "model was given a degraded input as well as a different network. Rely on "
            "the other direction.",
            "- A row marked \"one seed\" has no standard deviation. Do not "
            "attach one.", ""]
    return out


def sec_explainability() -> list[str]:
    rel = f"metrics/evidence/evidence_{PRIMARY}_summary.json"
    d = read(rel)
    out = ["## 8. Explainability (over the evidence packs)", ""]
    if not d:
        return out + missing("The evidence-pack summary.",
                             "python scripts/summarise_evidence.py")
    s = d["summary"]
    t, m = s["topological_weight"], s["family_named_to_the_model"]
    out += [source(rel, d), "",
            f"{s['n_packs']} evidence packs: {s['n_true_attacks']} true attacks and "
            f"{s['n_false_positives']} false positives (benign flows the detector "
            f"flagged). Checkpoint: {d.get('checkpoint')}. The packs are a stratified "
            f"sample of *flagged* flows, not of traffic.", "",
            "### Share of each decision assigned to the neighbourhood channel", "",
            "| Group | Packs | Median | 25th–75th percentile | Mean |", "|---|---|---|---|---|"]
    def row(label, x):
        if x:
            out.append(f"| {label} | {x['n']} | {x['median']:.4f} | "
                       f"{x['p25']:.4f}–{x['p75']:.4f} | {x['mean']:.4f} |")
    row("all packs", t["all"]); row("true attacks", t["true_attacks"])
    row("false positives", t["false_positives"])
    for fam, x in t["by_true_family"].items():
        if fam != "Benign":
            row(fam, x)
    feats = s["features_shown_to_the_model"]
    out += ["", f"### Features most often among the top {feats['top_k']} attributed", "",
            "| Feature | Share of packs | Times it raised suspicion | "
            "Times it lowered suspicion |", "|---|---|---|---|"]
    for r in feats["ranked"][:10]:
        out.append(f"| {r['feature']} | {r['share_of_packs']:.0%} | "
                   f"{r['raises_suspicion']} | {r['lowers_suspicion']} |")
    out += ["", "A feature and its encoded variants (a port and its port bucket) "
            "count as one feature per pack but as separate entries in the last two "
            "columns, so those can exceed the pack count.",
            "", "### The attack family named in the pack", "",
            f"- Correct for {f(m['correct_on_true_attacks'])} of "
            f"{m['n_true_attacks']} true attacks in the sample",
            f"- False positives were named as: {m['false_positives_named_as']}",
            f"- Ambiguous packs (confidence below 0.9): {m['ambiguous_packs']}", "",
            "### Influence of single neighbouring flows", ""]
    o = s.get("neighbour_occlusion")
    if o:
        mi = o["max_abs_influence"]
        out += [f"- Up to {o['tested_per_detection_cap']} neighbouring flows were "
                f"removed one at a time per detection (median tested: "
                f"{o['neighbours_tested']['median']:.0f}; median sharing a host: "
                f"{o['neighbours_sharing_a_host']['median']:.0f})",
                f"- Largest change in the detection score from removing one "
                f"neighbour: median {mi['median']:.2e}, 95th percentile "
                f"{mi['p95']:.2e}, maximum {mi['max']:.2e}",
                f"- Packs with any neighbour above the {o['threshold']} threshold: "
                f"{o['packs_with_an_influential_neighbour']} of {s['n_packs']}", ""]
    else:
        out += missing("These packs were made before neighbour occlusion was "
                       "recorded.", "python scripts/make_evidence.py")
    return out


def _prompt_sensitivity() -> list[str]:
    """v3 against v4, each model against itself on the packs it answered under
    BOTH prompts. A per-file mean would compare prompts over different packs:
    deepseek-v4-pro was capped at 40 packs on v4 (D51) and lost some to the
    output limit, so its pairing is smaller than the others'."""
    import statistics
    rep = RESULTS / "reports"
    runs = {}
    for f in sorted(rep.glob("reports_ollama_*_v[34].json")):
        d = json.loads(f.read_text())
        runs[(d["model"], d["prompt_version"])] = {
            r["detection_id"]: r for r in d["reports"]
            if r.get("ok") and not r.get("degenerate")}
    pairs = {m: set(runs[(m, "v3")]) & set(runs[(m, "v4")])
             for m in sorted({m for m, _ in runs})
             if (m, "v3") in runs and (m, "v4") in runs}
    if not pairs:
        return []
    out = ["", "### Prompt sensitivity: v3 against v4 on the same packs", "",
           "Each model is compared with itself, on the packs it answered under both "
           "prompts (failed and degenerate reports left out on either side). "
           "deepseek-v4-pro:0813 was capped at 40 packs on v4 after runaway "
           "generations (D51), so its pairing is small.", "",
           "| Model | Prompt | Packs | Groundedness | Direction faithful | Borderline "
           "conveyed | Uncertainty conveyed | Names predicted class | Mean words |",
           "|---|---|---|---|---|---|---|---|---|"]
    def stat(rows, fn, nd=4):
        vals = [fn(r) for r in rows]
        vals = [v for v in vals if v is not None]
        return f"{statistics.mean(vals):.{nd}f}" if vals else "n/a"
    for model, common in pairs.items():
        for v in ("v3", "v4"):
            rows = [runs[(model, v)][i] for i in sorted(common)]
            out.append(
                f"| {model} | {v} | {len(rows)} | "
                f"{stat(rows, lambda r: (r.get('scores') or {}).get('groundedness'))} | "
                f"{stat(rows, lambda r: None if not (r.get('direction') or {}).get('n_checked') else float(r['direction']['direction_faithful']))} | "
                f"{stat(rows, lambda r: None if not r.get('borderline') else float(r['borderline']['borderline_conveyed']))} | "
                f"{stat(rows, lambda r: None if not r.get('uncertainty') else float(r['uncertainty']['uncertainty_conveyed']))} | "
                f"{stat(rows, lambda r: None if not r.get('classification') else float(r['classification']['names_predicted']))} | "
                f"{stat(rows, lambda r: (r.get('scores') or {}).get('report_words'), 0)} |")
    return out


def _list_price_total() -> list[str]:
    """What the whole sweep would cost at the service's list prices, failed
    calls included (a runaway generation is billed in full). The study ran on
    the free allowance, then a flat-fee subscription; neither itemises."""
    total, done = {}, {}
    for f in sorted((RESULTS / "reports").glob("reports_ollama_*_v[34].json")):
        d = json.loads(f.read_text())
        v = d["prompt_version"]
        for r in d["reports"]:
            c = r.get("reference_cost_usd") or 0.0
            total[v] = total.get(v, 0.0) + c
            if r.get("ok") and not r.get("degenerate"):
                done[v] = done.get(v, 0.0) + c
    if not total:
        return []
    parts = ", ".join(f"{v} ${total[v]:.2f}" for v in sorted(total))
    return ["", f"List-price equivalent of every language-model call: "
            f"${sum(total.values()):.2f} ({parts}); ${sum(done.values()):.2f} of it "
            f"for reports that completed, the rest on failed or runaway generations. "
            f"The calls ran first on Ollama's free allowance (D45) and then on a "
            f"flat-fee subscription (D50); neither itemises per call, so what each "
            f"call actually cost is unknown."]


def _hidden_output() -> list[str]:
    """How much each model generated beyond the note. The hosted service bills
    reasoning as output tokens but the adapter keeps only the note, so the
    ratio of output tokens to note words shows unseen reasoning: plain English
    runs at roughly 1.3-1.7 tokens a word."""
    import statistics
    rows = []
    for f in sorted((RESULTS / "reports").glob("reports_ollama_*_v[34].json")):
        d = json.loads(f.read_text())
        rs = [r for r in d["reports"] if r.get("ok") and not r.get("degenerate")
              and r.get("output_tokens")]
        if not rs:
            continue
        tok = statistics.median(r["output_tokens"] for r in rs)
        words = statistics.median(len(r["report"].split()) for r in rs)
        ratio = statistics.median(r["output_tokens"] / max(1, len(r["report"].split()))
                                  for r in rs)
        rows.append((d["prompt_version"], d["model"], len(rs), tok, words, ratio))
    if not rows:
        return []
    out = ["", "### Output beyond the note", "",
           "The service counts a model's reasoning as output tokens; only the note is "
           "kept and scored. Plain English needs about 1.3-1.7 tokens a word, so a "
           "higher ratio is reasoning nobody reads. Medians over completed reports.", "",
           "| Model | Prompt | Reports | Output tokens | Note words | Output tokens per note word |",
           "|---|---|---|---|---|---|"]
    for v, m, n, tok, words, ratio in sorted(rows):
        out.append(f"| {m} | {v} | {n} | {tok:,.0f} | {words:,.0f} | {ratio:.1f} |")
    return out


def sec_llm() -> list[str]:
    out = ["## 9. Language-model comparison", ""]
    rows = script("summarise_llm_study").rows()
    real = [r for r in rows if r["provider"] != "stub"]
    sampling = read(f"evidence/evidence_{PRIMARY}_sampling.json")
    if sampling:
        out += [f"Evidence packs: {sampling['n_selected']} "
                f"({sampling['false_positives_taken']} false positives). By true "
                f"family: {sampling['taken_by_true_family']}.", ""]
    if not real:
        out += missing("No language model has been run; only the template control "
                       "exists.", "python scripts/run_llm_study.py")
    if not rows:
        return out
    out += ["*Source: every `results/reports/reports_*.json`, pooled by "
            "`scripts/summarise_llm_study.py`.*", "",
            "| Model | Prompt | Reports | Failed | Groundedness | Fabricated-address "
            "reports | Groundedness on false positives | Names predicted class | "
            "Class faithful | Substitutes class | Attributed features cited | Cites none | "
            "Latency p50 / p95 (s) | Charged (USD) | List price (USD) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    money = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "unknown"   # noqa: E731
    for r in sorted(rows, key=lambda x: (x["prompt"] or "", x["model"] or "")):
        out.append(
            f"| {r['provider']}/{r['model']} | {r['prompt']} | {r['n_ok']} | "
            f"{r['n_failed']} | {f(r['groundedness'])} | {r['fabricated_addr_reports']} | "
            f"{f(r['groundedness_on_fp'])} (n = {r['n_false_positive']}) | "
            f"{f(r['class_named'])} | {f(r['class_faithful'])} | {f(r['class_substituted'])} | "
            f"{f(r['attr_coverage'])} | {f(r['attr_cites_none'])} | "
            f"{f(r['latency_p50'], 2)} / {f(r['latency_p95'], 2)} | "
            f"{money(r['cost'])} | {money(r['reference_cost'])} |")
    out += _prompt_sensitivity()
    out += _hidden_output()
    out += _list_price_total()
    out += ["", "### Direction of reasons, and borderline detections (D49)", "",
            "| Model | Prompt | Uncertainty conveyed (ambiguous category) | Reports with a "
            "checkable reason | Direction faithful | Direction errors | Borderline "
            "detections | Borderline conveyed |",
            "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["prompt"] or "", x["model"] or "")):
        out.append(
            f"| {r['provider']}/{r['model']} | {r['prompt']} | {f(r['uncertainty_rate'])} | "
            f"{r.get('direction_checked')} | {f(r.get('direction_faithful'))} | "
            f"{r.get('direction_errors')} | {r.get('borderline_n')} | "
            f"{f(r.get('borderline_conveyed'))} |")
    degen = [(r["model"], r["prompt"], r.get("n_degenerate")) for r in rows if r.get("n_degenerate")]
    leaked = [(r["model"], r["prompt"], r.get("reasoning_leaked")) for r in rows if r.get("reasoning_leaked")]
    if leaked:
        out += ["", "Reports that opened with the model's own working-out instead of the "
                "note (C38): " + "; ".join(f"{m} {v}: {n}" for m, v, n in leaked) + "."]
    failed = [(r["model"], r["prompt"], r["n_failed"]) for r in rows if r["n_failed"]]
    if failed or degen:
        out += ["", "Failed reports by model (empty after the output limit, network, or "
                "error) and degenerate reports (collapsed into repeated characters, C37; "
                "kept out of every quality mean): "
                + "; ".join(f"{m} {v}: {n} failed" for m, v, n in failed)
                + ("; " if failed and degen else "")
                + "; ".join(f"{m} {v}: {n} degenerate" for m, v, n in degen) + "."]
    out += ["", "- The stub is a template with no language model: the control arm.",
            "- The direction check reads a reason's direction from clause-level "
            "wording and skips any clause it cannot read, so its error count is a "
            "lower bound.",
            "- \"Borderline\" is decided by the facts layer by fixed rules: margin over "
            "the threshold below a tenth of the room above it, category confidence "
            "below 0.6, or a second category at 0.05 or more.",
            "- Groundedness measures \"invented nothing\", not \"reasoned correctly\".",
            "- The class and feature checks are lexical: they match names and "
            "phrases, are blind to negation, and are generous about what counts as "
            "citing a feature.",
            "- \"Class faithful\" means the report names the predicted category and no "
            "category the detector gave below 0.05. It is lexical: \"may be scanning\" "
            "or naming injection beside cross-site scripting (a kind of injection) "
            "counts against it. \"Substitutes class\" -- another category named "
            "instead of the prediction -- is the failure the study looks for.",
            "- Latency is response time from a hosted service; it is not model speed.",
            "- A cost of \"unknown\" is not zero.", ""]
    for r in real:
        sess = r.get("sessions") or []
        model = (sess[-1].get("model") or {}) if sess else {}
        if model.get("digest"):
            gpus = {g["name"] for s in sess for g in (s.get("gpus") or [])}
            size = model.get("parameter_size")
            size = (f"{int(size) / 1e9:,.1f}B parameters" if str(size).isdigit()
                    else size or "size not reported")
            hw = (f"GPU {', '.join(sorted(gpus))}" if gpus
                  else "hardware not disclosed by the hosted service")
            out.append(f"- {r['model']} ({r['prompt']}): digest `{model['digest'][:19]}`, "
                       f"{size}, {model.get('quantization_level') or 'quantisation not reported'}, "
                       f"{hw}, {len(sess)} session(s)")
    return out + [""]


def sec_replay() -> list[str]:
    rel = f"metrics/replay/replay_{PRIMARY}.json"
    d = read(rel)
    out = ["## 10. Replay (the demonstration harness)", ""]
    if not d:
        return out + missing("Replay metrics at full scale.", "python scripts/replay.py")
    s = d["stats"]
    return out + [source(rel, d), "",
                  "Offline replay of the held-out split. Not a real-time measurement.", "",
                  f"- Windows replayed: {s['windows']} ({s['flows']:,} flows)",
                  f"- Alerts: {s['alerts']:,} ({s['alerts_per_1000_flows']} per 1,000 flows)",
                  f"- Precision {s['precision']}, recall {s['recall']}",
                  f"- Windows in which the language model would be invoked: "
                  f"{s['trigger_rate']:.1%}",
                  f"- Scoring time per window: median {s['scoring_ms_median']} ms, "
                  f"95th percentile {s['scoring_ms_p95']} ms", ""]


def sec_figures() -> list[str]:
    d = read("figures/results_manifest.json")
    out = ["## 11. Figures", ""]
    if not d:
        return out + missing("The figures manifest.", "python scripts/make_figures.py")
    for key, fig in d["figures"].items():
        if fig["status"] == "drawn":
            out += [f"- **`results/figures/{fig['file']}`** — {fig['caption']}"]
        else:
            out += [f"- *{key}: not drawn yet ({fig['reason']})*"]
    return out + [""]


def sec_process() -> list[str]:
    tests = sum(src.read_text().count("\ndef test_")
                for src in (REPO_ROOT / "tests").glob("test_*.py"))
    return ["## 12. Process", "",
            f"- Automated tests in the repository: {tests}",
            "- Counts of logged decisions and defects are kept in the project notes "
            "(Decision Register, Defect Register), not in this file.", ""]


ARCHIVE = "superseded/2026-10-10_duration_artefact"


def _seed_stat(values: list) -> str:
    """mean ± sd (n = k) over per-seed values, in the file's own convention."""
    import statistics
    v = [x for x in values if isinstance(x, (int, float))]
    if not v:
        return "n/a"
    if len(v) == 1:
        return f"{v[0]:.4f} (one seed)"
    return f"{statistics.mean(v):.4f} ± {statistics.pstdev(v):.4f} (n = {len(v)})"


def _gnn_variants(doc: dict | None) -> dict:
    return (doc or {}).get("results", {})


def sec_step_back() -> list[str]:
    """D47 and D48: the data-quality findings, the duplicate-flow evaluation, and
    what the duration artefact was worth to each model."""
    out = ["## 2a. Data quality, duplicates, and the artefact comparison (D47, D48)", ""]
    for ds in (PRIMARY, SECONDARY):
        q = read(f"metrics/data_quality_{ds}.json")
        if not q:
            continue
        w = q["validity_flags"].get("duration_wrapped", {})
        fams = sorted(w.get("share_by_family", {}).items(), key=lambda kv: -kv[1])[:4]
        dup = q.get("duplicates", {}).get("features_only", {})
        lab = q.get("labels", {})
        out += [f"### {ds}", "", source(f"metrics/data_quality_{ds}.json", q), "",
                f"- Flow duration wrapped (32-bit counter overflow): {w.get('n', 0):,} rows, "
                f"{w.get('share', 0):.2%}. Most affected: "
                + ", ".join(f"{k} {v:.1%}" for k, v in fams) + ".",
                "- In every row that is not wrapped the duration is exactly 0, so the three "
                "duration columns carry no genuine timing. They and the four throughput "
                "columns computed from them were removed from the model's inputs (D47).",
                f"- Test flows identical to some training flow (all measured columns): "
                f"{dup.get('test_rows_also_in_train_share', 0):.1%}.",
                f"- Flows whose identical twin carries another label: "
                f"{dup.get('rows_whose_identical_twin_has_another_label', 0):,}."]
        for k, label in (("involves_external_address", "touching an address outside the testbed"),
                         ("involves_192_168_1_1", "touching 192.168.1.1")):
            v = lab.get(k) or {}
            if v.get("share_of_rows"):
                out.append(f"- Flows {label}: {v['share_of_rows']:.2%} of rows; attack rate "
                           f"{v['attack_rate_when_true']} when true against "
                           f"{v['attack_rate_when_false']} when false.")
        pre = read("metrics/preprocess_NF-ToN-IoT-v2.json" if ds == PRIMARY
                   else "metrics/preprocess.json") or {}
        share = pre.get("share_identical_to_a_training_row")
        if share:
            out.append(f"- Test flows whose model input matches a training flow "
                       f"(the D48 \"novel only\" subset excludes these): {share['test']:.1%}.")
        out.append("")

    # Novel only, beside the headline, for every model.
    out += ["### Scores on novel test flows only (D48)", "",
            "PR-AUC at 4% prevalence (floor 0.04). \"All\" is the headline; \"novel "
            "only\" drops test flows whose model input matches a training flow.", "",
            "| Dataset | Model | All test flows | Novel only |", "|---|---|---|---|"]
    rows = 0
    for ds in (PRIMARY, SECONDARY):
        for name, v in _gnn_variants(read(f"metrics/gnn/gnn_{ds}.json")).items():
            runs = v.get("runs", [])
            nov = [(r.get("novel_only") or {}).get("at_4%", {}).get("pr_auc") for r in runs]
            if any(x is not None for x in nov):
                out.append(f"| {ds} | GNN {name} | {_seed_stat([r['at_4%']['pr_auc'] for r in runs])} "
                           f"| {_seed_stat(nov)} |"); rows += 1
        b = read(f"metrics/baselines/baselines_{ds}.json") or {}
        for key, v in sorted(b.get("results", {}).items()):
            runs = v.get("runs", [])
            nov = [(r.get("novel_only") or {}).get("at_4%", {}).get("pr_auc") for r in runs]
            if any(x is not None for x in nov):
                out.append(f"| {ds} | {key} | {_seed_stat([r['at_4%']['pr_auc'] for r in runs])} "
                           f"| {_seed_stat(nov)} |"); rows += 1
    if not rows:
        out += ["", "**NOT YET MEASURED.** No result carries the novel-only block yet."]
    out.append("")

    # With and without the artefact.
    out += ["### What the duration artefact was worth (D47)", "",
            f"\"With\" is the archived run (`results/{ARCHIVE}/`), which saw the seven "
            "artefact columns; \"without\" is the current run. The rows are identical "
            "and in the same order, so the feature list is the only difference on "
            f"{PRIMARY}. On {SECONDARY} the earlier preprocessing also lacked the "
            "outlier clip, so that comparison has two differences.", "",
            "| Dataset | Model | PR-AUC with the artefact | PR-AUC without | Difference |",
            "|---|---|---|---|---|"]
    rows = 0
    for ds in (PRIMARY, SECONDARY):
        old = _gnn_variants(read(f"{ARCHIVE}/metrics/gnn/gnn_{ds}.json"))
        new = _gnn_variants(read(f"metrics/gnn/gnn_{ds}.json"))
        for name in old:
            if name in new:
                a = old[name]["aggregate"]["pr_auc"]["mean"]; b_ = new[name]["aggregate"]["pr_auc"]["mean"]
                out.append(f"| {ds} | GNN {name} | {a:.4f} | {b_:.4f} | {b_ - a:+.4f} |"); rows += 1
        ob = (read(f"{ARCHIVE}/metrics/baselines/baselines_{ds}.json") or {}).get("results", {})
        nb = (read(f"metrics/baselines/baselines_{ds}.json") or {}).get("results", {})
        for key in sorted(ob):
            if key in nb and ob[key].get("schema") == BASELINE_SCHEMA:
                a = ob[key]["aggregate"]["pr_auc"]["mean"]; b_ = nb[key]["aggregate"]["pr_auc"]["mean"]
                out.append(f"| {ds} | {key} | {a:.4f} | {b_:.4f} | {b_ - a:+.4f} |"); rows += 1
    oz = (read(f"{ARCHIVE}/metrics/zeroday/zeroday_{PRIMARY}.json") or {}).get("results", {})
    nz = (read(f"metrics/zeroday/zeroday_{PRIMARY}.json") or {}).get("results", {})
    if oz and nz:
        out += ["", f"Leave-one-attack-out recall on {PRIMARY}, with and without the artefact:", "",
                "| Held-out family | Recall with | Recall without | Difference |", "|---|---|---|---|"]
        for fam in sorted(oz, key=lambda k: -oz[k]["test_positives"]):
            if fam in nz:
                a = oz[fam]["aggregate"]["recall"]["mean"]; b_ = nz[fam]["aggregate"]["recall"]["mean"]
                out.append(f"| {fam} | {a:.4f} | {b_:.4f} | {b_ - a:+.4f} |"); rows += 1
    if not rows:
        out += ["", "**NOT YET MEASURED.** The retrained results do not exist yet."]
    return out + [""]


def build() -> str:
    now = datetime.now(timezone.utc).isoformat()[:19]
    head = [
        "# Report facts", "",
        f"*Exported {now} UTC by `scripts/export_report_facts.py`. Every number below "
        f"was read from a results file at export time. None was typed in.*", "",
        "**For whoever drafts from this file (human or model):**", "",
        "1. Use these numbers exactly as written. Do not round further, recompute, "
        "or combine them.",
        "2. A section marked **NOT YET MEASURED** has no numbers. Write `[NUM?]`.",
        "3. A section marked **SUPERSEDED** has numbers that will be replaced. Do "
        "not quote them in final text.",
        "4. `0.9887 ± 0.0030 (n = 3)` means mean ± standard deviation over 3 seeds.",
        "5. Every PR-AUC is at the stated prevalence. A random model scores the "
        "prevalence itself.", "",
    ]
    body = (sec_datasets() + sec_step_back() + sec_baselines()
            + sec_ablation(PRIMARY, "3", f"Topology ablation on {PRIMARY}")
            + sec_vs_xgboost()
            + sec_ablation(SECONDARY, "5", f"Topology ablation on {SECONDARY} "
                                           f"(the saturated dataset)")
            + sec_zeroday() + sec_transfer() + sec_explainability() + sec_llm()
            + sec_replay() + sec_figures() + sec_process())
    return "\n".join(head + body) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=RESULTS / "report_facts.md")
    args = ap.parse_args()
    text = build()
    args.out.write_text(text)
    gaps = text.count("**NOT YET MEASURED.**")
    stale = text.count("SUPERSEDED.")
    shown = args.out.relative_to(REPO_ROOT) if args.out.is_relative_to(REPO_ROOT) else args.out
    print(f"written -> {shown}  ({len(text.split()):,} words)")
    print(f"  sections not yet measured: {gaps}")
    print(f"  sections superseded by a pending re-run: {stale}\n")


if __name__ == "__main__":
    main()
