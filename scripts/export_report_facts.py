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
            "Substitutes class | Attributed features cited | Cites none | "
            "Latency p50 / p95 (s) | Charged (USD) | List price (USD) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    money = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "unknown"   # noqa: E731
    for r in sorted(rows, key=lambda x: (x["prompt"] or "", x["model"] or "")):
        out.append(
            f"| {r['provider']}/{r['model']} | {r['prompt']} | {r['n_ok']} | "
            f"{r['n_failed']} | {f(r['groundedness'])} | {r['fabricated_addr_reports']} | "
            f"{f(r['groundedness_on_fp'])} (n = {r['n_false_positive']}) | "
            f"{f(r['class_faithful'])} | {f(r['class_substituted'])} | "
            f"{f(r['attr_coverage'])} | {f(r['attr_cites_none'])} | "
            f"{f(r['latency_p50'], 2)} / {f(r['latency_p95'], 2)} | "
            f"{money(r['cost'])} | {money(r['reference_cost'])} |")
    out += ["", "- The stub is a template with no language model: the control arm.",
            "- Groundedness measures \"invented nothing\", not \"reasoned correctly\".",
            "- The class and feature checks are lexical: they match names and "
            "phrases, are blind to negation, and are generous about what counts as "
            "citing a feature.",
            "- Latency is comparable within a hosting arm only.",
            "- A cost of \"unknown\" is not zero.", ""]
    for r in real:
        sess = r.get("sessions") or []
        model = (sess[-1].get("model") or {}) if sess else {}
        if model.get("digest"):
            gpus = {g["name"] for s in sess for g in (s.get("gpus") or [])} or {"none"}
            out.append(f"- {r['model']}: digest `{model['digest'][:19]}`, "
                       f"{model.get('parameter_size')}, {model.get('quantization_level')}, "
                       f"GPU {', '.join(sorted(gpus))}, {len(sess)} session(s)")
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
    body = (sec_datasets() + sec_baselines()
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
