"""The Streamlit demo -- a thin view over `ReplayEngine` (Phase 8, D7).

Run it with:

    streamlit run src/gnnids/ui/app.py

**Everything with logic in it lives elsewhere.** Replay is `ui/replay.py`,
evidence packs are `explain/evidence.py`, reports are `llm/`. This module
arranges widgets. That split is what makes the demo testable: the video showcase
is half the assessed deliverable, and a failure the night before it should be a
rendering bug, not a logic one.

The panels follow the pipeline the project actually implements, left to right:
traffic arrives -> a flow is flagged -> evidence is assembled -> a report is
written. A viewer should be able to read the architecture off the screen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _configs() -> tuple[dict, dict, dict, dict]:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "replay.yaml").read_text())
    gnn_cfg = yaml.safe_load((REPO_ROOT / cfg["gnn_config"]).read_text())
    pre_cfg = yaml.safe_load((REPO_ROOT / gnn_cfg["preprocess_config"]).read_text())
    ds_cfg = yaml.safe_load((REPO_ROOT / pre_cfg["dataset_config"]).read_text())
    return cfg, gnn_cfg, pre_cfg, ds_cfg


@st.cache_resource(show_spinner=False)
def _replay(window: int, n_windows: int, checkpoint: str) -> dict:
    """One replay, run once and kept.

    Streamlit reruns this whole script on every interaction. Until 2026-10-03
    the replay itself ran inside the script behind a button, so choosing a
    different detection reran nothing -- the button was no longer "pressed" and
    the page fell back to its start screen -- and a cached engine would have
    added each rerun's windows to the same running totals. Caching the finished
    result fixes both: the numbers on screen are from one replay, however many
    times the page redraws.
    """
    from replay import build_engine

    cfg, gnn_cfg, pre_cfg, ds_cfg = _configs()
    engine, _, inputs = build_engine(
        cfg, gnn_cfg, pre_cfg, ds_cfg, REPO_ROOT / checkpoint, window, n_windows)
    detections = []
    for _, found in engine.run(n_windows):
        detections.extend(found)
    return {"detections": detections, "stats": engine.stats.as_dict(),
            "threshold": engine.threshold, "dataset": ds_cfg["name"],
            "inputs": inputs, "window_size": pre_cfg["graph"]["window_size"],
            "processed_dir": str(REPO_ROOT / pre_cfg["output"]["dir"])}


@st.cache_resource(show_spinner=False)
def _addresses(processed_dir: str):
    """Source and destination addresses for every flow, as Arrow columns."""
    import pyarrow.parquet as pq

    return pq.read_table(Path(processed_dir) / "meta.parquet",
                         columns=["IPV4_SRC_ADDR", "IPV4_DST_ADDR"])


def _evidence_file(dataset: str) -> tuple[Path | None, bool]:
    """The real pack file for this dataset, else the smoke one, flagged.

    Chosen by name. The old rule -- the alphabetically last file matching
    `*evidence*.json` -- began picking the sampling record once that existed,
    and a sampling record is not a list of packs.
    """
    folder = REPO_ROOT / "results" / "evidence"
    real, smoke = (folder / f"evidence_{dataset}.json",
                   folder / f"smoke_evidence_{dataset}.json")
    if real.exists():
        return real, False
    return (smoke, True) if smoke.exists() else (None, False)


def _reports_for(detection_id: str) -> list[dict]:
    """Every model's report on one detection, real models before the stub."""
    folder = REPO_ROOT / "results" / "reports"
    found = []
    for path in sorted(folder.glob("*reports_*.json")) if folder.exists() else []:
        doc = json.loads(path.read_text())
        if not isinstance(doc, dict):
            continue
        for row in doc.get("reports", []):
            if row["detection_id"] == detection_id and row["ok"]:
                found.append({**row, "provider": doc["provider"], "model": doc["model"],
                              "prompt_version": doc["prompt_version"],
                              "smoke": doc.get("smoke", False)})
    return sorted(found, key=lambda r: (r["provider"] == "stub", r["smoke"],
                                        r["model"], r["prompt_version"]))


def _neighbourhood(pack: dict, run: dict) -> None:
    """The graph around the flagged flow -- what the neighbourhood channel saw."""
    from gnnids.ui.topology import ego_graph, figure, window_bounds

    inputs, row = run["inputs"], pack["provenance"]["row_index"]
    split = inputs.splits[pack["provenance"].get("split", "test")]
    try:
        lo, hi = window_bounds(row, split["start"], split["stop"], run["window_size"])
    except ValueError as e:
        st.info(f"Cannot draw the neighbourhood: {e}")
        return

    table = _addresses(run["processed_dir"]).slice(lo, hi - lo)
    src_ip = table.column("IPV4_SRC_ADDR").to_pylist()
    dst_ip = table.column("IPV4_DST_ADDR").to_pylist()
    target = row - lo
    # The pack records the flow's addresses and its row. If they disagree with
    # the capture, the pack was made from different preprocessing and the graph
    # would be of some other flow -- say so rather than draw it.
    if (src_ip[target], dst_ip[target]) != (pack["flow"]["src_ip"], pack["flow"]["dst_ip"]):
        st.warning("This evidence pack does not match the preprocessed capture on "
                   "disk (it was made from a different preprocessing run), so its "
                   "neighbourhood cannot be drawn.")
        return

    graph = ego_graph(inputs.src[lo:hi], inputs.dst[lo:hi], target)
    names = {}
    for code, ip in zip(inputs.src[lo:hi].tolist(), src_ip):
        names.setdefault(code, ip)
    for code, ip in zip(inputs.dst[lo:hi].tolist(), dst_ip):
        names.setdefault(code, ip)
    st.plotly_chart(figure(graph, names), use_container_width=True)
    st.caption(
        f"**{graph['neighbours_sharing_a_host']:,}** other flows share a host with "
        f"this one, out of {graph['window_flows']:,} in its window. Drawn: the "
        f"{graph['hosts_shown']} busiest of {graph['hosts_in_neighbourhood']:,} "
        f"neighbouring hosts; line thickness grows with the number of flows "
        f"between two hosts. This is the neighbourhood the model's second "
        f"channel aggregates over.")


def main() -> None:
    st.set_page_config(page_title="GNN-LLM NIDS", layout="wide")
    st.title("Explainable Network Intrusion Detection")
    st.caption(
        "Replay of a held-out capture. All reported metrics come from offline "
        "batch evaluation — this is a demonstration harness, not a real-time "
        "system (D7).")

    with st.sidebar:
        st.header("Replay")
        window = st.select_slider("Flows per window", [2000, 5000, 10000], 2000)
        n_windows = st.slider("Windows to replay", 1, 40, 6)
        checkpoint = st.text_input(
            "Checkpoint", "results/checkpoints/NF-ToN-IoT-v2_full_seed0.pt")
        if st.button("Run replay", type="primary"):
            # Remembered across reruns, so the page keeps showing this replay
            # while the viewer picks detections and models below.
            st.session_state["replay"] = (window, n_windows, checkpoint)

    if "replay" not in st.session_state:
        st.info("Choose a window size and press **Run replay**.")
        return

    try:
        with st.spinner("Replaying the capture…"):
            run = _replay(*st.session_state["replay"])
    except FileNotFoundError:
        st.error(f"No checkpoint at {st.session_state['replay'][2]}. Train one first.")
        return

    st.write(f"**{run['dataset']}** · threshold "
             f"`{run['threshold']:.4f}` (chosen on validation, from the checkpoint)")

    detections, stats = run["detections"], run["stats"]
    cols = st.columns(5)
    for col, (label, key) in zip(cols, [
            ("Flows replayed", "flows"), ("Alerts", "alerts"),
            ("Alerts / 1k flows", "alerts_per_1000_flows"),
            ("Windows alerting", "trigger_rate"),
            ("Median scoring", "scoring_ms_median")]):
        v = stats[key]
        col.metric(label, f"{v:,}" if isinstance(v, int) else
                   (f"{v} ms" if key.endswith("ms_median") else v))

    # Without this the demo overstates the alert load enormously: this split is
    # ~63% attack natively, so "433 alerts per 1,000 flows" is a property of the
    # capture, not of the detector. The reported figures are at a standardised
    # 4% (D23), and a viewer reading the two as the same number would be badly
    # misled.
    st.caption(
        "These counts are at the capture's **native** attack rate, which is far "
        "higher than a real network's. The project's reported precision and "
        "recall are computed at a standardised 4% prevalence (D23) and are not "
        "these numbers. What replay adds is latency and alert volume.")

    if not detections:
        st.warning("No flow crossed the threshold in this stretch of traffic.")
        return

    st.subheader("Detections")
    ranked = sorted(detections, key=lambda d: -d.score)[:200]
    st.dataframe(
        [{"window": d.window_index, "score": round(d.score, 4),
          "ground truth": "attack" if d.true_label else "benign",
          "scoring ms": d.latency_ms.get("scoring")} for d in ranked],
        use_container_width=True, height=260)

    st.subheader("Evidence and report")
    st.caption(
        "The evidence pack is the only thing the language model sees. Anything "
        "in the report that is not in the pack was invented — which is what "
        "makes groundedness measurable.")

    packs_file, is_smoke = _evidence_file(run["dataset"])
    if packs_file is None:
        st.info("No evidence packs yet — run `scripts/make_evidence.py`.")
        return
    if is_smoke:
        st.warning("Showing **smoke-run** evidence packs: a real pack file does not "
                   "exist yet. Their numbers are meaningless.")
    packs = json.loads(packs_file.read_text())
    if not packs:
        st.info("The evidence file is empty.")
        return

    choice = st.selectbox(
        "Detection", range(len(packs)),
        format_func=lambda i: (f"{packs[i]['flow'].get('src_ip')} → "
                               f"{packs[i]['flow'].get('dst_ip')}  "
                               f"({packs[i]['detection']['predicted_class']}, "
                               f"score {packs[i]['detection']['score']:.3f}"
                               + (", benign in truth"
                                  if packs[i]["flow"].get("true_label") == "Benign"
                                  else "") + ")"))
    pack = packs[choice]

    st.markdown("**The flagged flow and its neighbourhood**")
    _neighbourhood(pack, run)

    left, right = st.columns(2)
    with left:
        st.markdown("**Why it was flagged**")
        cw = pack["attribution"].get("channel_weights")
        if cw:
            st.write(f"Decision weight — the flow's own features "
                     f"{cw['attribute']:.0%}, its network context "
                     f"{cw['topological']:.0%}")
        for f in pack["attribution"]["top_features"][:6]:
            arrow = "↑" if f["direction"] == "increases_suspicion" else "↓"
            st.write(f"{arrow} `{f['name']}` = {f['value']}")
        if pack["attribution"].get("neighbour_note"):
            st.caption(pack["attribution"]["neighbour_note"])
        with st.expander("Full evidence pack"):
            st.json(pack)

    with right:
        st.markdown("**Generated report**")
        reports = _reports_for(pack["detection_id"])
        if not reports:
            st.info("No report for this detection — run "
                    "`scripts/generate_reports.py`.")
            return
        # Same evidence, same prompt, different model: the comparison the study
        # makes in aggregate, shown for one detection.
        pick = st.selectbox(
            "Written by", range(len(reports)),
            format_func=lambda i: (
                f"{reports[i]['provider']} / {reports[i]['model']} · prompt "
                f"{reports[i]['prompt_version']}"
                + ("  (template, no model)" if reports[i]["provider"] == "stub" else "")
                + ("  [smoke]" if reports[i]["smoke"] else "")))
        report = reports[pick]
        scores = report.get("scores", {})
        cls, att = report.get("classification") or {}, report.get("attribution") or {}
        a, b, c = st.columns(3)
        a.metric("Groundedness", scores.get("groundedness"))
        b.metric("Names the category",
                 {True: "yes", False: "no"}.get(cls.get("names_predicted"), "n/a"))
        c.metric("Features cited",
                 f"{att['coverage']:.0%}" if att.get("coverage") is not None else "n/a")
        st.caption(f"{scores.get('report_words')} words · "
                   f"{report['latency_seconds']:.2f} s")
        if scores.get("fabricated_addresses"):
            st.error(f"Fabricated addresses: {scores['fabricated_addresses']}")
        ungrounded = [e for k in scores.get("per_class", {}).values()
                      for e in k.get("ungrounded", [])]
        if ungrounded:
            st.warning(f"Not in the evidence: {', '.join(ungrounded)}")
        st.markdown(report["report"])


# Streamlit executes this module top to bottom on every interaction, with
# __name__ == "__main__". A guard here would run main() twice.
main()
