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

import numpy as np
import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


@st.cache_resource
def _engine(window: int, n_windows: int, checkpoint: str):
    """Built once per session. Loading the model on every rerun would make the
    slider unusable, and Streamlit reruns the whole script on every interaction."""
    from replay import build_engine

    cfg = yaml.safe_load((REPO_ROOT / "configs" / "replay.yaml").read_text())
    gnn_cfg = yaml.safe_load((REPO_ROOT / cfg["gnn_config"]).read_text())
    pre_cfg = yaml.safe_load((REPO_ROOT / gnn_cfg["preprocess_config"]).read_text())
    ds_cfg = yaml.safe_load((REPO_ROOT / pre_cfg["dataset_config"]).read_text())
    engine, ckpt, inputs = build_engine(
        cfg, gnn_cfg, pre_cfg, ds_cfg, REPO_ROOT / checkpoint, window, n_windows)
    return engine, ckpt, inputs, ds_cfg


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
        go = st.button("Run replay", type="primary")

    if not go:
        st.info("Choose a window size and press **Run replay**.")
        return

    try:
        engine, ckpt, inputs, ds_cfg = _engine(window, n_windows, checkpoint)
    except FileNotFoundError:
        st.error(f"No checkpoint at {checkpoint}. Train one first.")
        return

    st.write(f"**{ds_cfg['name']}** · threshold "
             f"`{engine.threshold:.4f}` (chosen on validation, from the checkpoint)")

    progress = st.progress(0.0)
    detections = []
    for i, found in engine.run(n_windows):
        detections.extend(found)
        progress.progress((i + 1) / n_windows)
    progress.empty()

    stats = engine.stats.as_dict()
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

    packs_path = REPO_ROOT / "results" / "evidence"
    files = sorted(packs_path.glob("*evidence*.json")) if packs_path.exists() else []
    if not files:
        st.info("No evidence packs yet — run `scripts/make_evidence.py`.")
        return

    packs = json.loads(files[-1].read_text())
    if not packs:
        st.info("The evidence file is empty.")
        return

    choice = st.selectbox(
        "Detection", range(len(packs)),
        format_func=lambda i: (f"{packs[i]['flow'].get('src_ip')} → "
                               f"{packs[i]['flow'].get('dst_ip')}  "
                               f"({packs[i]['detection']['predicted_class']}, "
                               f"score {packs[i]['detection']['score']:.3f})"))
    pack = packs[choice]

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
        reports_dir = REPO_ROOT / "results" / "reports"
        rfiles = sorted(reports_dir.glob("*reports*.json")) if reports_dir.exists() else []
        report = None
        for rf in reversed(rfiles):
            for row in json.loads(rf.read_text()).get("reports", []):
                if row["detection_id"] == pack["detection_id"] and row["ok"]:
                    report = row
                    break
            if report:
                break
        if report is None:
            st.info("No report for this detection — run "
                    "`scripts/generate_reports.py`.")
        else:
            g = report["scores"].get("groundedness")
            st.write(f"Groundedness **{g}** · "
                     f"{report['scores'].get('report_words')} words · "
                     f"{report['latency_seconds']:.2f}s")
            if report["scores"].get("fabricated_addresses"):
                st.error(f"Fabricated addresses: "
                         f"{report['scores']['fabricated_addresses']}")
            st.markdown(report["report"])


# Streamlit executes this module top to bottom on every interaction, with
# __name__ == "__main__". A guard here would run main() twice.
main()
