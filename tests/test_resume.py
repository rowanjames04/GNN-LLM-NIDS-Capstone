"""Tests for resumable report generation (D35).

The self-hosted arm runs in a Colab session that can end at any pack. Two ways
to get this wrong, both silent: losing completed reports on a disconnect, and
**merging a resumed log into the wrong run** -- a file mixing two prompt versions
would pass every downstream check and measure nothing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnnids.llm.resume import ReportLog, ResumeMismatch  # noqa: E402

ID = {"provider": "ollama", "model": "m", "prompt_version": "v1",
      "packs_source": "packs.json"}


def _row(i: int, ok: bool = True) -> dict:
    return {"detection_id": f"id-{i}", "ok": ok, "report": "r" if ok else ""}


def test_rows_written_before_a_crash_are_recovered(tmp_path):
    log = ReportLog(tmp_path / "x.partial.jsonl", ID)
    log.start({})
    log.append(_row(0))
    log.append(_row(1))

    rows, retried = ReportLog(tmp_path / "x.partial.jsonl", ID).load(
        ["id-0", "id-1", "id-2"])

    assert set(rows) == {"id-0", "id-1"}
    assert retried == 0


def test_a_log_from_a_different_prompt_version_is_refused(tmp_path):
    ReportLog(tmp_path / "x.partial.jsonl", ID).start({"id-0": _row(0)})

    other = dict(ID, prompt_version="v2")
    with pytest.raises(ResumeMismatch):
        ReportLog(tmp_path / "x.partial.jsonl", other).load(["id-0"])


def test_a_log_holding_packs_that_are_no_longer_in_the_set_is_refused(tmp_path):
    """The packs were regenerated between runs -- the D33 hazard."""
    ReportLog(tmp_path / "x.partial.jsonl", ID).start({"id-9": _row(9)})

    with pytest.raises(ResumeMismatch):
        ReportLog(tmp_path / "x.partial.jsonl", ID).load(["id-0", "id-1"])


def test_a_torn_last_line_costs_one_report_not_the_run(tmp_path):
    path = tmp_path / "x.partial.jsonl"
    log = ReportLog(path, ID)
    log.start({"id-0": _row(0)})
    with path.open("a") as f:
        f.write('{"detection_id": "id-1", "ok": tr')      # killed mid-write

    rows, _ = ReportLog(path, ID).load(["id-0", "id-1"])

    assert set(rows) == {"id-0"}


def test_failed_rows_are_kept_unless_retry_is_asked_for_and_then_counted(tmp_path):
    """Silently regenerating failures until they succeed would make a model
    look more reliable than it is."""
    path = tmp_path / "x.partial.jsonl"
    ReportLog(path, ID).start({"id-0": _row(0), "id-1": _row(1, ok=False)})

    kept, n0 = ReportLog(path, ID).load(["id-0", "id-1"])
    retry, n1 = ReportLog(path, ID).load(["id-0", "id-1"], retry_failed=True)

    assert set(kept) == {"id-0", "id-1"} and n0 == 0
    assert set(retry) == {"id-0"} and n1 == 1


def _pack(i: int) -> dict:
    return {
        "schema_version": "1.0", "detection_id": f"id-{i}",
        "flow": {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "IN_BYTES": 48,
                 "true_label": "scanning"},
        "detection": {"score": 0.96, "threshold": 0.85, "flagged": True,
                      "predicted_class": "scanning", "class_confidence": 0.99,
                      "class_distribution": {"scanning": 0.99, "Benign": 0.01}},
        "attribution": {"channel_weights": {"attribute": 0.6, "topological": 0.4},
                        "top_features": [], "influential_neighbours": []},
        "context": {"window_flow_count": 10},
        "causal_path": None,
        "guidance": {"attack_family": "scanning", "mitre_techniques": ["T1046"]},
        "provenance": {"checkpoint": "seed0"},
    }


def test_an_interrupted_run_resumes_and_finishes_with_every_pack(tmp_path):
    """End to end on the free stub: a partial log with one report already in it
    means the re-run generates only the rest, and the final file has all of
    them in pack order with the log removed."""
    packs = tmp_path / "packs.json"
    packs.write_text(json.dumps([_pack(i) for i in range(3)]))
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "provider: stub\nprompt_version: v1\nn_reports: 3\n"
        f"packs: {packs}\nprovider_options: {{stub: {{}}}}\n"
        f"smoke: {{packs: {packs}, n_reports: 2}}\n"
        f"output: {{dir: {tmp_path}}}\n")
    partial = tmp_path / "reports_stub_template-v1_v1.partial.jsonl"
    ReportLog(partial, {"provider": "stub", "model": "template-v1",
                        "prompt_version": "v1", "packs_source": "packs.json"}
              ).start({"id-1": dict(_row(1), marker="from-before-the-crash",
                                    scores={"groundedness": 1.0,
                                            "fabricated_addresses": [],
                                            "report_words": 1},
                                    jargon={"clean": True}, uncertainty=None,
                                    latency_seconds=0.0, cost_usd=None,
                                    determinism=None)})

    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "generate_reports.py"),
         "--config", str(cfg), "--packs", str(packs)],
        capture_output=True, text=True, timeout=180)

    assert r.returncode == 0, r.stderr
    assert "resuming: 1 of 3" in r.stdout
    out = json.loads((tmp_path / "reports_stub_template-v1_v1.json").read_text())
    assert [x["detection_id"] for x in out["reports"]] == ["id-0", "id-1", "id-2"]
    assert out["reports"][1].get("marker") == "from-before-the-crash"
    assert not partial.exists()
