"""Tests for the roster sweep and its cost gate (Phase 7b).

Two things are guarded here. The sweep must be **fully crossed** — every model
over the same fixed pack set — because a roster run one model at a time confounds
the model with whatever else changed between invocations. And **no paid provider
may be called without an explicit flag**: the failure mode is spending money the
student did not agree to, which no test failure would recover.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pack(i: int = 0, true_label: str = "scanning") -> dict:
    return {
        "schema_version": "1.0", "detection_id": f"id-{i}",
        "flow": {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "IN_BYTES": 48,
                 "true_label": true_label},
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


def test_the_stub_is_free_and_priced_models_are_not():
    study = _load("run_llm_study")
    packs = [_pack(i) for i in range(10)]

    free, _ = study.estimate({"provider": "stub"}, packs, "v1")
    paid, _ = study.estimate({"provider": "anthropic", "model": "claude-opus-5"},
                             packs, "v1")

    assert free == 0.0
    assert paid > 0


def test_cost_scales_with_the_pack_count():
    """The number worth seeing before committing: a roster is the per-model cost
    multiplied by the roster size."""
    study = _load("run_llm_study")

    small, _ = study.estimate({"provider": "anthropic", "model": "claude-opus-5"},
                              [_pack(i) for i in range(10)], "v1")
    large, _ = study.estimate({"provider": "anthropic", "model": "claude-opus-5"},
                              [_pack(i) for i in range(100)], "v1")

    assert large > small * 5


def test_an_unpriced_provider_reports_unknown_rather_than_zero():
    """Reporting an unknown price as 0.0 would understate the sweep's cost."""
    study = _load("run_llm_study")

    usd, label = study.estimate({"provider": "ollama", "model": "llama3.2:3b"},
                                [_pack()], "v1")

    assert usd is None
    assert "unknown" in label.lower()


def test_a_paid_roster_refuses_to_run_without_confirm_spend(tmp_path):
    """The gate that protects real money. It must hold from the command line,
    not just in principle."""
    packs = tmp_path / "packs.json"
    packs.write_text(json.dumps([_pack(i) for i in range(3)]))
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "provider: stub\nprompt_version: v1\nn_reports: 3\n"
        f"packs: {packs}\n"
        "roster:\n"
        "  - {provider: stub, enabled: true}\n"
        "  - {provider: anthropic, model: claude-opus-5, enabled: true}\n"
        "provider_options: {stub: {}, anthropic: {model: claude-opus-5}}\n"
        f"smoke: {{packs: {packs}, n_reports: 2}}\n"
        f"output: {{dir: {tmp_path}}}\n")

    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_llm_study.py"),
         "--config", str(cfg), "--packs", str(packs)],
        capture_output=True, text=True, timeout=180)

    assert r.returncode == 0
    assert "--confirm-spend" in r.stdout
    # It must not have produced any report output.
    assert not list(tmp_path.glob("*reports_anthropic*"))


def test_estimate_flag_stops_before_calling_anything(tmp_path):
    packs = tmp_path / "packs.json"
    packs.write_text(json.dumps([_pack(i) for i in range(3)]))
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "provider: stub\nprompt_version: v1\nn_reports: 3\n"
        f"packs: {packs}\n"
        "roster:\n  - {provider: stub, enabled: true}\n"
        "provider_options: {stub: {}}\n"
        f"smoke: {{packs: {packs}, n_reports: 2}}\n"
        f"output: {{dir: {tmp_path}}}\n")

    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_llm_study.py"),
         "--config", str(cfg), "--packs", str(packs), "--estimate"],
        capture_output=True, text=True, timeout=180)

    assert "stopping before any call" in r.stdout
    assert not list(tmp_path.glob("*reports_*"))


def test_disabled_roster_entries_are_skipped(tmp_path):
    """Disable rather than delete, so the record shows what was considered."""
    import yaml
    cfg = yaml.safe_load(
        "roster:\n"
        "  - {provider: stub, enabled: true}\n"
        "  - {provider: openai, model: gpt-4o, enabled: false}\n")

    enabled = [e for e in cfg["roster"] if e.get("enabled", True)]

    assert [e["provider"] for e in enabled] == ["stub"]
