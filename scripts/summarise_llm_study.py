"""Collate every LLM run into the comparison table (Phase 7b).

Reads `results/reports/*reports_*.json` and produces the cross-model table the
Discussion needs. Written for the same reason as `summarise_transfer.py`: the
headline rests on numbers from many runs, and transcribing those by hand into a
report is how an invalid comparison gets published without anyone lying.

**Groundedness is reported beside the count of reports it averages**, and
fabricated addresses are counted separately from the aggregate -- an invented IP
would send an analyst to the wrong host and must not be averaged away against a
rounded number.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "results" / "reports"


def rows() -> list[dict]:
    out = []
    for f in sorted(REPORTS.glob("*reports_*.json")):
        if f.name.startswith("smoke_"):
            continue
        d = json.loads(f.read_text())
        s, reports = d.get("summary", {}), d.get("reports", [])
        ok = [r for r in reports if r.get("ok")]
        fp = [r for r in ok if r.get("true_label") == "Benign"]

        def mean(rs, key):
            vals = [r["scores"].get(key) for r in rs
                    if isinstance(r.get("scores", {}).get(key), (int, float))]
            return st.mean(vals) if vals else None

        out.append({
            "provider": d.get("provider"), "model": d.get("model"),
            "prompt": d.get("prompt_version"),
            "n": len(reports), "n_ok": len(ok), "n_failed": len(reports) - len(ok),
            "groundedness": s.get("groundedness_mean"),
            "fabricated_addr_reports": s.get("reports_with_fabricated_addresses"),
            "jargon_reports": s.get("reports_with_jargon"),
            "words": s.get("report_words_mean"),
            "latency": s.get("latency_seconds_mean"),
            "cost": s.get("total_cost_usd"),
            # Reports from before C21 carry no list-price field. The stub is free
            # by construction, so its figure is known; nothing else is inferred.
            "reference_cost": s.get("reference_cost_usd",
                                    0.0 if d.get("provider") == "stub" else None),
            "uncertainty_rate": s.get("uncertainty_conveyed_rate"),
            # The false-positive arm, scored separately: on detections the
            # detector got WRONG, does the model still invent nothing?
            "n_false_positive": len(fp),
            "groundedness_on_fp": mean(fp, "groundedness"),
            "determinism": (d.get("summary", {}).get("determinism") or {}),
            "sessions": (d.get("provenance") or {}).get("sessions") or [],
        })
    return out


def main() -> None:
    data = rows()
    if not data:
        raise SystemExit(f"no LLM results in {REPORTS}. Run scripts/run_llm_study.py.")

    fmt = lambda v, s=".4f": format(v, s) if isinstance(v, (int, float)) else "n/a"
    # C21: a missing price prints as unknown, never as n/a or 0 -- "cost$" is
    # what was charged, "list$" what it costs at list price on a billed key.
    money = lambda v: format(v, ".4f") if isinstance(v, (int, float)) else "unknown"
    print("=" * 108)
    print("  PHASE 7b -- LLM COMPARISON (identical evidence packs, model is the only variable)")
    print("=" * 108)
    print(f"  {'model':<30} {'prompt':<7} {'n':>4} {'grounded':>9} {'fab.addr':>9} "
          f"{'jargon':>7} {'words':>7} {'lat(s)':>7} {'cost$':>8} {'list$':>8}")
    for r in sorted(data, key=lambda x: (-(x["groundedness"] or 0), x["model"] or "")):
        name = f"{r['provider']}/{r['model']}"
        print(f"  {name:<30} {r['prompt'] or '-':<7} {r['n_ok']:>4} "
              f"{fmt(r['groundedness']):>9} {str(r['fabricated_addr_reports']):>9} "
              f"{str(r['jargon_reports']):>7} {fmt(r['words'], '.0f'):>7} "
              f"{fmt(r['latency'], '.2f'):>7} {money(r['cost']):>8} "
              f"{money(r['reference_cost']):>8}")

    print(f"\n  {'model':<30} {'FP packs':>9} {'grounded on FPs':>17} "
          f"{'uncertainty conveyed':>22} {'failed':>7}")
    for r in sorted(data, key=lambda x: x["model"] or ""):
        name = f"{r['provider']}/{r['model']}"
        print(f"  {name:<30} {r['n_false_positive']:>9} "
              f"{fmt(r['groundedness_on_fp']):>17} {fmt(r['uncertainty_rate']):>22} "
              f"{r['n_failed']:>7}")

    print(f"\n  {'model':<30} {'determinism applied':<24} note")
    for r in sorted(data, key=lambda x: x["model"] or ""):
        d = r["determinism"]
        applied = ("temperature+seed" if d.get("temperature") is not None and d.get("seed") is not None
                   else "temperature" if d.get("temperature") is not None
                   else "seed" if d.get("seed") is not None else "none")
        print(f"  {r['provider']}/{r['model']:<24} {applied:<24} {d.get('note', '')}")
    print("\n  The roster is NOT uniform on determinism: sampling controls were "
          "removed on the\n  current Claude generation. Report this asymmetry "
          "rather than claiming temperature 0.")

    _print_provenance(data)

    stub = [r for r in data if r["provider"] == "stub"]
    if stub and len(data) > 1:
        floor = stub[0]["groundedness"]
        print(f"\n  Control (template stub) groundedness: {fmt(floor)} — by construction "
              f"it invents nothing.\n  A model at or below this floor is adding fluency, "
              f"not faithfulness.")
    if any(r["n_false_positive"] == 0 for r in data):
        print("\n  WARNING: at least one run contains no false-positive packs. The "
              "study's\n  sharpest question cannot be answered from it — check the "
              "sampling plan.")
    print("\n  Groundedness measures INVENTED NOTHING, not REASONED CORRECTLY.")
    print("  A fluent, wrong interpretation built from real facts scores 1.0.\n")


def _print_provenance(data: list[dict]) -> None:
    """What each run executed on (D35). A self-hosted result is quoted by its
    digest and quantisation, and its latency only beside the GPU it ran on."""
    print(f"\n  {'model':<30} {'sessions':>8}  {'digest':<19} {'quant':<8} "
          f"{'params':<7} {'gpu':<22} commit")
    for r in sorted(data, key=lambda x: x["model"] or ""):
        ss = r["sessions"]
        if not ss:
            print(f"  {r['provider']}/{r['model']:<24} {'-':>8}  (no provenance -- "
                  f"generated before D35)")
            continue
        m = ss[-1].get("model") or {}
        gpus = {g["name"] for s in ss for g in (s.get("gpus") or [])} or {"none"}
        commits = {(s.get("git") or {}).get("commit") for s in ss} - {None}
        dirty = any((s.get("git") or {}).get("dirty") for s in ss)
        print(f"  {r['provider']}/{r['model']:<24} {len(ss):>8}  "
              f"{(m.get('digest') or '-')[:19]:<19} "
              f"{m.get('quantization_level') or '-':<8} "
              f"{m.get('parameter_size') or '-':<7} {', '.join(sorted(gpus)):<22} "
              f"{','.join(c[:8] for c in sorted(commits)) or '-'}"
              + ("  DIRTY TREE" if dirty else ""))
    print("\n  Latency is hardware-bound: the self-hosted arm ran on an assigned "
          "Colab GPU and\n  the cloud arms include network time. Compare latency "
          "within an arm, never across.")


if __name__ == "__main__":
    main()
