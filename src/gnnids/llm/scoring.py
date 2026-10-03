"""Every automated score for one report, and their aggregation, in one place.

`generate_reports.py` scores a report the moment it is generated, and
`rescore_reports.py` re-applies the same functions to reports already on disk.
Keeping both behind one function is what makes re-scoring trustworthy: a paid
run never has to be repeated because a metric was added or corrected afterwards,
and a re-scored file cannot drift from a freshly generated one.

All scoring is arithmetic over the evidence pack and the report text. No model
is called, so re-scoring is free and deterministic.
"""

from __future__ import annotations

import numpy as np

from .fidelity import check_attribution_fidelity, check_classification_fidelity
from .groundedness import check_no_jargon, check_uncertainty_conveyed, score_report

SCORING_VERSION = "2026-10-03"      # bumped when a score's meaning changes


def score_all(pack: dict, text: str, ok: bool = True) -> dict:
    """The score fields of one report row. Empty for a failed generation, so a
    failure is never averaged in as a zero."""
    if not ok:
        return {"scores": {}, "jargon": {}, "uncertainty": None,
                "classification": None, "attribution": None}
    return {
        "scores": score_report(pack, text),
        "jargon": check_no_jargon(text),
        "uncertainty": check_uncertainty_conveyed(pack, text),
        "classification": check_classification_fidelity(pack, text),
        "attribution": check_attribution_fidelity(pack, text),
    }


def _percentile(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 3) if values else None


def summarise(rows: list[dict]) -> dict:
    """Aggregate the metric families that survive aggregation.

    Groundedness is averaged over successful reports only, and the failure count
    is reported beside it -- a model whose errored runs were dropped would look
    better than one that returned a flawed report, which is backwards.
    """
    ok = [r for r in rows if r["ok"]]

    def total(key):
        # C21: an unknown cost is unknown, not zero. Summing None as 0.0 printed
        # free-tier Gemini and Colab Ollama as $0.0000 -- a price they do not
        # have, in the one column where the study compares prices.
        vals = [r.get(key) for r in ok]
        if any(v is None for v in vals):
            return None
        return round(sum(vals), 6)

    def mean(key, source="scores"):
        vals = [r[source][key] for r in ok
                if isinstance((r.get(source) or {}).get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None

    def rate(source, key):
        # Over the reports the check applies to; None where it applied to none.
        vals = [r[source][key] for r in ok if r.get(source)]
        return round(sum(bool(v) for v in vals) / len(vals), 4) if vals else None

    unc = [r["uncertainty"] for r in ok if r.get("uncertainty")]
    lat = [r["latency_seconds"] for r in ok]
    gen = [r["timing"]["generation_seconds"] for r in ok
           if (r.get("timing") or {}).get("generation_seconds") is not None]
    out_tokens = [r["output_tokens"] for r in ok
                  if (r.get("timing") or {}).get("generation_seconds") is not None
                  and r.get("output_tokens")]
    return {
        "n": len(rows), "n_ok": len(ok), "n_failed": len(rows) - len(ok),
        "groundedness_mean": mean("groundedness"),
        "reports_with_fabricated_addresses": sum(
            1 for r in ok if r["scores"]["fabricated_addresses"]),
        "reports_with_jargon": sum(1 for r in ok if not r["jargon"]["clean"]),
        "report_words_mean": mean("report_words"),
        # Classification fidelity: the report names the predicted category and
        # no category the evidence did not offer.
        "classification_faithful_rate": rate("classification", "faithful"),
        "classification_substituted_rate": rate("classification", "substituted"),
        # Attribution fidelity: share of the attributed features a report cites,
        # and how often it cites none at all.
        "attribution_coverage_mean": mean("coverage", "attribution"),
        "attribution_cites_none_rate": rate("attribution", "cites_none"),
        "latency_seconds_mean": round(sum(lat) / len(lat), 3) if lat else None,
        # The mean hides the tail an analyst waits on. p95 is the design's figure.
        "latency_seconds_p50": _percentile(lat, 50),
        "latency_seconds_p95": _percentile(lat, 95),
        # Self-hosted only: time the model spent generating, as reported by the
        # server. On a GPU runtime this is the GPU-seconds the study asks for.
        "generation_seconds_total": round(sum(gen), 2) if gen else None,
        "output_tokens_per_second": (
            round(sum(out_tokens) / sum(gen), 2) if gen and sum(gen) > 0 else None),
        "total_cost_usd": total("cost_usd"),
        "reference_cost_usd": total("reference_cost_usd"),
        "ambiguous_detections": len(unc),
        "uncertainty_conveyed_rate": (
            round(sum(u["uncertainty_conveyed"] for u in unc) / len(unc), 4)
            if unc else None),
        "scoring_version": SCORING_VERSION,
    }
