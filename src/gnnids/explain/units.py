"""Turn standardised model inputs back into the quantities they measure.

The model sees z-scores of log-transformed, clipped features. The evidence pack
must not: an LLM handed `IN_BYTES: 1.83` has no idea whether that is bytes,
kilobytes or standard deviations, and the one thing it will reliably do is
choose a unit and state it confidently. Every number that reaches the pack is
therefore inverted back to its real unit first.

The forward pipeline is  clip -> log1p(x + shift) -> (x - mean) / std
(see `gnnids.data.transforms.FeaturePipeline`), so the inverse runs backwards
through it. **Clipping is not invertible** -- a value winsorised at the 99.999th
percentile (D22) comes back as the clip bound, not its original magnitude -- so
inverted values at the bound are flagged rather than presented as measurements.
"""

from __future__ import annotations

import numpy as np


class UnitRestorer:
    """Inverts one fitted `FeaturePipeline`'s continuous transforms."""

    def __init__(self, pipeline_meta: dict) -> None:
        self.names: list[str] = list(pipeline_meta["continuous_names"])
        self.mean = pipeline_meta["mean"]
        self.std = pipeline_meta["std"]
        self.log_shift = pipeline_meta.get("log_shift", {})
        self.clip_hi = pipeline_meta.get("clip_hi", {})

    def restore(self, name: str, z: float) -> dict:
        """One standardised value back to its measured quantity.

        Returns the value plus whether it sits at the winsorisation bound, so a
        consumer can say "at least 4.1 GB" rather than stating a clipped number
        as exact.
        """
        if name not in self.mean:
            return {"value": _round(z), "unit": "standardised", "exact": True}

        x = z * self.std[name] + self.mean[name]
        if name in self.log_shift:
            x = np.expm1(x) - self.log_shift[name]

        hi = self.clip_hi.get(name)
        at_bound = hi is not None and x >= hi - 1e-6
        return {
            "value": _round(x),
            "unit": "measured",
            # Clipping (D22) is lossy in one direction only: a flagged value is
            # a floor, not an estimate.
            "exact": not at_bound,
            **({"note": "winsorised at the 99.999th percentile; this is a lower bound"}
               if at_bound else {}),
        }

    def restore_row(self, z_row: np.ndarray) -> dict[str, dict]:
        n = min(len(self.names), len(z_row))
        return {self.names[i]: self.restore(self.names[i], float(z_row[i]))
                for i in range(n)}


def _round(x: float) -> float | int:
    x = float(x)
    if not np.isfinite(x):
        return 0.0
    # Counts and byte totals read as integers; rates and durations do not.
    # The near-integer case matters: features are stored as float16 and passed
    # through log1p and back, so a packet count of 1 returns as 1.0003. Handing
    # the LLM "1.0003 packets" invites it to reason about the fraction.
    # Relative as well as absolute, because the noise scales with the value:
    # a true 48 comes back as 48.0135 and a true 1 as 1.0003, both ~0.03% off.
    if abs(x) < 1e9 and abs(x - round(x)) <= max(0.01, abs(x) * 0.001):
        return int(round(x))
    return int(round(x)) if abs(x) >= 100 else round(x, 4)


def feature_names(pipeline_meta: dict) -> list[str]:
    """Names for every column the model sees, continuous then one-hot.

    Without this the one-hot block is anonymous, and any categorical that lands
    in a detection's top features gets dropped from the pack rather than
    explained -- which is how a scan detected on `L7_PROTO=DNS` reaches the LLM
    with its actual reason missing.

    One-hot names are `COLUMN=value`, using the fitted vocabulary so the value
    is the one that appeared in the capture (protocol 6, not slot 1). Index 0 is
    the unseen/out-of-vocabulary slot in every block.
    """
    names = list(pipeline_meta["continuous_names"])
    vocab = pipeline_meta.get("vocab", {})
    for col, card in zip(pipeline_meta["categorical_names"],
                         pipeline_meta["cardinalities"]):
        inverse = {int(slot): value for value, slot in vocab.get(col, {}).items()}
        for slot in range(card):
            label = inverse.get(slot, "other" if slot == 0 else str(slot))
            names.append(f"{col}={label}")
    return names
