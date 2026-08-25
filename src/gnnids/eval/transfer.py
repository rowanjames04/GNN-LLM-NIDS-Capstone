"""Cross-dataset transfer: score a trained model on a network it never saw (P9).

Promoted from Tier 2 to **Tier 1** (D18) because it is the project's only real
external-validity check. Phases 1 and 3 found that NF-UNSW-NB15-v2 is full of
provenance fingerprints -- three raw features separate the classes at |AUC| ~
0.99 purely by encoding which generator produced the flow -- and removing the
ones we *measured* does not prove none remain. A single-feature audit cannot see
a shortcut spread across a combination of features, and a saturated benchmark
cannot distinguish a model that learned behaviour from one that learned
provenance. Only a different network can.

**The pipeline is frozen.** Scalers and categorical vocabularies are the ones
fitted on the *source* dataset's training split. Refitting on the target would
be a form of test-set access dressed up as preprocessing: the model would be
handed the target's own means and variances, and any transfer it then showed
would partly be that gift rather than generalisation.

The consequence is measured rather than assumed. Target values the source
vocabulary never saw fall to the OTHER bucket by design -- and how often that
happens is a **reported statistic**, because a transfer result computed where
most categorical values landed in OTHER is measuring a degraded input, not a
different network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.schema import FeatureSchema
from ..data.transforms import OTHER, FeaturePipeline
from ..graph.inputs import one_hot


def oov_report(pipeline: FeaturePipeline, df: pd.DataFrame) -> dict:
    """How much of the target fell outside the source's vocabulary.

    Reported beside every transfer number. A model asked to score a network
    whose protocols it has never seen is being tested on mush, and the resulting
    figure -- good or bad -- would say nothing about generalisation.
    """
    out = {}
    for col in pipeline.schema.categorical:
        vocab = pipeline.vocab_[col]
        mapped = df[col].astype(str).map(vocab)
        n_oov = int(mapped.isna().sum() + (mapped == OTHER).sum())
        values = set(df[col].astype(str).unique())
        out[col] = {
            "oov_rate": round(n_oov / max(len(df), 1), 5),
            "distinct_in_target": len(values),
            "distinct_in_source_vocab": len(vocab),
            "shared": len(values & set(vocab)),
        }
    out["_worst_oov_rate"] = round(
        max((v["oov_rate"] for v in out.values() if isinstance(v, dict)), default=0.0), 5)
    return out


def assert_alignment(pipeline: FeaturePipeline, schema: FeatureSchema,
                     source_meta: dict) -> None:
    """The columns must line up, in order, or the transfer is measuring noise.

    `transform` reads continuous columns from the frozen pipeline (safe) but
    iterates **categoricals from the schema**, which is derived from the target
    dataframe. If the two datasets order their categorical columns differently,
    the one-hot blocks land in the wrong feature positions and the model scores
    a scrambled input -- producing exactly what a genuine failure to transfer
    looks like, with no error anywhere.

    Given that a near-zero transfer result is a plausible *real* outcome here
    (D18 exists because these datasets may be full of fingerprints), the two
    cases must be impossible to confuse. So this is asserted on every run.
    """
    derived = (list(schema.categorical)
               + [f"{c}_bucket" for c in schema.ports]
               + [f"{c}_present" for c in schema.conditional])
    if derived != source_meta["categorical_names"]:
        raise AssertionError(
            "categorical column order differs between source and target:\n"
            f"  source: {source_meta['categorical_names']}\n"
            f"  target: {derived}\n"
            "The one-hot blocks would be misaligned and the transfer result "
            "meaningless.")
    if list(pipeline.continuous_names_) != source_meta["continuous_names"]:
        raise AssertionError("continuous column order changed since fitting")


def transfer_inputs(
    source_pipeline_path: Path,
    target_parquet: Path,
    schema: FeatureSchema,
    label_column: str = "Label",
    max_rows: int | None = None,
) -> dict:
    """Target dataset encoded through the SOURCE dataset's frozen pipeline."""
    pipeline = FeaturePipeline.load(Path(source_pipeline_path), schema)

    columns = (list(dict.fromkeys(
        pipeline.continuous_names_ + list(schema.categorical) + list(schema.ports)
        + list(schema.conditional)
        + ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", label_column])))
    df = pd.read_parquet(target_parquet, columns=columns)
    if max_rows is not None and len(df) > max_rows:
        # An even spread, never a prefix: these files are ordered by capture
        # scenario, so a prefix samples one attack family (B4).
        df = df.iloc[np.linspace(0, len(df) - 1, max_rows).astype(int)]

    meta = json.loads(Path(source_pipeline_path).read_text())
    assert_alignment(pipeline, schema, meta)
    oov = oov_report(pipeline, df)
    encoded = pipeline.transform(df)
    feats = np.hstack([
        encoded["continuous"], one_hot(encoded["categorical"], meta["cardinalities"])
    ]).astype(np.float16)

    n = len(df)
    codes = pd.factorize(np.concatenate([
        df.IPV4_SRC_ADDR.to_numpy(), df.IPV4_DST_ADDR.to_numpy()]))[0].astype(np.int32)

    return {
        "src": codes[:n], "dst": codes[n:],
        "edge_features": feats,
        "y": df[label_column].to_numpy().astype(np.int64),
        # The multi-class head cannot transfer: attack vocabularies differ
        # between datasets, so there is no shared label space. Zeros here are a
        # placeholder the binary evaluation never reads, matching the reason
        # leave-one-attack-out is also binary-only.
        "y_multiclass": np.zeros(n, dtype=np.int64),
        "n_rows": n,
        "n_hosts": int(codes.max()) + 1 if n else 0,
        "oov": oov,
        "prevalence": round(float(df[label_column].mean()), 5),
    }
