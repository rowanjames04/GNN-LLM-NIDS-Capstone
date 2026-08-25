"""Tests for Phase 9 cross-dataset transfer.

The hazard here is unusual and worth stating: **a genuine failure to transfer
and a feature-misalignment bug look identical from the outside.** Both produce a
PR-AUC at the prevalence floor with no error raised. D18 exists precisely
because near-zero transfer is a plausible real outcome on these datasets, so the
two cases have to be impossible to confuse -- hence the alignment assertion, and
hence these tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnnids.data.schema import FeatureSchema  # noqa: E402
from gnnids.data.transforms import FeaturePipeline  # noqa: E402
from gnnids.eval.transfer import assert_alignment, oov_report, transfer_inputs  # noqa: E402


def _schema():
    return FeatureSchema(continuous=["IN_BYTES", "OUT_BYTES"],
                         categorical=["PROTOCOL", "L7_PROTO"],
                         ports=[], conditional=[])


def _fitted(schema):
    df = pd.DataFrame({
        "IN_BYTES": [10.0, 20.0, 30.0, 40.0], "OUT_BYTES": [1.0, 2.0, 3.0, 4.0],
        "PROTOCOL": ["6", "6", "17", "17"], "L7_PROTO": ["0", "7", "0", "7"],
    })
    return FeaturePipeline(schema).fit(df), df


def test_alignment_passes_when_the_order_matches():
    schema = _schema()
    p, _ = _fitted(schema)
    meta = {"categorical_names": ["PROTOCOL", "L7_PROTO"],
            "continuous_names": ["IN_BYTES", "OUT_BYTES"]}

    assert_alignment(p, schema, meta)


def test_reordered_categoricals_are_refused_rather_than_scored():
    """Swapped order puts one-hot blocks in the wrong feature positions. The
    model would score a scrambled input and return a plausible near-zero."""
    schema = _schema()
    p, _ = _fitted(schema)
    meta = {"categorical_names": ["L7_PROTO", "PROTOCOL"],   # swapped
            "continuous_names": ["IN_BYTES", "OUT_BYTES"]}

    with pytest.raises(AssertionError, match="misaligned"):
        assert_alignment(p, schema, meta)


def test_a_changed_continuous_order_is_refused():
    schema = _schema()
    p, _ = _fitted(schema)
    meta = {"categorical_names": ["PROTOCOL", "L7_PROTO"],
            "continuous_names": ["OUT_BYTES", "IN_BYTES"]}

    with pytest.raises(AssertionError, match="continuous column order"):
        assert_alignment(p, schema, meta)


def test_out_of_vocabulary_values_are_measured_not_ignored():
    """A transfer number computed where most values fell to OTHER measures a
    degraded input, not a different network."""
    schema = _schema()
    p, _ = _fitted(schema)
    target = pd.DataFrame({
        "IN_BYTES": [1.0, 2.0], "OUT_BYTES": [1.0, 2.0],
        "PROTOCOL": ["6", "6"],            # both seen in the source
        "L7_PROTO": ["999", "888"],        # neither seen
    })

    got = oov_report(p, target)

    assert got["PROTOCOL"]["oov_rate"] == 0.0
    assert got["L7_PROTO"]["oov_rate"] == 1.0
    assert got["L7_PROTO"]["shared"] == 0
    assert got["_worst_oov_rate"] == 1.0


def test_unseen_categoricals_fall_back_rather_than_raising():
    """Required for transfer to be possible at all -- a target network will
    always carry protocol values the source never saw."""
    schema = _schema()
    p, _ = _fitted(schema)
    target = pd.DataFrame({
        "IN_BYTES": [1.0], "OUT_BYTES": [1.0],
        "PROTOCOL": ["132"], "L7_PROTO": ["999"]})

    encoded = p.transform(target)

    assert encoded["categorical"].shape == (1, 2)


def test_transfer_inputs_produces_a_binary_only_label_space(tmp_path):
    """Attack vocabularies differ between datasets, so the 9-way head has no
    shared label space and its placeholder must never be read as a family."""
    schema = _schema()
    p, df = _fitted(schema)
    df = df.assign(IPV4_SRC_ADDR=["10.0.0.1"] * 4, IPV4_DST_ADDR=["10.0.0.2"] * 4,
                   Label=[0, 1, 0, 1])
    parquet = tmp_path / "target.parquet"
    df.to_parquet(parquet)
    pipeline_path = tmp_path / "pipeline.json"
    p.save(pipeline_path)

    got = transfer_inputs(pipeline_path, parquet, schema)

    assert set(np.unique(got["y"])) == {0, 1}
    assert set(np.unique(got["y_multiclass"])) == {0}
    assert got["prevalence"] == 0.5
    assert got["n_rows"] == 4


def test_transfer_inputs_subsamples_evenly_not_as_a_prefix(tmp_path):
    """These files are ordered by capture scenario, so a prefix samples one
    attack family (B4)."""
    schema = _schema()
    p, _ = _fitted(schema)
    n = 100
    df = pd.DataFrame({
        "IN_BYTES": np.arange(n, dtype=float), "OUT_BYTES": np.arange(n, dtype=float),
        "PROTOCOL": ["6"] * n, "L7_PROTO": ["0"] * n,
        "IPV4_SRC_ADDR": ["10.0.0.1"] * n, "IPV4_DST_ADDR": ["10.0.0.2"] * n,
        # All positives sit in the last quarter, as a scenario-ordered file.
        "Label": [0] * 75 + [1] * 25,
    })
    parquet = tmp_path / "t.parquet"
    df.to_parquet(parquet)
    pipeline_path = tmp_path / "p.json"
    p.save(pipeline_path)

    got = transfer_inputs(pipeline_path, parquet, schema, max_rows=20)

    assert got["n_rows"] == 20
    # A prefix would have caught zero positives.
    assert got["y"].sum() > 0
