"""Tests for campaign bookkeeping: which runs may be merged, and where they go.

The failures this guards against all produced files that parsed and looked
right: a variant run overwriting the headline results (C20), and runs from two
different configurations merged into one table.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnnids.training.campaign import (  # noqa: E402
    checkpoint_name, fingerprint, read_if_matching, stem,
)

CFG = {
    "preprocess_config": "configs/preprocess_toniot.yaml",
    "model": {"hidden_dim": 128, "n_gnn_layers": 2},
    "train": {"lr": 0.001, "max_epochs": 40, "early_stopping": False,
              "n_seeds": 3, "device": "auto"},
    "eval": {"target_prevalence": 0.04},
    "ablations": {"full": {"use_channel1": True, "use_channel2": True},
                  "gnn_layer_sweep": [1, 2, 3]},
}


def _with(path: tuple, value):
    cfg = copy.deepcopy(CFG)
    node = cfg
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return cfg


def test_asking_for_more_seeds_does_not_change_what_a_run_is():
    """--extend from 3 seeds to 10 must be able to reuse seeds 0-2."""
    assert fingerprint(_with(("train", "n_seeds"), 10)) == fingerprint(CFG)
    assert fingerprint(_with(("train", "device"), "cpu")) == fingerprint(CFG)


def test_anything_that_changes_the_result_changes_the_fingerprint():
    base = fingerprint(CFG)

    assert fingerprint(_with(("model", "n_gnn_layers"), 3)) != base
    assert fingerprint(_with(("train", "early_stopping"), True)) != base
    assert fingerprint(_with(("train", "max_epochs"), 60)) != base
    assert fingerprint(_with(("eval", "target_prevalence"), 0.1)) != base
    assert fingerprint(CFG, extra={"held_out": "scanning"}) != base


def test_a_tagged_variant_never_shares_a_name_with_the_headline_run():
    assert stem("NF-ToN-IoT-v2", None) == "NF-ToN-IoT-v2"
    assert stem("NF-ToN-IoT-v2", "layers3") == "NF-ToN-IoT-v2__layers3"
    assert (checkpoint_name("NF-ToN-IoT-v2", "full", 0, None)
            == "NF-ToN-IoT-v2_full_seed0.pt")       # unchanged: evidence.yaml reads it
    assert (checkpoint_name("NF-ToN-IoT-v2", "full", 0, "layers3")
            != checkpoint_name("NF-ToN-IoT-v2", "full", 0, None))


def test_a_partial_from_another_configuration_is_not_reused(tmp_path):
    part = tmp_path / "partial.json"
    part.write_text(json.dumps({"fingerprint": fingerprint(CFG), "results": {}}))

    assert read_if_matching(part, fingerprint(CFG)) is not None
    assert read_if_matching(part, fingerprint(_with(("model", "n_gnn_layers"), 3))) is None


def test_a_results_file_from_before_fingerprints_is_not_reused(tmp_path):
    """Every result written before 2026-10-03 lacks the field. Reusing one
    would mix an early-stopped run into a fixed-budget campaign."""
    old = tmp_path / "gnn.json"
    old.write_text(json.dumps({"results": {"full": {"runs": [{}]}}}))

    assert read_if_matching(old, fingerprint(CFG)) is None


def test_a_torn_or_missing_file_is_absent_not_an_error(tmp_path):
    torn = tmp_path / "torn.json"
    torn.write_text('{"fingerprint": "abc", "resul')

    assert read_if_matching(torn, "abc") is None
    assert read_if_matching(tmp_path / "nope.json", "abc") is None


# --- C32 / C33: the data is part of what a run is -----------------------------

def _proc(tmp_path, mean):
    d = tmp_path / "proc"
    d.mkdir(exist_ok=True)
    (d / "pipeline.json").write_text(json.dumps({"mean": {"IN_BYTES": mean},
                                                 "cardinalities": [2], "continuous_names": ["IN_BYTES"]}))
    (d / "splits.json").write_text(json.dumps({"train": {"start": 0, "stop": 4}}))
    return d


def test_a_change_to_the_preprocessed_data_changes_the_fingerprint(tmp_path):
    from gnnids.data.version import data_version

    v1 = data_version(_proc(tmp_path, 1.0))
    v2 = data_version(_proc(tmp_path, 2.0))
    assert v1 != v2
    assert fingerprint(CFG, data_version=v1) != fingerprint(CFG, data_version=v2)
    assert fingerprint(CFG, data_version=v1) == fingerprint(CFG, data_version=v1)


def test_a_checkpoint_from_other_data_is_refused(tmp_path, monkeypatch):
    import pytest

    from gnnids.data import version

    monkeypatch.setattr(version, "version_for_preprocess_config", lambda p: "new")
    version.check_checkpoint({"data_version": "new"}, "x.yaml", "test")   # accepted
    for stale in ({"data_version": "old"}, {}):
        with pytest.raises(SystemExit, match="Retrain"):
            version.check_checkpoint(stale, "x.yaml", "test")


def test_the_edge_feature_cache_is_rebuilt_when_the_data_changes(tmp_path):
    import numpy as np
    import pandas as pd

    from gnnids.graph.inputs import load_graph_inputs

    d = _proc(tmp_path, 1.0)
    (d / "attack_families.json").write_text(json.dumps({"Benign": 0, "dos": 1}))
    pd.DataFrame({"IPV4_SRC_ADDR": ["a", "b", "a", "c"], "IPV4_DST_ADDR": ["b", "c", "c", "a"],
                  "Label": [0, 1, 0, 1], "Attack": ["Benign", "dos", "Benign", "dos"]}
                 ).to_parquet(d / "meta.parquet")
    np.savez(d / "features.npz", continuous=np.ones((4, 1), np.float32),
             categorical=np.zeros((4, 1), np.int64))
    assert float(load_graph_inputs(d, "t", verbose=False).edge_features[0, 0]) == 1.0

    # New preprocessing: different statistics, different features.
    _proc(tmp_path, 2.0)
    np.savez(d / "features.npz", continuous=np.full((4, 1), 7, np.float32),
             categorical=np.zeros((4, 1), np.int64))
    assert float(load_graph_inputs(d, "t", verbose=False).edge_features[0, 0]) == 7.0
