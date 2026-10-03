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
