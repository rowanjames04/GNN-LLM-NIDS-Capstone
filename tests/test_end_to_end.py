"""The class-balance counts and the end-to-end tally of detections and notes."""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


balance = _load("class_balance")
e2e = _load("end_to_end")


def test_rows_purged_between_splits_are_not_counted():
    attack = pd.Series(["Benign", "dos", "Benign", "dos", "dos", "Benign"])
    splits = {"train": {"start": 0, "stop": 2}, "val": {"start": 3, "stop": 4},
              "test": {"start": 4, "stop": 6}}

    counts = balance.count_classes(attack, splits)

    assert counts == {"Benign": {"train": 1, "val": 0, "test": 1},
                      "dos": {"train": 1, "val": 1, "test": 1}}
    assert balance.summarise(counts)["n_flows"] == 5


def _report(**over):
    r = {"ok": True, "degenerate": False,
         "scores": {"n_ungrounded": 0, "fabricated_addresses": []},
         "classification": {"names_predicted": True, "substituted": False},
         "direction": {"n_errors": 0}, "borderline": None, "uncertainty": None}
    r.update(over)
    return r


def test_a_note_passes_only_when_every_check_holds():
    assert e2e.passed_every_check(_report())
    assert not e2e.passed_every_check(_report(ok=False))
    assert not e2e.passed_every_check(_report(direction={"n_errors": 1}))
    assert not e2e.passed_every_check(_report(borderline={"borderline_conveyed": False}))
    assert not e2e.passed_every_check(
        _report(scores={"n_ungrounded": 0, "fabricated_addresses": ["92.168.1.1"]}))
    # a check that does not apply to the note is not held against it
    assert e2e.passed_every_check(_report(direction=None))


def test_detection_outcome_separates_wrong_family_from_false_positive():
    pack = {"flow": {"true_label": "mitm"}, "detection": {"predicted_class": "dos"}}
    assert e2e.detection_outcome(pack) == "attack_wrong_family"
    pack["flow"]["true_label"] = "dos"
    assert e2e.detection_outcome(pack) == "attack_right_family"
