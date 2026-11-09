"""The split guard: no overlap, benchmarks only from earlier days, and the sealed days
kept out of training, evaluation and other samples until the sealed test is scored."""

from __future__ import annotations

import argparse

import pytest

from gnssguard.model.splits import SEALED_DAYS, SplitError, check_split, manifest, refuse_sealed


def test_overlap_is_refused():
    with pytest.raises(SplitError, match="train and test days overlap"):
        check_split(["2026-10-01", "2026-10-26"], ["2026-10-25"], ["2026-10-26"])
    with pytest.raises(SplitError, match="validation and test"):
        check_split(["2026-10-01"], ["2026-10-26"], ["2026-10-26"])


def test_benchmark_needs_earlier_days_deployment_may_use_later_ones():
    later = ["2026-10-01", "2026-10-30"]
    with pytest.raises(SplitError, match="before the first test day"):
        check_split(later, ["2026-10-25"], ["2026-10-26"], "benchmark", sealed_scored=False)
    check_split(later, ["2026-10-25"], ["2026-10-26"], "deployment", sealed_scored=False)


def test_sealed_days_are_refused_until_scored():
    for purpose in ("benchmark", "deployment"):  # test day after: only the sealed rule bites
        with pytest.raises(SplitError, match="sealed test days"):
            check_split(["2026-10-01", "2026-11-07"], ["2026-10-02"], ["2026-11-25"],
                        purpose, sealed_scored=False)
    check_split(["2026-10-01", "2026-11-07"], ["2026-10-02"], ["2026-10-26"], "deployment",
                sealed_scored=True)
    with pytest.raises(SplitError):
        refuse_sealed(["2026-11-18"], "evaluation", sealed_scored=False)
    refuse_sealed(["2026-11-19", "2026-11-04"], "evaluation", sealed_scored=False)
    assert SEALED_DAYS[0] == "2026-11-05" and SEALED_DAYS[-1] == "2026-11-18"


def test_manifest_is_versioned_and_hashed():
    m = manifest(["b", "a"], ["c"], ["d"], "benchmark")
    assert m["schema"] == "split/1" and m["train"] == ["a", "b"] and len(m["hash"]) == 12
    assert manifest(["a", "b"], ["c"], ["d"], "benchmark")["hash"] == m["hash"]


def test_default_split_takes_only_earlier_usable_days(tmp_path, monkeypatch):
    from gnssguard.cli import _common

    days = ["2026-10-20", "2026-10-21", "2026-10-22", "2026-10-26", "2026-10-30",
            "2026-11-06"]
    monkeypatch.setattr("gnssguard.analysis.coverage.partitions",
                        lambda cells: [("baltic", d, None) for d in days])
    monkeypatch.chdir(tmp_path)  # no results/sealed_test.json here
    ns = argparse.Namespace(cells="x", test_dates="2026-10-26", gold="none.csv",
                            val_dates=None, train_dates=None, purpose="benchmark")
    tr, va, te = _common.split_days(ns)
    assert tr == ["2026-10-20", "2026-10-21"] and va == ["2026-10-22"] and te == ["2026-10-26"]
    ns.purpose = "deployment"
    tr, va, _ = _common.split_days(ns)
    assert va == ["2026-10-30"] and "2026-11-06" not in tr  # later days, never sealed ones
    ns.train_dates = "2026-11-06"
    with pytest.raises(SystemExit, match="sealed test days"):
        _common.split_days(ns)


def test_evaluate_and_samplers_refuse_sealed_days(tmp_path, monkeypatch, capsys):
    from gnssguard import cli

    monkeypatch.chdir(tmp_path)
    for argv in (["review-sample", "--dates", "2026-11-07", "--out", "r.csv",
                  "--groups", "g.csv"],
                 ["holdout-sample", "--date", "2026-11-10", "--out", "h.csv",
                  "--strata", "s.csv"]):
        assert cli.main(argv) == 1
        assert "sealed test days" in capsys.readouterr().err
