"""The command line: every command parses, lives in its area's module, and the guards
that protect hand labels and the sealed test hold."""

from __future__ import annotations

import argparse
import datetime as dt
import json

import pytest

from gnssguard import cli
from gnssguard.cli import _common, checks, labels, pipeline, product


def _commands() -> dict[str, argparse.ArgumentParser]:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers()
    for m in cli.MODULES:
        m.register(sub)
    return dict(sub.choices)


EXPECTED = {
    pipeline: {"ingest", "coverage", "cells", "label", "map", "synth", "train", "evaluate",
               "rolling"},
    labels: {"gold-sample", "gold-tool", "holdout-sample", "holdout-eval", "review-sample",
             "review-eval", "relabel", "score-bands", "sealed-sample", "trust-report",
             "disagreement-sample", "disagreement-eval"},
    checks: {"event-check", "incidents", "opensky-probe", "opensky-fetch", "opensky-check",
             "ground-fetch", "ground-check", "benchmark"},
    product: {"publish", "daily", "health"},
}


def test_every_command_is_in_its_module():
    cmds = _commands()
    assert set(cmds) == set().union(*EXPECTED.values())
    for module, names in EXPECTED.items():
        for n in names:
            func = cmds[n].get_default("func")
            assert func.__module__ == module.__name__, n


@pytest.mark.parametrize("name", sorted(set().union(*EXPECTED.values())))
def test_help_works(name, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main([name, "--help"])
    assert e.value.code == 0
    assert f"usage: gnssguard {name}" in capsys.readouterr().out


def test_split_and_dates():
    assert _common.split(" a, b,,c ") == ["a", "b", "c"]
    assert _common.split(None) is None
    ns = argparse.Namespace(date=None, start="2026-10-30", end="2026-11-02")
    assert _common.dates(ns) == [dt.date(2026, 10, 30) + dt.timedelta(days=i) for i in range(4)]
    with pytest.raises(SystemExit):
        _common.dates(argparse.Namespace(date=None, start="2026-11-02", end="2026-10-30"))


@pytest.mark.parametrize("argv, key_flag", [
    (["gold-sample"], None),
    (["holdout-sample", "--date", "2026-11-01"], "--strata"),
    (["review-sample", "--dates", "2026-11-01"], "--groups"),
    (["sealed-sample", "--kind", "test", "--dates", "2026-11-05"], "--groups"),
    (["disagreement-sample", "--dates", "2026-10-01"], "--groups"),
])
def test_samplers_never_overwrite_labels(argv, key_flag, tmp_path, capsys):
    out = tmp_path / "labels.csv"
    groups = tmp_path / "groups.csv"  # the key file, kept apart from the labeler
    out.write_text("labels a person typed\n", encoding="utf-8")
    extra = ["--out", str(out)] + ([key_flag, str(groups)] if key_flag else [])
    assert cli.main(argv + extra) == 1
    assert out.read_text(encoding="utf-8") == "labels a person typed\n"
    assert not groups.exists()
    assert "Use --force" in capsys.readouterr().err


def test_disagreement_sample_refuses_sealed_days(tmp_path, capsys):
    argv = ["disagreement-sample", "--dates", "2026-11-04,2026-11-05",
            "--out", str(tmp_path / "d.csv"), "--groups", str(tmp_path / "g.csv")]
    assert cli.main(argv) == 1
    assert "sealed test days" in capsys.readouterr().err
    assert not (tmp_path / "d.csv").exists()


def test_health_command_writes_status(tmp_path, capsys):
    assert cli.main(["health", "--out", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("data: ")
    res = json.loads((tmp_path / "data" / "health.json").read_text(encoding="utf-8"))
    assert res["freshness"]["status"] in {"fresh", "stale"}
