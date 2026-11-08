"""An interrupted publish (killed runner, out of memory, full disk) must never leave a
half-written day or index: the previous version stays, and the next run cleans up."""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from gnssguard.product import publish as P

DATE = "2026-11-07"
REGIONS = ["baltic", "central_europe"]


class _StubDetectors:
    meta = {"train_dates": ["2026-09-29", "2026-10-24"], "versions": {}}
    features: list[str] = []

    @classmethod
    def load(cls, model_dir):
        return cls()


@pytest.fixture
def stub(monkeypatch, tmp_path):
    """publish_day without a model or data: every region is empty, the files still go out."""
    monkeypatch.setattr(P, "Detectors", _StubDetectors)
    monkeypatch.setattr(P, "load_cells", lambda *a, **k: pd.DataFrame())
    (tmp_path / "model").mkdir()
    return tmp_path


def _publish(tmp_path, **kw):
    return P.publish_day(DATE, REGIONS, tmp_path / "processed", tmp_path / "cells",
                         tmp_path / "model", tmp_path / "public", config=None,
                         advisories_csv=None, **kw)


def _snapshot(folder):
    return {p.relative_to(folder).as_posix(): p.read_bytes()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def test_publish_writes_day_then_index(stub):
    s = _publish(stub)
    data = stub / "public" / "data"
    assert (data / DATE / "summary.json").exists()
    assert json.loads((data / DATE / "summary.json").read_text(encoding="utf-8")) == s
    assert [d["date"] for d in P.read_index(stub / "public")] == [DATE]
    assert not list(data.glob(".staging-*")) and not list(data.glob("*.old"))
    assert not list(data.glob("*.tmp"))


def test_interrupted_republish_keeps_the_previous_version(stub, monkeypatch):
    _publish(stub)
    data = stub / "public" / "data"
    before = _snapshot(data)

    def boom(*a, **k):
        raise MemoryError("runner killed")
    monkeypatch.setattr(P, "outside_reports", boom)  # late in the run: files already written
    with pytest.raises(MemoryError):
        _publish(stub)

    # the published day and the index are byte for byte what they were
    after = {k: v for k, v in _snapshot(data).items() if not k.startswith(".staging-")}
    assert after == before
    assert list(data.glob(".staging-*"))  # the half-written copy is set aside, not live

    monkeypatch.undo()
    monkeypatch.setattr(P, "Detectors", _StubDetectors)
    monkeypatch.setattr(P, "load_cells", lambda *a, **k: pd.DataFrame())
    _publish(stub)  # the next run removes it and publishes normally
    assert not list(data.glob(".staging-*"))
    assert (data / DATE / "summary.json").exists()


def test_interrupted_first_publish_is_not_listed(stub, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(P, "outside_reports", boom)
    with pytest.raises(OSError):
        _publish(stub)
    # not in the index, so the nightly job sees it as missing and runs it again
    assert P.read_index(stub / "public") == []
    assert not (stub / "public" / "data" / DATE).exists()


def test_stop_between_the_two_renames_restores_the_old_day(tmp_path):
    data = tmp_path / "data"
    (data / f"{DATE}.old").mkdir(parents=True)
    (data / f"{DATE}.old" / "summary.json").write_text("{}", encoding="utf-8")
    (data / f".staging-{DATE}").mkdir()
    done = P.recover(tmp_path)
    assert (data / DATE / "summary.json").exists()
    assert not (data / f"{DATE}.old").exists()
    assert sorted(done) == [f"removed .staging-{DATE}", f"restored {DATE}"]


def test_failed_index_write_keeps_the_old_index(tmp_path, monkeypatch):
    P.update_index(tmp_path, {"date": "2026-11-06", "published": True,
                              "regions": {"baltic": {"status": "published"}}})
    p = tmp_path / "data" / "index.json"
    before = p.read_bytes()

    def fail(src, dst):
        raise OSError("killed during the write")
    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        P.update_index(tmp_path, {"date": DATE, "published": True,
                                  "regions": {"baltic": {"status": "published"}}})
    assert p.read_bytes() == before  # still valid JSON, still the old days
    monkeypatch.undo()
    P.recover(tmp_path)
    assert not list((tmp_path / "data").glob("*.tmp"))
