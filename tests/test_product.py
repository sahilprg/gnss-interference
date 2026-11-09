import datetime as dt
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gnssguard.features.cells import GridSpec
from gnssguard.product import quality
from gnssguard.product.daily import days_to_run
from gnssguard.product.publish import (
    LABEL_CODES,
    _write_csv,
    cell_table,
    map_payload,
    public_incidents,
    read_index,
    region_stats,
    update_index,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = GridSpec(1.0, 15)
DAY = "2026-10-26"


def table(rows):
    t = pd.DataFrame(rows, columns=["cell_lat", "cell_lon", "w", "usable", "label"])
    t["window"] = pd.Timestamp(DAY, tz="UTC") + pd.to_timedelta(t["w"] * 15, unit="min")
    t["usable_aircraft"] = t["usable"].where(t["usable"] >= 3)
    t["p_jamming"], t["p_spoofing"] = 0.9, 0.01
    t["region"], t["date"] = "baltic", DAY
    return t.drop(columns=["w", "usable"])


def test_map_payload_codes_and_privacy():
    t = table([(55.0, 20.0, 0, 5, "jamming"), (55.0, 20.0, 1, 2, "low_coverage"),
               (56.0, 21.0, 95, 4, "normal")])
    p = map_payload(t, DAY, "baltic", SPEC)
    assert p["windows"] == 96 and len(p["cells"]) == 2
    c = p["cells"][0]
    assert c["label"][:3] == [LABEL_CODES["jamming"], LABEL_CODES["low_coverage"], 0]
    assert c["aircraft"][:2] == [5, None]  # no exact count for 1-2 aircraft
    assert p["cells"][1]["label"][95] == LABEL_CODES["normal"]
    assert "icao" not in json.dumps(p)


def test_csv_has_no_identifiers_and_skips_empty(tmp_path):
    t = table([(55.0, 20.0, 0, 5, "jamming"), (55.0, 20.0, 1, 0, "no_data")])
    _write_csv(tmp_path / "x.csv.gz", t)
    with gzip.open(tmp_path / "x.csv.gz", "rt", encoding="utf-8") as f:
        text = f.read()
    header = text.splitlines()[0].split(",")
    assert header == ["date", "region", "cell_lat", "cell_lon", "window_start",
                      "usable_aircraft", "p_jamming", "p_spoofing", "label"]
    assert len(text.splitlines()) == 2  # the no-data window is not listed


def test_region_stats_use_judged_cells_only():
    t = table([(55.0, 20.0, 0, 5, "jamming"), (55.0, 20.0, 1, 4, "normal"),
               (55.0, 20.0, 2, 1, "low_coverage")])
    s = region_stats(t, aircraft=12)
    assert s["judged_cells"] == 2 and s["low_coverage_cells"] == 1
    assert s["flagged_share"] == 0.5 and s["jamming_share"] == 0.5
    t2 = table([(55.0, 20.0, 0, 5, "jamming"), (55.0, 20.0, 1, 4, "normal"),
                (55.0, 20.0, 2, 1, "jamming_sparse"), (55.0, 20.0, 3, 2, "low_coverage")])
    s2 = region_stats(t2, aircraft=12)
    # sparse flags are listed, but the shares stay on cells with 3+ aircraft
    assert s2["low_coverage_cells"] == 2 and s2["sparse_flagged_cells"] == 1
    assert s2["judged_cells"] == 2 and s2["flagged_share"] == 0.5


def test_cell_table_marks_sparse_flags_and_hides_counts():
    class Det:
        def predict(self, cells, calibrated=True):
            return pd.DataFrame({"p_jamming": 0.9, "p_spoofing": 0.0,
                                 "label": ["jamming", "jamming", "normal", "jamming"]},
                                index=cells.index)

    cells = pd.DataFrame({"region": "baltic", "date": DAY, "cell_lat": 55.0,
                          "cell_lon": [20.0, 21.0, 22.0, 23.0],
                          "window": pd.Timestamp(DAY, tz="UTC"),
                          "n_ac_usable": [5, 2, 1, 0]})
    t = cell_table(cells, Det())
    assert t["label"].tolist() == ["jamming", "jamming_sparse", "low_coverage", "no_data"]
    assert t["usable_aircraft"].isna().tolist() == [False, True, True, True]
    assert set(t["label"]) <= set(LABEL_CODES)


def test_incidents_coarse_and_only_confident_sources():
    w = pd.Timestamp(DAY, tz="UTC")
    inc = pd.DataFrame({"cell_lat": [55.0, 55.0, 56.0, 50.0], "cell_lon": [20.0, 21.0, 20.0, 9.0],
                        "window": [w, w, w, w], "incident_id": [0, 0, 0, 1]})
    est = pd.DataFrame({"incident_id": [0], "lat": [55.437], "lon": [20.561],
                        "uncertainty_cal_km": [23.0], "confident": [True]})
    out = public_incidents("baltic", est, inc, SPEC)
    assert len(out) == 1  # the single-cell incident is not listed
    src = out[0]["source"]
    assert (src["lat"], src["lon"]) == (55.4, 20.6)
    assert src["radius_km"] == 50.0  # never smaller than 50 km
    wide = public_incidents("baltic", est.assign(uncertainty_cal_km=250.0), inc, SPEC)[0]
    assert wide["source"] is None and wide["confidence"] == "too_uncertain"
    low = public_incidents("baltic", est.assign(confident=False), inc, SPEC)[0]
    assert low["source"] is None and low["confidence"] == "low"
    assert low["area"] == {"lat_min": 55.0, "lat_max": 57.0, "lon_min": 20.0, "lon_max": 22.0}

    # a centre outside the region box: no aircraft there to constrain it, not drawn
    box = (53.0, 60.5, 18.0, 28.5)
    assert public_incidents("baltic", est, inc, SPEC, box)[0]["source"] is not None
    away = public_incidents("baltic", est.assign(lon=16.1), inc, SPEC, box)[0]
    assert away["source"] is None and away["confidence"] == "outside_coverage"


def _hist(n, aircraft=1000, judged=500):
    return [{"date": f"2026-10-{10 + i}", "regions": {"baltic": {
        "status": "published", "aircraft": aircraft, "judged_cells": judged}}} for i in range(n)]


def test_quality_gates():
    ok = {"baltic": {"aircraft": 900, "judged_cells": 400, "flagged_share": 0.4},
          "central_europe": {"aircraft": 4000, "judged_cells": 2600, "flagged_share": 0.006}}
    st, day = quality.check_day(ok, _hist(5), complete_day=True)
    assert st["baltic"]["status"] == "published" and not day
    assert st["central_europe"]["notes"]  # no history yet for that region: passes, noted
    low = {**ok, "baltic": {"aircraft": 300, "judged_cells": 400, "flagged_share": 0.4}}
    st, _ = quality.check_day(low, _hist(5), complete_day=True)
    assert st["baltic"]["status"] == "withheld" and "low aircraft" in st["baltic"]["reasons"][0]
    alarm = {**ok, "central_europe": {"aircraft": 4000, "judged_cells": 2600,
                                      "flagged_share": 0.08}}
    st, day = quality.check_day(alarm, _hist(5), complete_day=True)
    # an output alarm marks the day for review but does not hide it
    assert not day and all(v["status"] == "published" for v in st.values())
    assert all("control alarm" in v["review"][0] for v in st.values())
    st, day = quality.check_day(ok, _hist(5), complete_day=False)
    assert all(v["status"] == "withheld" for v in st.values()) and "incomplete" in day[0]


def test_index_and_days_to_run(tmp_path):
    for d, pub, reason in (("2026-10-28", True, ""), ("2026-10-29", False, "control alarm"),
                           ("2026-10-30", False, "incomplete download: ...")):
        s = {"date": d, "published": pub, "day_reasons": [reason] if reason else [],
             "regions": {"baltic": {"status": "published" if pub else "withheld"}}}
        (tmp_path / "data" / d).mkdir(parents=True)
        (tmp_path / "data" / d / "summary.json").write_text(json.dumps(s), encoding="utf-8")
        update_index(tmp_path, s)
    assert [d["date"] for d in read_index(tmp_path)] == ["2026-10-30", "2026-10-29", "2026-10-28"]
    todo = days_to_run(tmp_path, dt.date(2026, 11, 1), lookback=5)
    # 27 and 31 are new; 29 and 30 were withheld and are retried (a bounded number of
    # times); 28 is complete. The newest day runs first.
    assert todo == [dt.date(2026, 10, 31), dt.date(2026, 10, 27), dt.date(2026, 10, 29),
                    dt.date(2026, 10, 30)]
    idx = json.loads((tmp_path / "data" / "index.json").read_text(encoding="utf-8"))
    assert idx["latest"] == "2026-10-28"


def test_build_site(tmp_path):
    from gnssguard.product.site import build_site

    out = build_site(tmp_path, ROOT / "configs" / "regions.yaml", ["baltic", "punjab"])
    for f in ("index.html", "about.html", "data.html", "app.js", "style.css", ".nojekyll"):
        assert (out / f).exists()
    bm = json.loads((out / "data" / "basemap" / "baltic.json").read_text(encoding="utf-8"))
    assert bm["land"] and bm["borders"]
    regions = json.loads((out / "data" / "regions.json").read_text(encoding="utf-8"))["regions"]
    assert [r["name"] for r in regions] == ["baltic", "punjab"]
    html = (out / "index.html").read_text(encoding="utf-8")
    assert 'integrity="sha256-' in html  # pinned third-party script


@pytest.mark.parametrize("v", [np.nan, None])
def test_payload_handles_missing_probabilities(v):
    t = table([(55.0, 20.0, 0, 1, "low_coverage")])
    t["p_jamming"] = v
    p = map_payload(t, DAY, "baltic", SPEC)
    assert p["cells"][0]["p_jamming"][0] is None


def _summary(date, ce_share, disagree=0.005, judged=1000, train_end="2026-10-24"):
    reg = lambda share: {"status": "published", "flagged_share": share,  # noqa: E731
                         "rule_disagreement": disagree, "judged_cells": judged, "aircraft": 900}
    return {"date": date, "published": True, "day_reasons": [],
            "model": {"train_dates": ["2026-09-29", train_end]},
            "regions": {"central_europe": reg(ce_share), "baltic": reg(0.4)}}


def _write_days(out, summaries):
    for s in summaries:
        d = out / "data" / s["date"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(s), encoding="utf-8")
        update_index(out, s)


def test_health_ok_then_alarms(tmp_path):
    from gnssguard.product.health import assess

    days = [f"2026-11-{i:02d}" for i in range(1, 15)]
    _write_days(tmp_path, [_summary(d, 0.006) for d in days])
    res = assess(tmp_path, today=dt.date(2026, 11, 15))
    assert res["status"] == "ok" and res["signals"]["model_age_days"] == 22
    assert res["freshness"]["status"] == "fresh" and res["freshness"]["days_behind"] == 1
    drift = tmp_path / "drift"
    _write_days(drift, [_summary(d, 0.006 if i < 7 else 0.03, disagree=0.005 if i < 7 else 0.05,
                                 judged=1000 if i < 7 else 300)
                        for i, d in enumerate(days)])
    reasons = " ".join(assess(drift, today=dt.date(2026, 11, 15))["reasons"])
    assert "calm central_europe" in reasons and "disagree" in reasons and "judged cells" in reasons
    old = tmp_path / "old"
    _write_days(old, [_summary("2026-12-20", 0.006)])
    assert "days before today" in assess(old, today=dt.date(2026, 12, 21))["reasons"][0]


def test_a_stopped_pipeline_is_stale_not_ok(tmp_path):
    # review of 7 Nov 2026: health used to be measured against the newest published day,
    # so a pipeline that stopped publishing still reported "ok"
    from gnssguard.product.health import assess

    _write_days(tmp_path, [_summary(f"2026-11-{i:02d}", 0.006) for i in range(1, 15)])
    res = assess(tmp_path, today=dt.date(2026, 12, 1), backlog=16)
    assert res["freshness"]["status"] == "stale" and res["freshness"]["days_behind"] == 17
    assert res["signals"]["model_age_days"] == 38  # measured to today, not to Nov 14


def test_backlog_persists_past_the_lookback(tmp_path):
    # review of 7 Nov 2026: with a fixed 3-day look-back, an outage longer than that left
    # permanent gaps; the backlog now keeps every missing day since the first published one
    from gnssguard.product.daily import backlog

    for d in ("2026-10-20", "2026-10-21"):
        s = {"date": d, "published": True, "day_reasons": [],
             "regions": {"baltic": {"status": "published"}}}
        (tmp_path / "data" / d).mkdir(parents=True)
        (tmp_path / "data" / d / "summary.json").write_text(json.dumps(s), encoding="utf-8")
        update_index(tmp_path, s)
    today = dt.date(2026, 11, 1)  # a ten-day outage after the 21st
    assert len(backlog(tmp_path, today, lookback=3)) == 10  # 22 Oct to 31 Oct
    todo = days_to_run(tmp_path, today, lookback=3, max_days=4)
    # this run: the newest missing day first (a timeout then loses an old day, not
    # yesterday), then the oldest; the rest wait for later runs
    assert todo == [dt.date(2026, 10, 31), dt.date(2026, 10, 22), dt.date(2026, 10, 23),
                    dt.date(2026, 10, 24)]


def test_outside_reports_daily_check(tmp_path):
    from gnssguard.product.publish import outside_reports

    adv = tmp_path / "adv.csv"
    adv.write_text(
        "advisory_id,reference,issuer,lat_min,lat_max,lon_min,lon_max,start_utc,end_utc,"
        "source_url,notes\n"
        "A1,NOTAM X,Test,54,57,19,22,2026-10-01T00:00:00Z,,https://example.org,box\n"
        "A2,Old,Test,54,57,19,22,2026-01-01T00:00:00Z,2026-02-01T00:00:00Z,https://example.org,x\n",
        encoding="utf-8")
    rows = [(54.0 + i % 3, 19.0 + i % 3, 0, 5, "jamming") for i in range(12)]
    rows += [(58.0, 25.0 + i % 3, 0, 5, "normal") for i in range(12)]
    rows += [(55.0, 20.0, 1, 2, "low_coverage")]
    t = table(rows)
    out = outside_reports(DAY, {"baltic": t}, adv, SPEC)
    assert [r["report"] for r in out] == ["A1"]  # A2 was not in force that day
    assert out[0]["judged_inside"] == 12 and out[0]["verdict"] == "seen"
    assert out[0]["share_inside"] == 1.0 and out[0]["share_outside"] == 0.0


def test_model_hash_and_run_config_are_stable(tmp_path):
    from gnssguard.product.publish import file_hash, run_config

    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")
    (tmp_path / "y_jam.txt").write_text("tree", encoding="utf-8")
    h = file_hash(tmp_path)
    assert len(h) == 12 and h == file_hash(tmp_path)
    (tmp_path / "y_jam.txt").write_text("other tree", encoding="utf-8")
    assert file_hash(tmp_path) != h
    a = run_config(["baltic", "punjab"], SPEC)
    assert a == run_config(["punjab", "baltic"], SPEC) and a["label_threshold"] == 0.5


def test_provenance_same_model_same_hash_on_every_os(tmp_path):
    # review of 7 Nov 2026: published files lacked the commit, region boxes and feature
    # schema, and a Windows checkout reported a different model hash than the site
    from gnssguard.product.publish import model_hash, run_config

    lin, win = tmp_path / "lin", tmp_path / "win"
    for d, nl in ((lin, b"\n"), (win, b"\r\n")):
        d.mkdir()
        for name in ("meta.json", "y_jam.txt", "y_spoof.txt"):
            (d / name).write_bytes(b"line one" + nl + b"line two" + nl)
    (win / "notes.txt").write_text("not part of the model", encoding="utf-8")
    assert model_hash(lin) == model_hash(win)
    boxes = {"baltic": (53.0, 60.5, 18.0, 28.5), "punjab": (29.0, 33.5, 72.0, 76.5)}
    a = run_config(["baltic"], SPEC, boxes, ["f1", "f2"])
    assert a["region_boxes"] == {"baltic": [53.0, 60.5, 18.0, 28.5]}
    assert a["points_schema"].startswith("points/") and len(a["features_hash"]) == 12
    assert "commit" in a and "dirty" in a
    moved = run_config(["baltic"], SPEC, {"baltic": (53.0, 61.0, 18.0, 28.5)}, ["f1", "f2"])
    assert moved["hash"] != a["hash"]  # a changed region box is a different computation


def test_hourly_gap_marks_review_not_withhold():
    hist = []
    for d in ("2026-10-01", "2026-10-02", "2026-10-03"):
        hist.append({"date": d, "regions": {"baltic": {
            "status": "published", "aircraft": 100, "judged_cells": 100, "flagged_share": 0.1,
            "hourly_cells": [50] * 24}}})
    today = {"baltic": {"aircraft": 100, "judged_cells": 100, "flagged_share": 0.1,
                        "hourly_cells": [50] * 3 + [5, 5] + [50] * 19}}
    st, day = quality.check_day(today, hist, complete_day=True)
    assert st["baltic"]["status"] == "published" and not day
    assert "hours 03, 04 UTC" in st["baltic"]["review"][0]
    assert quality.hourly_gaps({"hourly_cells": [50] * 24}, hist, "baltic") == []


def _fake_daily(monkeypatch, fail_on=(), withhold=()):
    from gnssguard.product import daily

    def publish(iso, regions, *a, **k):
        if iso in fail_on:
            raise RuntimeError("model file corrupt")
        st = {r: {"status": "withheld" if (iso, r) in withhold else "published"}
              for r in regions}
        s = {"date": iso, "published": True, "day_reasons": [], "regions": st}
        update_index(k.get("out") or a[3], s)
        return s
    monkeypatch.setattr(daily, "ingest_day", lambda *a, **k: None)
    monkeypatch.setattr(daily, "build_all", lambda *a, **k: None)
    monkeypatch.setattr(daily, "publish_day", publish)
    monkeypatch.setattr(daily, "build_site", lambda *a, **k: None)
    monkeypatch.setattr(daily, "load_regions", lambda *a, **k: [])
    return daily


def test_a_failing_day_does_not_stop_the_others_and_is_retried_at_most_3_times(
        tmp_path, monkeypatch):
    daily = _fake_daily(monkeypatch, fail_on={"2026-10-30"})
    kw = dict(processed="p", cells_dir="c", model_dir="m", config="x", regions=["baltic"],
              lookback=3, today=dt.date(2026, 11, 1))
    done = daily.run(tmp_path, **kw)
    assert [s["date"] for s in done] == ["2026-10-31", "2026-10-29"]  # 30 failed, others went on
    q = daily.read_queue(tmp_path)
    assert q["2026-10-30"]["state"] == "failed" and q["2026-10-30"]["attempts"] == 1
    assert "RuntimeError" in q["2026-10-30"]["last_error"]
    for _ in range(4):
        daily.run(tmp_path, **kw)
    assert daily.read_queue(tmp_path)["2026-10-30"]["attempts"] == 3  # then given up
    assert "2026-10-30" in daily.given_up(tmp_path)
    assert dt.date(2026, 10, 30) not in daily.backlog(tmp_path, dt.date(2026, 11, 1), 3)


def test_withheld_regions_are_retried(tmp_path, monkeypatch):
    daily = _fake_daily(monkeypatch, withhold={("2026-10-31", "punjab")})
    kw = dict(processed="p", cells_dir="c", model_dir="m", config="x",
              regions=["baltic", "punjab"], lookback=1, today=dt.date(2026, 11, 1))
    daily.run(tmp_path, **kw)
    assert daily.read_queue(tmp_path)["2026-10-31"]["regions_withheld"] == ["punjab"]
    assert daily.backlog(tmp_path, dt.date(2026, 11, 1), 1) == [dt.date(2026, 10, 31)]
    daily = _fake_daily(monkeypatch)  # the problem passed: next run publishes punjab
    daily.run(tmp_path, **kw)
    assert daily.read_queue(tmp_path)["2026-10-31"]["state"] == "complete"
    assert daily.backlog(tmp_path, dt.date(2026, 11, 1), 1) == []


def test_health_reports_failed_and_given_up_days(tmp_path, monkeypatch):
    from gnssguard.product.health import write_health

    daily = _fake_daily(monkeypatch, fail_on={"2026-10-30"})
    kw = dict(processed="p", cells_dir="c", model_dir="m", config="x", regions=["baltic"],
              lookback=3, today=dt.date(2026, 11, 1))
    daily.run(tmp_path, **kw)
    res = write_health(tmp_path, today=dt.date(2026, 11, 1))
    assert res["processing"]["status"] == "attention"
    assert "2026-10-30" in res["processing"]["failing"]
    for _ in range(3):
        daily.run(tmp_path, **kw)
    res = write_health(tmp_path, today=dt.date(2026, 11, 1))
    assert list(res["processing"]["given_up"]) == ["2026-10-30"]
    md = (tmp_path / "data" / "health.md").read_text(encoding="utf-8")
    assert "given up after 3 attempts" in md
