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
    assert all(v["status"] == "withheld" for v in st.values()) and "control alarm" in day[0]
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
    # 27 and 31 are new; 30 is retried (incomplete download); 28 done; 29 would fail again
    assert todo == [dt.date(2026, 10, 27), dt.date(2026, 10, 30), dt.date(2026, 10, 31)]
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
    res = assess(tmp_path)
    assert res["status"] == "ok" and res["signals"]["model_age_days"] == 21
    drift = tmp_path / "drift"
    _write_days(drift, [_summary(d, 0.006 if i < 7 else 0.03, disagree=0.005 if i < 7 else 0.05,
                                 judged=1000 if i < 7 else 300)
                        for i, d in enumerate(days)])
    reasons = " ".join(assess(drift)["reasons"])
    assert "calm central_europe" in reasons and "disagree" in reasons and "judged cells" in reasons
    old = tmp_path / "old"
    _write_days(old, [_summary("2026-12-20", 0.006)])
    assert "days older" in assess(old)["reasons"][0]


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
