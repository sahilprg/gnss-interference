from types import SimpleNamespace

import pandas as pd

from gnssguard.labels.events import load_events
from gnssguard.product.incident import incident, index_page, page

HEADER = ("event_id,type,lat_min,lat_max,lon_min,lon_max,start_utc,end_utc,source_url,notes,"
          "evidence_level,region\n")
ROW = ("{id},jamming,54.0,56.0,20.0,22.0,{d}T00:00:00Z,{e}T00:00:00Z,https://example.org/a,"
       "\"test, with a comma\",observed_anomaly,baltic\n")
EVENTS = (HEADER + ROW.format(id="EV01", d="2026-11-03", e="2026-11-04")
          + ROW.format(id="EV02", d="2026-11-05", e="2026-11-06"))


def day_cells(date, jam_share):
    rows = []
    for w in pd.date_range(date, periods=8, freq="15min", tz="UTC"):
        for lat in (54, 55):
            for lon in (20, 21):
                flagged = len(rows) % 10 < jam_share * 10
                rows.append({"region": "baltic", "date": date, "cell_lat": float(lat),
                             "cell_lon": float(lon), "window": w, "n_ac_usable": 5,
                             "n_ac_degraded": 4 if flagged else 0, "n_ac_nic0": 3 if flagged else 0,
                             "n_ac_altdev": 0, "n_ac_speed_mismatch": 0,
                             "model": "jamming" if flagged else "normal",
                             "rules": "jamming" if flagged else "normal",
                             "p_jamming_cal": 0.95 if flagged else 0.02,
                             "p_spoofing_cal": 0.0})
    return pd.DataFrame(rows)


class FakeScored:
    def __init__(self, tmp_path, days):
        self.days, self.model_dir = days, tmp_path
        (tmp_path / "m.txt").write_text("model", encoding="utf-8")
        self.det = SimpleNamespace(meta={"train_dates": ["2026-10-01"]})

    def available(self, region):
        return sorted(self.days)

    def day(self, region, date):
        return self.days[date]

    def cell_hash(self, region, date):
        return "abc"


def test_event_day_against_ordinary_days(tmp_path):
    f = tmp_path / "events.csv"
    f.write_text(EVENTS, encoding="utf-8")
    events = load_events(f)
    days = {"2026-11-03": day_cells("2026-11-03", 0.8),
            "2026-11-05": day_cells("2026-11-05", 0.9),  # another event: left out
            **{f"2026-10-{d:02d}": day_cells(f"2026-10-{d:02d}", 0.1) for d in range(1, 6)}}
    p = incident(FakeScored(tmp_path, days), next(events.itertuples(index=False)), events)
    assert p["status"] == "found" and p["check"]["model"]["share"] > 0.75
    base = p["ordinary_days"]
    assert base["days"] == 5 and base["days_at_or_above_event"] == 0
    assert base["training_days"] == 1 and "stands out" in p["reading"]
    flagged = {r["what"]: r for r in p["evidence"]}["NIC 0 (no integrity at all)"]
    assert flagged["flagged"] == 3 and flagged["normal"] == 0
    html = page(p)
    assert "EV01" in html and "test, with a comma" in html and "<svg" in html
    assert "aircraft identifiers" in html and "icao" not in html.lower()
    assert "EV01.html" in index_page([p])


def test_routine_box_is_said_so(tmp_path):
    f = tmp_path / "events.csv"
    f.write_text(EVENTS, encoding="utf-8")
    events = load_events(f)
    days = {"2026-11-03": day_cells("2026-11-03", 0.5),
            **{f"2026-10-{d:02d}": day_cells(f"2026-10-{d:02d}", 0.6) for d in range(1, 6)}}
    p = incident(FakeScored(tmp_path, days), next(events.itertuples(index=False)), events)
    assert p["ordinary_days"]["days_at_or_above_event"] == 5 and "routine" in p["reading"]


def test_no_data_page(tmp_path):
    f = tmp_path / "events.csv"
    f.write_text(EVENTS, encoding="utf-8")
    events = load_events(f)
    p = incident(FakeScored(tmp_path, {}), next(events.itertuples(index=False)), events)
    assert p["status"] == "no data" and "No cell files" in page(p)


def test_site_copies_incident_pages(tmp_path):
    from gnssguard.product.site import build_site

    inc = tmp_path / "inc"
    inc.mkdir()
    (inc / "index.html").write_text("<html>incidents</html>", encoding="utf-8")
    (inc / "EV01.json").write_text("{}", encoding="utf-8")
    out = build_site(tmp_path / "site", regions=["baltic"], incidents=inc)
    copied = (out / "incidents" / "index.html").read_text(encoding="utf-8")
    assert copied == "<html>incidents</html>"
    assert 'href="incidents/index.html"' in (out / "index.html").read_text(encoding="utf-8")
