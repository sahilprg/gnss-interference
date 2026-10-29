import json
import re

import numpy as np
import pandas as pd
import pytest
from helpers import T0, flight, points

from gnssguard.features.cells import GridSpec, _haversine_km, build_from_points
from gnssguard.labels import synthetic as S
from gnssguard.viz.map import build_map

SPEC = GridSpec(1.0, 15)
START, END = T0.isoformat(), (T0 + pd.Timedelta(hours=1)).isoformat()


def test_radio_horizon():
    # 10 km aircraft, 30 m mast: 4.12 * (100 + 5.48) = 434.6 km
    assert S.radio_horizon_km(10_000 / 0.3048, 30) == pytest.approx(434.6, abs=0.5)
    # a low aircraft is reached only from much closer
    assert S.radio_horizon_km(1_000, 30) < 150


def test_jamming_core_ring_and_line_of_sight():
    near = flight("near", 50.0, 10.0, dlon=0.0)          # at the jammer
    far = flight("far", 50.0, 13.0, dlon=0.0)            # ~215 km away, beyond range
    low = flight("low", 50.0, 11.3, dlon=0.0, alt=500)   # ~93 km, but below the horizon
    ev = S.Jamming("J1", 50.0, 10.0, START, END, range_km=150, height_m=10, p_drop=0.0,
                   p_geom_lost=0.0)
    out, truth = S.inject(points(near, far, low), [ev], seed=0)
    by = out.groupby("icao")
    assert (by.get_group("near")["nic"] == 0).all()
    assert (by.get_group("far")["nic"] == 8).all() and (by.get_group("low")["nic"] == 8).all()
    assert set(truth["icao"]) == {"near"} and (truth["inj_event"] == "J1").all()


def test_jamming_respects_time_window_and_drop():
    f = flight("a", 50.0, 10.0, dlon=0.0, n=60, step_s=120)  # two hours
    ev = S.Jamming("J1", 50.0, 10.0, START, END, range_km=200, p_drop=1.0)
    out, truth = S.inject(points(f), [ev], seed=0)
    assert len(out) == 30  # first hour dropped entirely, second hour untouched
    assert (out["nic"] == 8).all() and truth["dropped"].all()


def test_spoofing_drift_moves_toward_target_and_stops_there():
    f = flight("a", 50.0, 10.0, dlon=0.0, n=60, step_s=60)  # hovering, one hour
    ev = S.Spoofing("S1", 50.0, 10.0, 50.0, START, END, target_lat=50.0, target_lon=11.0,
                    mode="drift", drift_kt=400, p_nic_kept=1.0)
    out, _ = S.inject(points(f), [ev], seed=0)
    d_target = _haversine_km(out["lat"], out["lon"], 50.0, 11.0)
    assert d_target.iloc[0] == pytest.approx(71.5, abs=1)  # starts at the true position
    assert np.all(np.diff(d_target) <= 1e-6) and d_target.iloc[-1] < 0.5  # ends at target
    assert (out["nic"] == 8).all()  # integrity kept: hard to see
    assert (out["true_lat"] == 50.0).all()


def test_truth_cells_and_evaluation_end_to_end():
    fl = [flight(f"a{i}", 50.1 + i / 10, 10.1, dlon=0.002) for i in range(6)]
    ev = S.Jamming("J1", 50.4, 10.2, START, END, range_km=200, p_drop=0.0)
    out, _ = S.inject(points(*fl), [ev], seed=0)
    truth = S.truth_cells(out, SPEC)
    assert truth["n_ac_affected"].iloc[0] == 6 and truth["inj_type"].iloc[0] == "jamming"
    cw, _ = build_from_points(out, SPEC)
    pred = cw.assign(p_jamming=0.99, majority_vote_jamming=1.0, spoof_votes=0, label="jamming")
    summary, per = S.evaluate(pred, truth, SPEC)
    assert summary["recall_jamming"] == 1.0 and per["recall"].iloc[0] == 1.0


def test_map_html_embeds_data_safely():
    fl = [flight(f"a{i}", 50.1 + i / 10, 10.1) for i in range(4)]
    cw, _ = build_from_points(points(*fl), SPEC)
    cw = cw.assign(region="test</script><b>", date="2026-10-26", cell_deg=1.0, window_min=15)
    html = build_map(cw)
    assert html.count("</script>") == 3  # leaflet, data, app ... and no injected one
    blob = re.search(r'id="data">(.*?)</script>', html, re.S).group(1)
    data = json.loads(blob.replace("<\\/", "</"))
    assert data["region"] == "test</script><b>" and len(data["cells"]) == 1
    assert data["rows"][0][0][2] == 4  # four usable aircraft in the only cell
    assert "integrity=" in html  # pinned, integrity-checked Leaflet
