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


def test_truth_keeps_aircraft_whose_points_were_all_dropped():
    # review bug 1: an aircraft that lost every position is the most affected, not absent
    fl = [flight(f"a{i}", 50.1 + i / 10, 10.1, dlon=0.002) for i in range(6)]
    ev = S.Jamming("J1", 50.4, 10.2, START, END, range_km=200, p_drop=1.0, core_frac=1.0)
    out, injected = S.inject(points(*fl), [ev], seed=0)
    assert out.empty and injected["dropped"].all()
    old_way = S.truth_cells(out, SPEC)
    assert old_way.empty  # what the code used to do: the event disappears from truth
    truth = S.truth_cells(out, SPEC, truth=injected)
    row = truth.iloc[0]
    assert row["n_ac_jam"] == 6 and row["n_ac_jam_observed"] == 0
    pred = pd.DataFrame(columns=["cell_lat", "cell_lon", "window", "p_jamming",
                                 "majority_vote_jamming", "spoof_votes", "label"])
    summary, _ = S.evaluate(pred, truth, SPEC)
    assert summary["recall_jamming"] == 0.0  # counted, and missed: nothing was observable
    assert summary["unobservable_jamming"] == len(truth)
    assert summary["recall_jamming_observable"] is None


def test_mixed_cell_is_positive_for_both_types():
    # review bug 2: one jammed and one spoofed group in the same cell-window
    jam = [flight(f"j{i}", 50.1 + i / 20, 10.1, dlon=0.0) for i in range(3)]
    spf = [flight(f"s{i}", 50.6 + i / 20, 10.6, dlon=0.0) for i in range(3)]
    ev_j = S.Jamming("J1", 50.15, 10.1, START, END, range_km=30, height_m=500, p_drop=0.0,
                     core_frac=1.0)
    ev_s = S.Spoofing("S1", 50.65, 10.6, 20.0, START, END, target_lat=50.65,
                      target_lon=10.65, mode="jump", p_nic_kept=1.0)
    out, injected = S.inject(points(*jam, *spf), [ev_j, ev_s], seed=0)
    truth = S.truth_cells(out, SPEC, truth=injected)
    cell = truth[(truth["cell_lat"] == 50) & (truth["cell_lon"] == 10)].iloc[0]
    assert cell["n_ac_jam"] == 3 and cell["n_ac_spoof"] == 3 and cell["inj_type"] == "both"
    cw, _ = build_from_points(out, SPEC)
    pred = cw.assign(p_jamming=0.99, majority_vote_jamming=1.0, spoof_votes=0, label="jamming")
    summary, per = S.evaluate(pred, truth, SPEC)
    assert summary["truth_cell_windows_jamming"] > 0 and summary["truth_cell_windows_spoofing"] > 0
    assert summary["recall_jamming"] == 1.0 and summary["recall_spoofing"] == 0.0
    assert set(per["inj_type"]) == {"jamming", "spoofing"}


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
    assert data["basemap"]["land"], "embedded coastlines around central Europe"
    assert "tile.openstreetmap.org" not in html  # OSM refuses pages opened from disk


def test_basemap_clip():
    from gnssguard.viz.basemap import _clip_ring, background

    square = [[0, 0], [10, 0], [10, 10], [0, 10]]  # [lon, lat]
    clipped = _clip_ring(square, (2, 5, -5, 4))  # lat 2..5, lon -5..4
    lons, lats = [p[0] for p in clipped], [p[1] for p in clipped]
    assert min(lons) == 0 and max(lons) == 4 and min(lats) == 2 and max(lats) == 5
    assert _clip_ring(square, (20, 30, 20, 30)) == []

    bg = background((53.0, 60.5, 18.0, 28.5))  # Baltic
    pts = [p for poly in bg["land"] for ring in poly for p in ring]
    assert pts and all(-2 <= p[0] <= 48.5 and 45 <= p[1] <= 68.5 for p in pts)
