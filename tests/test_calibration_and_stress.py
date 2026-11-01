import numpy as np
import pandas as pd
import pytest
from helpers import T0, flight, points

from gnssguard.features.cells import _haversine_km
from gnssguard.labels import synthetic as S
from gnssguard.model.calibrate import Calibrator, ece, fit_isotonic, mid_ece, reliability

START, END = T0.isoformat(), (T0 + pd.Timedelta(hours=1)).isoformat()


# -- calibration ---------------------------------------------------------------------------

def test_isotonic_is_monotone_and_fixes_overconfidence():
    rng = np.random.default_rng(0)
    p = rng.random(20_000)
    y = (rng.random(20_000) < p ** 3).astype(float)  # the model says p, truth is p**3
    cal = fit_isotonic(p, y)
    assert np.all(np.diff(cal.y) >= 0) and np.all(np.diff(cal.x) > 0)
    fixed = cal(p)
    assert ece(fixed, y) < 0.02 < ece(p, y)
    assert fixed[np.argsort(p)].tolist() == sorted(fixed.tolist())  # order kept
    assert cal(0.8) == pytest.approx(0.8 ** 3, abs=0.06)


def test_isotonic_merges_tiny_blocks_and_roundtrips():
    assert fit_isotonic([0.1, 0.2, 0.9], [0, 1, 0], min_block_weight=1).y == [0.0, 0.5]
    cal = fit_isotonic([0.1, 0.2, 0.9], [0, 1, 0], min_block_weight=20)
    assert len(cal.x) == 1 and cal(0.95) == pytest.approx(1 / 3)
    again = Calibrator.from_dict(cal.to_dict())
    assert again(0.5) == cal(0.5)
    assert Calibrator([], [])(0.7) == 0.7  # empty calibrator is the identity


def test_reliability_and_mid_ece():
    p = np.array([0.02] * 100 + [0.75] * 10)
    y = np.array([0.0] * 100 + [0.0] * 10)
    t = reliability(p, y)
    assert t.loc[t["bin"] == "0.7-0.8", "cells"].item() == 10
    assert ece(p, y) == pytest.approx((100 * 0.02 + 10 * 0.75) / 110)
    assert mid_ece(p, y) == pytest.approx(0.75)  # only the uncertain cells


# -- stress attacks --------------------------------------------------------------------------

def test_intermittent_jammer_is_off_between_bursts():
    f = flight("a", 50.0, 10.0, dlon=0.0, n=60, step_s=60)  # hovering for an hour
    ev = S.Jamming("J1", 50.0, 10.0, START, END, range_km=200, p_drop=0.0, p_geom_lost=0.0,
                   on_min=10, off_min=20)
    out, truth = S.inject(points(f), [ev], seed=0)
    minute = ((out["ts"] - T0).dt.total_seconds() // 60).astype(int)
    hit = out["nic"] == 0
    assert (hit == (minute % 30 < 10)).all()
    assert len(truth) == 20


def test_moving_jammer_follows_its_track():
    ev = S.Jamming("J1", 50.0, 10.0, START, END, speed_kt=60, heading_deg=90)
    ts = pd.Series(pd.to_datetime([T0, T0 + pd.Timedelta(minutes=30),
                                   T0 + pd.Timedelta(hours=1)], utc=True))
    lat, lon = ev.position_at(ts)
    assert lon[1] == pytest.approx(10.0) and lon[0] < 10.0 < lon[2]  # (lat, lon) = midpoint
    assert _haversine_km(lat[0], lon[0], lat[2], lon[2]) == pytest.approx(111.1, abs=1)
    # an aircraft near the end of the track is jammed only late in the event
    f = flight("a", 50.0, 10.75, dlon=0.0, n=60, step_s=60, alt=1000)
    out, _ = S.inject(points(f), [S.Jamming("J1", 50.0, 10.0, START, END, range_km=40,
                                            p_drop=0.0, speed_kt=60, heading_deg=90)], seed=0)
    hit = (out["nic"] < 8).to_numpy()
    assert not hit[:10].any() and hit[-10:].all()


def test_stress_profile_adds_pairs_bursts_motion_and_mlat():
    t0 = pd.Timestamp("2026-10-26T08:00Z")
    pts = points(*[flight(f"a{i}", 50 + i / 10, 10, alt=30000, t0=t0, n=200, step_s=60)
                   for i in range(5)])
    evs = [e for s in range(6) for e in S.random_events(pts, 4, 2, seed=s, profile="stress")]
    jam = [e for e in evs if e.type == "jamming"]
    assert any(e.event_id.endswith("b") for e in jam)
    assert any(e.on_min for e in jam) and any(e.speed_kt for e in jam)
    assert all(e.p_mlat == 0.5 for e in jam)
    # the standard profile draws no stress features
    std = [e for e in S.random_events(pts, 4, 2, seed=0) if e.type == "jamming"]
    assert all(e.on_min is None and e.speed_kt == 0 and e.p_mlat == 0 for e in std)


# -- localization uncertainty and rolling folds --------------------------------------------

def test_fit_and_apply_uncertainty():
    from gnssguard.model.localize import calibrate_uncertainty, fit_uncertainty

    rng = np.random.default_rng(0)
    raw = rng.uniform(10, 40, 200)
    loc = pd.DataFrame({"found": True, "score": 0.95, "uncertainty_km": raw,
                        "fit_error_km": raw * rng.uniform(0, 2.4, 200)})
    fit = fit_uncertainty(loc, target=0.9)
    assert fit["coverage"] >= 0.9 and 1.5 <= fit["scale"] <= 2.5
    est = calibrate_uncertainty(loc.assign(score=[0.6] + [0.95] * 199), fit["scale"],
                                fit["floor_km"])
    assert (est["uncertainty_cal_km"] >= est["uncertainty_km"]).all()
    assert not est["confident"].iloc[0] and est["confident"].iloc[1:].all()


def test_rolling_fold_uses_only_the_past():
    from gnssguard.model.rolling import fold_table

    days = ["2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23"]
    t = pd.DataFrame({"date": np.repeat(days, 3), "source": "synthetic", "y_jam": 0.0})
    ft, lm = fold_table(t, "2026-10-23")
    assert sorted(ft["date"].unique()) == days[:3]  # nothing from the test day or later
    assert set(ft.loc[ft["split"] == "val", "date"]) == {"2026-10-22"}
    assert set(ft.loc[ft["split"] == "train", "date"]) == {"2026-10-20", "2026-10-21"}
    assert fold_table(t, "2026-10-21") is None  # fewer than two earlier days


def test_two_source_localization_finds_both_jammers():
    from gnssguard.model.localize import localize

    rng = np.random.default_rng(3)
    lat = rng.uniform(48, 54, 3000)
    lon = rng.uniform(8, 20, 3000)
    alt = rng.uniform(20_000, 38_000, 3000)
    jams = [(50.0, 10.5), (52.0, 17.5)]  # about 520 km apart
    deg = np.zeros(3000, dtype=bool)
    for jl, jo in jams:
        deg |= _haversine_km(lat, lon, jl, jo) <= 150
    pts = pd.DataFrame({"lat": lat, "lon": lon, "alt_ft": alt, "degraded": deg})
    one = localize(pts)
    two = localize(pts, two_sources=True)
    assert one["n_sources"] == 1 and one["score"] < 0.85
    assert two["n_sources"] == 2 and two["score"] >= one["score"] + 0.03
    found = [(two["lat"], two["lon"]), (two["lat2"], two["lon2"])]
    for jl, jo in jams:
        assert min(_haversine_km(a, b, jl, jo) for a, b in found) < 30
