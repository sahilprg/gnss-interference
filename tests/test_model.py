import numpy as np
import pandas as pd
import pytest

from gnssguard.features.cells import _haversine_km
from gnssguard.labels.synthetic import _destination, radio_horizon_km
from gnssguard.model.dataset import FEATURES, add_context
from gnssguard.model.evaluate import wilson
from gnssguard.model.localize import incidents, localize
from gnssguard.model.train import Detectors, train

W0 = pd.Timestamp("2026-10-26T10:00Z")
W1 = W0 + pd.Timedelta(minutes=15)


def cell(lat, lon, w, usable, degraded, **kw):
    base = {c: 0 for c in FEATURES if c.startswith("n_")}
    base.update(cell_lat=lat, cell_lon=lon, window=w, n_aircraft=usable, n_ac_usable=usable,
                n_ac_degraded=degraded, share_ac_degraded=degraded / usable if usable else np.nan,
                share_ac_nacp_low=0.0, share_ac_altdev=np.nan, share_pts_degraded=0.0,
                mean_alt_ft=30000.0, max_alt_ft=35000.0, cell_deg=1.0, window_min=15)
    base.update(kw)
    return base


def test_add_context_neighbours_and_time():
    c = pd.DataFrame([
        cell(50, 10, W0, 4, 4),
        cell(50, 11, W0, 5, 1),      # east neighbour
        cell(51, 11, W0, 3, 3),      # north-east neighbour, hot
        cell(53, 10, W0, 9, 9),      # too far away
        cell(50, 10, W1, 6, 3),      # same cell, next window
    ])
    x = add_context(c).set_index(["cell_lat", "cell_lon", "window"])
    r = x.loc[(50, 10, W0)]
    assert r.nb_cells == 2 and r.nb_ac_usable == 8 and r.nb_ac_degraded == 4
    assert r.nb_hot == 1 and r.nb_max_share == 1.0
    assert r.next_n_ac_degraded == 3 and r.prev_n_ac_usable == 0
    assert np.isnan(r.prev_share_ac_degraded)
    assert x.loc[(50, 10, W1)].prev_n_ac_degraded == 4


def test_incidents_connect_in_space_and_time():
    f = pd.DataFrame({"cell_lat": [50, 51, 51, 58], "cell_lon": [10, 11, 11, 20],
                      "window": [W0, W0, W1, W0]})
    inc = incidents(f)
    ids = inc["incident_id"].tolist()
    assert ids[0] == ids[1] == ids[2] and ids[3] != ids[0]


def test_localize_recovers_synthetic_jammer():
    rng = np.random.default_rng(0)
    j_lat, j_lon, power = 55.0, 21.0, 220.0
    n = 1500
    lat, lon = _destination(np.full(n, j_lat), np.full(n, j_lon),
                            rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 450, n))
    alt = rng.uniform(3000, 40000, n)
    d = _haversine_km(lat, lon, j_lat, j_lon)
    degraded = d <= np.minimum(power, radio_horizon_km(alt, 30))
    pts = pd.DataFrame({"lat": lat, "lon": lon, "alt_ft": alt, "degraded": degraded})
    est = localize(pts)
    assert _haversine_km(est["lat"], est["lon"], j_lat, j_lon) < 25
    assert abs(est["range_km"] - power) <= 50
    assert est["uncertainty_km"] >= 11  # never finer than the grid


def test_train_predict_save_load(tmp_path):
    rng = np.random.default_rng(1)
    n = 3000
    df = pd.DataFrame({f: rng.random(n) for f in FEATURES})
    df["n_ac_usable"] = rng.integers(0, 10, n)
    df["y_jam"] = (df["share_ac_degraded"] > 0.6).astype(float)
    df["y_spoof"] = (df["n_ac_speed_mismatch"] > 0.9).astype(float)
    df["split"] = np.where(np.arange(n) < 2400, "train", "val")
    df["source"], df["weight"], df["date"] = "real", 1.0, "2026-10-20"
    det = train(df, rounds=200, early_stop=20)
    p = det.predict(df)
    assert set(p.columns) == {"p_jamming", "p_spoofing", "label"}
    acc = ((p["p_jamming"] >= 0.5) == df["y_jam"].astype(bool)).mean()
    assert acc > 0.95
    assert (p.loc[df["n_ac_usable"] == 0, "label"] == "unknown").all()
    det.save(tmp_path / "m")
    again = Detectors.load(tmp_path / "m").predict(df)
    assert np.allclose(again["p_jamming"], p["p_jamming"])


def test_hard_profile_differs_and_standard_is_unchanged():
    import sys

    sys.path.insert(0, "tests")
    from helpers import flight, points

    from gnssguard.labels.synthetic import random_events

    t0 = pd.Timestamp("2026-10-26T08:00Z")
    pts = points(*[flight(f"a{i}", 50 + i / 10, 10, alt=30000, t0=t0, n=200, step_s=60)
                   for i in range(5)])
    std = random_events(pts, 3, 3, seed=5)
    again = random_events(pts, 3, 3, seed=5, profile="standard")
    hard = random_events(pts, 3, 3, seed=5, profile="hard")
    assert [e.__dict__ for e in std] == [e.__dict__ for e in again]  # reproducible default
    dur = [(pd.Timestamp(e.end) - pd.Timestamp(e.start)).total_seconds() / 60 for e in hard]
    assert max(dur) <= 45
    jam = [e for e in hard if e.type == "jamming"]
    assert all(60 <= e.range_km <= 140 and e.core_frac == 0.25 for e in jam)
    assert all(e.mode == "drift" and e.drift_kt <= 150 for e in hard if e.type == "spoofing")


def test_wilson():
    lo, hi = wilson(45, 50)
    assert 0.78 < lo < 0.9 < hi < 0.97
    assert wilson(0, 0)[0] != wilson(0, 0)[0]  # nan


@pytest.mark.parametrize("n", [3])
def test_localize_needs_degraded_aircraft(n):
    pts = pd.DataFrame({"lat": [50.0] * n, "lon": [10.0] * n, "alt_ft": [30000.0] * n,
                        "degraded": [False] * n})
    assert localize(pts) is None
