import pandas as pd
from helpers import T0, flight, points

from gnssguard.features.cells import GridSpec, build_from_points

SPEC = GridSpec(1.0, 15)


def one_window(cw):
    assert len(cw) == 1, cw
    return cw.iloc[0]


def test_counts_distinct_aircraft_and_degraded_share():
    df = points(
        flight("a1", 50.1, 10.1, nic=0, nac_p=0),
        flight("a2", 50.2, 10.1, nic=0, nac_p=0),
        flight("a3", 50.3, 10.1, nic=8),
        flight("a4", 50.4, 10.1, nic=8),
    )
    c = one_window(build_from_points(df, SPEC)[0])
    assert c.n_aircraft == 4 and c.n_ac_usable == 4
    assert c.n_ac_degraded == 2 and c.n_ac_nic0 == 2 and c.n_ac_nacp_low == 2
    assert c.share_ac_degraded == 0.5
    assert c.n_points == 120


def test_low_altitude_and_stale_quality_are_not_usable():
    df = points(flight("a1", 50.1, 10.1, alt=2000), flight("a2", 50.2, 10.1))
    df.loc[df.icao == "a2", "q_age_s"] = 90.0
    c = one_window(build_from_points(df, SPEC)[0])
    assert c.n_aircraft == 2 and c.n_ac_usable == 0


def test_chronic_aircraft_excluded():
    # "bad" reports NIC 0 in four windows where three other aircraft are healthy
    frames = []
    for k in range(4):
        t = T0 + pd.Timedelta(minutes=15 * k)
        frames += [flight("bad", 50.1, 10.1, nic=0, t0=t, n=5),
                   *[flight(f"ok{k}{j}", 50.2 + j / 10, 10.1, t0=t, n=5) for j in range(3)]]
    cw, chronic = build_from_points(points(*frames), SPEC)
    assert chronic == {"bad"}
    assert (cw.n_ac_degraded == 0).all() and (cw.n_chronic == 1).all()


def test_mlat_fallback_counts_as_quality_reading():
    a = flight("a1", 50.1, 10.1, nic=0)
    a.loc[10:, "src"] = "mlat"  # same aircraft, positions now from MLAT, NIC still 0
    c = one_window(build_from_points(points(a, flight("a2", 50.2, 10.1)), SPEC)[0])
    assert c.n_ac_mlat_fallback == 1
    assert c.n_ac_usable == 2 and c.n_ac_degraded == 1


def test_jump_and_speed_mismatch():
    a = flight("a1", 50.1, 10.1)
    a.loc[15:, "lon"] += 0.6  # ~43 km in 10 s: impossible
    b = flight("b1", 50.5, 10.1, n=10, dlon=0.04, gs=250)  # positions imply ~550 kt
    c = one_window(build_from_points(points(a, b), SPEC)[0])
    assert c.n_ac_jump == 1
    assert c.n_ac_speed_mismatch == 2  # the jump pair also disagrees with ground speed


def test_altitude_deviation_against_local_norm():
    normal = [flight(f"n{i}", 50.1 + i / 20, 10.1, alt_geom_offset=600) for i in range(6)]
    odd = flight("odd", 50.9, 10.1, alt_geom_offset=3000)
    c = one_window(build_from_points(points(*normal, odd), SPEC)[0])
    assert c.n_ac_altpair == 7 and c.n_ac_altdev == 1


def test_no_altitude_baseline_from_too_few_aircraft():
    few = [flight(f"n{i}", 50.1 + i / 20, 10.1) for i in range(3)]
    odd = flight("odd", 50.9, 10.1, alt_geom_offset=3000)
    c = one_window(build_from_points(points(*few, odd), SPEC)[0])
    assert c.n_ac_altpair == 0
