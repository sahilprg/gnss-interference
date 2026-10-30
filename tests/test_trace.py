from conftest import BASE_TS, REGIONS, det, make_trace

from gnssguard.ingest.trace import COLUMNS, iter_rows


def rows_by_region(trace):
    out = {}
    for region, row in iter_rows(trace, REGIONS):
        out.setdefault(region, []).append(dict(zip(COLUMNS, row, strict=True)))
    return out


def test_region_filter_and_overlap(sample_traces):
    got = rows_by_region(sample_traces[0])
    assert len(got["north"]) == 3  # points at 50.5, 51.5 and 51.6
    assert len(got["overlap"]) == 2  # points at 51.5 and 51.6


def test_quality_forward_fill_and_age(sample_traces):
    north = rows_by_region(sample_traces[0])["north"]
    first, second = north[:2]
    # first in-region point has no details; inherits nic=8 from the point 10 s earlier
    assert first["nic"] == 8 and first["nac_p"] == 9
    assert first["q_age_s"] == 10.0
    # second point reports new quality itself
    assert second["nic"] == 0 and second["q_age_s"] == 0.0
    assert second["stale"] is True


def test_ground_and_flags(sample_traces):
    last = rows_by_region(sample_traces[0])["overlap"][-1]
    assert last["on_ground"] is True and last["alt_baro"] is None
    assert last["new_leg"] is True
    assert last["icao"] == "abc123"
    assert last["ts"] == int((BASE_TS + 30) * 1000)


def test_geometric_altitude_flag_and_missing_position(sample_traces):
    rows = rows_by_region(sample_traces[1])["north"]
    assert len(rows) == 1  # the point without lat/lon is dropped
    r = rows[0]
    assert r["alt_baro"] is None and r["alt_geom"] == 12000
    assert r["src"] == "mlat"
    assert r["nic"] is None and r["q_age_s"] is None


def test_no_hits_outside_regions(sample_traces):
    assert rows_by_region(sample_traces[2]) == {}


def test_drops_non_forward_points_but_keeps_their_quality():
    pts = [
        [0.0, 50.5, 10.5, 30000, 400.0, 90.0, 0, 0, None, "adsb_icao"],
        [0.0, 50.5, 10.5, 30000, 400.0, 90.0, 0, 0, det(3, 4), "adsb_icao"],    # duplicate
        [0.2, 50.5, 10.5, 30000, 400.0, 90.0, 0, 0, None, "adsb_icao"],         # kept
        [0.1, 50.5, 10.6, 30000, 400.0, 90.0, 0, 0, None, "adsb_icao"],         # backwards
        [1.0, 50.5, 10.7, 30000, 400.0, 90.0, 0, 0, None, "adsb_icao"],
    ]
    stats = {}
    rows = [dict(zip(COLUMNS, r, strict=True))
            for _, r in iter_rows(make_trace("abc", pts), REGIONS[:1], stats=stats)]
    assert [r["ts"] for r in rows] == [int((BASE_TS + s) * 1000) for s in (0.0, 0.2, 1.0)]
    assert stats["points_not_forward"] == 2
    assert rows[1]["nic"] == 3  # NIC from the dropped duplicate is still carried forward


def test_empty_trace():
    assert list(iter_rows({"timestamp": BASE_TS, "trace": []}, REGIONS)) == []
