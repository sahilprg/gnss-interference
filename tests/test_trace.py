from conftest import BASE_TS, REGIONS

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


def test_empty_trace():
    assert list(iter_rows({"timestamp": BASE_TS, "trace": []}, REGIONS)) == []
