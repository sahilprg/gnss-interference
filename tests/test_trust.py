import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gnssguard.model.trust import aircraft_overlap, day_bootstrap, reliability_by


def test_day_bootstrap_uses_whole_days():
    v = pd.Series([1.0, 1.0, 1.0, 0.0])
    d = pd.Series(["a", "a", "a", "b"])  # day a mean 1, day b mean 0
    m, lo, hi = day_bootstrap(v, d)
    assert m == 0.5 and lo == 0.0 and hi == 1.0


def test_reliability_bands_and_groups():
    m = pd.DataFrame({"score": [0.0, 0.01, 0.95, 0.99], "human": [False, True, True, True],
                      "n_ac_usable": [4, 12, 4, 12], "region": ["a", "a", "b", "b"]})
    r = reliability_by(m, None).set_index("score_band")
    assert r.loc["<0.1", "cells"] == 2 and r.loc["<0.1", "human_interference_share"] == 0.5
    assert r.loc[">=0.9", "human_interference_share"] == 1.0
    a = reliability_by(m, "aircraft")
    assert set(a["aircraft"]) == {"3-5", "11+"}


def test_aircraft_overlap(tmp_path):
    for day, icaos in (("2026-10-01", ["a", "b"]), ("2026-10-02", ["b", "c", "d", "e"])):
        f = tmp_path / "region=x" / f"date={day}"
        f.mkdir(parents=True)
        pq.write_table(pa.table({"icao": icaos}), f / "part-0.parquet")
    ov = aircraft_overlap(tmp_path, ["2026-10-01"], ["2026-10-02"], ["x"])
    assert ov["aircraft"].iloc[0] == 4 and ov["seen_in_training"].iloc[0] == 0.25
