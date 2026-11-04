import pandas as pd

from gnssguard.model.eventcheck import check, placebo, summarize, verdict

W = pd.Timestamp("2025-03-05T10:00Z")


def cells(n_inside=12, region="baltic"):
    rows = [{"region": region, "date": "2025-03-05", "cell_lat": 54.0, "cell_lon": 20.0 + i % 3,
             "window": W + pd.Timedelta(minutes=15 * i), "n_ac_usable": 5}
            for i in range(n_inside)]
    rows.append({"region": region, "date": "2025-03-05", "cell_lat": 58.0, "cell_lon": 27.0,
                 "window": W, "n_ac_usable": 5})  # outside the box
    rows.append({"region": "central_europe", "date": "2025-03-05", "cell_lat": 48.0,
                 "cell_lon": 9.0, "window": W, "n_ac_usable": 5})
    return pd.DataFrame(rows)


def events(kind="jamming", region="baltic"):
    return pd.DataFrame([{"event_id": "E1", "type": kind, "lat_min": 53.5, "lat_max": 55.5,
                          "lon_min": 19.5, "lon_max": 23.0,
                          "start_utc": pd.Timestamp("2025-03-05T00:00Z"),
                          "end_utc": pd.Timestamp("2025-03-06T00:00Z"),
                          "evidence_level": "observed_anomaly", "region": region}])


def test_verdict_thresholds():
    assert verdict(3, 3) == "found" and verdict(12, 2) == "not found"
    assert verdict(9, 2) == "too few cells"


def test_type_must_match_and_only_inside_counts():
    c = cells()
    jam = pd.Series(["jamming"] * 4 + ["normal"] * 8 + ["jamming", "normal"])
    spoof = pd.Series(["spoofing"] * 2 + ["normal"] * 10 + ["spoofing", "normal"])
    r = check(c, {"model": jam, "rules": spoof}, events("jamming"))
    got = dict(zip(r["predictor"], r["verdict"], strict=True))
    assert got == {"model": "found", "rules": "not found"}
    assert r.set_index("predictor").loc["model", "judged"] == 12
    r2 = check(c, {"rules": spoof}, events("unknown"))
    assert r2["verdict"].iloc[0] == "not found" and r2["matching"].iloc[0] == 2


def test_outside_regions_not_checked_and_summary_and_placebo():
    c = cells()
    lab = pd.Series(["jamming"] * len(c))
    r = check(c, {"model": lab}, pd.concat([events(), events(region="")]))
    assert list(r["verdict"]) == ["found", "not checked"]
    s = summarize(r)
    assert s[s["group"] == "all"]["found"].iloc[0] == 1
    p = placebo(c, {"model": lab})
    assert p["flagged_share"].iloc[0] == 1.0 and p["judged"].iloc[0] == 1
