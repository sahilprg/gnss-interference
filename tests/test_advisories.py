import pandas as pd
import pytest

from gnssguard.labels.advisories import advisory_check, load_advisories

HEADER = ("advisory_id,reference,issuer,lat_min,lat_max,lon_min,lon_max,start_utc,end_utc,"
          "source_url,notes\n")


def test_load_and_check(tmp_path):
    p = tmp_path / "adv.csv"
    p.write_text(HEADER + "A1,NOTAM X,Testland,50,52,10,12,2026-10-01T00:00:00Z,,"
                 "https://example.org,approximate\n"
                 "A2,NOTAM Y,Testland,50,52,10,12,2026-11-01T00:00:00Z,,https://example.org,\n",
                 encoding="utf-8")
    adv = load_advisories(p)
    assert pd.isna(adv.loc[0, "end_utc"])  # still in force

    labels = pd.DataFrame({
        "region": "r", "date": "2026-10-26", "cell_deg": 1.0,
        "cell_lat": [50.0, 51.0, 54.0, 55.0, 56.0, 57.0],
        "cell_lon": [10.0, 11.0, 10.0, 10.0, 10.0, 10.0],
        "label": ["jamming", "normal", "normal", "normal", "jamming", "unknown"]})
    out = advisory_check(labels, adv)
    assert list(out["advisory_id"]) == ["A1"]  # A2 starts after the data
    r = out.iloc[0]
    assert (r.judged_inside, r.flagged_inside) == (2, 1)
    assert (r.judged_outside, r.flagged_outside) == (3, 1)  # 'unknown' not judged
    assert r.ratio == pytest.approx(1.5)


def test_validation(tmp_path):
    p = tmp_path / "adv.csv"
    p.write_text(HEADER + "A1,N,T,52,50,10,12,2026-10-01T00:00:00Z,,https://example.org,\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="lat_min < lat_max"):
        load_advisories(p)
    p.write_text(HEADER + "A1,N,T,50,52,10,12,2026-10-01T00:00:00Z,,,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source_url"):
        load_advisories(p)


def test_gold_tool_payload(tmp_path):
    import json
    import re
    import sys

    sys.path.insert(0, "tests")
    from helpers import flight, points

    from gnssguard.features.cells import build_all
    from gnssguard.labels.goldtool import build_payload, write_tool

    proc = tmp_path / "processed" / "region=r" / "date=2026-10-26"
    proc.mkdir(parents=True)
    df = points(*[flight(f"a{i}", 50.1 + i / 10, 10.1, nic=0 if i < 2 else 8) for i in range(4)])
    df.to_parquet(proc / "part-0.parquet")
    build_all(tmp_path / "processed", tmp_path / "cells")
    gold = tmp_path / "gold.csv"
    gold.write_text("gold_id,region,date,cell_lat,cell_lon,window,label,labeler,notes\n"
                    "G0001,r,2026-10-26,50.0,10.0,2026-10-26 10:00:00+00:00,,,\n", encoding="utf-8")
    data = build_payload(gold, tmp_path / "cells")
    item = data["items"][0]
    focal = [r for r in item["ctx"] if r[:3] == [0, 0, 1]][0]
    usable, degraded = focal[3 + data["metrics"].index("n_ac_usable")], \
        focal[3 + data["metrics"].index("n_ac_degraded")]
    assert (usable, degraded) == (4, 2)
    assert data["rows"][0]["window"] == "2026-10-26 10:00:00+00:00"  # kept verbatim
    html = write_tool(gold, tmp_path / "cells", tmp_path / "tool.html").read_text(encoding="utf-8")
    blob = re.search(r'id="data">(.*?)</script>', html, re.S).group(1)
    assert json.loads(blob.replace("<\\/", "</"))["file_id"] == data["file_id"]
    assert "p_jamming" not in html and "spoof_votes" not in html  # blind


def test_shipped_file_loads():
    adv = load_advisories("configs/advisories.csv")
    assert len(adv) == 7 and adv["advisory_id"].is_unique
