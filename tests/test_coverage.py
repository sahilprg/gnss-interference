import datetime as dt
import shutil

from conftest import REGIONS

from gnssguard.analysis.coverage import scan
from gnssguard.ingest.pipeline import ingest_day


def test_scan_accumulates_across_days(tmp_path, split_archive):
    out = tmp_path / "processed"
    ingest_day(dt.date(2026, 10, 26), REGIONS, out, sources=split_archive)
    one, _ = scan(out)

    # duplicate the day under a second date: counts double, distinct aircraft do not
    for region_dir in out.glob("region=*"):
        shutil.copytree(region_dir / "date=2026-10-26", region_dir / "date=2026-10-27")
    two, hourly = scan(out)

    a, b = one.set_index("region"), two.set_index("region")
    assert (b["points"] == 2 * a["points"]).all()
    assert (b["aircraft"] == a["aircraft"]).all()
    assert (b["days"] == 2).all()
    assert a.loc["north", "points"] == 4 and b.loc["north", "points"] == 8
    assert set(hourly.columns) == {"region", "hour", "aircraft", "points", "share_degraded"}


def test_scan_filters(tmp_path, split_archive):
    out = tmp_path / "processed"
    ingest_day(dt.date(2026, 10, 26), REGIONS, out, sources=split_archive)
    report, _ = scan(out, regions=["overlap"])
    assert report["region"].tolist() == ["overlap"]
    empty, _ = scan(out, dates=["2020-01-01"])
    assert empty.empty
