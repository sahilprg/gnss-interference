import datetime as dt
import json

import pyarrow.dataset as ds
from conftest import REGIONS

from gnssguard.ingest.pipeline import ingest_day, is_done, manifest_path
from gnssguard.ingest.releases import parse_preferred
from gnssguard.ingest.stream import MultiPartStream

DAY = dt.date(2026, 10, 26)


def test_multipart_stream_concatenates(split_archive):
    expected = b"".join(open(p, "rb").read() for p in split_archive)
    s = MultiPartStream(split_archive)
    got = b""
    while chunk := s.read(7777):
        got += chunk
    assert got == expected and s.total_read == len(expected)


def test_ingest_day_end_to_end(tmp_path, split_archive):
    out = tmp_path / "processed"
    m = ingest_day(DAY, REGIONS, out, sources=split_archive, max_decode_error_share=0.5)
    assert m["trace_files"] == 4
    assert m["decode_errors"] == 1
    assert m["rows_per_region"] == {"north": 4, "overlap": 2}
    assert m["complete_day"] is True
    assert is_done(out, DAY)
    assert not list(out.glob(".tmp-*"))

    table = ds.dataset(out, format="parquet", partitioning="hive").to_table()
    assert table.num_rows == 6
    assert set(table.column("region").to_pylist()) == {"north", "overlap"}


def test_ingest_skips_finished_day(tmp_path, split_archive):
    out = tmp_path / "processed"
    ingest_day(DAY, REGIONS, out, sources=split_archive, max_decode_error_share=0.5)
    before = manifest_path(out, DAY).read_text(encoding="utf-8")
    ingest_day(DAY, REGIONS, out, sources=["/does/not/exist"])  # must not open sources
    assert manifest_path(out, DAY).read_text(encoding="utf-8") == before


def test_sample_run_is_not_marked_done(tmp_path, split_archive):
    out = tmp_path / "processed"
    m = ingest_day(DAY, REGIONS, out, sources=split_archive, max_traces=1)
    assert m["complete_day"] is False
    assert not is_done(out, DAY)
    assert json.loads(manifest_path(out, DAY).read_text(encoding="utf-8"))["trace_files"] == 1


def test_too_many_decode_errors_mark_the_day_incomplete(tmp_path, split_archive):
    out = tmp_path / "processed"
    m = ingest_day(DAY, REGIONS, out, sources=split_archive)  # 1 of 4 files unreadable
    assert m["decode_errors"] == 1 and m["complete_day"] is False
    assert not is_done(out, DAY)


def test_changed_region_box_needs_a_new_run(tmp_path, split_archive):
    import dataclasses

    out = tmp_path / "processed"
    ingest_day(DAY, REGIONS, out, sources=split_archive, max_decode_error_share=0.5)
    assert is_done(out, DAY, REGIONS) and is_done(out, DAY, REGIONS[:1])
    moved = [dataclasses.replace(REGIONS[0], lat_max=REGIONS[0].lat_max + 1)]
    assert not is_done(out, DAY, moved)


def test_parse_preferred():
    base = "https://github.com/adsblol/globe_history_2026/releases/download"
    text = (
        f"{base}/v2026.10.26-planes-readsb-prod-0/v2026.10.26-planes-readsb-prod-0.tar.ab,"
        f"{base}/v2026.10.26-planes-readsb-prod-0/v2026.10.26-planes-readsb-prod-0.tar.aa\n"
        f"{base}/v2025.12.17-planes-readsb-mlatonly-0/v2025.12.17-planes-readsb-mlatonly-0.tar\n"
    )
    got = parse_preferred(text)
    assert list(got) == [DAY]
    assert got[DAY][0].endswith(".tar.aa")  # parts sorted into order
