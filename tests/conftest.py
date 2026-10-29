import gzip
import io
import json
import tarfile

import pytest

from gnssguard.config import Region

BASE_TS = 1792972800.0  # 2026-10-26 00:00:00 UTC

REGIONS = [
    Region("north", 50.0, 52.0, 10.0, 12.0),
    Region("overlap", 51.0, 53.0, 11.0, 13.0),
]


def make_trace(icao: str, points: list) -> dict:
    return {"icao": icao, "t": "A320", "dbFlags": 0, "timestamp": BASE_TS, "trace": points}


def det(nic: int, nac_p: int, **extra) -> dict:
    return {"type": "adsb_icao", "nic": nic, "nac_p": nac_p, "sil": 3, "rc": 186,
            "version": 2, **extra}


@pytest.fixture
def sample_traces():
    a = make_trace("ABC123", [
        # outside both regions; carries quality that must be forward-filled
        [0.0, 49.0, 9.0, 35000, 450.0, 90.0, 0, 0, det(8, 9), "adsb_icao", 35100],
        [10.0, 50.5, 10.5, 35000, 450.0, 90.0, 0, 0, None, "adsb_icao", 35100],  # north
        [20.0, 51.5, 11.5, 35000, 450.0, 90.0, 1, 0, det(0, 0), "adsb_icao", 35100],  # both
        [30.0, 51.6, 11.6, "ground", 5.0, 90.0, 2, None, None, "adsb_icao", None],
    ])
    b = make_trace("def456", [
        [5.0, 50.1, 10.1, 12000, 300.0, 180.0, 8, 0, None, "mlat", None],  # geometric alt
        [6.0, None, None, 12000, None, None, 0, 0, None, "mode_s", None],  # no position
    ])
    c = make_trace("aaaaaa", [[0.0, 10.0, 10.0, 1000, 100.0, 0.0, 0, 0, None, "adsb_icao"]])
    return [a, b, c]


@pytest.fixture
def split_archive(tmp_path, sample_traces):
    """Tar with heatmap junk + traces, split into two parts like adsb.lol releases."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        def add(name, data):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        add("./heatmap/00.bin.ttf", b"\x00" * 300_000)
        add("./acas/acas.csv.gz", gzip.compress(b"x"))
        for tr in sample_traces:
            icao = tr["icao"].lower()
            add(f"./traces/{icao[-2:]}/trace_full_{icao}.json",
                gzip.compress(json.dumps(tr).encode()))
        add("./traces/zz/trace_full_broken.json", b"\x1f\x8bnot-gzip")
    data = buf.getvalue()
    cut = len(data) // 2 + 123  # deliberately not on a tar block boundary
    p1, p2 = tmp_path / "day.tar.aa", tmp_path / "day.tar.ab"
    p1.write_bytes(data[:cut])
    p2.write_bytes(data[cut:])
    return [str(p1), str(p2)]
