import gzip

import numpy as np
import pandas as pd

from gnssguard.ground.check import agreement, compare, station_windows
from gnssguard.ground.rinex import ecef_to_latlon, read_snr

HEADER = """\
     3.04           OBSERVATION DATA    M                   RINEX VERSION / TYPE
LAMA                                                        MARKER NAME
  3524522.8000  1329693.5000  5129846.4000                  APPROX POSITION XYZ
G    4 C1C L1C D1C S1C                                      SYS / # / OBS TYPES
E    2 C1C S1C                                              SYS / # / OBS TYPES
                                                            END OF HEADER
"""


def obs(v):
    return f"{v:14.3f}  "


def rinex(tmp_path, epochs):
    lines = [HEADER]
    for t, snrs in epochs:
        lines.append(f"> {t:%Y %m %d %H %M} {t.second:10.7f}  0{len(snrs):3d}\n")
        for i, s in enumerate(snrs):
            lines.append(f"G{i + 1:02d}" + obs(2.2e7) + obs(1.1e8) + obs(-100.0) + obs(s) + "\n")
    f = tmp_path / "LAMA00POL_R_20263050000_01D_30S_MO.rnx.gz"
    f.write_bytes(gzip.compress("".join(lines).encode()))
    return f


def test_reads_s1c_and_position(tmp_path):
    t0 = pd.Timestamp("2026-11-01T00:00:00Z")
    f = rinex(tmp_path, [(t0, [45.0, 40.0, 0.0]), (t0 + pd.Timedelta(seconds=30), [44.0])])
    df, info = read_snr(f)
    assert len(df) == 3 and df["snr"].max() == 45.0  # zero means no reading
    assert info["snr_code"] == "S1C" and abs(info["lat"] - 53.89) < 0.05
    lat, lon = ecef_to_latlon(3524522.8, 1329693.5, 5129846.4)
    assert abs(lon - 20.67) < 0.05


def test_drop_against_same_window_of_other_days(tmp_path):
    epochs = []
    for d in range(3):
        t = pd.Timestamp("2026-11-01T00:00:00Z") + pd.Timedelta(days=d)
        level = 30.0 if d == 2 else 45.0  # day 3 jammed in its first window
        epochs.append((t, [level] * 8))
        epochs.append((t + pd.Timedelta(minutes=20), [45.0] * 8))
    w, _ = station_windows([rinex(tmp_path, epochs)])
    hit = w[w["ground_affected"]]
    assert len(hit) == 1 and hit["cn0_drop"].iloc[0] == 15.0


def test_agreement_and_day_shift():
    win = pd.date_range("2026-11-01", periods=4 * 96, freq="15min", tz="UTC")
    g = pd.DataFrame({"window": win, "cn0_drop": 0.0, "ground_affected": False})
    g.loc[[10, 200], "ground_affected"] = True
    g.loc[[10, 200], "cn0_drop"] = 6.0
    air = pd.DataFrame({"window": win, "judged": 5, "jam_share": 0.0})
    air.loc[[10, 200], "jam_share"] = 0.4
    r = agreement(g.merge(air, on="window"))
    assert r["aircraft_flag_when_ground_affected"] == 1.0 and r["aircraft_flag_otherwise"] == 0
    shifted = compare(g, air, shift_days=1)
    assert shifted["aircraft_flag_when_ground_affected"] == 0.0
    assert np.isclose(compare(g, air)["spearman_cn0_drop_vs_jam_share"], 1.0)


def test_reads_hatanaka_compressed(tmp_path):
    import pytest

    hatanaka = pytest.importorskip("hatanaka")
    t0 = pd.Timestamp("2026-11-01T00:00:00Z")
    plain = rinex(tmp_path, [(t0, [45.0, 40.0]), (t0 + pd.Timedelta(seconds=30), [44.0])])
    crx = hatanaka.rnx2crx(gzip.decompress(plain.read_bytes()))
    f = tmp_path / "LAMA00POL_R_20263050000_01D_30S_MO.crx.gz"
    f.write_bytes(gzip.compress(crx if isinstance(crx, bytes) else crx.encode()))
    df, _ = read_snr(f)
    assert df["snr"].tolist() == [45.0, 40.0, 44.0]


def test_fetch_takes_the_first_mirror_that_has_the_file(tmp_path):
    import datetime as dt

    from gnssguard.ground.fetch import fetch_day

    name = "LAMA00POL_R_20263050000_01D_30S_MO.crx.gz"

    class R:
        def __init__(self, text="", content=b""):
            self.text, self.content = text, content
            self.status_code, self.url = (200 if content else 404), ""

        def raise_for_status(self):
            pass

    class S:
        calls = []

        def get(self, url, **kw):
            self.calls.append(url)
            if "w3id.org" in url:
                return R(content=b"")  # no persistent id for this day
            if url.endswith(name):
                return R(content=b"x")
            if "bkg" in url:
                return R(text="<a>nothing here</a>")
            return R(text=f'<a href="{name}">{name}</a>')

    def no_ftp(*a, **k):
        raise OSError("offline")

    got = fetch_day(["LAMA00POL", "XXXX00POL"], dt.date(2026, 11, 1), tmp_path, S(), no_ftp)
    assert got["XXXX00POL"] == "missing" and got["LAMA00POL"].endswith(name)
    assert (tmp_path / "2026-11-01" / name).read_bytes() == b"x"


def test_fetch_uses_ftp_first(tmp_path):
    import datetime as dt

    from gnssguard.ground.fetch import fetch_day

    name = "LAMA00POL_R_20263050000_01D_30S_MO.crx.gz"

    class F:
        def __init__(self, host, timeout):
            assert host == "igs-ftp.bkg.bund.de"

        def login(self):
            pass

        def nlst(self, d):
            assert d == "/EUREF/obs/2026/305/"
            return [d + name, d + "OTHR00XXX_R_20263050000_01D_30S_MO.crx.gz"]

        def retrbinary(self, cmd, write):
            assert cmd.endswith(name)
            write(b"data")

        def quit(self):
            pass

    class NoHttp:
        def get(self, url, **k):
            assert "w3id.org" in url or "epncb.oma.be/pub/RINEX" in url, \
                "only the EUREF data centre is tried before FTP"

            class R:
                status_code, content, url, text = 404, b"", "", ""
            return R()

    got = fetch_day(["LAMA00POL"], dt.date(2026, 11, 1), tmp_path, NoHttp(), F)
    assert (tmp_path / "2026-11-01" / name).read_bytes() == b"data"
    assert "missing" not in got.values()


def test_fetch_follows_the_persistent_id(tmp_path):
    import datetime as dt

    from gnssguard.ground.fetch import fetch_day

    class R:
        status_code = 200
        content = b"\x1f\x8b" + b"x" * 2000
        url = "https://epncb.oma.be/x/LAMA00POL_R_20263050000_01D_30S_MO.crx.gz"

    class S:
        def get(self, url, **kw):
            assert url == "https://w3id.org/EUREF-HDC/2026/305/LAMA00POL_2026305_01D_30S_R3"
            return R()

    def no_ftp(*a, **k):
        raise AssertionError("FTP not needed")

    got = fetch_day(["LAMA00POL"], dt.date(2026, 11, 1), tmp_path, S(), no_ftp)
    assert got["LAMA00POL"].endswith("LAMA00POL_R_20263050000_01D_30S_MO.crx.gz")


def test_ftp_refused_is_skipped(tmp_path):
    import datetime as dt

    from gnssguard.ground.fetch import fetch_day

    class S:
        def get(self, url, **kw):
            class R:
                status_code, content, url, text = 404, b"", "", ""

                def raise_for_status(self):
                    pass
            return R()

    def eof(*a, **k):
        raise EOFError

    got = fetch_day(["LAMA00POL"], dt.date(2026, 11, 1), tmp_path, S(), eof)
    assert got == {"LAMA00POL": "missing"}


def test_fetch_follows_the_landing_page_link(tmp_path):
    import datetime as dt

    from gnssguard.ground.fetch import fetch_day

    name = "LAMA00POL_S_20263050000_01D_30S_MO.crx.gz"
    link = f"https://epncb.oma.be/pub/RINEX/2026/305/{name}"
    data = b"\x1f\x8b" + b"x" * 2000

    class R:
        def __init__(self, status, content=b"", text=""):
            self.status_code, self.content, self.text, self.url = status, content, text, ""

    class S:
        calls = []

        def get(self, url, **kw):
            self.calls.append(url)
            if "w3id.org" in url:
                return R(200, b"<!DOCTYPE html>", f'Download URL <a href="{link}">{name}</a>')
            return R(200, data) if url == link else R(404)

    def no_ftp(*a, **k):
        raise AssertionError("FTP not needed")

    s = S()
    got = fetch_day(["LAMA00POL"], dt.date(2026, 11, 1), tmp_path, s, no_ftp)
    assert got["LAMA00POL"].endswith(name) and s.calls == [s.calls[0], link]
    assert (tmp_path / "2026-11-01" / name).read_bytes() == data
    s.calls.clear()  # a second run reuses the saved file
    assert fetch_day(["LAMA00POL"], dt.date(2026, 11, 1), tmp_path, s, no_ftp) == got
    assert s.calls == []


def test_event_records_are_skipped(tmp_path):
    t0 = pd.Timestamp("2026-11-01T00:00:00Z")
    f = rinex(tmp_path, [(t0, [45.0, 40.0])])
    text = gzip.decompress(f.read_bytes()).decode()
    event = (">" + " " * 30 + "4  2\n"
             + "RECEIVER RESTARTED".ljust(60) + "COMMENT\n"
             + "G99 not an observation".ljust(60) + "COMMENT\n")
    later = (f"> {t0:%Y %m %d %H} 00 30.0000000  0  1\n"
             + "G05" + obs(2.2e7) + obs(1.1e8) + obs(-100.0) + obs(41.0) + "\n")
    f.write_bytes(gzip.compress((text + event + later).encode()))
    df, info = read_snr(f)
    assert df["snr"].tolist() == [45.0, 40.0, 41.0] and info["event_records"] == 1
