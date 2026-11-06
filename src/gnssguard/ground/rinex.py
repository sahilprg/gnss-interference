"""Minimal RINEX 3 observation reader: GPS L1 C/A signal strength (S1C) per satellite
and epoch, and the station position from the header. Enough for a C/N0 time series; not
a general RINEX library."""

from __future__ import annotations

import gzip
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd

SNR_CODES = ("S1C", "S1W", "S1X")  # first one present is used


def ecef_to_latlon(x: float, y: float, z: float) -> tuple[float, float]:
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - e2))
    for _ in range(5):
        n = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
        lat = math.atan2(z + e2 * n * math.sin(lat), p)
    return math.degrees(lat), math.degrees(lon)


def _open_text(path: Path) -> io.TextIOBase:
    """Decompress by content, not by name: gzip, Unix compress (.Z) and Hatanaka."""
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    elif raw[:2] == b"\x1f\x9d":
        import ncompress

        raw = ncompress.decompress(raw)
    if b"COMPACT RINEX" in raw[:200]:
        import hatanaka

        raw = hatanaka.crx2rnx(raw)
        raw = raw.encode() if isinstance(raw, str) else raw
    return io.StringIO(raw.decode("ascii", errors="replace"))


def read_snr(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Return (rows of time, sat, snr) for GPS, and header info {lat, lon, marker}."""
    fh = _open_text(Path(path))
    gps_types: list[str] = []
    info: dict = {}
    pending_g = 0
    for line in fh:
        label = line[60:].strip()
        if label == "SYS / # / OBS TYPES":
            if line[0] == "G":
                pending_g = int(line[3:6])
                gps_types = line[7:58].split()
            elif line[0] == " " and pending_g > len(gps_types):
                gps_types += line[7:58].split()
        elif label == "APPROX POSITION XYZ":
            x, y, z = (float(v) for v in line[:42].split())
            info["lat"], info["lon"] = ecef_to_latlon(x, y, z)
        elif label == "MARKER NAME":
            info["marker"] = line[:60].strip()
        elif label == "END OF HEADER":
            break
    code = next((c for c in SNR_CODES if c in gps_types), None)
    info["snr_code"] = code
    if code is None:
        return pd.DataFrame(columns=["time", "sat", "snr"]), info
    k = gps_types.index(code)
    times, sats, vals = [], [], []
    t = None
    skip = 0
    info["event_records"] = info["bad_epochs"] = 0
    for line in fh:
        if skip:  # header lines carried by an event record
            skip -= 1
            continue
        if line.startswith(">"):
            flag = line[31:32].strip()
            if flag and flag in "2345":
                # event record (antenna moved, new site, header lines follow, external
                # event): time fields may be blank; the next n lines are not observations
                info["event_records"] += 1
                try:
                    skip = int(line[32:35])
                except ValueError:
                    skip = 0
                t = None
                continue
            try:
                f = line[1:].split()
                sec = float(f[5])
                t = pd.Timestamp(int(f[0]), int(f[1]), int(f[2]), int(f[3]), int(f[4]),
                                 int(sec), tz="UTC")
            except (IndexError, ValueError):
                info["bad_epochs"] += 1
                t = None
            if flag == "6":  # cycle-slip records repeat satellites, not new readings
                t = None
            continue
        if t is None or not line.startswith("G"):
            continue
        field = line[3 + 16 * k: 3 + 16 * k + 14].strip()
        if field:
            try:
                v = float(field)
            except ValueError:
                continue
            if v > 0:
                times.append(t)
                sats.append(line[:3])
                vals.append(v)
    df = pd.DataFrame({"time": times, "sat": sats, "snr": np.asarray(vals, dtype=float)})
    return df, info
