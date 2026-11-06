"""Phase 2 exploration (no decision): position-spoofing signatures on real days.

Per cell-window, counted over distinct aircraft (ADS-B positions, airborne):

* converge: the aircraft reports a position within CONVERGE_M of another aircraft's
  position at the same time (within CONVERGE_S) while their barometric altitudes differ
  by more than CONVERGE_FT: two aircraft cannot be there, so at least one position is
  false (several aircraft pulled to one spoofed point).
* loop: at over 10,000 ft and over 150 kt, the course derived from successive positions
  turns by more than LOOP_DEG in total within LOOP_MIN minutes (spoofed circles; real
  holding patterns are slower and lower, and are reported with a matching track).
* offset_jump: the aircraft's GPS-minus-baro altitude differs from its own median for
  the day by more than OFFSET_FT (a spoofed position usually carries a false height).

Usage: python scripts/position_spoof_check.py 2026-01-15 delhi
"""

import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gnssguard.features.cells import load_cells
from gnssguard.model.dataset import with_context
from gnssguard.model.train import Detectors

CONVERGE_M, CONVERGE_S, CONVERGE_FT = 300.0, 5, 1000
LOOP_DEG, LOOP_MIN = 300.0, 10
OFFSET_FT = 1000
COLS = ["icao", "ts", "lat", "lon", "alt_baro", "alt_geom", "on_ground", "gs", "src"]


def load(day, region):
    p = pq.read_table(f"data/processed/region={region}/date={day}", columns=COLS).to_pandas()
    p = p[~p["on_ground"].astype(bool) & p["src"].isin(["adsb_icao", "adsb_icao_nt"])]
    p = p.sort_values(["icao", "ts"]).reset_index(drop=True)
    ts = pd.to_datetime(p["ts"], utc=True)
    p["t"] = (ts - pd.Timestamp(0, tz="UTC")).dt.total_seconds().astype("int64")
    p["window"] = ts.dt.floor("15min")
    p["cell_lat"], p["cell_lon"] = np.floor(p["lat"]), np.floor(p["lon"])
    return p


def converge(p):
    """Aircraft-windows whose position coincides with a different aircraft's."""
    q = p[p["alt_baro"].notna()].copy()
    q["tb"] = q["t"] // CONVERGE_S
    # coarse spatial bucket of about 1 km, then exact distance within neighbours
    q["xb"], q["yb"] = (q["lon"] * 100).astype(int), (q["lat"] * 100).astype(int)
    hits = set()
    for _, g in q.groupby(["tb", "yb"]):
        if g["icao"].nunique() < 2:
            continue
        lat, lon = g["lat"].to_numpy(), g["lon"].to_numpy()
        alt, ic = g["alt_baro"].to_numpy(), g["icao"].to_numpy()
        dy = (lat[:, None] - lat) * 111_320
        dx = (lon[:, None] - lon) * 111_320 * np.cos(np.deg2rad(lat[:, None]))
        close = (np.hypot(dx, dy) < CONVERGE_M) & (np.abs(alt[:, None] - alt) > CONVERGE_FT) \
            & (ic[:, None] != ic)
        for i in np.flatnonzero(close.any(1)):
            hits.add(g.index[i])
    p["converge"] = p.index.isin(list(hits))
    return p


def loops(p):
    lat, lon = np.deg2rad(p["lat"].to_numpy()), np.deg2rad(p["lon"].to_numpy())
    same = p["icao"].eq(p["icao"].shift()).to_numpy()
    dlon = np.diff(lon, prepend=lon[0])
    y = np.sin(dlon) * np.cos(lat)
    x = np.cos(np.roll(lat, 1)) * np.sin(lat) - np.sin(np.roll(lat, 1)) * np.cos(lat) * np.cos(dlon)
    course = np.rad2deg(np.arctan2(y, x))
    p["course"] = np.where(same, course, np.nan)
    turn = ((p["course"] - p["course"].shift() + 180) % 360 - 180).abs()
    ok = same & np.roll(same, 1) & (p["alt_baro"] > 10000).to_numpy() & (p["gs"] > 150).to_numpy()
    p["turn"] = turn.where(ok, 0).fillna(0)
    p["minute_bin"] = p["t"] // (LOOP_MIN * 60)
    tot = p.groupby(["icao", "minute_bin"])["turn"].transform("sum")
    p["loop"] = tot > LOOP_DEG
    return p


def offsets(p):
    off = p["alt_geom"] - p["alt_baro"]
    med = off.groupby(p["icao"]).transform("median")
    p["offset_jump"] = ((off - med).abs() > OFFSET_FT) & (p["alt_baro"] > 10000)
    return p


def main(day, regions):
    det = Detectors.load("models/detector")
    out = []
    for region in regions:
        c = with_context(load_cells("data/cells", [day], [region])).reset_index(drop=True)
        c["label"] = det.predict(c)["label"].astype(str)
        c.loc[c["n_ac_usable"].fillna(0) < 3, "label"] = "not judged"
        c["window"] = pd.to_datetime(c["window"], utc=True)
        p = offsets(loops(converge(load(day, region))))
        a = p.groupby(["window", "cell_lat", "cell_lon", "icao"])[
            ["converge", "loop", "offset_jump"]].any().reset_index()
        cw = a.groupby(["window", "cell_lat", "cell_lon"])[
            ["converge", "loop", "offset_jump"]].sum().reset_index()
        m = c[["window", "cell_lat", "cell_lon", "label"]].merge(
            cw, on=["window", "cell_lat", "cell_lon"], how="left").fillna(0)
        m["region"] = region
        out.append(m)
    m = pd.concat(out, ignore_index=True)
    for k in ("converge", "loop", "offset_jump"):
        m[f"{k}_2plus"] = m[k] >= 2
    print(f"{day}")
    print(m.groupby(["region", "label"]).agg(
        cells=("label", "size"), converge_any=("converge", lambda s: (s > 0).mean()),
        converge_2plus=("converge_2plus", "mean"), loop_any=("loop", lambda s: (s > 0).mean()),
        offset_any=("offset_jump", lambda s: (s > 0).mean()),
        offset_2plus=("offset_jump_2plus", "mean")).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","))
