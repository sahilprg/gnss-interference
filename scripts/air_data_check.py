"""Phase 2 exploration (no decision): do the points/2 air-data fields show spoofing?

For each day and region, per aircraft and cell-window (above 10,000 ft, ADS-B, ground
speed and true airspeed over 150 kt, air data under 10 s old):

* heading-track gap: |true heading - GNSS track|. Wind turns the track away from the
  heading by at most about 20 degrees for an airliner; a median gap over 45 degrees means
  the GNSS velocity does not match where the aircraft points.
* wind residual: readsb's derived wind against the median wind of other aircraft in the
  3x3 cells around it, same 4,000 ft band and window.

Usage: python scripts/air_data_check.py 2026-11-01 baltic,east_med,black_sea,punjab,central_europe
Needs the days stored in point format points/2 (docs/data.md).
"""

import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gnssguard.features.cells import load_cells
from gnssguard.model.dataset import with_context
from gnssguard.model.train import Detectors

COLS = ["icao", "ts", "lat", "lon", "alt_baro", "on_ground", "gs", "track", "src", "nic",
        "tas", "true_heading", "wind_dir", "wind_speed", "air_age_s"]


def aircraft_windows(day: str, region: str) -> pd.DataFrame:
    p = pq.read_table(f"data/processed/region={region}/date={day}", columns=COLS).to_pandas()
    p = p[~p["on_ground"].astype(bool) & p["src"].isin(["adsb_icao", "adsb_icao_nt"])
          & (p["alt_baro"] > 10000) & (p["gs"] > 150) & (p["tas"] > 150)
          & (p["air_age_s"] <= 10) & p["true_heading"].notna() & p["track"].notna()]
    p["gap"] = ((p["true_heading"] - p["track"] + 180) % 360 - 180).abs()
    wd = np.deg2rad(p["wind_dir"].to_numpy())
    p["wu"], p["wv"] = -p["wind_speed"] * np.sin(wd), -p["wind_speed"] * np.cos(wd)
    p["window"] = pd.to_datetime(p["ts"], unit="ms", utc=True).dt.floor("15min")
    p["cell_lat"], p["cell_lon"] = np.floor(p["lat"]), np.floor(p["lon"])
    p["band"] = (p["alt_baro"] // 4000).astype(int)
    a = p.groupby(["window", "cell_lat", "cell_lon", "band", "icao"]).agg(
        gap=("gap", "median"), nic=("nic", "median"), wu=("wu", "median"),
        wv=("wv", "median")).reset_index()
    a["resid"] = np.nan
    for _, g in a[a["wu"].notna()].groupby(["window", "band"]):
        la, lo = g["cell_lat"].to_numpy(), g["cell_lon"].to_numpy()
        near = (np.abs(la[:, None] - la) <= 1) & (np.abs(lo[:, None] - lo) <= 1)
        np.fill_diagonal(near, False)
        u, v = g["wu"].to_numpy(), g["wv"].to_numpy()
        for j, i in enumerate(g.index):
            if near[j].sum() >= 2:
                a.loc[i, "resid"] = np.hypot(u[j] - np.median(u[near[j]]),
                                             v[j] - np.median(v[near[j]]))
    return a


def main(day: str, regions: list[str]) -> None:
    det = Detectors.load("models/detector")
    out = []
    for region in regions:
        c = with_context(load_cells("data/cells", [day], [region])).reset_index(drop=True)
        c["label"] = det.predict(c)["label"].astype(str)
        c.loc[c["n_ac_usable"].fillna(0) < 3, "label"] = "not judged"
        c["window"] = pd.to_datetime(c["window"], utc=True)
        a = aircraft_windows(day, region).merge(
            c[["window", "cell_lat", "cell_lon", "label"]],
            on=["window", "cell_lat", "cell_lon"], how="left")
        a["region"] = region
        out.append(a)
    a = pd.concat(out, ignore_index=True)
    a["gap_over_45"] = a["gap"] > 45
    a["resid_over_40"] = a["resid"] > 40
    print(f"{day}: {len(a):,} aircraft-cell-windows with fresh air data")
    print(a.groupby(["region", "label"]).agg(
        aircraft_windows=("gap", "size"), gap_over_45=("gap_over_45", "sum"),
        wind_checked=("resid", "count"), resid_over_40=("resid_over_40", "sum"),
        median_resid_kt=("resid", "median")).round(1).to_string())
    big = a[a["gap_over_45"]]
    if len(big):
        print(f"\ngap over 45 deg: {big['icao'].nunique()} aircraft, median NIC "
              f"{big['nic'].median():.0f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","))
