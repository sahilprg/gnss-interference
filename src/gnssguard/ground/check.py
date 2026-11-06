"""Ground check 1 (docs/evaluation_protocol.md): EUREF station C/N0 against the map.

Per station and 15-minute window: median GPS L1 C/A C/N0 and satellites tracked, against
their usual value (median of the same window of the day over the checked days). A window
is ground-affected if C/N0 is at least DROP_DB below usual or satellites tracked at least
SATS_DROP below usual. The aircraft side is the share of judged cell-windows within
RADIUS_KM of the station labeled jamming or both by the published model.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from gnssguard.ground.rinex import read_snr

log = logging.getLogger(__name__)

DROP_DB = 3.0
SATS_DROP = 0.30
RADIUS_KM = 150.0
WINDOW = "15min"


def station_windows(files: list[Path]) -> tuple[pd.DataFrame, dict]:
    frames, info = [], {}
    for f in files:
        try:
            df, inf = read_snr(f)
        except Exception as e:  # one unreadable file is reported, not fatal
            log.warning("%s: unreadable (%s: %s), left out", f, type(e).__name__, e)
            continue
        if inf.get("event_records") or inf.get("bad_epochs"):
            log.info("%s: %d event records, %d unreadable epoch lines skipped", f,
                     inf.get("event_records", 0), inf.get("bad_epochs", 0))
        info = info or inf
        frames.append(df)
    d = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if d.empty:
        return pd.DataFrame(), info
    per_epoch = d.groupby("time").agg(cn0=("snr", "median"), sats=("sat", "nunique"))
    per_epoch["window"] = per_epoch.index.floor(WINDOW)
    w = per_epoch.groupby("window").agg(cn0=("cn0", "median"), sats=("sats", "median"))
    w = w.reset_index()
    w["slot"] = w["window"].dt.hour * 4 + w["window"].dt.minute // 15
    usual = w.groupby("slot")[["cn0", "sats"]].median().rename(
        columns={"cn0": "cn0_usual", "sats": "sats_usual"})
    w = w.join(usual, on="slot")
    w["cn0_drop"] = w["cn0_usual"] - w["cn0"]
    w["sats_drop"] = 1 - w["sats"] / w["sats_usual"].clip(lower=1)
    w["ground_affected"] = (w["cn0_drop"] >= DROP_DB) | (w["sats_drop"] >= SATS_DROP)
    return w, info


def km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


def aircraft_windows(cells: pd.DataFrame, labels: pd.Series, lat: float, lon: float,
                     cell_deg: float = 1.0) -> pd.DataFrame:
    dense = cells["n_ac_usable"].fillna(0) >= 3
    d = km(lat, lon, cells["cell_lat"] + cell_deg / 2, cells["cell_lon"] + cell_deg / 2)
    near = dense & (d <= RADIUS_KM)
    c = cells[near].assign(jam=labels[near].isin(["jamming", "both"]).to_numpy())
    c["window"] = pd.to_datetime(c["window"], utc=True)
    return c.groupby("window").agg(judged=("jam", "size"), jam_share=("jam", "mean")
                                   ).reset_index()


def agreement(m: pd.DataFrame) -> dict:
    """How often aircraft flag nearby when the ground is affected vs when it is not."""
    m = m[m["judged"] > 0]
    a, n = m[m["ground_affected"]], m[~m["ground_affected"]]
    pa = float((a["jam_share"] > 0).mean()) if len(a) else None
    pn = float((n["jam_share"] > 0).mean()) if len(n) else None
    rho = m["cn0_drop"].corr(m["jam_share"], method="spearman") if len(m) > 2 else None
    return {"windows": len(m), "ground_affected": len(a),
            "aircraft_flag_when_ground_affected": None if pa is None else round(pa, 3),
            "aircraft_flag_otherwise": None if pn is None else round(pn, 3),
            "ratio": round(pa / pn, 2) if pa is not None and pn else None,
            "spearman_cn0_drop_vs_jam_share": None if rho is None or np.isnan(rho)
            else round(float(rho), 3)}


def compare(ground: pd.DataFrame, air: pd.DataFrame, shift_days: int = 0) -> dict:
    return agreement(shift_merge(ground, air, shift_days))


def shift_merge(ground: pd.DataFrame, air: pd.DataFrame, shift_days: int = 0) -> pd.DataFrame:
    a = air.copy()
    if shift_days:
        days = sorted(a["window"].dt.normalize().unique())
        span = len(days)
        a["window"] = a["window"] + pd.Timedelta(days=shift_days)
        last = days[-1] + pd.Timedelta(days=1)
        wrap = a["window"] >= last
        a.loc[wrap, "window"] -= pd.Timedelta(days=span)
    return ground.merge(a, on="window", how="inner")


def run(station_files: dict[str, list[Path]], cells: pd.DataFrame, labels: pd.Series,
        out: Path) -> pd.DataFrame:
    rows, merged, pairs = [], [], []
    for st, files in station_files.items():
        g, info = station_windows(files)
        if g.empty or "lat" not in info:
            rows.append({"station": st, "status": "no data"})
            continue
        air = aircraft_windows(cells, labels, info["lat"], info["lon"])
        pairs.append((g, air))
        res = compare(g, air)
        shifted = [compare(g, air, k)["ratio"] for k in range(1, 10)]
        shifted = [x for x in shifted if x is not None]
        rows.append({"station": st, "status": "ok", "lat": round(info["lat"], 3),
                     "lon": round(info["lon"], 3), "snr_code": info.get("snr_code"),
                     "days": g["window"].dt.normalize().nunique(), **res,
                     "ratio_days_shifted_median": round(float(np.median(shifted)), 2)
                     if shifted else None})
        mm = g.merge(air, on="window", how="inner")
        mm["station"] = st
        merged.append(mm)
    res = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "ground_check.csv", index=False)
    if merged:
        allm = pd.concat(merged, ignore_index=True)
        allm.to_csv(out / "ground_windows.csv", index=False)
        shifted = []
        for k in range(1, 10):  # every station shifted by the same k days, then pooled
            parts = [shift_merge(g, air, k) for g, air in pairs]
            r = agreement(pd.concat(parts, ignore_index=True))["ratio"]
            if r is not None:
                shifted.append(r)
        pooled = {"station": "all", "status": "pooled", **agreement(allm),
                  "ratio_days_shifted_median": round(float(np.median(shifted)), 2)
                  if shifted else None,
                  "ratio_days_shifted_max": max(shifted) if shifted else None}
        res = pd.concat([res, pd.DataFrame([pooled])], ignore_index=True)
        res.to_csv(out / "ground_check.csv", index=False)
    return res
