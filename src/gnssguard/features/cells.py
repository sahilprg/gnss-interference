"""Cell-window feature table: the unit the rules and the detector work on.

Each row is one grid cell (default 1 deg x 1 deg) in one time window (default 15 min)
of one region-day. Counts are of distinct aircraft, not points, so one chatty aircraft
cannot dominate a cell.

Steps for one region-day:
1. Point flags: usable quality reading, degraded NIC, low NACp, impossible jumps,
   MLAT fallback, GPS-vs-baro altitude deviation from the local norm.
2. Aircraft-window rows: one per (cell, window, aircraft).
3. Chronic aircraft: degraded where other aircraft nearby are healthy (equipment faults).
   They are excluded from the quality counts.
4. Cell-window aggregation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gnssguard.analysis.coverage import partitions

log = logging.getLogger(__name__)

ADSB_SOURCES = ("adsb_icao", "adsb_icao_nt")
FRESH_QUALITY_S = 30.0  # quality fields older than this are ignored
MIN_ALT_FT = 3_000  # below this, GPS start-up and airport effects dominate
DEGRADED_NIC = 7  # NIC >= 7 is normal (containment radius < 0.2 NM)
LOW_NACP = 8  # NACp >= 8 is normal (95% accuracy < 93 m)
JUMP_KT = 1_000  # implied speed no airliner can fly
JUMP_MAX_GAP_S = 120  # only compare consecutive points close in time
SPEED_PAIR_S = (5, 30)  # point pairs used to compare position change with ground speed
SPEED_MISMATCH_KT = 150  # implied speed vs reported ground speed
ALT_BAND_FT = 5_000  # altitude bands for the GPS-vs-baro baseline
ALTDEV_FT = 1_000  # deviation from local norm that counts as abnormal
BASELINE_MIN_AC = 5  # healthy aircraft needed to trust an hour-band altitude baseline
CHRONIC_MIN_ROWS = 3  # windows where others were healthy, needed to judge an aircraft
CHRONIC_SHARE = 0.8  # degraded in this share of those windows => equipment fault

POINT_COLS = ["icao", "ts", "lat", "lon", "alt_baro", "alt_geom", "on_ground", "gs", "src",
              "nic", "nac_p", "q_age_s"]


@dataclass(frozen=True)
class GridSpec:
    # 1 deg x 15 min keeps >= 3 usable aircraft in most busy cells even in sparse regions
    # (see docs/labels.md for the comparison); jamming footprints span hundreds of km.
    cell_deg: float = 1.0
    window_min: int = 15

    def cell_floor(self, v: pd.Series) -> pd.Series:
        return (np.floor(v / self.cell_deg) * self.cell_deg).round(4)


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    h = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(h.clip(0, 1)))


def prepare_points(df: pd.DataFrame, spec: GridSpec = GridSpec()) -> pd.DataFrame:
    """Airborne points with cell keys and boolean point flags."""
    df = df.loc[~df["on_ground"].astype(bool)].copy()
    df["icao"] = df["icao"].astype(str)
    df["src"] = df["src"].astype(str)
    df = df.sort_values(["icao", "ts"], kind="stable").reset_index(drop=True)

    df["cell_lat"] = spec.cell_floor(df["lat"])
    df["cell_lon"] = spec.cell_floor(df["lon"])
    df["window"] = df["ts"].dt.floor(f"{spec.window_min}min")

    is_adsb = df["src"].isin(ADSB_SOURCES)

    # MLAT fallback: an ADS-B-equipped aircraft positioned by ground multilateration.
    # In real data ~99% of these points follow the aircraft's own report of NIC 0:
    # the tracker stops trusting its GPS position. The quality fields still come from
    # the aircraft's ADS-B messages, so these points count as quality readings below;
    # leaving them out would hide the most strongly affected aircraft.
    adsb_aircraft = df.loc[is_adsb, "icao"].unique()
    df["mlat_fallback"] = df["src"].eq("mlat") & df["icao"].isin(adsb_aircraft)

    alt = df["alt_baro"].astype("float64").fillna(df["alt_geom"].astype("float64"))
    df["alt_ft"] = alt
    usable = ((is_adsb | df["mlat_fallback"]) & df["q_age_s"].le(FRESH_QUALITY_S)
              & df["nic"].notna() & (alt >= MIN_ALT_FT))
    df["usable"] = usable
    df["degraded"] = usable & (df["nic"] < DEGRADED_NIC)
    df["nic0"] = usable & (df["nic"] == 0)
    df["nacp_low"] = usable & df["nac_p"].notna() & (df["nac_p"] < LOW_NACP)

    # Consecutive-point kinematics per aircraft.
    same_ac = df["icao"].eq(df["icao"].shift())
    dt = (df["ts"] - df["ts"].shift()).dt.total_seconds().where(same_ac)
    dist = _haversine_km(df["lat"].shift(), df["lon"].shift(), df["lat"], df["lon"])
    dist = pd.Series(dist, index=df.index).where(same_ac)
    speed_kt = dist / (dt / 3600.0) / 1.852
    pair_ok = (dt > 0) & (dt <= JUMP_MAX_GAP_S) & (dist > 1.0)
    prev_src = df["src"].shift().where(same_ac)
    fast = pair_ok & (speed_kt > JUMP_KT)
    df["jump"] = fast & is_adsb & prev_src.isin(ADSB_SOURCES)
    df["switch_jump"] = fast & prev_src.notna() & (prev_src != df["src"])

    # Position change vs the aircraft's own reported ground speed. Ground speed comes
    # from the aircraft's navigation system; a spoofed GPS position moves differently.
    # In calm traffic no pair 5-30 s apart disagrees by more than 150 kt (docs/labels.md).
    both_adsb = is_adsb & prev_src.isin(ADSB_SOURCES)
    close = (dt >= SPEED_PAIR_S[0]) & (dt <= SPEED_PAIR_S[1])
    gs = df["gs"].astype("float64")
    df["speed_mismatch"] = (both_adsb & close & (gs > 100)
                            & ((dist / (dt / 3600.0) / 1.852) - gs).abs().gt(SPEED_MISMATCH_KT))

    # GPS altitude minus baro altitude, compared with healthy aircraft at the same hour
    # and altitude band (the normal difference depends on weather).
    diff = df["alt_geom"].astype("float64") - df["alt_baro"].astype("float64")
    band = (df["alt_baro"].astype("float64") // ALT_BAND_FT)
    hour = df["ts"].dt.floor("h")
    healthy = usable & ~df["degraded"] & diff.notna() & band.notna()
    hb = pd.DataFrame({"hour": hour[healthy], "band": band[healthy], "diff": diff[healthy],
                       "icao": df.loc[healthy, "icao"]}).groupby(["hour", "band"])
    base = hb["diff"].median().where(hb["icao"].nunique() >= BASELINE_MIN_AC).rename("base")
    base_v = pd.DataFrame({"hour": hour, "band": band}).join(base, on=["hour", "band"])["base"]
    dev = diff - base_v
    df["altpair"] = usable & dev.notna()
    df["altdev"] = df["altpair"] & (dev.abs() > ALTDEV_FT)
    return df


AC_KEYS = ["cell_lat", "cell_lon", "window", "icao"]
CELL_KEYS = ["cell_lat", "cell_lon", "window"]


def aircraft_windows(pts: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell, window, aircraft) with per-aircraft flags."""
    g = pts.groupby(AC_KEYS, sort=False, observed=True)
    ac = g.agg(
        points=("ts", "size"),
        usable_pts=("usable", "sum"),
        degraded_pts=("degraded", "sum"),
        nic0_pts=("nic0", "sum"),
        nacp_low_pts=("nacp_low", "sum"),
        altpair_pts=("altpair", "sum"),
        altdev_pts=("altdev", "sum"),
        jump=("jump", "any"),
        switch_jump=("switch_jump", "any"),
        speed_mismatch=("speed_mismatch", "any"),
        mlat_fallback=("mlat_fallback", "any"),
        alt_ft=("alt_ft", "median"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
    ).reset_index()
    u = ac["usable_pts"]
    ac["usable"] = u > 0
    ac["degraded"] = ac["usable"] & (ac["degraded_pts"] >= 0.5 * u)
    ac["nic0"] = ac["usable"] & (ac["nic0_pts"] >= 0.5 * u)
    ac["nacp_low"] = ac["usable"] & (ac["nacp_low_pts"] >= 0.5 * u)
    ac["altpair"] = ac["altpair_pts"] > 0
    ac["altdev"] = ac["altpair"] & (ac["altdev_pts"] >= 0.5 * ac["altpair_pts"])
    return ac


def find_chronic(ac: pd.DataFrame) -> set[str]:
    """Aircraft degraded where the other aircraft in the same cell-window are healthy."""
    usable = ac[ac["usable"]]
    tot = usable.groupby(CELL_KEYS, observed=True).agg(
        n_u=("usable", "size"), n_d=("degraded", "sum"))
    x = usable.join(tot, on=CELL_KEYS)
    others_u = x["n_u"] - 1
    others_d = x["n_d"] - x["degraded"].astype(int)
    calm = (others_u >= 2) & (others_d / others_u.clip(lower=1) < 0.2)
    judged = x[calm].groupby("icao")["degraded"].agg(["size", "mean"])
    bad = judged[(judged["size"] >= CHRONIC_MIN_ROWS) & (judged["mean"] >= CHRONIC_SHARE)]
    return set(bad.index)


def cell_windows(ac: pd.DataFrame, chronic: set[str], spec: GridSpec) -> pd.DataFrame:
    """Aggregate aircraft-window rows into cell-window features."""
    ac = ac.copy()
    ac["chronic"] = ac["icao"].isin(chronic)
    q = ~ac["chronic"]  # quality counts exclude equipment-fault aircraft
    for c in ("usable", "degraded", "nic0", "nacp_low", "altpair", "altdev"):
        ac[f"q_{c}"] = ac[c] & q
    ac["u_alt"] = ac["alt_ft"].where(ac["q_usable"])

    g = ac.groupby(CELL_KEYS, observed=True)
    cw = g.agg(
        n_points=("points", "sum"),
        n_aircraft=("icao", "size"),
        n_chronic=("chronic", "sum"),
        n_ac_usable=("q_usable", "sum"),
        n_ac_degraded=("q_degraded", "sum"),
        n_ac_nic0=("q_nic0", "sum"),
        n_ac_nacp_low=("q_nacp_low", "sum"),
        n_ac_altpair=("q_altpair", "sum"),
        n_ac_altdev=("q_altdev", "sum"),
        n_ac_jump=("jump", "sum"),
        n_ac_switch_jump=("switch_jump", "sum"),
        n_ac_speed_mismatch=("speed_mismatch", "sum"),
        n_ac_mlat_fallback=("mlat_fallback", "sum"),
        usable_pts=("usable_pts", "sum"),
        degraded_pts=("degraded_pts", "sum"),
        mean_alt_ft=("u_alt", "mean"),
        max_alt_ft=("u_alt", "max"),
    ).reset_index()

    def share(num, den):
        return (cw[num] / cw[den]).where(cw[den] > 0)

    cw["share_ac_degraded"] = share("n_ac_degraded", "n_ac_usable")
    cw["share_ac_nacp_low"] = share("n_ac_nacp_low", "n_ac_usable")
    cw["share_ac_altdev"] = share("n_ac_altdev", "n_ac_altpair")
    cw["share_pts_degraded"] = share("degraded_pts", "usable_pts")
    cw["center_lat"] = cw["cell_lat"] + spec.cell_deg / 2
    cw["center_lon"] = cw["cell_lon"] + spec.cell_deg / 2
    int_cols = [c for c in cw.columns if c.startswith("n_") or c.endswith("_pts")]
    cw[int_cols] = cw[int_cols].astype("int32")
    return cw


def build_from_points(df: pd.DataFrame, spec: GridSpec = GridSpec()) -> tuple[pd.DataFrame, set]:
    """Points of one region-day -> (cell-window table, chronic aircraft)."""
    pts = prepare_points(df, spec)
    ac = aircraft_windows(pts)
    chronic = find_chronic(ac)
    return cell_windows(ac, chronic, spec), chronic


def build_all(processed: str | Path, out: str | Path, spec: GridSpec = GridSpec(),
              dates: list[str] | None = None, regions: list[str] | None = None,
              force: bool = False) -> list[dict]:
    """Build cell-window tables for every stored region-day. One region-day in memory."""
    out = Path(out)
    summary = []
    for region, date, folder in partitions(processed, dates, regions):
        target = out / f"region={region}" / f"date={date}" / "part-0.parquet"
        if target.exists() and not force:
            continue
        df = pq.read_table(folder, columns=POINT_COLS).to_pandas()
        cw, chronic = build_from_points(df, spec)
        del df
        # region and date live in the folder names (Hive partitions), not in the file,
        # so the folder can also be read directly with pandas / pyarrow / DuckDB.
        cw["cell_deg"] = spec.cell_deg
        cw["window_min"] = spec.window_min
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        pq.write_table(pa.Table.from_pandas(cw, preserve_index=False), tmp, compression="zstd")
        tmp.replace(target)
        s = {"region": region, "date": date, "cell_windows": len(cw),
             "chronic_aircraft": len(chronic)}
        log.info("cells %s %s: %d cell-windows, %d chronic aircraft",
                 region, date, len(cw), len(chronic))
        summary.append(s)
    return summary


def load_cells(cells: str | Path, dates: list[str] | None = None,
               regions: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for region, date, folder in partitions(cells, dates, regions):
        t = pq.read_table(folder / "part-0.parquet").to_pandas()
        t.insert(0, "date", date)
        t.insert(0, "region", region)
        frames.append(t)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
