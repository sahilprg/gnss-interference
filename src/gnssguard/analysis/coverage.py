"""Week-1 coverage check: is there enough data per region to detect anything?

For each region it reports traffic volume, how often fresh GPS quality fields are
available, and a first-look share of airborne ADS-B points with degraded NIC.
The degraded share is a quick indicator, not a label.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

COLS = ["icao", "ts", "alt_baro", "on_ground", "src", "nic", "nac_p", "q_age_s"]

# Thresholds for the first-look indicator (documented in docs/data.md).
FRESH_QUALITY_S = 30.0  # quality fields reported within the last 30 s
MIN_ALT_FT = 10_000  # ignore low altitude, where reception and terrain effects dominate
DEGRADED_NIC = 7  # NIC >= 7 is normal; below it the position integrity is reduced


def load(processed: str | Path, dates: list[str] | None = None,
         regions: list[str] | None = None) -> pd.DataFrame:
    dataset = ds.dataset(processed, format="parquet", partitioning="hive")
    filt = None
    if dates:
        filt = ds.field("date").isin(dates)
    if regions:
        f2 = ds.field("region").isin(regions)
        filt = f2 if filt is None else filt & f2
    table = dataset.to_table(columns=COLS + ["region", "date"], filter=filt)
    df = table.to_pandas()
    for c in ("region", "date"):
        df[c] = df[c].astype(str)
    return df


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """One row per region with volume and quality-availability metrics."""
    df = df.copy()
    df["hour"] = df["ts"].dt.floor("h")
    airborne_adsb = (~df["on_ground"]) & (df["src"] == "adsb_icao") & (
        df["alt_baro"] >= MIN_ALT_FT
    )
    fresh = df["q_age_s"].le(FRESH_QUALITY_S) & df["nic"].notna()
    usable = airborne_adsb & fresh

    rows = []
    for region, g in df.groupby("region"):
        idx = g.index
        u = usable.loc[idx]
        per_hour = g.groupby("hour")["icao"].nunique()
        nic_u = g.loc[u[u].index, "nic"]
        rows.append(
            {
                "region": region,
                "days": g["date"].nunique(),
                "aircraft": g["icao"].nunique(),
                "points": len(g),
                "aircraft_per_hour_median": float(per_hour.median()) if len(per_hour) else 0.0,
                "share_adsb": round(float((g["src"] == "adsb_icao").mean()), 3),
                "share_mlat": round(float((g["src"] == "mlat").mean()), 3),
                "share_fresh_quality": round(float(fresh.loc[idx].mean()), 3),
                "usable_points": int(u.sum()),
                "share_degraded_nic": round(float((nic_u < DEGRADED_NIC).mean()), 4)
                if len(nic_u) else None,
            }
        )
    return pd.DataFrame(rows).sort_values("aircraft", ascending=False).reset_index(drop=True)


def hourly_degraded(df: pd.DataFrame) -> pd.DataFrame:
    """Per region and hour: aircraft seen and share of usable points with degraded NIC."""
    df = df[(~df["on_ground"]) & (df["src"] == "adsb_icao") & (df["alt_baro"] >= MIN_ALT_FT)
            & df["q_age_s"].le(FRESH_QUALITY_S) & df["nic"].notna()].copy()
    df["hour"] = df["ts"].dt.floor("h")
    df["degraded"] = df["nic"] < DEGRADED_NIC
    out = df.groupby(["region", "hour"]).agg(
        aircraft=("icao", "nunique"),
        points=("nic", "size"),
        share_degraded=("degraded", "mean"),
    )
    return out.reset_index()
