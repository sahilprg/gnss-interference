"""Week-1 coverage check: is there enough data per region to detect anything?

For each region it reports traffic volume, how often fresh GPS quality fields are
available, and a first-look share of airborne ADS-B points with degraded NIC.
The degraded share is a quick indicator, not a label.

The report reads one region-day file at a time and keeps only running totals, so memory
stays around 1 GB however many days are stored (one busy region-day is the peak).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

COLS = ["icao", "ts", "alt_baro", "on_ground", "src", "nic", "nac_p", "q_age_s"]

# Thresholds for the first-look indicator (documented in docs/data.md).
FRESH_QUALITY_S = 30.0  # quality fields reported within the last 30 s
MIN_ALT_FT = 10_000  # ignore low altitude, where reception and terrain effects dominate
DEGRADED_NIC = 7  # NIC >= 7 is normal; below it the position integrity is reduced


def partitions(processed: str | Path, dates: list[str] | None = None,
               regions: list[str] | None = None) -> Iterator[tuple[str, str, Path]]:
    """Yield (region, date, folder) for each stored region-day, in sorted order."""
    for folder in sorted(Path(processed).glob("region=*/date=*")):
        region = folder.parent.name.split("=", 1)[1]
        date = folder.name.split("=", 1)[1]
        if (regions and region not in regions) or (dates and date not in dates):
            continue
        yield region, date, folder


def read_partition(folder: Path, columns: list[str] = COLS) -> pd.DataFrame:
    """Read one region-day. Strings become categoricals to save memory."""
    table = pq.read_table(folder, columns=columns)
    return table.to_pandas(strings_to_categorical=True)


def load(processed: str | Path, dates: list[str] | None = None,
         regions: list[str] | None = None) -> pd.DataFrame:
    """Load everything into one DataFrame. Fine for a few days; for many, use scan()."""
    dataset = ds.dataset(processed, format="parquet", partitioning="hive")
    filt = None
    if dates:
        filt = ds.field("date").isin(dates)
    if regions:
        f2 = ds.field("region").isin(regions)
        filt = f2 if filt is None else filt & f2
    df = dataset.to_table(columns=COLS + ["region", "date"], filter=filt).to_pandas()
    for c in ("region", "date"):
        df[c] = df[c].astype(str)
    return df


def _masks(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(fresh, usable): fresh quality fields; usable = fresh + airborne ADS-B above MIN_ALT."""
    fresh = df["q_age_s"].le(FRESH_QUALITY_S) & df["nic"].notna()
    airborne_adsb = (~df["on_ground"].astype(bool)) & (df["src"] == "adsb_icao") & (
        df["alt_baro"] >= MIN_ALT_FT
    )
    return fresh, fresh & airborne_adsb


@dataclass
class _Totals:
    days: set = field(default_factory=set)
    aircraft: set = field(default_factory=set)
    per_hour: dict = field(default_factory=dict)
    points: int = 0
    adsb: int = 0
    mlat: int = 0
    fresh: int = 0
    usable: int = 0
    degraded: int = 0

    def add(self, date: str, df: pd.DataFrame) -> pd.DataFrame:
        fresh, usable = _masks(df)
        hours = df["ts"].dt.floor("h")
        self.days.add(date)
        self.aircraft.update(df["icao"].astype(str).unique())
        for hour, n in df.groupby(hours, observed=True)["icao"].nunique().items():
            self.per_hour[hour] = max(self.per_hour.get(hour, 0), int(n))
        self.points += len(df)
        self.adsb += int((df["src"] == "adsb_icao").sum())
        self.mlat += int((df["src"] == "mlat").sum())
        self.fresh += int(fresh.sum())
        self.usable += int(usable.sum())

        u = df.loc[usable, ["icao", "nic"]].assign(hour=hours[usable])
        u["degraded"] = u["nic"] < DEGRADED_NIC
        self.degraded += int(u["degraded"].sum())
        return u.groupby("hour", observed=True).agg(
            aircraft=("icao", "nunique"),
            points=("nic", "size"),
            share_degraded=("degraded", "mean"),
        ).reset_index()

    def row(self, region: str) -> dict:
        def share(n: int, d: int, nd: int = 3):
            return round(n / d, nd) if d else None

        return {
            "region": region,
            "days": len(self.days),
            "aircraft": len(self.aircraft),
            "points": self.points,
            "aircraft_per_hour_median": float(pd.Series(self.per_hour).median())
            if self.per_hour else 0.0,
            "share_adsb": share(self.adsb, self.points),
            "share_mlat": share(self.mlat, self.points),
            "share_fresh_quality": share(self.fresh, self.points),
            "usable_points": self.usable,
            "share_degraded_nic": share(self.degraded, self.usable, 4),
        }


def scan(processed: str | Path, dates: list[str] | None = None,
         regions: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (coverage report, hourly degraded table), reading one region-day at a time."""
    totals: dict[str, _Totals] = {}
    hourly = []
    for region, date, folder in partitions(processed, dates, regions):
        df = read_partition(folder)
        h = totals.setdefault(region, _Totals()).add(date, df)
        hourly.append(h.assign(region=region))
        del df
    if not totals:
        return pd.DataFrame(), pd.DataFrame()
    report = pd.DataFrame([t.row(r) for r, t in totals.items()])
    report = report.sort_values("aircraft", ascending=False).reset_index(drop=True)
    hourly_df = pd.concat(hourly, ignore_index=True)[
        ["region", "hour", "aircraft", "points", "share_degraded"]
    ]
    return report, hourly_df
