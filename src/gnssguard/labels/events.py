"""Match officially reported interference events to cell-windows.

`configs/events.csv` is filled by hand from public sources (aviation authority
bulletins, NOTAMs, regulator or airline statements). One row per event:

    event_id, type, lat_min, lat_max, lon_min, lon_max, start_utc, end_utc, source_url, notes,
    evidence_level, region

type: jamming | spoofing | unknown. Times are ISO 8601 in UTC, e.g. 2026-10-26T06:00:00Z.
Every row must cite a source_url. Events check the system; they are never used to train
it or to tune thresholds.

evidence_level (optional; docs/labels.md, "Event catalog"):

* observed_anomaly: seen in ADS-B data by an outside analyst. Not independent of this
  project's input.
* suspected_mechanism: an authority, airline or crew states jamming or spoofing happened,
  without a measurement.
* independently_corroborated: a measurement that does not come from ADS-B (ground
  spectrum monitoring, direction finding, a ground GNSS receiver).
* unresolved: reported, cause or occurrence unclear.

region: the study region the box lies in (empty: outside all regions, not checked).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

EVENT_COLUMNS = ["event_id", "type", "lat_min", "lat_max", "lon_min", "lon_max",
                 "start_utc", "end_utc", "source_url", "notes"]
EVENT_TYPES = {"jamming", "spoofing", "unknown"}
EVIDENCE_LEVELS = {"observed_anomaly", "suspected_mechanism", "independently_corroborated",
                   "unresolved"}
OPTIONAL_COLUMNS = ["evidence_level", "region"]


def load_events(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    ev = pd.read_csv(path, dtype={"event_id": str, "type": str, "source_url": str,
                                  "notes": str, "evidence_level": str, "region": str})
    for c in OPTIONAL_COLUMNS:
        if c not in ev:
            ev[c] = pd.NA
    missing = set(EVENT_COLUMNS) - set(ev.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if ev.empty:
        return ev[EVENT_COLUMNS + OPTIONAL_COLUMNS]
    bad_type = set(ev["type"].str.lower()) - EVENT_TYPES
    if bad_type:
        raise ValueError(f"{path}: unknown event types {sorted(bad_type)}")
    if ev["source_url"].isna().any() or (ev["source_url"].str.strip() == "").any():
        raise ValueError(f"{path}: every event needs a source_url")
    ev["type"] = ev["type"].str.lower()
    ev["start_utc"] = pd.to_datetime(ev["start_utc"], utc=True)
    ev["end_utc"] = pd.to_datetime(ev["end_utc"], utc=True)
    if (ev["end_utc"] <= ev["start_utc"]).any():
        raise ValueError(f"{path}: end_utc must be after start_utc")
    bad_level = set(ev["evidence_level"].dropna().str.lower()) - EVIDENCE_LEVELS
    if bad_level:
        raise ValueError(f"{path}: unknown evidence levels {sorted(bad_level)}")
    ev["evidence_level"] = ev["evidence_level"].str.lower()
    return ev[EVENT_COLUMNS + OPTIONAL_COLUMNS]


def match_events(cells: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Columns `event_id` and `event_type` for cell-windows overlapping an event.

    A cell-window matches when its cell overlaps the event box and its window overlaps
    the event period. Overlapping events: the first listed wins.
    """
    out = pd.DataFrame({"event_id": pd.Series(pd.NA, index=cells.index, dtype="string"),
                        "event_type": pd.Series(pd.NA, index=cells.index, dtype="string")})
    if events.empty or cells.empty:
        return out
    deg = cells["cell_deg"] if "cell_deg" in cells else 1.0
    win = pd.to_timedelta(cells["window_min"] if "window_min" in cells else 15, unit="min")
    w0 = pd.to_datetime(cells["window"], utc=True)
    w1 = w0 + win
    for ev in events.itertuples(index=False):
        hit = ((cells["cell_lat"] < ev.lat_max) & (cells["cell_lat"] + deg > ev.lat_min)
               & (cells["cell_lon"] < ev.lon_max) & (cells["cell_lon"] + deg > ev.lon_min)
               & (w0 < ev.end_utc) & (w1 > ev.start_utc) & out["event_id"].isna())
        out.loc[hit, "event_id"] = ev.event_id
        out.loc[hit, "event_type"] = ev.type
    return out
