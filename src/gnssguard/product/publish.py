"""Turn one processed day into small public files (schema `gnssguard-public/1`).

Layout under the output folder (served as is by GitHub Pages):

    data/index.json                      every published day, newest first
    data/<date>/summary.json             per-region status, quality checks, headline numbers
    data/<date>/<region>.json            map payload: per cell, one value per 15-min window
    data/<date>/<region>.csv.gz          the same, one row per cell-window (for analysis)
    data/<date>/incidents.json           jamming incidents with coarse source areas

What is published, and what is not (docs/model_card.md):

* Cell-level aggregates only. No aircraft identifiers, callsigns or tracks.
* A cell-window with 3+ usable aircraft gets a label (normal, jamming, spoofing, both)
  and calibrated probabilities. With 1-2 it is "low coverage": the probabilities are
  given but no label, and the exact aircraft count is not. With none it is "no data",
  never "normal".
* A jammer estimate is a circle (centre rounded to 0.1°, radius = stated 90%
  uncertainty, 50-200 km) and only for confident fits whose circle is at most 200 km
  wide in radius; otherwise the incident
  is described by the area of its flagged cells.
* A region-day that fails a quality gate (product/quality.py) is withheld: only its
  status and reasons are published.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from gnssguard.features.cells import POINT_COLS, GridSpec, load_cells
from gnssguard.model.dataset import with_context
from gnssguard.model.localize import aircraft_points, incidents, localize_incidents
from gnssguard.model.train import Detectors
from gnssguard.product import quality

SCHEMA = "gnssguard-public/1"
MIN_AIRCRAFT = 3
MIN_RADIUS_KM = 50.0
# A wider 90% circle says little about where the jammer is (the regions are 400-900 km
# across): such incidents are shown by their affected area only.
MAX_RADIUS_KM = 200.0
MIN_INCIDENT_CELLS = 3  # single flagged cells are listed in the cell files, not as incidents
CENTRE_ROUND_DEG = 0.1
LABEL_CODES = {"no_data": 0, "low_coverage": 1, "normal": 2, "jamming": 3, "spoofing": 4,
               "both": 5}
FLAGGED = ("jamming", "spoofing", "both")
MODEL_CARD = "https://github.com/sahilprg/gnss-interference/blob/main/docs/model_card.md"
NOTICE = ("Research output. Not for navigation, flight planning or any operational "
          "decision. Not an attribution of interference to any party.")


def cell_table(cells: pd.DataFrame, det: Detectors) -> pd.DataFrame:
    """Public per-cell-window table: calibrated probabilities and the public label."""
    p = det.predict(cells, calibrated=True)
    usable = cells["n_ac_usable"].fillna(0).astype(int)
    label = p["label"].astype(str).where(usable >= MIN_AIRCRAFT, "low_coverage")
    label = label.where(usable >= 1, "no_data")
    return pd.DataFrame({
        "region": cells["region"].to_numpy(), "date": cells["date"].to_numpy(),
        "cell_lat": cells["cell_lat"].round(4).to_numpy(),
        "cell_lon": cells["cell_lon"].round(4).to_numpy(),
        "window": pd.to_datetime(cells["window"], utc=True).to_numpy(),
        # exact counts only where they are not about one or two aircraft
        "usable_aircraft": usable.where(usable >= MIN_AIRCRAFT).to_numpy(),
        "p_jamming": p["p_jamming"].round(3).to_numpy(),
        "p_spoofing": p["p_spoofing"].round(3).to_numpy(),
        "label": label.to_numpy(),
    })


def region_stats(table: pd.DataFrame, aircraft: int) -> dict:
    dense = table["label"].isin(["normal", *FLAGGED])
    low = table["label"] == "low_coverage"
    flagged = table["label"].isin(FLAGGED)
    return {
        "aircraft": int(aircraft), "judged_cells": int(dense.sum()),
        "low_coverage_cells": int(low.sum()),
        "flagged_share": round(float(flagged[dense].mean()), 4) if dense.any() else 0.0,
        "jamming_share": round(float(table.loc[dense, "label"].isin(["jamming", "both"])
                                     .mean()), 4) if dense.any() else 0.0,
        "spoofing_share": round(float(table.loc[dense, "label"].isin(["spoofing", "both"])
                                      .mean()), 4) if dense.any() else 0.0,
    }


def map_payload(table: pd.DataFrame, date: str, region: str, spec: GridSpec) -> dict:
    """Compact map file: per cell, arrays with one entry per window of the day."""
    start = pd.Timestamp(date, tz="UTC")
    n_win = int(24 * 60 / spec.window_min)
    idx = ((pd.to_datetime(table["window"], utc=True) - start)
           / pd.Timedelta(minutes=spec.window_min)).astype(int)
    cells = []
    for (lat, lon), g in table.assign(w=idx).groupby(["cell_lat", "cell_lon"], sort=True):
        lab = [0] * n_win
        pj: list = [None] * n_win
        ps: list = [None] * n_win
        n: list = [None] * n_win
        for r in g.itertuples(index=False):
            if 0 <= r.w < n_win:
                lab[r.w] = LABEL_CODES[r.label]
                pj[r.w], ps[r.w] = _num(r.p_jamming), _num(r.p_spoofing)
                n[r.w] = None if pd.isna(r.usable_aircraft) else int(r.usable_aircraft)
        cells.append({"lat": float(lat), "lon": float(lon), "label": lab,
                      "p_jamming": pj, "p_spoofing": ps, "aircraft": n})
    return {"schema": SCHEMA, "date": date, "region": region, "cell_deg": spec.cell_deg,
            "window_min": spec.window_min, "windows": n_win,
            "label_codes": LABEL_CODES, "cells": cells, "notice": NOTICE}


def _num(v) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def public_incidents(region: str, est: pd.DataFrame, inc: pd.DataFrame,
                     spec: GridSpec) -> list[dict]:
    """Incidents with coarse areas; a source circle only for confident fits."""
    out = []
    for iid, g in inc.groupby("incident_id"):
        if len(g) < MIN_INCIDENT_CELLS:
            continue
        e = est[est["incident_id"] == iid] if not est.empty else est
        item = {
            "region": region,
            "start": pd.Timestamp(g["window"].min()).isoformat(),
            "end": (pd.Timestamp(g["window"].max())
                    + pd.Timedelta(minutes=spec.window_min)).isoformat(),
            "cell_windows": int(len(g)),
            "area": {"lat_min": float(g["cell_lat"].min()),
                     "lat_max": float(g["cell_lat"].max() + spec.cell_deg),
                     "lon_min": float(g["cell_lon"].min()),
                     "lon_max": float(g["cell_lon"].max() + spec.cell_deg)},
            "source": None, "confidence": "none",
        }
        if not e.empty:
            r = e.iloc[0]
            radius = 10 * math.ceil(float(r["uncertainty_cal_km"]) / 10)
            if not bool(r["confident"]):
                item["confidence"] = "low"
            elif radius > MAX_RADIUS_KM:
                item["confidence"] = "too_uncertain"
            else:
                item["confidence"] = "confident"
                item["source"] = {
                    "lat": _coarse(r["lat"]), "lon": _coarse(r["lon"]),
                    "radius_km": float(max(MIN_RADIUS_KM, radius)),
                    "meaning": "90% uncertainty area of a single ground jammer fitted to "
                               "which aircraft were affected; not an attribution"}
        out.append(item)
    return out


def _coarse(v) -> float:
    return round(round(float(v) / CENTRE_ROUND_DEG) * CENTRE_ROUND_DEG, 2)


def _aircraft(processed: Path, region: str, date: str) -> tuple[int, pd.DataFrame | None]:
    folder = processed / f"region={region}" / f"date={date}"
    if not folder.exists():
        return 0, None
    pts = pq.read_table(folder, columns=POINT_COLS).to_pandas()
    air = pts[~pts["on_ground"].astype(bool)]
    return int(air["icao"].nunique()), pts


def publish_day(date: str, regions: list[str], processed: str | Path, cells_dir: str | Path,
                model_dir: str | Path, out: str | Path, *, spec: GridSpec = GridSpec(),
                locate: bool = True) -> dict:
    """Publish one day. Returns the day summary (also written to summary.json)."""
    processed, out = Path(processed), Path(out)
    det = Detectors.load(model_dir)
    manifest_path = processed / "_manifests" / f"{date}.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {})
    history = [d for d in read_index(out) if d["date"] < date]
    history = [read_summary(out, d["date"]) for d in reversed(history)]
    history = [h for h in history if h]

    tables, stats, points = {}, {}, {}
    cells = load_cells(cells_dir, [date], regions)
    if not cells.empty:
        cells = with_context(cells)
    for region in regions:
        n_air, pts = _aircraft(processed, region, date)
        c = cells[cells["region"] == region] if not cells.empty else cells
        t = cell_table(c.reset_index(drop=True), det) if not c.empty else pd.DataFrame(
            columns=["region", "date", "cell_lat", "cell_lon", "window", "usable_aircraft",
                     "p_jamming", "p_spoofing", "label"])
        tables[region], stats[region], points[region] = t, region_stats(t, n_air), pts

    status, day_reasons = quality.check_day(stats, history,
                                            bool(manifest.get("complete_day", False)))
    day_dir = out / "data" / date
    day_dir.mkdir(parents=True, exist_ok=True)
    all_incidents = []
    for region in regions:
        t, st = tables[region], status[region]
        for f in (day_dir / f"{region}.json", day_dir / f"{region}.csv.gz"):
            f.unlink(missing_ok=True)  # a re-run that now withholds must not leave old files
        if st["status"] != "published":
            continue
        (day_dir / f"{region}.json").write_text(
            json.dumps(map_payload(t, date, region, spec), separators=(",", ":")),
            encoding="utf-8")
        _write_csv(day_dir / f"{region}.csv.gz", t)
        if locate and points[region] is not None:
            dense_jam = (t["label"].isin(["jamming", "both"])).to_numpy()
            if dense_jam.any():
                flagged = t.loc[dense_jam, ["cell_lat", "cell_lon", "window"]]
                inc = incidents(flagged, spec)
                est = localize_incidents(aircraft_points(points[region], spec), inc, spec)
                all_incidents += public_incidents(region, est, inc, spec)
        points[region] = None
    (day_dir / "incidents.json").write_text(json.dumps(
        {"schema": SCHEMA, "date": date, "incidents": all_incidents, "notice": NOTICE},
        indent=1), encoding="utf-8")

    summary = {
        "schema": SCHEMA, "date": date,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "published": any(s["status"] == "published" for s in status.values()),
        "day_reasons": day_reasons,
        "model": {"train_dates": [det.meta.get("train_dates", [None])[0],
                                  det.meta.get("train_dates", [None])[-1]],
                  "versions": det.meta.get("versions", {}),
                  "card": MODEL_CARD},
        "regions": {r: {**stats[r], **status[r]} for r in regions},
        "incidents": len(all_incidents),
        "notice": NOTICE,
    }
    (day_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    update_index(out, summary)
    return summary


def _write_csv(path: Path, t: pd.DataFrame) -> None:
    rows = t[t["label"] != "no_data"].copy()
    rows["window_start"] = pd.to_datetime(rows["window"], utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    cols = ["date", "region", "cell_lat", "cell_lon", "window_start", "usable_aircraft",
            "p_jamming", "p_spoofing", "label"]
    rows["usable_aircraft"] = rows["usable_aircraft"].astype("Int64")
    # gzip: a third of the size; pandas and most tools read .csv.gz directly
    rows[cols].to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n",
                      compression={"method": "gzip", "mtime": 0})


def read_index(out: str | Path) -> list[dict]:
    p = Path(out) / "data" / "index.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("days", [])


def read_summary(out: str | Path, date: str) -> dict | None:
    p = Path(out) / "data" / date / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def update_index(out: str | Path, summary: dict) -> None:
    days = {d["date"]: d for d in read_index(out)}
    days[summary["date"]] = {
        "date": summary["date"], "published": summary["published"],
        "regions": {r: v["status"] for r, v in summary["regions"].items()}}
    ordered = sorted(days.values(), key=lambda d: d["date"], reverse=True)
    latest = next((d["date"] for d in ordered if d["published"]), None)
    p = Path(out) / "data" / "index.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"schema": SCHEMA, "latest": latest, "days": ordered}, indent=1),
                 encoding="utf-8")


def published_dates(out: str | Path) -> set[str]:
    return {d["date"] for d in read_index(out)}


