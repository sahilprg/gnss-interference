"""Official interference advisories: standing warnings for whole areas.

Unlike `events.csv` (one incident at a known place and time), an advisory such as a
NOTAM saying "expect GPS interference in Cyprus airspace" covers a large area for
months. It cannot say which cell-window was affected, so it is never used as a cell
label. It answers a coarser question:

    Within a study region, is the flagged share higher inside the warned area than
    outside it, over the period the warning was in force?

`configs/advisories.csv`, one row per area box (one warning may need several boxes):

    advisory_id, reference, issuer, lat_min, lat_max, lon_min, lon_max,
    start_utc, end_utc, source_url, notes

end_utc may be empty: the warning is still in force. Boxes are rough rectangles, not
the official airspace boundaries; say so in `notes`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ADVISORY_COLUMNS = ["advisory_id", "reference", "issuer", "lat_min", "lat_max", "lon_min",
                    "lon_max", "start_utc", "end_utc", "source_url", "notes"]
FLAGGED = ("jamming", "spoofing", "both")


def load_advisories(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=ADVISORY_COLUMNS)
    adv = pd.read_csv(path, dtype={"advisory_id": str, "reference": str, "issuer": str,
                                   "source_url": str, "notes": str}, keep_default_na=False)
    missing = set(ADVISORY_COLUMNS) - set(adv.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if adv.empty:
        return adv[ADVISORY_COLUMNS]
    if (adv["source_url"].str.strip() == "").any():
        raise ValueError(f"{path}: every advisory needs a source_url")
    if adv["advisory_id"].duplicated().any():
        raise ValueError(f"{path}: advisory_id must be unique (use a suffix per box)")
    for c in ("lat_min", "lat_max", "lon_min", "lon_max"):
        adv[c] = pd.to_numeric(adv[c])
    if ((adv["lat_min"] >= adv["lat_max"]) | (adv["lon_min"] >= adv["lon_max"])).any():
        raise ValueError(f"{path}: each box needs lat_min < lat_max and lon_min < lon_max")
    adv["start_utc"] = pd.to_datetime(adv["start_utc"], utc=True)
    adv["end_utc"] = pd.to_datetime(adv["end_utc"].replace("", None), utc=True)
    return adv[ADVISORY_COLUMNS]


def advisory_check(labels: pd.DataFrame, advisories: pd.DataFrame) -> pd.DataFrame:
    """Flagged share inside vs outside each advisory box, per study region it touches.

    A cell is inside when its centre is in the box. Only cell-windows some rule could
    judge (label not 'unknown') and dates while the advisory was in force count.
    """
    cols = ["advisory_id", "reference", "region", "days", "judged_inside", "flagged_inside",
            "share_inside", "judged_outside", "flagged_outside", "share_outside", "ratio"]
    if advisories.empty or labels.empty:
        return pd.DataFrame(columns=cols)
    lab = labels[labels["label"].astype(str) != "unknown"].copy()
    deg = lab["cell_deg"] if "cell_deg" in lab else 1.0
    lab["clat"] = lab["cell_lat"] + deg / 2
    lab["clon"] = lab["cell_lon"] + deg / 2
    lab["flagged"] = lab["label"].astype(str).isin(FLAGGED)
    day = pd.to_datetime(lab["date"], utc=True)

    rows = []
    for a in advisories.itertuples(index=False):
        end = a.end_utc if pd.notna(a.end_utc) else pd.Timestamp.max.tz_localize("UTC")
        live = (day + pd.Timedelta(days=1) > a.start_utc) & (day < end)
        inside = (lab["clat"].between(a.lat_min, a.lat_max)
                  & lab["clon"].between(a.lon_min, a.lon_max))
        for region in sorted(lab.loc[live & inside, "region"].unique()):
            sel = live & (lab["region"] == region)
            i, o = lab[sel & inside], lab[sel & ~inside]
            si = float(i["flagged"].mean()) if len(i) else np.nan
            so = float(o["flagged"].mean()) if len(o) else np.nan
            rows.append({
                "advisory_id": a.advisory_id, "reference": a.reference, "region": region,
                "days": int(lab.loc[sel, "date"].nunique()),
                "judged_inside": len(i), "flagged_inside": int(i["flagged"].sum()),
                "share_inside": round(si, 4),
                "judged_outside": len(o), "flagged_outside": int(o["flagged"].sum()),
                "share_outside": round(so, 4),
                "ratio": round(si / so, 2) if so and so > 0 else None,
            })
    return pd.DataFrame(rows, columns=cols)
