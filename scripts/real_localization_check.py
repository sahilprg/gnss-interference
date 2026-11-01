"""Sanity check of jammer localization on REAL Baltic data (no labels needed).

Public reporting most often names the Kaliningrad region as a source of GNSS jamming
over the southern Baltic. If the localization works on real data, confident estimates
for real Baltic incidents should cluster there more tightly than the plain centroid of
the degraded aircraft does. This is a plausibility check, not attribution.

    python scripts/real_localization_check.py reports/model 2026-10-20 2026-10-25
"""

import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gnssguard.features.cells import POINT_COLS, _haversine_km, load_cells
from gnssguard.model.dataset import with_context
from gnssguard.model.localize import aircraft_points, incidents, localize_incidents
from gnssguard.model.train import Detectors

KALININGRAD = (54.71, 20.51)
out, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
days = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end)]
det = Detectors.load("data/models/detector")
frames = []
for day in days:
    cells = with_context(load_cells("data/cells", [day], ["baltic"]))
    if cells.empty:
        continue
    flag = det.predict(cells)["label"].astype(str).isin(["jamming", "both"])
    pts = pq.read_table(f"data/processed/region=baltic/date={day}",
                        columns=POINT_COLS).to_pandas()
    est = localize_incidents(aircraft_points(pts), incidents(cells[flag]))
    del pts
    if not est.empty:
        frames.append(est.assign(date=day))
    print(day, len(est), "incidents", flush=True)
est = pd.concat(frames, ignore_index=True)
est["fit_km"] = _haversine_km(est["lat"], est["lon"], *KALININGRAD)
if "lat2" in est:
    d2 = _haversine_km(est["lat2"], est["lon2"], *KALININGRAD)
    est["fit_km"] = np.where(est["n_sources"] == 2, np.fmin(est["fit_km"], d2), est["fit_km"])
est["centroid_km"] = _haversine_km(est["centroid_lat"], est["centroid_lon"], *KALININGRAD)
est.to_csv(f"{out}/real_baltic_localization.csv", index=False)
rows = []
for name, g in (("all incidents", est), ("confident", est[est["confident"]])):
    rows.append({"set": name, "incidents": len(g),
                 "fit_median_km": round(float(g["fit_km"].median()), 1),
                 "fit_within_100km": round(float((g["fit_km"] <= 100).mean()), 3),
                 "centroid_median_km": round(float(g["centroid_km"].median()), 1),
                 "centroid_within_100km": round(float((g["centroid_km"] <= 100).mean()), 3)})
res = pd.DataFrame(rows)
res.to_csv(f"{out}/real_baltic_localization_summary.csv", index=False)
print(res.to_string(index=False))
