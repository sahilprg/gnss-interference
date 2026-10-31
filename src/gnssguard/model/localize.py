"""Estimate where a jammer is, from which aircraft it affected and which it did not.

1. Incidents: flagged cell-windows joined into connected groups (touching cells, in the
   same or adjacent windows).
2. For each incident, take every usable aircraft-window in its windows and nearby:
   position, altitude, degraded or not.
3. Two estimates:
   * centroid: mean position of the degraded aircraft (baseline)
   * horizon fit: a ground jammer reaches an aircraft only within radio line of sight,
         reach = min(R, 4.12 * (sqrt(h_aircraft_m) + sqrt(h_jammer_m)))
     so high aircraft are affected from farther away than low ones. A grid search over
     jammer position and power range R picks the hypothesis that best separates degraded
     from healthy aircraft (balanced accuracy). Near-best hypotheses give the raw
     uncertainty.
4. Honest uncertainty. The raw spread is too small (it covered the truth for only 75%
   of synthetic test events). It is scaled and floored with constants fitted on
   synthetic events from earlier days (`fit_uncertainty`), aiming at 90% coverage. A fit
   that separates degraded from healthy aircraft poorly (score < MIN_SCORE) usually
   means several jammers were merged into one incident; such estimates are marked
   `confident = False` and should be shown as an area, not a point.

On real data only coarse estimates should be published (docs/model.md).
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from gnssguard.features.cells import (
    CELL_KEYS,
    GridSpec,
    _haversine_km,
    aircraft_windows,
    prepare_points,
)
from gnssguard.labels.synthetic import Jamming, radio_horizon_km

R_GRID_KM = np.arange(50, 451, 25)
JAMMER_HEIGHT_M = 30.0

# Fitted with `fit_uncertainty` on 168 synthetic jamming events (standard, hard and
# stress profiles, 3 seeds each) injected on 22-25 Oct 2026, never on the test day:
# 90.7% of the 107 confident estimates fell inside 3.75 x the raw spread. See docs/model.md.
UNCERTAINTY_SCALE = 3.75
UNCERTAINTY_FLOOR_KM = 0.0
MIN_SCORE = 0.85
SCALE_GRID = np.arange(1.0, 6.01, 0.25)
FLOOR_GRID_KM = np.arange(0.0, 80.1, 5.0)


def aircraft_points(points: pd.DataFrame, spec: GridSpec = GridSpec()) -> pd.DataFrame:
    """One row per usable aircraft-window: position, altitude, degraded flag."""
    ac = aircraft_windows(prepare_points(points, spec))
    ac = ac[ac["usable"] & ac["alt_ft"].notna()]
    return ac[[*CELL_KEYS, "icao", "lat", "lon", "alt_ft", "degraded"]].reset_index(drop=True)


def incidents(flagged: pd.DataFrame, spec: GridSpec = GridSpec()) -> pd.DataFrame:
    """Label connected groups of flagged cell-windows. Returns flagged rows + incident_id."""
    f = flagged[[*CELL_KEYS]].copy()
    f["window"] = pd.to_datetime(f["window"], utc=True)
    step, dw = spec.cell_deg, pd.Timedelta(minutes=spec.window_min)
    keys = list(zip(f["cell_lat"].round(4), f["cell_lon"].round(4), f["window"], strict=True))
    index = {k: i for i, k in enumerate(keys)}
    comp = np.full(len(keys), -1)
    nxt = 0
    for i in range(len(keys)):
        if comp[i] >= 0:
            continue
        comp[i] = nxt
        queue = deque([i])
        while queue:
            la, lo, w = keys[queue.popleft()]
            for dl in (-step, 0, step):
                for dn in (-step, 0, step):
                    for dt in (-dw, pd.Timedelta(0), dw):
                        j = index.get((round(la + dl, 4), round(lo + dn, 4), w + dt))
                        if j is not None and comp[j] < 0:
                            comp[j] = nxt
                            queue.append(j)
        nxt += 1
    return f.assign(incident_id=comp)


def localize(pts: pd.DataFrame, *, jammer_height_m: float = JAMMER_HEIGHT_M,
             search_deg: float = 4.0, step_deg: float = 0.1,
             r_grid: np.ndarray = R_GRID_KM, near_best: float = 0.005) -> dict | None:
    """Estimate one jammer from aircraft points (lat, lon, alt_ft, degraded)."""
    deg = pts["degraded"].to_numpy(bool)
    if deg.sum() < 3:
        return None
    lat, lon = pts["lat"].to_numpy(), pts["lon"].to_numpy()
    c_lat, c_lon = float(lat[deg].mean()), float(lon[deg].mean())
    k = max(np.cos(np.radians(c_lat)), 0.2)
    glat = np.arange(c_lat - search_deg, c_lat + search_deg + 1e-9, step_deg)
    glon = np.arange(c_lon - search_deg / k, c_lon + search_deg / k + 1e-9, step_deg / k)
    cand_lat, cand_lon = (a.ravel() for a in np.meshgrid(glat, glon, indexing="ij"))
    horizon = radio_horizon_km(pts["alt_ft"].to_numpy(), jammer_height_m)
    n_deg, n_ok = deg.sum(), (~deg).sum()

    best = np.full((len(cand_lat), len(r_grid)), -np.inf, dtype=np.float32)
    for s in range(0, len(cand_lat), 800):
        d = _haversine_km(cand_lat[s:s + 800, None], cand_lon[s:s + 800, None],
                          lat[None, :], lon[None, :]).astype(np.float32)
        for j, r in enumerate(r_grid):
            inside = d <= np.minimum(r, horizon)[None, :]
            tpr = (inside & deg).sum(1) / n_deg
            tnr = ((~inside) & ~deg).sum(1) / n_ok if n_ok else np.full(len(d), 0.5)
            best[s:s + 800, j] = (tpr + tnr) / 2
    top = best.max()
    ci, rj = np.nonzero(best >= top - near_best)
    est_lat, est_lon = float(cand_lat[ci].mean()), float(cand_lon[ci].mean())
    spread = _haversine_km(est_lat, est_lon, cand_lat[ci], cand_lon[ci])
    return {"lat": est_lat, "lon": est_lon, "range_km": float(np.median(r_grid[rj])),
            "score": float(top),
            # never claim better than the search grid can resolve
            "uncertainty_km": float(max(np.percentile(spread, 90), step_deg * 111.2)),
            "centroid_lat": c_lat, "centroid_lon": c_lon,
            "n_degraded": int(n_deg), "n_healthy": int(n_ok)}


def localize_incidents(ac: pd.DataFrame, inc: pd.DataFrame, spec: GridSpec = GridSpec(),
                       margin_deg: float = 3.0, min_cells: int = 3) -> pd.DataFrame:
    """One estimate per incident with at least `min_cells` flagged cell-windows."""
    ac = ac.assign(window=pd.to_datetime(ac["window"], utc=True))
    rows = []
    for iid, g in inc.groupby("incident_id"):
        if len(g) < min_cells:
            continue
        sel = (ac["window"].isin(g["window"].unique())
               & ac["lat"].between(g["cell_lat"].min() - margin_deg,
                                   g["cell_lat"].max() + spec.cell_deg + margin_deg)
               & ac["lon"].between(g["cell_lon"].min() - margin_deg,
                                   g["cell_lon"].max() + spec.cell_deg + margin_deg))
        est = localize(ac[sel])
        if est is None:
            continue
        rows.append({"incident_id": int(iid), "cell_windows": len(g),
                     "start": g["window"].min(), "end": g["window"].max(), **est})
    return calibrate_uncertainty(pd.DataFrame(rows))


def calibrate_uncertainty(est: pd.DataFrame, scale: float | None = None,
                          floor_km: float | None = None, min_score: float = MIN_SCORE
                          ) -> pd.DataFrame:
    """Add uncertainty_cal_km (scaled, floored raw spread) and the confident flag."""
    if est.empty:
        return est
    scale = UNCERTAINTY_SCALE if scale is None else scale
    floor_km = UNCERTAINTY_FLOOR_KM if floor_km is None else floor_km
    return est.assign(uncertainty_cal_km=np.maximum(est["uncertainty_km"] * scale, floor_km),
                      confident=est["score"] >= min_score)


def fit_uncertainty(loc: pd.DataFrame, target: float = 0.9,
                    scales: np.ndarray = SCALE_GRID, floors: np.ndarray = FLOOR_GRID_KM
                    ) -> dict:
    """Smallest (scale, floor) whose interval covers the truth for `target` of the
    confident, found events, choosing the narrowest median interval among those."""
    f = loc[loc["found"] & (loc["score"] >= MIN_SCORE)]
    best = None
    for s in scales:
        for fl in floors:
            width = np.maximum(f["uncertainty_km"] * s, fl)
            cover = float((f["fit_error_km"] <= width).mean())
            if cover >= target and (best is None or width.median() < best["median_km"]):
                best = {"scale": float(s), "floor_km": float(fl), "coverage": round(cover, 3),
                        "median_km": float(width.median()), "events": len(f)}
    return best or {}


def match_to_events(est: pd.DataFrame, events: list, spec: GridSpec = GridSpec(),
                    max_km: float = 600.0) -> pd.DataFrame:
    """For each synthetic jamming event, the overlapping incident nearest the truth."""
    dw = pd.Timedelta(minutes=spec.window_min)
    rows = []
    for e in events:
        if not isinstance(e, Jamming):
            continue
        t0, t1 = pd.Timestamp(e.start), pd.Timestamp(e.end)
        row = {"event_id": e.event_id, "range_km": e.range_km, "found": False}
        if not est.empty:
            live = est[(est["start"] < t1) & (est["end"] + dw > t0)]
            if not live.empty:
                d = _haversine_km(live["centroid_lat"], live["centroid_lon"], e.lat, e.lon)
                j = int(np.argmin(d))
                if d.iloc[j] <= max_km:
                    m = live.iloc[j]
                    fit_err = float(_haversine_km(m.lat, m.lon, e.lat, e.lon))
                    row.update(found=True, fit_error_km=fit_err,
                               centroid_error_km=float(d.iloc[j]),
                               uncertainty_km=float(m.uncertainty_km),
                               uncertainty_cal_km=float(m.uncertainty_cal_km),
                               confident=bool(m.confident),
                               fitted_range_km=float(m.range_km), score=float(m.score),
                               n_degraded=int(m.n_degraded), n_healthy=int(m.n_healthy),
                               cell_windows=int(m.cell_windows))
        rows.append(row)
    return pd.DataFrame(rows)
