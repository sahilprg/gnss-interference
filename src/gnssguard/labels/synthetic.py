"""Inject synthetic GNSS interference into clean flights, with exact ground truth.

Real data never says where a jammer was. Synthetic attacks do: every event records its
true location, period and type, and every affected point is marked. They are used to
test what the rules (and later the models) can and cannot catch, and to measure source
localization. Results on synthetic data are an upper bound and are always reported
separately from real-data results.

Jamming
    A ground transmitter at (lat, lon, height). An aircraft is affected only within
    radio line of sight, so higher aircraft are affected from farther away:
        horizon_km = 4.12 * (sqrt(aircraft_height_m) + sqrt(jammer_height_m))
    and within the jammer's power range. Inside the core (distance / reach <= core_frac)
    GPS is lost: NIC and NACp drop to 0, GPS altitude may disappear, some position
    messages are lost, and optionally the tracker falls back to MLAT (as seen in real
    Baltic data). In the outer ring the effect is partial and probabilistic.

Spoofing
    Inside an area, a fake GPS signal pulls reported positions toward a target point,
    either gradually ("drift", at drift_kt) or at once ("jump"). The aircraft's
    reported ground speed stays true, and NIC often stays high, which is what makes
    spoofing hard to see.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from gnssguard.features.cells import CELL_KEYS, GridSpec, _haversine_km

EARTH_R_KM = 6371.0


def radio_horizon_km(aircraft_alt_ft, jammer_height_m: float):
    h_ac = np.clip(np.asarray(aircraft_alt_ft, dtype="float64") * 0.3048, 0, None)
    return 4.12 * (np.sqrt(h_ac) + np.sqrt(max(jammer_height_m, 0.0)))


def _bearing(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    return np.arctan2(np.sin(dl) * np.cos(p2),
                      np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))


def _destination(lat, lon, bearing_rad, dist_km):
    p1, l1 = np.radians(lat), np.radians(lon)
    d = np.asarray(dist_km, dtype="float64") / EARTH_R_KM
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(bearing_rad))
    l2 = l1 + np.arctan2(np.sin(bearing_rad) * np.sin(d) * np.cos(p1),
                         np.cos(d) - np.sin(p1) * np.sin(p2))
    return np.degrees(p2), (np.degrees(l2) + 540) % 360 - 180


@dataclass
class Jamming:
    event_id: str
    lat: float
    lon: float
    start: str
    end: str
    range_km: float = 200.0
    height_m: float = 30.0
    core_frac: float = 0.6
    p_drop: float = 0.15
    p_geom_lost: float = 0.5
    p_mlat: float = 0.0
    type: str = field(default="jamming", init=False)


@dataclass
class Spoofing:
    event_id: str
    lat: float
    lon: float
    radius_km: float
    start: str
    end: str
    target_lat: float
    target_lon: float
    mode: str = "drift"  # drift | jump
    drift_kt: float = 250.0
    p_nic_kept: float = 0.7
    spoof_alt_ft: float | None = None  # jump mode: GPS altitude of the fake location
    type: str = field(default="spoofing", init=False)


Event = Jamming | Spoofing


def inject(points: pd.DataFrame, events: list[Event], seed: int = 0
           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (points with interference applied, truth table of affected points).

    The truth table holds every affected point (including dropped ones) with its
    true position, so truth can be placed in the cells where aircraft really were.
    Later events do not affect points an earlier event already affected.
    """
    rng = np.random.default_rng(seed)
    df = points.copy().reset_index(drop=True)
    for c in ("nic", "nac_p"):
        df[c] = df[c].astype("float64")
    df["alt_geom"] = df["alt_geom"].astype("float64")
    df["q_age_s"] = df["q_age_s"].astype("float64")
    df["src"] = df["src"].astype(str)
    df["true_lat"], df["true_lon"] = df["lat"], df["lon"]
    df["inj_type"] = ""
    df["inj_event"] = ""
    drop = np.zeros(len(df), dtype=bool)

    alt_ft = df["alt_baro"].astype("float64").fillna(df["alt_geom"])
    airborne = ~df["on_ground"].astype(bool) & alt_ft.notna()
    ts = df["ts"]

    for ev in events:
        t0, t1 = pd.Timestamp(ev.start), pd.Timestamp(ev.end)
        live = airborne & (ts >= t0) & (ts < t1) & (df["inj_type"] == "")
        d = pd.Series(_haversine_km(df["true_lat"], df["true_lon"], ev.lat, ev.lon),
                      index=df.index)
        if isinstance(ev, Jamming):
            reach = np.minimum(ev.range_km, radio_horizon_km(alt_ft, ev.height_m))
            inside = live & (d <= reach)
            s = d / reach
            core = inside & (s <= ev.core_frac)
            p_ring = ((1 - s) / (1 - ev.core_frac)).clip(0, 1)
            partial = inside & ~core & (rng.random(len(df)) < p_ring)
            hit = core | partial

            df.loc[core, ["nic", "nac_p"]] = 0.0
            n_part = int(partial.sum())
            df.loc[partial, "nic"] = rng.choice(7, n_part, p=[.4, .1, .1, .1, .1, .1, .1])
            df.loc[partial, "nac_p"] = rng.integers(0, 8, n_part)
            df.loc[hit, "q_age_s"] = 0.0
            geom_lost = core & (rng.random(len(df)) < ev.p_geom_lost)
            df.loc[geom_lost, "alt_geom"] = np.nan
            to_mlat = core & (rng.random(len(df)) < ev.p_mlat)
            df.loc[to_mlat, "src"] = "mlat"
            drop |= (core & (rng.random(len(df)) < ev.p_drop)).to_numpy()
        else:
            hit = live & (d <= ev.radius_km)
            if hit.any():
                idx = df.index[hit]
                lat0, lon0 = df.loc[idx, "true_lat"], df.loc[idx, "true_lon"]
                if ev.mode == "jump":
                    noise_b = rng.uniform(0, 2 * np.pi, len(idx))
                    noise_d = np.abs(rng.normal(0, 0.3, len(idx)))
                    nl, no = _destination(np.full(len(idx), ev.target_lat),
                                          np.full(len(idx), ev.target_lon), noise_b, noise_d)
                    if ev.spoof_alt_ft is not None:
                        df.loc[idx, "alt_geom"] = ev.spoof_alt_ft
                else:
                    first = ts[hit].groupby(df.loc[hit, "icao"]).transform("min")
                    hours = (ts[hit] - first).dt.total_seconds() / 3600.0
                    to_target = _haversine_km(lat0, lon0, ev.target_lat, ev.target_lon)
                    move = np.minimum(ev.drift_kt * 1.852 * hours, to_target)
                    b = _bearing(lat0, lon0, ev.target_lat, ev.target_lon)
                    nl, no = _destination(lat0, lon0, b, move)
                df.loc[idx, "lat"], df.loc[idx, "lon"] = nl, no
                lose = rng.random(len(idx)) >= ev.p_nic_kept
                df.loc[idx[lose], "nic"] = rng.integers(0, 7, int(lose.sum()))
                df.loc[idx, "q_age_s"] = 0.0
        df.loc[hit, "inj_type"] = ev.type
        df.loc[hit, "inj_event"] = ev.event_id

    affected = df["inj_type"] != ""
    truth = df.loc[affected, ["icao", "ts", "true_lat", "true_lon", "inj_type", "inj_event"]]
    truth = truth.assign(alt_ft=alt_ft[affected], dropped=drop[affected.to_numpy()])
    out = df.loc[~drop].reset_index(drop=True)
    return out, truth.reset_index(drop=True)


# Attack profiles. "hard" is a deliberately different distribution used only for testing:
# short, weak jammers that mostly degrade aircraft partially, and slow spoofing drifts
# that keep integrity high. A model that only learned the "standard" generator's habits
# will fall down here.
PROFILES = {
    "standard": {"durations": [45, 60, 90, 120, 180], "jam_range": (120, 300), "core_frac": 0.6,
                 "p_drop": 0.15, "spoof_radius": (60, 150), "drift": (150, 400),
                 "p_nic_kept": 0.7, "modes": ["drift", "jump"]},
    "hard": {"durations": [15, 30, 45], "jam_range": (60, 140), "core_frac": 0.25,
             "p_drop": 0.3, "spoof_radius": (40, 80), "drift": (60, 150),
             "p_nic_kept": 0.9, "modes": ["drift"]},
}


def random_events(points: pd.DataFrame, n_jamming: int, n_spoofing: int, seed: int = 0,
                  min_sep_km: float = 450.0, profile: str = "standard") -> list[Event]:
    """Place events where there is traffic, separated in space or time."""
    pr = PROFILES[profile]
    rng = np.random.default_rng(seed)
    alt = points["alt_baro"].astype("float64")
    pool = points[~points["on_ground"].astype(bool) & (alt >= 20_000)
                  & points["ts"].dt.hour.between(5, 19)]
    if pool.empty:
        raise ValueError("no cruising traffic between 05 and 20 UTC to place events in")
    events: list[Event] = []
    placed: list[tuple[float, float, pd.Timestamp, pd.Timestamp]] = []
    kinds = ["jamming"] * n_jamming + ["spoofing"] * n_spoofing
    for i, kind in enumerate(kinds):
        for _ in range(200):
            p = pool.iloc[int(rng.integers(len(pool)))]
            start = p["ts"].floor("15min")
            end = start + pd.Timedelta(minutes=int(rng.choice(pr["durations"])))
            clash = any(
                _haversine_km(p["lat"], p["lon"], la, lo) < min_sep_km and start < e and end > s
                for la, lo, s, e in placed)
            if not clash:
                break
        placed.append((p["lat"], p["lon"], start, end))
        eid = f"{kind[0].upper()}{i + 1:02d}"
        if kind == "jamming":
            events.append(Jamming(
                eid, float(p["lat"]), float(p["lon"]), start.isoformat(), end.isoformat(),
                range_km=float(rng.uniform(*pr["jam_range"])),
                height_m=float(rng.uniform(10, 100)), core_frac=pr["core_frac"],
                p_drop=pr["p_drop"], p_mlat=0.0))
        else:
            b = rng.uniform(0, 2 * np.pi)
            tl, to = _destination(p["lat"], p["lon"], b, rng.uniform(80, 200))
            mode = str(rng.choice(pr["modes"]))
            events.append(Spoofing(
                eid, float(p["lat"]), float(p["lon"]), float(rng.uniform(*pr["spoof_radius"])),
                start.isoformat(), end.isoformat(), float(tl), float(to), mode=mode,
                drift_kt=float(rng.uniform(*pr["drift"])), p_nic_kept=pr["p_nic_kept"],
                spoof_alt_ft=300.0 if mode == "jump" else None))
    return events


def truth_cells(points: pd.DataFrame, spec: GridSpec, min_share: float = 0.5) -> pd.DataFrame:
    """Affected aircraft per cell-window.

    points: the output of `inject` (all points, with inj_type / true_lat / true_lon).
    An aircraft counts as affected in a cell-window when at least `min_share` of its
    points there were affected. Jamming is placed by true position. Spoofing is placed
    by true position AND by reported position, because its evidence (impossible motion)
    appears where the fake positions are.
    """
    p = points[~points["on_ground"].astype(bool)]
    win = p["ts"].dt.floor(f"{spec.window_min}min")
    frames = []
    for lat_col, lon_col, only_spoof in (("true_lat", "true_lon", False), ("lat", "lon", True)):
        t = pd.DataFrame({
            "cell_lat": spec.cell_floor(p[lat_col]), "cell_lon": spec.cell_floor(p[lon_col]),
            "window": win, "icao": p["icao"].astype(str), "inj_type": p["inj_type"],
            "inj_event": p["inj_event"], "aff": p["inj_type"] != ""})
        keys = [*CELL_KEYS, "icao"]
        size = t.groupby(keys, observed=True).size().rename("n")
        hit = t[t["aff"]]
        if only_spoof:
            hit = hit[hit["inj_type"] == "spoofing"]
        if hit.empty:
            continue
        a = hit.groupby(keys, observed=True).agg(
            k=("aff", "size"), inj_type=("inj_type", "first"), inj_event=("inj_event", "first"))
        a = a.join(size)
        a = a[a["k"] >= min_share * a["n"]].reset_index()
        frames.append(a)
    if not frames:
        return pd.DataFrame(columns=[*CELL_KEYS, "n_ac_affected", "inj_type", "inj_event"])
    a = pd.concat(frames, ignore_index=True).drop_duplicates([*CELL_KEYS, "icao"])
    return (a.groupby(CELL_KEYS).agg(n_ac_affected=("icao", "nunique"),
                                      inj_type=("inj_type", "first"),
                                      inj_event=("inj_event", "first")).reset_index())


INTERFERENCE_LABELS = ("jamming", "spoofing", "both")


def evaluate(pred: pd.DataFrame, truth_cw: pd.DataFrame, spec: GridSpec,
             min_affected: int = 2) -> tuple[dict, pd.DataFrame]:
    """Detection of synthetic events at cell-window level.

    pred: cell-windows with `p_jamming`, `majority_vote_jamming`, `spoof_votes`, `label`.
    Truth positive: a cell-window with >= min_affected affected aircraft.
    Jamming is scored with p_jamming >= 0.5, spoofing with spoof_votes >= 1.
    False positives count only away from events (outside a one-cell, one-window buffer
    around any affected cell-window) and only where some rule could judge.
    """
    m = pred.merge(truth_cw, on=CELL_KEYS, how="outer")
    m["n_ac_affected"] = m["n_ac_affected"].fillna(0)
    m["inj_type"] = m["inj_type"].fillna("")
    jam_det = m["p_jamming"].fillna(0) >= 0.5
    jam_mv = m["majority_vote_jamming"].fillna(0) >= 0.5
    spoof_det = m["spoof_votes"].fillna(0) >= 1
    m["det"] = np.where(m["inj_type"] == "spoofing", spoof_det, jam_det)
    m["det_mv"] = np.where(m["inj_type"] == "spoofing", spoof_det, jam_mv)
    m["det_any"] = m["label"].isin(INTERFERENCE_LABELS)
    pos = m["n_ac_affected"] >= min_affected

    # Buffer: one cell around each affected cell-window, and one window before and after
    # (spoofed tracks snap back to the true position when an event ends, which shows up
    # as a jump in the next window).
    near = set()
    step = spec.cell_deg
    dw = pd.Timedelta(minutes=spec.window_min)
    for r in truth_cw.itertuples(index=False):
        for dl in (-step, 0, step):
            for dn in (-step, 0, step):
                for dt in (-dw, pd.Timedelta(0), dw):
                    near.add((round(r.cell_lat + dl, 4), round(r.cell_lon + dn, 4),
                              r.window + dt))
    keys = zip(m["cell_lat"].round(4), m["cell_lon"].round(4), m["window"], strict=True)
    away = ~pd.Series([k in near for k in keys], index=m.index)
    judged = m["label"].notna() & (m["label"] != "unknown")

    per_event = (m[pos].groupby(["inj_event", "inj_type"])
                 .agg(truth_cell_windows=("det", "size"), recall=("det", "mean"),
                      recall_majority_vote=("det_mv", "mean"), recall_any_label=("det_any", "mean"))
                 .round(3).reset_index())

    def rate(mask, col="det"):
        return round(float(m.loc[mask, col].mean()), 3) if mask.any() else None

    summary = {
        "truth_cell_windows": int(pos.sum()),
        "recall_jamming": rate(pos & (m["inj_type"] == "jamming")),
        "recall_spoofing": rate(pos & (m["inj_type"] == "spoofing")),
        "false_positive_rate_away_from_events": rate(away & judged, "det_any"),
        "judged_cell_windows_away_from_events": int((away & judged).sum()),
    }
    return summary, per_event


def save_events(events: list[Event], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(e) for e in events], indent=2), encoding="utf-8")


def run_experiment(processed: str | Path, region: str, date: str, *, n_jamming: int = 4,
                   n_spoofing: int = 4, seeds: list[int] = (0,), p_mlat: float = 0.0,
                   real_cells: str | Path | None = None, out_dir: str | Path = "data/synthetic",
                   report_dir: str | Path = "reports/synthetic",
                   spec: GridSpec = GridSpec()) -> pd.DataFrame:
    """Inject attacks into one (preferably calm) region-day and score the rules on it.

    The label model is fitted on the synthetic cells together with the real cell tables
    in `real_cells` (other region-days), as it would be in practice.
    """
    import pyarrow.parquet as pq

    from gnssguard.features.cells import POINT_COLS, build_from_points, load_cells
    from gnssguard.labels.labelmodel import DawidSkene
    from gnssguard.labels.pipeline import score
    from gnssguard.labels.rules import INTEGRITY_RULES, apply_rules

    folder = Path(processed) / f"region={region}" / f"date={date}"
    if not folder.exists():
        raise FileNotFoundError(f"{folder} not found; ingest that day first")
    pts = pq.read_table(folder, columns=POINT_COLS).to_pandas()
    real = load_cells(real_cells) if real_cells else pd.DataFrame()
    if not real.empty:
        real = real[~((real["region"] == region) & (real["date"] == date))]

    out_dir, report_dir = Path(out_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    summaries, per_events = [], []
    for seed in seeds:
        events = random_events(pts, n_jamming, n_spoofing, seed=seed)
        for e in events:
            if isinstance(e, Jamming):
                e.p_mlat = p_mlat
        inj, truth = inject(pts, events, seed=seed)
        cw, _ = build_from_points(inj, spec)
        fit_on = pd.concat([real, cw], ignore_index=True) if not real.empty else cw
        model = DawidSkene().fit(apply_rules(fit_on)[INTEGRITY_RULES])
        cw = pd.concat([cw, score(apply_rules(cw), model)], axis=1)
        truth_cw = truth_cells(inj, spec)
        summary, per = evaluate(cw, truth_cw, spec)
        summaries.append({"seed": seed, **summary})
        per_events.append(per.assign(seed=seed))

        run = out_dir / f"{region}_{date}_seed{seed}"
        run.mkdir(parents=True, exist_ok=True)
        save_events(events, run / "events.json")
        truth.to_parquet(run / "truth_points.parquet", index=False)
        truth_cw.to_parquet(run / "truth_cells.parquet", index=False)
        cw.assign(region=region, date=date, cell_deg=spec.cell_deg,
                  window_min=spec.window_min).to_parquet(run / "cells_labeled.parquet",
                                                         index=False)
    result = pd.DataFrame(summaries)
    result.to_csv(report_dir / f"{region}_{date}_summary.csv", index=False)
    pd.concat(per_events).to_csv(report_dir / f"{region}_{date}_per_event.csv", index=False)
    return result
