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
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
    on_min: float | None = None  # duty cycle: on for on_min, off for off_min, repeating
    off_min: float = 0.0
    speed_kt: float = 0.0  # moving jammer
    heading_deg: float = 0.0
    type: str = field(default="jamming", init=False)

    def position_at(self, ts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Jammer position at each timestamp (fixed unless speed_kt > 0)."""
        n = len(ts)
        if not self.speed_kt:
            return np.full(n, self.lat), np.full(n, self.lon)
        t0, t1 = pd.Timestamp(self.start), pd.Timestamp(self.end)
        mid = t0 + (t1 - t0) / 2
        hours = ((ts - mid).dt.total_seconds() / 3600.0).to_numpy()
        dist = self.speed_kt * 1.852 * hours
        b = np.where(dist >= 0, np.radians(self.heading_deg), np.radians(self.heading_deg + 180))
        return _destination(np.full(n, self.lat), np.full(n, self.lon), b, np.abs(dist))

    def is_on(self, ts: pd.Series) -> np.ndarray:
        if not self.on_min:
            return np.ones(len(ts), dtype=bool)
        mins = (ts - pd.Timestamp(self.start)).dt.total_seconds().to_numpy() / 60.0
        return (mins % (self.on_min + self.off_min)) < self.on_min


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
        if isinstance(ev, Jamming):
            idx = df.index[live]
            jl, jo = ev.position_at(ts[live])
            d = pd.Series(np.inf, index=df.index)
            d[idx] = _haversine_km(df.loc[idx, "true_lat"], df.loc[idx, "true_lon"], jl, jo)
            on = pd.Series(False, index=df.index)
            on[idx] = ev.is_on(ts[live])
            live = live & on
        else:
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
# values are numbers or (low, high) ranges, unpacked into rng.uniform(*...)
PROFILES: dict[str, dict[str, Any]] = {
    "standard": {"durations": [45, 60, 90, 120, 180], "jam_range": (120, 300), "core_frac": 0.6,
                 "p_drop": 0.15, "spoof_radius": (60, 150), "drift": (150, 400),
                 "p_nic_kept": 0.7, "modes": ["drift", "jump"]},
    "hard": {"durations": [15, 30, 45], "jam_range": (60, 140), "core_frac": 0.25,
             "p_drop": 0.3, "spoof_radius": (40, 80), "drift": (60, 150),
             "p_nic_kept": 0.9, "modes": ["drift"]},
    # "stress": what real jammers do that neither generator above does. Test only.
    # Half the jammers switch on and off, a third move (vehicle or ship), half come with
    # a second jammer 60-200 km away at the same time, and half of the aircraft that lose
    # GPS in the core fall back to MLAT positions (seen in real Baltic data).
    "stress": {"durations": [15, 30, 45, 60, 90, 120], "jam_range": (60, 300), "core_frac": 0.4,
               "p_drop": 0.3, "p_mlat": 0.5, "p_intermittent": 0.5, "on_min": [5, 10, 15],
               "off_min": [5, 10, 20], "p_moving": 0.33, "speed_kt": (15, 60),
               "p_paired": 0.5, "pair_km": (60, 200),
               "spoof_radius": (40, 150), "drift": (60, 400), "p_nic_kept": 0.8,
               "modes": ["drift", "jump"]},
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
            jams = [(eid, float(p["lat"]), float(p["lon"]))]
            if "p_paired" in pr and rng.random() < pr["p_paired"]:
                la, lo = _destination(p["lat"], p["lon"], rng.uniform(0, 2 * np.pi),
                                      rng.uniform(*pr["pair_km"]))
                jams.append((f"{eid}b", float(la), float(lo)))
            for jid, la, lo in jams:
                ev = Jamming(jid, la, lo, start.isoformat(), end.isoformat(),
                             range_km=float(rng.uniform(*pr["jam_range"])),
                             height_m=float(rng.uniform(10, 100)), core_frac=pr["core_frac"],
                             p_drop=pr["p_drop"], p_mlat=pr.get("p_mlat", 0.0))
                if "p_intermittent" in pr and rng.random() < pr["p_intermittent"]:
                    ev.on_min = float(rng.choice(pr["on_min"]))
                    ev.off_min = float(rng.choice(pr["off_min"]))
                if "p_moving" in pr and rng.random() < pr["p_moving"]:
                    ev.speed_kt = float(rng.uniform(*pr["speed_kt"]))
                    ev.heading_deg = float(rng.uniform(0, 360))
                events.append(ev)
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


TRUTH_COLUMNS = [*CELL_KEYS, "n_ac_affected", "n_ac_jam", "n_ac_spoof", "n_ac_jam_observed",
                 "n_ac_spoof_observed", "inj_type", "inj_event_jam", "inj_event_spoof"]


def with_dropped(points: pd.DataFrame, truth: pd.DataFrame | None) -> pd.DataFrame:
    """The injected points plus the points the attack removed (from `inject`'s truth
    table), so truth is counted where aircraft really were, not only where they were
    still seen. Dropped points come only from jamming, so their reported position is
    their true position."""
    if truth is None or "dropped" not in truth or not truth["dropped"].any():
        return points
    d = truth[truth["dropped"]]
    extra = pd.DataFrame({"icao": d["icao"].astype(str).to_numpy(), "ts": d["ts"].to_numpy(),
                          "lat": d["true_lat"].to_numpy(), "lon": d["true_lon"].to_numpy(),
                          "true_lat": d["true_lat"].to_numpy(),
                          "true_lon": d["true_lon"].to_numpy(),
                          "inj_type": d["inj_type"].to_numpy(),
                          "inj_event": d["inj_event"].to_numpy(), "on_ground": False,
                          "observed": False})
    cols = ["icao", "ts", "lat", "lon", "true_lat", "true_lon", "inj_type", "inj_event",
            "on_ground"]
    base = points[cols].assign(icao=points["icao"].astype(str), observed=True)
    out = pd.concat([base, extra], ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out


def _affected(p: pd.DataFrame, spec: GridSpec, min_share: float) -> pd.DataFrame:
    """(cell-window, icao, type) for aircraft affected by that type in at least
    `min_share` of their points there. Jamming is placed by true position; spoofing by
    true AND reported position (its evidence appears where the fake positions are)."""
    win = p["ts"].dt.floor(f"{spec.window_min}min")
    frames = []
    for lat_col, lon_col, only_spoof in (("true_lat", "true_lon", False), ("lat", "lon", True)):
        t = pd.DataFrame({
            "cell_lat": spec.cell_floor(p[lat_col]), "cell_lon": spec.cell_floor(p[lon_col]),
            "window": win, "icao": p["icao"].astype(str), "inj_type": p["inj_type"],
            "inj_event": p["inj_event"]})
        keys = [*CELL_KEYS, "icao"]
        size = t.groupby(keys, observed=True).size().rename("n")
        hit = t[t["inj_type"] != ""]
        if only_spoof:
            hit = hit[hit["inj_type"] == "spoofing"]
        if hit.empty:
            continue
        a = hit.groupby([*keys, "inj_type"], observed=True).agg(
            k=("inj_event", "size"), inj_event=("inj_event", "first"))
        a = a.reset_index().join(size, on=keys)
        frames.append(a[a["k"] >= min_share * a["n"]])
    if not frames:
        return pd.DataFrame(columns=[*CELL_KEYS, "icao", "inj_type", "inj_event"])
    return (pd.concat(frames, ignore_index=True)
            .drop_duplicates([*CELL_KEYS, "icao", "inj_type"])[[*CELL_KEYS, "icao", "inj_type",
                                                                 "inj_event"]])


def truth_cells(points: pd.DataFrame, spec: GridSpec, min_share: float = 0.5,
                truth: pd.DataFrame | None = None) -> pd.DataFrame:
    """Affected aircraft per cell-window, counted per interference type.

    points: the output of `inject` (with inj_type / true_lat / true_lon). Pass `inject`'s
    truth table as `truth` so that points the attack removed still count (aircraft that
    lost their position entirely are the most affected, not unaffected). `*_observed`
    counts only aircraft with surviving points, so recall can be reported both end to
    end and on what was observable. A cell-window with both types affected has
    `inj_type` "both" and is a positive for both detectors.
    """
    p = with_dropped(points, truth)
    p = p[~p["on_ground"].astype(bool)]
    if "observed" not in p:
        p = p.assign(observed=True)
    all_ = _affected(p, spec, min_share)
    seen = _affected(p[p["observed"].astype(bool)], spec, min_share)
    if all_.empty:
        return pd.DataFrame(columns=TRUTH_COLUMNS)

    def count(df, kind, name):
        d = df[df["inj_type"] == kind]
        return d.groupby(CELL_KEYS)["icao"].nunique().rename(name)

    def event(kind, name):
        d = all_[all_["inj_type"] == kind]
        return d.groupby(CELL_KEYS)["inj_event"].first().rename(name)

    out = pd.concat([all_.groupby(CELL_KEYS)["icao"].nunique().rename("n_ac_affected"),
                     count(all_, "jamming", "n_ac_jam"), count(all_, "spoofing", "n_ac_spoof"),
                     count(seen, "jamming", "n_ac_jam_observed"),
                     count(seen, "spoofing", "n_ac_spoof_observed"),
                     event("jamming", "inj_event_jam"), event("spoofing", "inj_event_spoof")],
                    axis=1).reset_index()
    for c in ("n_ac_jam", "n_ac_spoof", "n_ac_jam_observed", "n_ac_spoof_observed"):
        out[c] = out[c].fillna(0).astype(int)
    out["inj_type"] = np.select([(out["n_ac_jam"] > 0) & (out["n_ac_spoof"] > 0),
                                 out["n_ac_jam"] > 0], ["both", "jamming"], "spoofing")
    return out[TRUTH_COLUMNS]


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
    for c in ("n_ac_affected", "n_ac_jam", "n_ac_spoof", "n_ac_jam_observed",
              "n_ac_spoof_observed"):
        m[c] = m[c].fillna(0) if c in m else 0
    m["inj_type"] = m["inj_type"].fillna("")
    jam_det = m["p_jamming"].fillna(0) >= 0.5
    jam_mv = m["majority_vote_jamming"].fillna(0) >= 0.5
    spoof_det = m["spoof_votes"].fillna(0) >= 1
    m["det_any"] = m["label"].isin(INTERFERENCE_LABELS)
    pos_jam = m["n_ac_jam"] >= min_affected
    pos_spoof = m["n_ac_spoof"] >= min_affected
    pos = pos_jam | pos_spoof
    # observable: the cell-window exists in the predictions and enough affected aircraft
    # still have points there; the rest can only be missed (counted, never hidden)
    exists = m["label"].notna()
    obs_jam = pos_jam & exists & (m["n_ac_jam_observed"] >= min_affected)
    obs_spoof = pos_spoof & exists & (m["n_ac_spoof_observed"] >= min_affected)

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

    per = []
    for kind, mask, det, det_mv, ev in (("jamming", pos_jam, jam_det, jam_mv, "inj_event_jam"),
                                        ("spoofing", pos_spoof, spoof_det, spoof_det,
                                         "inj_event_spoof")):
        if mask.any() and ev in m:
            per.append(pd.DataFrame({"inj_event": m.loc[mask, ev], "inj_type": kind,
                                     "det": det[mask], "det_mv": det_mv[mask],
                                     "det_any": m.loc[mask, "det_any"]}))
    per_event = (pd.concat(per).groupby(["inj_event", "inj_type"])
                 .agg(truth_cell_windows=("det", "size"), recall=("det", "mean"),
                      recall_majority_vote=("det_mv", "mean"),
                      recall_any_label=("det_any", "mean")).round(3).reset_index()
                 if per else pd.DataFrame(columns=["inj_event", "inj_type"]))

    def rate(mask, det):
        return round(float(det[mask].mean()), 3) if mask.any() else None

    summary = {
        "truth_cell_windows": int(pos.sum()),
        "truth_cell_windows_jamming": int(pos_jam.sum()),
        "truth_cell_windows_spoofing": int(pos_spoof.sum()),
        "unobservable_jamming": int((pos_jam & ~obs_jam).sum()),
        "unobservable_spoofing": int((pos_spoof & ~obs_spoof).sum()),
        "recall_jamming": rate(pos_jam, jam_det),
        "recall_jamming_observable": rate(obs_jam, jam_det),
        "recall_spoofing": rate(pos_spoof, spoof_det),
        "recall_spoofing_observable": rate(obs_spoof, spoof_det),
        "false_positive_rate_away_from_events": rate(away & judged, m["det_any"]),
        "judged_cell_windows_away_from_events": int((away & judged).sum()),
    }
    return summary, per_event


def save_events(events: list[Event], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(e) for e in events], indent=2), encoding="utf-8")


def run_experiment(processed: str | Path, region: str, date: str, *, n_jamming: int = 4,
                   n_spoofing: int = 4, seeds: Sequence[int] = (0,), p_mlat: float = 0.0,
                   real_cells: str | Path | None = None, out_dir: str | Path = "data/synthetic",
                   report_dir: str | Path = "reports/synthetic",
                   spec: GridSpec = GridSpec()) -> pd.DataFrame:
    """Inject attacks into one (preferably calm) region-day and score the rules on it.

    The label model is fitted on the synthetic cells together with the real cell tables
    in `real_cells` (other region-days), as it would be in practice.
    """

    from gnssguard.features.cells import build_from_points, load_cells, read_points
    from gnssguard.labels.labelmodel import DawidSkene
    from gnssguard.labels.pipeline import score
    from gnssguard.labels.rules import INTEGRITY_RULES, apply_rules

    folder = Path(processed) / f"region={region}" / f"date={date}"
    if not folder.exists():
        raise FileNotFoundError(f"{folder} not found; ingest that day first")
    pts = read_points(folder)
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
        truth_cw = truth_cells(inj, spec, truth=truth)
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
