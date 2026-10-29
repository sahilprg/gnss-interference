"""Training table for the detectors: features with context, and targets.

Features per cell-window
* own: the cell-window's counts and shares from `features/cells.py`
* neighbours (`nb_`): the 8 surrounding cells in the same window
* before (`prev_`) and after (`next_`): the same cell one window earlier and later

No latitude, longitude, region or time of day: the model must learn from evidence,
not memorise that some places are usually jammed.

Targets
* real days: weak labels. The label model is fitted on the training days only and
  then applied to every day. Cells no rule could judge are left out.
* synthetic: attacks injected into a calm region, with exact truth (`labels/synthetic`).

Splits are by day: training days, a validation day for early stopping, and the test
day(s), which are kept sealed for the final evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gnssguard.features.cells import CELL_KEYS, POINT_COLS, GridSpec, build_from_points
from gnssguard.labels import synthetic as S
from gnssguard.labels.labelmodel import DawidSkene
from gnssguard.labels.pipeline import score
from gnssguard.labels.rules import INTEGRITY_RULES, apply_rules

log = logging.getLogger(__name__)

OWN = ["n_aircraft", "n_ac_usable", "n_ac_degraded", "share_ac_degraded", "n_ac_nic0",
       "n_ac_nacp_low", "share_ac_nacp_low", "n_ac_mlat_fallback", "n_ac_jump",
       "n_ac_switch_jump", "n_ac_speed_mismatch", "n_ac_altpair", "n_ac_altdev",
       "share_ac_altdev", "share_pts_degraded", "n_chronic", "mean_alt_ft", "max_alt_ft"]
_NB_SUM = ["n_ac_usable", "n_ac_degraded", "n_ac_nic0", "n_ac_mlat_fallback",
           "n_ac_speed_mismatch", "n_ac_jump", "n_ac_altdev"]
NEIGHBOUR = ["nb_cells", "nb_ac_usable", "nb_ac_degraded", "nb_ac_nic0", "nb_ac_mlat_fallback",
             "nb_ac_speed_mismatch", "nb_ac_jump", "nb_ac_altdev", "nb_share_degraded",
             "nb_max_share", "nb_hot"]
_T = ["n_ac_usable", "n_ac_degraded", "n_ac_mlat_fallback", "n_ac_speed_mismatch",
      "n_ac_jump", "share_ac_degraded"]
BEFORE = [f"prev_{c}" for c in _T] + ["prev_nb_share_degraded"]
AFTER = [f"next_{c}" for c in _T] + ["next_nb_share_degraded"]
FEATURES = OWN + NEIGHBOUR + BEFORE + AFTER
GROUPS = {"own": OWN, "neighbours": NEIGHBOUR, "before": BEFORE, "after": AFTER}

HOT_MIN_USABLE = 3


def _key(df: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([df["cell_lat"].round(4), df["cell_lon"].round(4),
                                      pd.to_datetime(df["window"], utc=True)])


def add_context(cells: pd.DataFrame) -> pd.DataFrame:
    """Neighbour and before/after features for ONE region-day of cell-windows."""
    c = cells.reset_index(drop=True).copy()
    c["window"] = pd.to_datetime(c["window"], utc=True)
    deg = float(c["cell_deg"].iloc[0]) if "cell_deg" in c else 1.0
    win = pd.Timedelta(minutes=int(c["window_min"].iloc[0]) if "window_min" in c else 15)
    src = c[CELL_KEYS + _NB_SUM].copy()
    src.index = _key(src)
    share = (src["n_ac_degraded"] / src["n_ac_usable"]).where(src["n_ac_usable"] >= HOT_MIN_USABLE)

    acc = {f"nb_ac_{k[5:]}": np.zeros(len(c)) for k in _NB_SUM}
    nb_cells = np.zeros(len(c))
    nb_hot = np.zeros(len(c))
    nb_max = np.full(len(c), np.nan)
    for dl in (-1, 0, 1):
        for dn in (-1, 0, 1):
            if dl == dn == 0:
                continue
            look = pd.MultiIndex.from_arrays([(c["cell_lat"] + dl * deg).round(4),
                                              (c["cell_lon"] + dn * deg).round(4), c["window"]])
            got = src.reindex(look)
            have = got["n_ac_usable"].notna().to_numpy()
            nb_cells += have
            for k in _NB_SUM:
                acc[f"nb_ac_{k[5:]}"] += got[k].fillna(0).to_numpy()
            s = share.reindex(look).to_numpy()
            nb_hot += np.nan_to_num(s) >= 0.5
            nb_max = np.fmax(nb_max, s)
    out = pd.DataFrame(acc)
    out["nb_cells"] = nb_cells
    out["nb_share_degraded"] = (out["nb_ac_degraded"] / out["nb_ac_usable"]).where(
        out["nb_ac_usable"] > 0)
    out["nb_max_share"] = nb_max
    out["nb_hot"] = nb_hot

    own = c[CELL_KEYS + _T].copy()
    own.index = _key(own)
    own["nb_share_degraded"] = out["nb_share_degraded"].to_numpy()
    for prefix, shift in (("prev", -win), ("next", win)):
        look = pd.MultiIndex.from_arrays([c["cell_lat"].round(4), c["cell_lon"].round(4),
                                          c["window"] + shift])
        got = own.reindex(look)
        for col in _T:
            v = got[col].to_numpy()
            out[f"{prefix}_{col}"] = v if col.startswith("share") else np.nan_to_num(v)
        out[f"{prefix}_nb_share_degraded"] = got["nb_share_degraded"].to_numpy()
    return pd.concat([c, out[NEIGHBOUR + BEFORE + AFTER]], axis=1)


def with_context(cells: pd.DataFrame) -> pd.DataFrame:
    """add_context for a table holding several region-days."""
    parts = [add_context(g) for _, g in cells.groupby(["region", "date"], sort=True)]
    return pd.concat(parts, ignore_index=True) if parts else cells


def fit_label_model(cells: pd.DataFrame, train_dates: list[str]) -> DawidSkene:
    votes = apply_rules(cells[cells["date"].isin(train_dates)])
    return DawidSkene().fit(votes[INTEGRITY_RULES])


def real_targets(cells: pd.DataFrame, model: DawidSkene) -> pd.DataFrame:
    """Weak-label targets; rows no rule could judge get NaN targets."""
    votes = apply_rules(cells)
    sc = score(votes, model)
    judged = sc["label"] != "unknown"
    y_jam = (sc["p_jamming"].fillna(0) >= 0.5).astype(float).where(judged)
    y_spoof = (sc["spoof_votes"] >= 1).astype(float).where(judged)
    return pd.DataFrame({"y_jam": y_jam, "y_spoof": y_spoof, "rule_label": sc["label"].astype(str),
                         "rule_p_jamming": sc["p_jamming"], "rule_spoof_votes": sc["spoof_votes"]},
                        index=cells.index)


def synthetic_rows(processed: str | Path, region: str, dates: list[str], seeds: list[int],
                   n_jamming: int = 4, n_spoofing: int = 4, spec: GridSpec = GridSpec(),
                   p_mlat: float = 0.0, profile: str = "standard"
                   ) -> tuple[pd.DataFrame, list[dict]]:
    """Cell-windows with context and exact targets from attacks injected into `region`.

    Truth positive: >= 2 affected aircraft. Negative: no affected aircraft and outside a
    one-cell, one-window buffer. Everything in between is left out (ambiguous).
    Also returns, per run, the injected points and events (for localization).
    """
    frames, runs = [], []
    for date in dates:
        folder = Path(processed) / f"region={region}" / f"date={date}"
        if not folder.exists():
            log.warning("no %s data for %s; skipping synthetic", region, date)
            continue
        pts = pq.read_table(folder, columns=POINT_COLS).to_pandas()
        for seed in seeds:
            events = S.random_events(pts, n_jamming, n_spoofing, seed=seed, profile=profile)
            for e in events:
                if isinstance(e, S.Jamming):
                    e.p_mlat = p_mlat
            inj, _ = S.inject(pts, events, seed=seed)
            cw, _ = build_from_points(inj, spec)
            cw = cw.assign(region=region, date=date, cell_deg=spec.cell_deg,
                           window_min=spec.window_min)
            cw = add_context(cw)
            truth = S.truth_cells(inj, spec)
            cw = cw.merge(truth, on=CELL_KEYS, how="left")
            cw["n_ac_affected"] = cw["n_ac_affected"].fillna(0)
            cw["inj_type"] = cw["inj_type"].fillna("")
            near = _near_events(cw, truth, spec)
            pos = cw["n_ac_affected"] >= 2
            neg = (cw["n_ac_affected"] == 0) & ~near
            cw["y_jam"] = np.where(pos & (cw["inj_type"] == "jamming"), 1.0,
                                   np.where(neg | (pos & (cw["inj_type"] != "jamming")), 0.0,
                                            np.nan))
            cw["y_spoof"] = np.where(pos & (cw["inj_type"] == "spoofing"), 1.0,
                                     np.where(neg | (pos & (cw["inj_type"] != "spoofing")), 0.0,
                                              np.nan))
            cw["seed"] = seed
            frames.append(cw)
            runs.append({"region": region, "date": date, "seed": seed, "events": events,
                         "points": inj, "truth": truth})
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return rows, runs


def _near_events(cw: pd.DataFrame, truth: pd.DataFrame, spec: GridSpec) -> pd.Series:
    near = set()
    step, dw = spec.cell_deg, pd.Timedelta(minutes=spec.window_min)
    for r in truth.itertuples(index=False):
        for dl in (-step, 0, step):
            for dn in (-step, 0, step):
                for dt in (-dw, pd.Timedelta(0), dw):
                    near.add((round(r.cell_lat + dl, 4), round(r.cell_lon + dn, 4),
                              r.window + dt))
    keys = zip(cw["cell_lat"].round(4), cw["cell_lon"].round(4), cw["window"], strict=True)
    return pd.Series([k in near for k in keys], index=cw.index)
