"""Trust report: intervals grouped by day, calibration against hand labels, aircraft
overlap between training and test days (docs/evaluation_protocol.md, "Trust report").
Reporting only: nothing here changes a model, rule or threshold."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gnssguard.features.cells import CELL_KEYS
from gnssguard.labels.holdout import FLAGGED, wilson

SCORE_BANDS = [0.0, 0.1, 0.5, 0.9, 1.0001]
SCORE_NAMES = ["<0.1", "0.1-0.5", "0.5-0.9", ">=0.9"]
AIRCRAFT_BANDS = [3, 6, 11, 10**6]
AIRCRAFT_NAMES = ["3-5", "6-10", "11+"]


def day_bootstrap(values: pd.Series, days: pd.Series, n: int = 4000, seed: int = 0
                  ) -> tuple[float, float, float]:
    """Mean of per-day means, with a 95% interval from resampling whole days."""
    per_day = values.groupby(days).mean().dropna().to_numpy()
    if len(per_day) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    boots = rng.choice(per_day, size=(n, len(per_day)), replace=True).mean(axis=1)
    return (round(float(per_day.mean()), 4), round(float(np.quantile(boots, 0.025)), 4),
            round(float(np.quantile(boots, 0.975)), 4))


def rolling_intervals(rolling_dir: str | Path) -> pd.DataFrame:
    d = Path(rolling_dir)
    syn = pd.read_csv(d / "rolling_synthetic.csv")
    rows = []
    for (region, profile, pred), g in syn.groupby(["synth_region", "profile", "predictor"]):
        for metric in ("recall_jamming", "recall_spoofing",
                       "false_positive_rate_away_from_events"):
            m, lo, hi = day_bootstrap(g[metric], g["test_day"])
            rows.append({"source": "synthetic", "region": region, "profile": profile,
                         "predictor": pred, "metric": metric, "mean": m, "ci95_low": lo,
                         "ci95_high": hi, "days": g["test_day"].nunique()})
    real = pd.read_csv(d / "rolling_real_flag_rates.csv")
    for (region, pred), g in real.groupby(["region", "predictor"]):
        m, lo, hi = day_bootstrap(g["flagged_share_dense"], g["test_day"])
        rows.append({"source": "real", "region": region, "profile": "",
                     "predictor": pred, "metric": "flagged_share_dense", "mean": m,
                     "ci95_low": lo, "ci95_high": hi, "days": g["test_day"].nunique()})
    return pd.DataFrame(rows)


def human_scored(gold_files: list[str], cells: pd.DataFrame, det) -> pd.DataFrame:
    """Decided hand labels joined to the published score of the same cell-window."""
    frames = []
    for f in gold_files:
        g = pd.read_csv(f, dtype={"label": str})
        g["set"] = Path(f).stem
        frames.append(g)
    g = pd.concat(frames, ignore_index=True)
    g = g[g["label"].notna() & ~g["label"].isin(["", "unsure"])].copy()
    g["window"] = pd.to_datetime(g["window"], utc=True)
    c = cells.copy()
    c["window"] = pd.to_datetime(c["window"], utc=True)
    p = det.predict(c, calibrated=True)
    c["score"] = np.maximum(p["p_jamming"], p["p_spoofing"])
    c["model_label"] = p["label"].astype(str)
    keys = ["region", "date", *CELL_KEYS]
    c["date"], g["date"] = c["date"].astype(str), g["date"].astype(str)
    m = g.merge(c[[*keys, "score", "model_label", "n_ac_usable"]], on=keys, how="inner")
    m = m[m["n_ac_usable"].fillna(0) >= 3].copy()
    m["human"] = m["label"].isin(FLAGGED)
    return m


def reliability_by(m: pd.DataFrame, by: str | None) -> pd.DataFrame:
    m = m.assign(band=pd.cut(m["score"], SCORE_BANDS, labels=SCORE_NAMES, right=False),
                 aircraft=pd.cut(m["n_ac_usable"], AIRCRAFT_BANDS, labels=AIRCRAFT_NAMES,
                                 right=False))
    keys = ["band"] if by is None else [by, "band"]
    rows = []
    for k, g in m.groupby(keys, observed=True):
        k = k if isinstance(k, tuple) else (k,)
        n, h = len(g), int(g["human"].sum())
        lo, hi = wilson(h, n)
        rows.append({**({by: k[0]} if by else {}), "score_band": k[-1], "cells": n,
                     "mean_score": round(float(g["score"].mean()), 3),
                     "human_interference_share": round(h / n, 3), "ci95_low": lo,
                     "ci95_high": hi})
    return pd.DataFrame(rows)


def aircraft_overlap(processed: str | Path, train_days: list[str], test_days: list[str],
                     regions: list[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    def icaos(day: str) -> set:
        out = set()
        for r in regions:
            f = Path(processed) / f"region={r}" / f"date={day}"
            if f.exists():
                out |= set(pq.read_table(f, columns=["icao"]).column("icao").to_pylist())
        return out

    seen = set()
    for d in train_days:
        seen |= icaos(d)
    rows = []
    for d in test_days:
        s = icaos(d)
        rows.append({"test_day": d, "aircraft": len(s),
                     "seen_in_training": round(len(s & seen) / len(s), 3) if s else None})
    return pd.DataFrame(rows)
