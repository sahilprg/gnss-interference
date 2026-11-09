"""Phase 2 development check (not the gate): the integrity-only baseline against the
rules and the model on the hand-labeled sets, at a matched false-alert rate.

Baseline: a cell-window is interference if at least k usable aircraft have degraded NIC
(n_ac_degraded >= k), or, as a share, n_ac_degraded / n_ac_usable >= s. k (or s) is the
loosest value whose flagged share in calm Central Europe over the training days is not
above the model's, so the three predictors raise about as many calm-region alarms.
The sealed test decides at the gate; this only shows where things stand.

Second baseline (declared 8 Nov 2026): a GPSJam-style accuracy score (gpsjam.org, FAQ):
per cell, 100 * (bad - 1) / all aircraft, where "bad" here is an aircraft whose NACp is
below 8 in at least half of its points (GPSJam names NACp but not its cut; the cut is
ours, the same as the rules'). Flagged when the score is above t: t = 0.10 (GPSJam's
"red") and the loosest multiple of 0.05 whose calm-region alarm rate is not above the
model's. GPSJam aggregates a whole day; here it is applied per 15-minute cell-window like
the other predictors (`gpsjam_day`: the same score over the cell's whole day).
"""

import sys

import numpy as np
import pandas as pd

from gnssguard.features.cells import CELL_KEYS, load_cells
from gnssguard.labels.holdout import FLAGGED, wilson
from gnssguard.model.baselines import gpsjam_day, gpsjam_score
from gnssguard.model.dataset import with_context
from gnssguard.model.evaluate import rule_frame
from gnssguard.model.train import Detectors

SETS = ["labels/gold.csv", "labels/gold_2026-11-01.csv", "labels/review_1.csv",
        "labels/calibration_1.csv"]


def main(model_dir="models/detector"):
    det = Detectors.load(model_dir)
    train = det.meta["train_dates"]
    calm = with_context(load_cells("data/cells", train, ["central_europe"]))
    calm = calm[calm["n_ac_usable"].fillna(0) >= 3].reset_index(drop=True)
    model_calm = float(det.predict(calm)["label"].astype(str).isin(FLAGGED).mean())
    deg, use = calm["n_ac_degraded"].fillna(0), calm["n_ac_usable"]
    k = next(k for k in range(1, 50) if (deg >= k).mean() <= model_calm)
    shares = np.round(np.arange(0.05, 1.0001, 0.05), 2)
    s = next(s for s in shares if ((deg / use) >= s).mean() <= model_calm)
    s_loose = round(s - 0.05, 2)  # the next looser share, just above the model's rate
    bad = calm["n_ac_nacp_low"].fillna(0)
    gj = (bad - 1).clip(lower=0) / use
    t = next(t for t in shares if (gj > t).mean() <= model_calm)
    print(f"GPSJam-style: t=0.10 flags {(gj > 0.10).mean():.2%}; matched t={t} flags "
          f"{(gj > t).mean():.2%}")
    print(f"calm Central Europe, training days: model flags {model_calm:.2%}; "
          f"baseline k={k} flags {(deg >= k).mean():.2%}; share s={s} flags "
          f"{((deg / use) >= s).mean():.2%}")

    frames = []
    for f in SETS:
        g = pd.read_csv(f, dtype={"label": str})
        g["set"] = f.split("/")[-1].removesuffix(".csv")
        frames.append(g)
    g = pd.concat(frames, ignore_index=True)
    g = g[g["label"].isin(["normal", *FLAGGED])].copy()
    g["window"] = pd.to_datetime(g["window"], utc=True)
    c = with_context(load_cells("data/cells", sorted(g["date"].astype(str).unique()), None))
    c = c.reset_index(drop=True)
    c["window"] = pd.to_datetime(c["window"], utc=True)
    c["model"] = det.predict(c)["label"].astype(str)
    c["rules"] = rule_frame(c, det)["label"].astype(str)
    c["baseline_count"] = np.where(c["n_ac_degraded"].fillna(0) >= k, "jamming", "normal")
    ratio = c["n_ac_degraded"].fillna(0) / c["n_ac_usable"].clip(lower=1)
    c["baseline_share"] = np.where(ratio >= s, "jamming", "normal")
    c["baseline_share_looser"] = np.where(ratio >= s_loose, "jamming", "normal")
    gj = gpsjam_score(c)
    c["gpsjam_red"] = np.where(gj > 0.10, "jamming", "normal")
    c["gpsjam_matched"] = np.where(gj > t, "jamming", "normal")
    c["gpsjam_day"] = np.where(gpsjam_day(c) > 0.10, "jamming", "normal")
    keys = ["region", "date", *CELL_KEYS]
    c["date"], g["date"] = c["date"].astype(str), g["date"].astype(str)
    preds = ["rules", "model", "baseline_count", "baseline_share", "baseline_share_looser",
             "gpsjam_red", "gpsjam_matched", "gpsjam_day"]
    m = g.merge(c[[*keys, "n_ac_usable", *preds[2:], "model", "rules"]], on=keys)
    m = m[m["n_ac_usable"].fillna(0) >= 3]
    m["human"] = m["label"].isin(FLAGGED)
    m["human_spoof"] = m["label"].isin(["spoofing", "both"])
    rows = []
    for p in preds:
        flag = m[p].isin(FLAGGED)
        tp, fp = int((flag & m["human"]).sum()), int((flag & ~m["human"]).sum())
        pos, neg = int(m["human"].sum()), int((~m["human"]).sum())
        sp = m["human_spoof"]
        sp_hit = int((m[p].isin(["spoofing", "both"]) & sp).sum())
        rows.append({"predictor": p, "recall": f"{tp}/{pos} ({tp / pos:.0%}, "
                     f"{wilson(tp, pos)[0]:.0%}-{wilson(tp, pos)[1]:.0%})",
                     "false_alarms": f"{fp}/{neg} ({fp / neg:.0%})",
                     "spoofing_found": f"{sp_hit}/{int(sp.sum())}"})
    print(f"\n{len(m)} decided hand-labeled cells with 3+ aircraft (4 sets; stratified "
          "samples, so these are not day-wide rates)")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
