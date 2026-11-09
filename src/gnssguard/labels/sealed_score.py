"""Scoring of the sealed test (questions 1-8), written before its labels exist.

Everything a result depends on is fixed here and in the protocol (Sealed test,
addendum 3), not chosen after the labels are seen:

* **Who counts as what.** The labeler's interference = jamming, spoofing or both;
  normal = normal; unsure = left out of every share (counted, and bounded by scoring all
  unsure as interference and all as normal for the primary endpoint).
* **Predictors** (frozen in the groups file at the draw): model (published label),
  rules (label model; "unknown" is no flag, never a correct "normal"), NIC share
  >= 0.25 and >= 0.30, GPSJam-style per window and per day (> 10%), and the model at a
  0.1 threshold (question 5).
* **Weights.** Each drawn cell stands for 1/pi cells of its group's pool (pi: its
  inclusion probability, simulated from the sampler's own rule at the draw). A T_random
  cell that lies in the flag or border population stands for itself only (weight 1),
  because those cells were removed from the later groups' pools; T_flag and T_border
  stand for the rest of their populations. Together they estimate totals over all
  judged cells (Horvitz-Thompson, conditional on the earlier groups).
* **Day weighting** (questions 1, 5, 6, 7, as declared): weights are divided by the
  day's number of judged cells, so every one of the 14 days counts the same. Questions
  2-4 estimate shares of all cell-windows of their population (no day weighting).
* **Intervals:** 95%, from 2000 bootstrap resamples of the days (day-clustered); paired
  comparisons use the same resamples for both predictors.
* **Primary endpoint** (the Phase 2 gate, question 6): the model beats the NIC-share
  baseline (s = 0.25) if its day-weighted recall is higher with the paired 95% interval
  of the difference above 0, and its false-alarm rate (share of the labeler's normal
  cells it flags) is not higher by more than 2 points. Everything else is reported, not
  a gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gnssguard.labels.holdout import FLAGGED, wilson
from gnssguard.model.baselines import GPSJAM_RED

INTERFERENCE = {"jamming", "spoofing", "both"}
DECIDED = INTERFERENCE | {"normal"}
SPOOF = {"spoofing", "both"}
BOOT = 2000
GATE_FA_MARGIN = 0.02


def predictors(t: pd.DataFrame) -> dict[str, pd.Series]:
    """Flag (True/False) per sampled cell for every compared predictor."""
    return {
        "model": t["model_label"].isin(FLAGGED),
        "rules": t["rule_label"].isin(FLAGGED),
        "nic_0.25": t["nic_share"] >= 0.25,
        "nic_0.30": t["nic_share"] >= 0.30,
        "gpsjam_window": t["gpsjam_window"] > GPSJAM_RED,
        "gpsjam_day": t["gpsjam_day"] > GPSJAM_RED,
        "model_t0.1": t["score"] >= 0.1,
    }


def load(gold: pd.DataFrame, groups: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """One row per sampled cell with its label, group, predictor values and weights."""
    t = gold.drop(columns=[c for c in ("group",) if c in gold]).merge(
        groups, on="gold_id", how="inner", validate="one_to_one")
    if len(t) != len(gold) or len(t) != len(groups):
        raise ValueError("labeling file and groups file do not list the same cells")
    t["label"] = t["label"].fillna("").astype(str).str.strip().str.lower()
    bad = sorted(set(t["label"]) - DECIDED - {"unsure"})
    if bad:
        raise ValueError(f"labels must be jamming/spoofing/both/normal/unsure, found {bad}")
    t["date"] = t["date"].astype(str)
    t["interference"] = t["label"].isin(INTERFERENCE)
    t["decided"] = t["label"].isin(DECIDED)
    in_lower = t["group"].eq("T_random") & (t["in_flag"].eq(1) | t["in_border"].eq(1))
    t["w_own"] = 1.0 / t["pi"]  # within the cell's own group
    # weight towards totals over all judged cells, and over the flag / border populations
    t["w_judged"] = np.where(t["group"].eq("T_sparse"), 0.0,
                             np.where(in_lower, 1.0, t["w_own"]))
    t["w_flag"] = np.where(t["group"].eq("T_flag"), t["w_own"],
                           np.where(t["group"].eq("T_random") & t["in_flag"].eq(1), 1.0, 0.0))
    t["w_border"] = np.where(t["group"].eq("T_border"), t["w_own"],
                             np.where(t["group"].eq("T_random") & t["in_border"].eq(1), 1.0, 0.0))
    t["w_sparse"] = np.where(t["group"].eq("T_sparse"), t["w_own"], 0.0)
    judged_day = (frames[frames["group"] == "judged"].assign(date=lambda f: f["date"].astype(str))
                  .groupby("date")["pool"].sum())
    t["w_day"] = t["w_judged"] / t["date"].map(judged_day).astype(float)
    return t


def _ratio(t: pd.DataFrame, w: str, num: pd.Series, den: pd.Series,
           days: list[str], rng) -> dict:
    """Weighted share num/den over the rows with den, with a day-clustered interval."""
    W = t[w].to_numpy(float)
    a, b = (W * num.to_numpy(bool)), (W * den.to_numpy(bool))
    by = t["date"].to_numpy()
    A = pd.Series(a).groupby(by).sum().reindex(days, fill_value=0.0).to_numpy()
    B = pd.Series(b).groupby(by).sum().reindex(days, fill_value=0.0).to_numpy()
    est = A.sum() / B.sum() if B.sum() > 0 else None
    boots = []
    for _ in range(BOOT):
        pick = rng.integers(0, len(days), len(days))
        if B[pick].sum() > 0:
            boots.append(A[pick].sum() / B[pick].sum())
    ci = [round(float(np.percentile(boots, q)), 4) for q in (2.5, 97.5)] if boots else None
    return {"estimate": None if est is None else round(float(est), 4), "ci95": ci,
            "cells": int(den.sum())}


def _paired(t: pd.DataFrame, w: str, num1: pd.Series, num2: pd.Series, den: pd.Series,
            days: list[str], rng) -> dict:
    """Difference of two weighted shares on the same cells, day-clustered interval."""
    W = t[w].to_numpy(float)
    by = t["date"].to_numpy()

    def per_day(x):
        return pd.Series(W * x.to_numpy(bool)).groupby(by).sum().reindex(days, fill_value=0.0
                                                                      ).to_numpy()
    A1, A2, B = per_day(num1), per_day(num2), per_day(den)
    if B.sum() == 0:
        return {"difference": None, "ci95": None}
    est = (A1.sum() - A2.sum()) / B.sum()
    boots = []
    for _ in range(BOOT):
        pick = rng.integers(0, len(days), len(days))
        if B[pick].sum() > 0:
            boots.append((A1[pick].sum() - A2[pick].sum()) / B[pick].sum())
    d1, d2 = num1 & den, num2 & den
    return {"difference": round(float(est), 4),
            "ci95": [round(float(np.percentile(boots, q)), 4) for q in (2.5, 97.5)],
            "discordant_cells": {"first_only": int((d1 & ~d2).sum()),
                                 "second_only": int((d2 & ~d1).sum())}}


def _wilson(k: int, n: int) -> dict:
    return {"k": k, "n": n, "share": round(k / n, 4) if n else None,
            "ci95": [round(x, 4) for x in wilson(k, n)] if n else None}


def score(gold: pd.DataFrame, groups: pd.DataFrame, frames: pd.DataFrame,
          seed: int = 20261119) -> dict:
    t = load(gold, groups, frames)
    days = sorted(frames.loc[frames["group"] == "judged", "date"].astype(str).unique())
    rng = np.random.default_rng(seed)
    preds = predictors(t)
    dec, interf = t["decided"], t["interference"]
    normal = dec & ~interf
    judged = t["w_judged"] > 0
    out: dict = {"cells": int(len(t)), "decided": int(dec.sum()),
                 "unsure": int((t["label"] == "unsure").sum()),
                 "by_group": t.groupby("group").agg(
                     sampled=("gold_id", "size"), decided=("decided", "sum"),
                     interference=("interference", "sum")).astype(int).to_dict("index"),
                 "days": days}

    def share(w, num, den):
        return _ratio(t, w, num, den, days, rng)

    # 1. agreement (interference or not) of model and rules with the labeler
    agree = {n: (p == interf) & dec for n, p in preds.items() if n in ("model", "rules")}
    rules_decides = ~t["rule_label"].isin(["unknown", "nan", ""])
    out["q1_agreement"] = {
        "model": share("w_day", agree["model"], dec & judged),
        "rules": share("w_day", agree["rules"] & rules_decides, dec & judged),
        "rules_where_they_decide": share("w_day", agree["rules"], dec & judged & rules_decides),
        "rules_no_verdict_share": share("w_day", ~rules_decides, judged),
    }
    # 2. false-alarm share of model flags; interference among borderline normal cells
    out["q2"] = {
        "false_alarm_share_of_model_flags": share("w_flag", normal, dec & (t["w_flag"] > 0)),
        "interference_share_of_borderline_normal": share(
            "w_border", interf, dec & (t["w_border"] > 0)),
    }
    # 3. precision of flags from 1-2 aircraft
    out["q3_sparse_flag_precision"] = share("w_sparse", interf, dec & (t["w_sparse"] > 0))
    # 4. calibration: interference share by score band, against the published table
    bands = {}
    for lo, hi, name in [(0.0, 0.1, "below 0.1"), (0.1, 0.9, "0.1 to 0.9"),
                         (0.9, 1.0001, "0.9 or more")]:
        inb = judged & (t["score"] >= lo) & (t["score"] < hi)
        m = inb & dec
        mean = (round(float(np.average(t.loc[m, "score"], weights=t.loc[m, "w_judged"])), 4)
                if m.any() else None)
        bands[name] = {**share("w_judged", interf, m), "mean_score": mean}
    sw = t.loc[dec & judged, "w_judged"]
    y = interf[dec & judged].astype(float)
    out["q4_calibration"] = {
        "bands": bands,
        "brier_raw_score": round(float(np.average((t.loc[dec & judged, "score"] - y) ** 2,
                                                  weights=sw)), 4) if len(sw) else None,
    }
    # 5. threshold 0.1 against 0.5 (agreement, day-weighted, paired)
    a01 = (preds["model_t0.1"] == interf) & dec
    out["q5_threshold"] = {
        "agreement_0.5": out["q1_agreement"]["model"],
        "agreement_0.1": share("w_day", a01, dec & judged),
        "difference_0.1_minus_0.5": _paired(t, "w_day", a01, agree["model"], dec & judged,
                                            days, rng),
    }
    # 6 and 7. recall, false-alarm rate, spoofing found, for every predictor
    spoof = t["label"].isin(SPOOF)
    model_spoof = t["model_label"].isin(SPOOF)
    rules_spoof = t["rule_label"].isin(SPOOF)
    table = {}
    for n, p in preds.items():
        table[n] = {"recall": share("w_day", p & interf, interf & judged),
                    "false_alarm_rate": share("w_day", p & normal, normal & judged)}
        found = {"model": model_spoof, "rules": rules_spoof}.get(n, pd.Series(False, index=t.index))
        table[n]["spoofing_found"] = {"k": int((found & spoof & judged).sum()),
                                      "n": int((spoof & judged).sum())}
    out["q6_q7_predictors"] = table
    # primary endpoint: model against NIC share 0.25
    rec = _paired(t, "w_day", preds["model"] & interf, preds["nic_0.25"] & interf,
                  interf & judged, days, rng)
    fa = _paired(t, "w_day", preds["model"] & normal, preds["nic_0.25"] & normal,
                 normal & judged, days, rng)
    passes = (rec["ci95"] is not None and rec["ci95"][0] > 0
              and fa["difference"] is not None and fa["difference"] <= GATE_FA_MARGIN)
    bounds = {}
    for name, as_interf in (("unsure_as_interference", True), ("unsure_as_normal", False)):
        unsure = t["label"] == "unsure"
        i2 = interf | (unsure & as_interf)
        n2 = normal | (unsure & (not as_interf))
        r2 = _paired(t, "w_day", preds["model"] & i2, preds["nic_0.25"] & i2, i2 & judged,
                     days, rng)
        f2 = _paired(t, "w_day", preds["model"] & n2, preds["nic_0.25"] & n2, n2 & judged,
                     days, rng)
        bounds[name] = {"recall_difference": r2, "false_alarm_difference": f2}
    out["primary_endpoint"] = {
        "question": "6: model against NIC share >= 0.25 (Phase 2 gate)",
        "recall_difference": rec, "false_alarm_difference": fa,
        "rule": "pass if the recall difference's 95% interval is above 0 and the "
                f"false-alarm difference is at most {GATE_FA_MARGIN:.2f}",
        "passes": bool(passes), "unsure_bounds": bounds,
    }
    # 8. one-aircraft cells: T_border cells the rules give no verdict (sample only)
    q8 = (t["group"] == "T_border") & (t["rule_label"] == "unknown")
    out["q8_one_aircraft_cells"] = {**_wilson(int((q8 & interf).sum()), int((q8 & dec).sum())),
                                    "unsure": int((q8 & (t["label"] == "unsure")).sum())}
    return out

