"""Evaluate the detectors against data they never trained on.

1. Gold set (hand labels on the test day): aggregate agreement only. No per-cell
   predictions for the test day are written anywhere, so labeling can stay blind.
2. Synthetic attacks injected into the test day of a calm region (new seeds), in three
   profiles (standard; hard; stress: intermittent, moving and paired jammers with MLAT
   fallback): recall for jamming and spoofing, false alarms away from events, recall at
   the edge vs the core of jamming footprints, source localization error and whether
   the stated uncertainty covers it, and probability calibration.
3. Official advisories: flagged share inside vs outside warned areas on the test day.
4. Ablations: the same evaluation for models trained without context features, without
   the "after" window (causal), on real data only, and on synthetic data only.

Every result is reported for the Phase 2 rules too, as the baseline.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pandas as pd

from gnssguard.features.cells import CELL_KEYS, GridSpec, load_cells
from gnssguard.labels import synthetic as S
from gnssguard.labels.advisories import advisory_check, load_advisories
from gnssguard.labels.pipeline import score
from gnssguard.labels.rules import apply_rules
from gnssguard.model.calibrate import ece
from gnssguard.model.dataset import (
    AFTER,
    BEFORE,
    FEATURES,
    NEIGHBOUR,
    OWN,
    synthetic_rows,
    with_context,
)
from gnssguard.model.localize import incidents, localize_incidents, match_to_events
from gnssguard.model.train import TARGETS, Detectors, train

log = logging.getLogger(__name__)
FLAGGED = ("jamming", "spoofing", "both")

VARIANTS = {
    "full": (FEATURES, ("real", "synthetic")),
    "no_context": (OWN, ("real", "synthetic")),
    "causal": (OWN + NEIGHBOUR + BEFORE, ("real", "synthetic")),
    "real_only": (FEATURES, ("real",)),
    "synthetic_only": (FEATURES, ("synthetic",)),
}
assert set(FEATURES) == set(OWN + NEIGHBOUR + BEFORE + AFTER)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def rule_frame(cells: pd.DataFrame, det: Detectors) -> pd.DataFrame:
    """Phase 2 rule labels, with the label model fitted on the training days."""
    sc = score(apply_rules(cells), det.label_model)
    return sc[["p_jamming", "majority_vote_jamming", "spoof_votes", "label"]]


def model_frame(cells: pd.DataFrame, det: Detectors) -> pd.DataFrame:
    p = det.predict(cells)
    return pd.DataFrame({"p_jamming": p["p_jamming"], "majority_vote_jamming": p["p_jamming"],
                         "spoof_votes": (p["p_spoofing"] >= 0.5).astype(int),
                         "label": p["label"]}, index=cells.index)


# -- 1. gold -----------------------------------------------------------------------------

def gold_eval(gold_csv: str | Path, test_cells: pd.DataFrame, predictors: dict) -> pd.DataFrame:
    """Agreement with hand labels. Only aggregate numbers leave this function."""
    g = pd.read_csv(gold_csv, dtype=str, keep_default_na=False)
    g = g[g["label"] != ""].copy()
    if g.empty:
        return pd.DataFrame()
    g["cell_lat"], g["cell_lon"] = g["cell_lat"].astype(float), g["cell_lon"].astype(float)
    g["window"] = pd.to_datetime(g["window"], utc=True)
    c = test_cells.assign(window=pd.to_datetime(test_cells["window"], utc=True))
    g = g.rename(columns={"label": "human_label"})[
        ["region", "date", *CELL_KEYS, "human_label"]]
    m = g.merge(c, on=["region", "date", *CELL_KEYS], how="inner")
    rows = []
    for name, fn in predictors.items():
        pred = fn(m)["label"].astype(str)
        human = m["human_label"]
        sure = human != "unsure"
        h_flag, p_flag = human.isin(FLAGGED), pred.isin(FLAGGED)
        n = int(sure.sum())
        k_bin = int((h_flag == p_flag)[sure].sum())
        k_exact = int((human == pred)[sure].sum())
        tp = int((h_flag & p_flag)[sure].sum())
        rows.append({
            "predictor": name, "labeled": len(m), "decided": n,
            "binary_agreement": round(k_bin / n, 3) if n else None,
            "binary_ci95": wilson(k_bin, n),
            "exact_agreement": round(k_exact / n, 3) if n else None,
            "exact_ci95": wilson(k_exact, n),
            "precision": round(tp / max(int(p_flag[sure].sum()), 1), 3),
            "recall": round(tp / max(int(h_flag[sure].sum()), 1), 3),
            "unsure_cells": int((~sure).sum()),
            "unsure_flagged": int(p_flag[~sure].sum()),
        })
    return pd.DataFrame(rows)


# -- 2. synthetic --------------------------------------------------------------------------

def synthetic_eval(runs_rows: pd.DataFrame, runs: list[dict], predictors: dict,
                   spec: GridSpec = GridSpec(), localize_with: tuple[str, ...] = ()) -> dict:
    out = {"summary": [], "edge": [], "localization": []}
    for run in runs:
        cw = runs_rows[runs_rows["seed"] == run["seed"]]
        cw = cw[cw["date"] == run["date"]].reset_index(drop=True)
        ac = run["aircraft"] if localize_with else None
        for name, fn in predictors.items():
            pred = fn(cw)
            frame = pd.concat([cw[CELL_KEYS], pred], axis=1)
            summ, _ = S.evaluate(frame, run["truth"], spec)
            out["summary"].append({"predictor": name, "seed": run["seed"], **summ})

            jam = (cw["inj_type"] == "jamming") & (cw["n_ac_affected"] >= 2)
            share = cw["n_ac_affected"] / cw["n_aircraft"].clip(lower=1)
            det = pred["p_jamming"].fillna(0) >= 0.5
            for zone, mask in (("edge (<50% affected)", jam & (share < 0.5)),
                               ("core (>=50% affected)", jam & (share >= 0.5))):
                out["edge"].append({"predictor": name, "seed": run["seed"], "zone": zone,
                                    "cells": int(mask.sum()), "detected": int((det & mask).sum())})
            if name in localize_with:
                inc = incidents(cw[det], spec)
                est = localize_incidents(ac, inc, spec)
                loc = match_to_events(est, run["events"], spec)
                out["localization"].append(loc.assign(predictor=name, seed=run["seed"]))
    summary = pd.DataFrame(out["summary"])
    edge = (pd.DataFrame(out["edge"]).groupby(["predictor", "zone"], as_index=False)
            [["cells", "detected"]].sum())
    edge["recall"] = (edge["detected"] / edge["cells"].clip(lower=1)).round(3)
    loc = (pd.concat(out["localization"], ignore_index=True) if out["localization"]
           else pd.DataFrame())
    return {"summary": summary, "edge": edge, "localization": loc}


def localization_summary(loc: pd.DataFrame) -> pd.DataFrame:
    if loc.empty:
        return loc
    rows = []
    keys = ["profile", "predictor"] if "profile" in loc else ["predictor"]
    for key, g in loc.groupby(keys):
        f = g[g["found"]]
        conf = f[f["confident"].astype(bool)] if "confident" in f else f
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key, strict=True))
        row.update({
            "events": len(g), "found": len(f),
            "fit_median_km": round(float(f["fit_error_km"].median()), 1),
            "fit_p90_km": round(float(f["fit_error_km"].quantile(0.9)), 1),
            "centroid_median_km": round(float(f["centroid_error_km"].median()), 1),
            "centroid_p90_km": round(float(f["centroid_error_km"].quantile(0.9)), 1),
            "raw_uncertainty_coverage": round(float(
                (f["fit_error_km"] <= f["uncertainty_km"]).mean()), 3)})
        if "uncertainty_cal_km" in f:
            row.update({
                "confident": len(conf),
                "confident_median_km": round(float(conf["fit_error_km"].median()), 1),
                "confident_p90_km": round(float(conf["fit_error_km"].quantile(0.9)), 1),
                "calibrated_coverage_confident": round(float(
                    (conf["fit_error_km"] <= conf["uncertainty_cal_km"]).mean()), 3),
                "calibrated_median_uncertainty_km": round(float(
                    conf["uncertainty_cal_km"].median()), 1),
                "low_confidence_median_km": round(float(
                    f.loc[~f["confident"].astype(bool), "fit_error_km"].median()), 1)})
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_report(rows_by_profile: dict[str, pd.DataFrame], det: Detectors
                       ) -> pd.DataFrame:
    """Raw vs calibrated probabilities against exact synthetic truth.

    mid_ece is measured on the cells the raw model is unsure about (0.05-0.95): almost
    every cell sits near 0 or 1, so plain ECE hides how wrong the uncertain ones are.
    """
    out = []
    for prof, rows in rows_by_profile.items():
        for target, col in TARGETS.items():
            r = rows[rows[target].notna()]
            if r.empty:
                continue
            y = r[target].to_numpy()
            p = det.predict(r, calibrated=True)
            raw, cal = p[col.replace("p_", "score_")].to_numpy(), p[col].to_numpy()
            unsure = (raw >= 0.05) & (raw < 0.95)
            out.append({"profile": prof, "target": col, "cells": len(r),
                        "unsure_cells": int(unsure.sum()),
                        "ece_raw": round(ece(raw, y), 4), "ece_calibrated": round(ece(cal, y), 4),
                        "mid_ece_raw": round(ece(raw[unsure], y[unsure]), 3),
                        "mid_ece_calibrated": round(ece(cal[unsure], y[unsure]), 3),
                        "brier_raw": round(float(((raw - y) ** 2).mean()), 5),
                        "brier_calibrated": round(float(((cal - y) ** 2).mean()), 5)})
    return pd.DataFrame(out)


# -- 3. advisories ---------------------------------------------------------------------------

def advisory_eval(test_cells: pd.DataFrame, advisories_csv: str | Path, predictors: dict
                  ) -> pd.DataFrame:
    adv = load_advisories(advisories_csv)
    if adv.empty:
        return pd.DataFrame()
    frames = []
    for name, fn in predictors.items():
        lab = test_cells[["region", "date", "cell_lat", "cell_lon", "cell_deg"]].assign(
            label=fn(test_cells)["label"].astype(str))
        frames.append(advisory_check(lab, adv).assign(predictor=name))
    return pd.concat(frames, ignore_index=True)


def real_flag_rates(test_cells: pd.DataFrame, predictors: dict, min_usable: int = 3
                    ) -> pd.DataFrame:
    """Flagged share of judged cell-windows per region on a real day.

    Reported twice: over cells with at least `min_usable` usable aircraft (where the
    rules can judge and the gold set was drawn, so rules and model are comparable), and
    over sparse cells with 1-2 usable aircraft, which only the model judges and which no
    hand label covers. The calm region's rate is the best real-data false-alarm estimate.
    """
    rows = []
    usable = test_cells["n_ac_usable"].fillna(0)
    for name, fn in predictors.items():
        lab = fn(test_cells)["label"].astype(str)
        flag = lab.isin(FLAGGED)
        judged = lab != "unknown"
        for region in sorted(test_cells["region"].unique()):
            r = judged & (test_cells["region"] == region)
            dense, sparse = r & (usable >= min_usable), r & (usable < min_usable)
            rows.append({"predictor": name, "region": region, "judged": int(r.sum()),
                         "flagged_share": round(float(flag[r].mean()), 4),
                         "judged_dense": int(dense.sum()),
                         "flagged_share_dense": round(float(flag[dense].mean()), 4)
                         if dense.any() else None,
                         "judged_sparse": int(sparse.sum()),
                         "flagged_share_sparse": round(float(flag[sparse].mean()), 4)
                         if sparse.any() else None})
    return pd.DataFrame(rows)


# -- runner ------------------------------------------------------------------------------------

def run(processed: str | Path, cells_dir: str | Path, table: pd.DataFrame, det: Detectors,
        test_dates: list[str], gold_csv: str | Path | None, advisories_csv: str | Path | None,
        report_dir: str | Path, *, synth_region: str = "central_europe",
        synth_seeds: tuple[int, ...] = (100, 101, 102, 103, 104), ablations: bool = True,
        spec: GridSpec = GridSpec()) -> dict:
    rep = Path(report_dir)
    rep.mkdir(parents=True, exist_ok=True)
    test_cells = with_context(load_cells(cells_dir, test_dates))

    models = {"full": det}
    if ablations:
        for name, (feats, sources) in VARIANTS.items():
            if name != "full":
                models[name] = train(table, feats, sources=sources, label_model=det.label_model)
    predictors = {"rules": lambda c: rule_frame(c, det)}
    predictors.update({f"model_{n}": (lambda c, m=m: model_frame(c, m)) for n, m in models.items()})

    results = {}
    if gold_csv and Path(gold_csv).exists():
        results["gold"] = gold_eval(gold_csv, test_cells, predictors)
    syn_rows, loc_frames = {}, []
    for prof, offset in (("standard", 0), ("hard", 100), ("stress", 200)):
        rows, runs = synthetic_rows(processed, synth_region, test_dates,
                                    [s + offset for s in synth_seeds], spec=spec, profile=prof,
                                    keep_runs=True)
        loc_with = ("rules", "model_full") if prof == "standard" else ("model_full",)
        res = synthetic_eval(rows, runs, predictors, spec, localize_with=loc_with)
        sfx = "" if prof == "standard" else f"_{prof}"
        results[f"synthetic{sfx}"], results[f"edge{sfx}"] = res["summary"], res["edge"]
        if not res["localization"].empty:
            loc_frames.append(res["localization"].assign(profile=prof))
        syn_rows[prof] = rows[list(dict.fromkeys([*det.features, "n_ac_usable", *TARGETS]))]
        del rows, runs
    loc = pd.concat(loc_frames, ignore_index=True) if loc_frames else pd.DataFrame()
    results.update(localization=localization_summary(loc), localization_events=loc)
    if det.calibrators:
        results["calibration"] = calibration_report(syn_rows, det)
    results["real_flag_rates"] = real_flag_rates(test_cells, predictors)
    if advisories_csv and Path(advisories_csv).exists():
        results["advisories"] = advisory_eval(test_cells, advisories_csv,
                                              {k: predictors[k] for k in ("rules", "model_full")})

    cols = ["recall_jamming", "recall_spoofing", "false_positive_rate_away_from_events"]
    results["synthetic_mean"] = (results["synthetic"].groupby("predictor")[cols]
                                 .mean().round(3).reset_index())
    for prof in ("hard", "stress"):
        results[f"synthetic_{prof}_mean"] = (results[f"synthetic_{prof}"].groupby("predictor")
                                             [cols].mean().round(3).reset_index())
    for name, df in results.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(rep / f"{name}.csv", index=False)
    run_info = {"test_dates": test_dates, "synthetic_region": synth_region,
                "synthetic_seeds": list(synth_seeds), "model": det.meta}
    (rep / "run.json").write_text(json.dumps(run_info, indent=2, default=str), encoding="utf-8")
    return results
