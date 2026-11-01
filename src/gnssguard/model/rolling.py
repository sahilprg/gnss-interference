"""Rolling evaluation: one sealed test day is too little to trust a number.

For each test day D in the chosen range, a model is trained on every stored day before
D - 1, validated (early stopping and calibration) on D - 1, and tested on D, so every
fold sees only the past, as the daily product will. Per fold:

* synthetic attacks injected into the calm region on day D, three profiles (standard,
  hard, stress), seeds never used in training: recall and false alarms, rules vs model
* real data on day D: flagged share per region, rules vs model. The calm region's rate
  is the real-data false-alarm estimate.

The training table from `gnssguard train` is reused (its synthetic rows are per day, and
its real rows are re-labelled with a label model fitted on each fold's training days).
Only aggregates are written. Gold labels stay in `gnssguard evaluate`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from gnssguard.features.cells import GridSpec, load_cells
from gnssguard.labels import synthetic as S
from gnssguard.labels.labelmodel import DawidSkene
from gnssguard.labels.rules import INTEGRITY_RULES, apply_rules
from gnssguard.model.dataset import real_targets, synthetic_rows, with_context
from gnssguard.model.evaluate import model_frame, real_flag_rates, rule_frame
from gnssguard.model.train import train

log = logging.getLogger(__name__)
PROFILE_SEEDS = {"standard": 700, "hard": 800, "stress": 900}
TARGET_COLS = ["y_jam", "y_spoof", "rule_label", "rule_p_jamming", "rule_spoof_votes"]


def fold_table(table: pd.DataFrame, test_day: str, unlabel_borderline: bool = False
               ) -> tuple[pd.DataFrame, DawidSkene | None] | None:
    """Rows before test_day: train = all but the last stored day, val = the last one."""
    days = sorted(d for d in table["date"].unique() if d < test_day)
    if len(days) < 2:
        return None
    train_days, val_day = days[:-1], days[-1]
    t = table[table["date"].isin(days)].copy()
    real = t["source"] == "real"
    lm = None
    if real.any():
        lm = DawidSkene().fit(apply_rules(t[real & t["date"].isin(train_days)])[INTEGRITY_RULES])
        t.loc[real, TARGET_COLS] = real_targets(t[real], lm, unlabel_borderline)[TARGET_COLS]
    t["split"] = (t["date"] == val_day).map({True: "val", False: "train"})
    return t, lm


def run(processed: str | Path, cells_dir: str | Path, table: pd.DataFrame,
        test_days: list[str], report_dir: str | Path, *,
        synth_region: str | list[str] = "central_europe", n_seeds: int = 3,
        spec: GridSpec = GridSpec(), unlabel_borderline: bool = False
        ) -> dict[str, pd.DataFrame]:
    test_regions = [synth_region] if isinstance(synth_region, str) else list(synth_region)
    rep = Path(report_dir)
    rep.mkdir(parents=True, exist_ok=True)
    synth_rows, flags, folds = [], [], []
    for day in test_days:
        ft = fold_table(table, day, unlabel_borderline)
        if ft is None:
            log.warning("%s: fewer than two earlier days; skipped", day)
            continue
        t, lm = ft
        det = train(t, label_model=lm)
        folds.append({"test_day": day, "train_days": len(det.meta["train_dates"]),
                      "val_day": det.meta["val_dates"][0],
                      **{f"{k}_trees": v["best_iteration"]
                         for k, v in det.meta["targets"].items()}})
        del t
        predictors = {"rules": lambda c, d=det: rule_frame(c, d),
                      "model": lambda c, d=det: model_frame(c, d)}

        cells = load_cells(cells_dir, [day])
        if not cells.empty:
            flags.append(real_flag_rates(with_context(cells), predictors).assign(test_day=day))
        for region in test_regions:
            for prof, base in PROFILE_SEEDS.items():
                seeds = [base + i for i in range(n_seeds)]
                rows, runs = synthetic_rows(processed, region, [day], seeds, spec=spec,
                                            profile=prof, keep_runs=True)
                for r in runs:
                    cw = rows[rows["seed"] == r["seed"]].reset_index(drop=True)
                    for name, fn in predictors.items():
                        frame = pd.concat([cw[["cell_lat", "cell_lon", "window"]], fn(cw)],
                                          axis=1)
                        summ, _ = S.evaluate(frame, r["truth"], spec)
                        synth_rows.append({"test_day": day, "synth_region": region,
                                           "profile": prof, "predictor": name,
                                           "seed": r["seed"], **summ})
                del rows, runs
        log.info("fold %s done", day)

    synthetic = pd.DataFrame(synth_rows)
    real = pd.concat(flags, ignore_index=True) if flags else pd.DataFrame()
    cols = ["recall_jamming", "recall_spoofing", "false_positive_rate_away_from_events"]
    per_day = (synthetic.groupby(["test_day", "synth_region", "profile", "predictor"])[cols]
               .mean().round(3).reset_index()) if not synthetic.empty else synthetic
    overall = (per_day.groupby(["synth_region", "profile", "predictor"])[cols]
               .agg(["mean", "min", "max"]).round(3)) if not per_day.empty else per_day
    if not overall.empty:
        overall.columns = [f"{a}_{b}" for a, b in overall.columns]
        overall = overall.reset_index()
    out = {"rolling_folds": pd.DataFrame(folds), "rolling_synthetic": per_day,
           "rolling_synthetic_overall": overall, "rolling_real_flag_rates": real}
    for name, df in out.items():
        if not df.empty:
            df.to_csv(rep / f"{name}.csv", index=False)
    return out
