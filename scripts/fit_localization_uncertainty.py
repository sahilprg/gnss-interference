"""Refit the localization uncertainty constants in src/gnssguard/model/localize.py.

Injects synthetic jamming (standard, hard and stress profiles, 3 seeds each) into the
calm control region on four days BEFORE the test day, localizes the incidents the saved
detector flags (with the optional second-jammer fit, so one run gives both the
one-jammer and the two-jammer results), and finds the smallest scale on the raw spread
that covers 90% of the confident estimates. Takes about an hour on 2 CPU cores.

    python scripts/fit_localization_uncertainty.py reports/model
"""

import sys

import pandas as pd

from gnssguard.model.dataset import synthetic_rows
from gnssguard.model.localize import (
    fit_uncertainty,
    incidents,
    localize_incidents,
    match_to_events,
)
from gnssguard.model.train import Detectors

out = sys.argv[1] if len(sys.argv) > 1 else "reports/model"
DAYS = ["2026-10-22", "2026-10-23", "2026-10-24", "2026-10-25"]
det = Detectors.load("data/models/detector")
frames = []
for prof, base in (("standard", 400), ("hard", 500), ("stress", 600)):
    for date in DAYS:
        rows, runs = synthetic_rows("data/processed", "central_europe", [date],
                                    [base, base + 1, base + 2], profile=prof, keep_runs=True)
        for run in runs:
            cw = rows[rows.seed == run["seed"]].reset_index(drop=True)
            flag = det.predict(cw)["p_jamming"] >= 0.5
            est = localize_incidents(run["aircraft"], incidents(cw[flag]), two_sources=True)
            loc = match_to_events(est, run["events"])
            ev = {e.event_id: e for e in run["events"]}
            loc["moving"] = [bool(getattr(ev[i], "speed_kt", 0)) for i in loc.event_id]
            loc["paired"] = [i.endswith("b") or f"{i}b" in ev for i in loc.event_id]
            frames.append(loc.assign(profile=prof, date=date, seed=run["seed"]))
        print(prof, date, flush=True)
loc = pd.concat(frames, ignore_index=True)
loc.to_csv(f"{out}/localization_fit_events.csv", index=False)
one = loc.assign(fit_error_km=loc["first_error_km"], uncertainty_km=loc["first_uncertainty_km"],
                 score=loc["first_score"])
f1, f2 = one[one["found"]], loc[loc["found"]]
print("one jammer : median error", round(f1["fit_error_km"].median(), 1), fit_uncertainty(one))
print("two jammers: median error", round(f2["fit_error_km"].median(), 1), fit_uncertainty(loc))
for prof in sorted(loc["profile"].unique()):
    a, b = f1[f1["profile"] == prof], f2[f2["profile"] == prof]
    n2 = int((b["n_sources"] == 2).sum())
    print(f"  {prof}: median error one {a['fit_error_km'].median():.1f} km, "
          f"two {b['fit_error_km'].median():.1f} km, two-source fits {n2}")
