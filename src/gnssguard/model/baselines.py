"""The simple predictors the model is compared with (declared 6 and 8 Nov 2026).

* NIC share: a cell-window is interference if n_ac_degraded / n_ac_usable >= s
  (s = 0.25 is the Phase 2 gate's baseline, s = 0.30 the stricter one).
* GPSJam-style (gpsjam.org FAQ): (bad - 1) / all aircraft, bad = NACp below 8, flagged
  above 10%; per 15-minute cell-window and over the cell's whole day.

One definition, used by the development check (scripts/baseline_check.py), the sealed
sampler (which freezes these values with the sample) and the sealed scoring.
"""

from __future__ import annotations

import pandas as pd

NIC_SHARES = (0.25, 0.30)
GPSJAM_RED = 0.10


def nic_share(c: pd.DataFrame) -> pd.Series:
    """Share of usable aircraft with degraded NIC."""
    return c["n_ac_degraded"].fillna(0) / c["n_ac_usable"].fillna(0).clip(lower=1)


def gpsjam_score(c: pd.DataFrame) -> pd.Series:
    """GPSJam's published formula per cell-window: (bad - 1) / all, bad = low NACp."""
    return ((c["n_ac_nacp_low"].fillna(0) - 1).clip(lower=0)
            / c["n_ac_usable"].fillna(0).clip(lower=1))


def gpsjam_day(c: pd.DataFrame) -> pd.Series:
    """The same score over the cell's whole day, as GPSJam aggregates it. Aircraft are
    counted once per window, so an aircraft seen in several windows counts several
    times (the cells keep no aircraft identities)."""
    key = [c["region"].astype(str), c["date"].astype(str), c["cell_lat"], c["cell_lon"]]
    bad = c.groupby(key)["n_ac_nacp_low"].transform("sum").fillna(0)
    alln = c.groupby(key)["n_ac_usable"].transform("sum").fillna(0)
    return (bad - 1).clip(lower=0) / alln.clip(lower=1)


def baseline_values(c: pd.DataFrame) -> pd.DataFrame:
    """The values the baselines decide on, one row per cell-window of c."""
    return pd.DataFrame({"nic_share": nic_share(c), "gpsjam_window": gpsjam_score(c),
                         "gpsjam_day": gpsjam_day(c)}, index=c.index)
