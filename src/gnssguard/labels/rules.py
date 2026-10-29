"""Weak labeling rules over the cell-window table.

Each rule votes INTERFERENCE (1), NORMAL (0) or ABSTAIN (-1) for every cell-window.
Rules abstain when there is too little evidence, so a quiet cell is not called normal.
Votes are combined by `labelmodel.py`.

Thresholds were set by comparing hotspot regions with calm control regions on real
data (docs/labels.md). They are deliberately simple and readable: the detector in
Phase 3 learns the subtle cases from these labels.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

INTERFERENCE, NORMAL, ABSTAIN = 1, 0, -1

MIN_USABLE = 3  # aircraft with fresh quality readings needed to judge a cell


def _vote(pos: pd.Series, neg: pd.Series) -> np.ndarray:
    v = np.full(len(pos), ABSTAIN, dtype=np.int8)
    v[neg.to_numpy(dtype=bool)] = NORMAL
    v[pos.to_numpy(dtype=bool)] = INTERFERENCE
    return v


def lf_nic_cluster(c: pd.DataFrame) -> np.ndarray:
    """Several aircraft lose position integrity together (not one faulty transponder)."""
    enough = c["n_ac_usable"] >= MIN_USABLE
    pos = enough & (c["n_ac_degraded"] >= 2) & (c["share_ac_degraded"] >= 0.5)
    neg = enough & (c["share_ac_degraded"] <= 0.1)
    return _vote(pos, neg)


def lf_nic_zero(c: pd.DataFrame) -> np.ndarray:
    """At least two aircraft report NIC 0: integrity completely lost."""
    enough = c["n_ac_usable"] >= MIN_USABLE
    pos = enough & (c["n_ac_nic0"] >= 2) & (c["n_ac_nic0"] >= 0.3 * c["n_ac_usable"])
    neg = enough & (c["n_ac_nic0"] == 0) & (c["n_ac_degraded"] == 0)
    return _vote(pos, neg)


def lf_nacp_cluster(c: pd.DataFrame) -> np.ndarray:
    """Several aircraft report poor position accuracy together."""
    enough = c["n_ac_usable"] >= MIN_USABLE
    pos = enough & (c["n_ac_nacp_low"] >= 2) & (c["share_ac_nacp_low"] >= 0.5)
    neg = enough & (c["share_ac_nacp_low"] <= 0.1)
    return _vote(pos, neg)


def lf_mlat_fallback(c: pd.DataFrame) -> np.ndarray:
    """ADS-B aircraft positioned by ground MLAT instead: their own position was unusable.

    Only votes for interference: no MLAT coverage is not evidence of normal GPS.
    """
    pos = c["n_ac_mlat_fallback"] >= 2
    return _vote(pos, pd.Series(False, index=c.index))


def lf_jumps(c: pd.DataFrame) -> np.ndarray:
    """Two or more aircraft make physically impossible position jumps."""
    pos = c["n_ac_jump"] >= 2
    return _vote(pos, pd.Series(False, index=c.index))


def lf_speed_mismatch(c: pd.DataFrame) -> np.ndarray:
    """Two or more aircraft whose positions move differently from their own ground speed.

    The spoofing signature: a fake GPS position drifts while the aircraft's navigation
    system reports its true speed. Never seen in calm traffic (docs/labels.md).
    """
    pos = c["n_ac_speed_mismatch"] >= 2
    return _vote(pos, pd.Series(False, index=c.index))


def lf_alt_mismatch(c: pd.DataFrame) -> np.ndarray:
    """GPS altitude disagrees with barometric altitude beyond the local norm."""
    enough = c["n_ac_altpair"] >= MIN_USABLE
    pos = enough & (c["n_ac_altdev"] >= 2) & (c["share_ac_altdev"] >= 0.3)
    neg = enough & (c["n_ac_altdev"] == 0)
    return _vote(pos, neg)


# Two families of evidence. Jamming removes GPS integrity; spoofing keeps integrity
# high but makes positions move wrongly. One binary label for both does not work: in
# tests the label model learned "interference = lost integrity" and ignored the motion
# rules, missing every synthetic spoofing event (docs/labels.md).
INTEGRITY_RULES = ["lf_nic_cluster", "lf_nic_zero", "lf_nacp_cluster", "lf_mlat_fallback"]
MOTION_RULES = ["lf_jumps", "lf_speed_mismatch", "lf_alt_mismatch"]

RULES: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
    "nic_cluster": lf_nic_cluster,
    "nic_zero": lf_nic_zero,
    "nacp_cluster": lf_nacp_cluster,
    "mlat_fallback": lf_mlat_fallback,
    "jumps": lf_jumps,
    "speed_mismatch": lf_speed_mismatch,
    "alt_mismatch": lf_alt_mismatch,
}


def apply_rules(cells: pd.DataFrame) -> pd.DataFrame:
    """Vote matrix: one int8 column per rule, prefixed `lf_`."""
    return pd.DataFrame({f"lf_{name}": fn(cells) for name, fn in RULES.items()},
                        index=cells.index)


def rule_summary(votes: pd.DataFrame) -> pd.DataFrame:
    """Coverage and vote split per rule, plus pairwise agreement with the others."""
    rows = []
    for col in votes.columns:
        v = votes[col]
        voted = v != ABSTAIN
        others = votes.drop(columns=col)
        both = voted & (others != ABSTAIN).any(axis=1)
        maj = (others.where(others != ABSTAIN).mean(axis=1) >= 0.5).astype(int)
        agree = (v[both] == maj[both]).mean() if both.any() else np.nan
        rows.append({
            "rule": col.removeprefix("lf_"),
            "coverage": round(float(voted.mean()), 4),
            "votes_interference": int((v == INTERFERENCE).sum()),
            "votes_normal": int((v == NORMAL).sum()),
            "agreement_with_others": None if np.isnan(agree) else round(float(agree), 3),
        })
    return pd.DataFrame(rows)
