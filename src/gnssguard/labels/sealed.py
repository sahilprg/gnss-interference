"""The sealed set for the Phase 1 gate (docs/evaluation_protocol.md, "Sealed set").

Two blind, stratified samples, each written as a labeling file plus a groups file that
the labeler does not open:

* calibration set (15-21 Sep 2026): by score band, for fitting the published score to
  human labels;
* sealed test set (5-18 Nov 2026): random, model flags, borderline normal cells, flags
  from 1-2 aircraft; scored once at the gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gnssguard.features.cells import CELL_KEYS
from gnssguard.labels.gold import GOLD_COLUMNS
from gnssguard.labels.holdout import FLAGGED
from gnssguard.labels.review import _spread
from gnssguard.labels.rules import MIN_USABLE
from gnssguard.model.dataset import borderline

CAL_SIZES = {"C_low": 40, "C_mid": 70, "C_high": 40}
TEST_SIZES = {"T_random": 100, "T_flag": 60, "T_border": 50, "T_sparse": 40}
GROUP_COLUMNS = ["gold_id", "group", "weight", "pool", "score", "rule_label", "model_label"]


def _write(cells, picks, meta, prefix, rng, score, rules, model):
    order = rng.permutation(len(picks))
    idx = [picks[i] for i in order]
    rows = cells.loc[idx, ["region", "date", *CELL_KEYS]].copy()
    rows["window"] = pd.to_datetime(rows["window"], utc=True)
    rows.insert(0, "gold_id", [f"{prefix}{i:04d}" for i in range(1, len(rows) + 1)])
    rows["label"], rows["labeler"], rows["notes"] = "", "", ""
    g = pd.DataFrame([meta[i] for i in order], columns=["group", "weight", "pool"])
    g.insert(0, "gold_id", rows["gold_id"].to_numpy())
    g["score"] = score.loc[idx].round(4).to_numpy()
    g["rule_label"] = rules.loc[idx].astype(str).to_numpy()
    g["model_label"] = model.loc[idx].astype(str).to_numpy()
    return rows[GOLD_COLUMNS].reset_index(drop=True), g[GROUP_COLUMNS]


def _key(cells: pd.DataFrame) -> pd.Series:
    return cells["region"].astype(str) + "|" + cells["date"].astype(str)


def calibration_sample(cells, score, rules, model, seed=20261104, sizes=None):
    sizes = {**CAL_SIZES, **(sizes or {})}
    rng = np.random.default_rng(seed)
    dense = cells["n_ac_usable"].fillna(0) >= MIN_USABLE
    pools = {"C_low": cells.index[dense & (score < 0.1)],
             "C_mid": cells.index[dense & (score >= 0.1) & (score < 0.9)],
             "C_high": cells.index[dense & (score >= 0.9)]}
    key = _key(cells)
    mid = _spread(pools["C_mid"], key, sizes["C_mid"], rng)
    extra = sizes["C_mid"] - len(mid)
    want = {"C_low": sizes["C_low"] + extra // 2, "C_high": sizes["C_high"] + extra - extra // 2}
    picks, meta = list(mid), [("C_mid", len(pools["C_mid"]) / max(len(mid), 1),
                               len(pools["C_mid"]))] * len(mid)
    for name in ("C_low", "C_high"):
        ch = _spread(pools[name], key, want[name], rng)
        picks += ch
        meta += [(name, len(pools[name]) / max(len(ch), 1), len(pools[name]))] * len(ch)
    return _write(cells, picks, meta, "K", rng, score, rules, model)


def sealed_test_sample(cells, score, rules, model, seed=20261119, sizes=None):
    sizes = {**TEST_SIZES, **(sizes or {})}
    rng = np.random.default_rng(seed)
    usable = cells["n_ac_usable"].fillna(0)
    dense = usable >= MIN_USABLE
    sparse = (usable >= 1) & (usable < MIN_USABLE)
    key = _key(cells)
    taken: set = set()
    picks, meta = [], []

    def take(name, mask, k, by):
        pool = cells.index[mask & ~cells.index.isin(list(taken))]
        ch = _spread(pool, by, k, rng)
        taken.update(ch)
        if name == "T_random":  # weight per region: its judged cells over its sample
            reg = cells.loc[ch, "region"].astype(str)
            pool_reg = cells.loc[pool, "region"].astype(str).value_counts()
            n_reg = reg.value_counts()
            w = [pool_reg[r] / n_reg[r] for r in reg]
        else:
            w = [len(pool) / max(len(ch), 1)] * len(ch)
        picks.extend(ch)
        meta.extend((name, wi, len(pool)) for wi in w)

    take("T_random", dense, sizes["T_random"], key)
    take("T_flag", dense & model.isin(FLAGGED), sizes["T_flag"], key)
    take("T_border", dense & (model == "normal") & ~rules.isin(FLAGGED) & borderline(cells),
         sizes["T_border"], key)
    take("T_sparse", sparse & model.isin(FLAGGED), sizes["T_sparse"], key)
    return _write(cells, picks, meta, "S", rng, score, rules, model)
