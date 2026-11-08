"""Disagreement set: cells where the model and the rules disagree, labeled blind, to say
which one is right (review of 7 Nov 2026: on real data the model reproduces the rules on
about 99% of cells, so an ordinary sample holds almost no disagreements).

Groups (each spread over regions and days; all cells taken when a pool is small):

* D1_model_flags: the model flags interference, the rules do not (normal or unknown);
* D2_rules_flag: the rules flag interference, the model says normal;
* D3_type: both flag, with a different type (jamming, spoofing, both);
* D4_rules_unsure: the rules give no verdict (unknown), the model says normal (by far
  the largest pool: who is right about the cells the rules leave open).

Written as a labeling file plus a groups file the labeler does not open, as the other
sets (docs/labels.md). Scoring (`disagreement_eval`): per group, who the labeler sided
with, with Wilson intervals; "unsure" labels are counted, never dropped silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gnssguard.labels.holdout import FLAGGED, wilson
from gnssguard.labels.review import _spread
from gnssguard.labels.rules import MIN_USABLE
from gnssguard.labels.sealed import _key, _write

SIZES = {"D1_model_flags": 30, "D2_rules_flag": 15, "D3_type": 15, "D4_rules_unsure": 20}


def disagreement_sample(cells: pd.DataFrame, score: pd.Series, rules: pd.Series,
                        model: pd.Series, seed: int = 20261108,
                        sizes: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    sizes = {**SIZES, **(sizes or {})}
    rng = np.random.default_rng(seed)
    dense = cells["n_ac_usable"].fillna(0) >= MIN_USABLE
    mf, rf = model.isin(FLAGGED), rules.isin(FLAGGED)
    pools = {"D1_model_flags": dense & mf & ~rf,
             "D2_rules_flag": dense & rf & (model == "normal"),
             "D3_type": dense & mf & rf & (model != rules),
             "D4_rules_unsure": dense & (rules == "unknown") & (model == "normal")}
    key = _key(cells)
    picks, meta = [], []
    for name, mask in pools.items():
        pool = cells.index[mask]
        ch = _spread(pool, key, sizes[name], rng)
        picks += ch
        meta += [(name, len(pool) / max(len(ch), 1), len(pool))] * len(ch)
    return _write(cells, picks, meta, "D", rng, score, rules, model)


def disagreement_eval(gold: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Per group: of the decided labels, how many side with the model and with the
    rules (interference or not; and the type in D3)."""
    m = gold[["gold_id", "label"]].merge(groups, on="gold_id")
    m["label"] = m["label"].fillna("")
    rows = []
    for name, g in m.groupby("group"):
        dec = g[g["label"].isin(["normal", *FLAGGED])]
        h = dec["label"].isin(FLAGGED)
        if name == "D3_type":
            model_right = int((dec["label"] == dec["model_label"]).sum())
            rules_right = int((dec["label"] == dec["rule_label"]).sum())
        else:
            model_right = int((h == dec["model_label"].isin(FLAGGED)).sum())
            rules_right = int((h == dec["rule_label"].isin(FLAGGED)).sum())
        n = len(dec)
        lo, hi = wilson(model_right, n)
        rows.append({"group": name, "sampled": len(g), "decided": n,
                     "unsure": int(len(g) - n), "pool": int(g["pool"].iloc[0]),
                     "model_right": model_right, "rules_right": rules_right,
                     "model_right_share": round(model_right / n, 3) if n else None,
                     "model_right_ci95": [lo, hi] if n else None})
    return pd.DataFrame(rows)
