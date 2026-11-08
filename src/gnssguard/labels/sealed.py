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


def calibration_sample(cells, score, rules, model, seed=20261104, sizes=None,
                       key_order="shuffled"):
    """calibration_1 (15-21 Sep) was drawn with key_order="alphabetical"."""
    sizes = {**CAL_SIZES, **(sizes or {})}
    rng = np.random.default_rng(seed)
    dense = cells["n_ac_usable"].fillna(0) >= MIN_USABLE
    pools = {"C_low": cells.index[dense & (score < 0.1)],
             "C_mid": cells.index[dense & (score >= 0.1) & (score < 0.9)],
             "C_high": cells.index[dense & (score >= 0.9)]}
    key = _key(cells)
    mid = _spread(pools["C_mid"], key, sizes["C_mid"], rng, key_order)
    extra = sizes["C_mid"] - len(mid)
    want = {"C_low": sizes["C_low"] + extra // 2, "C_high": sizes["C_high"] + extra - extra // 2}
    picks, meta = list(mid), [("C_mid", len(pools["C_mid"]) / max(len(mid), 1),
                               len(pools["C_mid"]))] * len(mid)
    for name in ("C_low", "C_high"):
        ch = _spread(pools[name], key, want[name], rng, key_order)
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
        ch = _spread(pool, by, k, rng)  # shuffled region-days (Correction 2)
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


BANDS = [(0.0, 0.1, "below 0.1"), (0.1, 0.9, "0.1 to 0.9"), (0.9, 1.0001, "0.9 or more")]


def band_pools(cells: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    """Pool size per score band and region-day among judged cells: what each reviewed
    cell stands for (the calibration sample was spread evenly over region-days within a
    band, not in proportion to their cells)."""
    dense = cells["n_ac_usable"].fillna(0) >= MIN_USABLE
    names = {lo: name for lo, _, name in BANDS}
    rows = []
    for lo, hi, _ in BANDS:
        m = dense & (score >= lo) & (score < hi)
        for key, n in _key(cells[m]).value_counts().items():
            rows.append({"band": names[lo], "key": key, "pool": int(n)})
    return pd.DataFrame(rows, columns=["band", "key", "pool"])


def score_bands(gold: pd.DataFrame, groups: pd.DataFrame, source: str,
                pools: pd.DataFrame | None = None, boot: int = 2000, seed: int = 0) -> dict:
    """Reviewer share of interference per published-score band, from the calibration set.

    A three-band table rather than a fitted curve: the scores sit almost all at the ends,
    and a curve through about 120 decided cells overfits (docs/evaluation_protocol.md,
    result of the calibration set).

    `interference_share` is the share among the reviewed cells (sample share, Wilson
    interval). With `pools` (band_pools), `population_share` weights each reviewed cell
    by the cells it stands for in its band and region-day, with an interval from
    resampling whole region-days (cells on one day are not independent), and
    `if_unsure_*` shows how far the excluded "unsure" labels could move it.
    """
    from gnssguard.labels.holdout import wilson

    cols = [c for c in ("gold_id", "region", "date", "label") if c in gold]
    m = gold[cols].merge(groups, on="gold_id")
    m["label"] = m["label"].fillna("")
    m["key"] = (m["region"].astype(str) + "|" + m["date"].astype(str)
                if {"region", "date"} <= set(m) else "")
    names = {lo: name for lo, _, name in BANDS}
    rng = np.random.default_rng(seed)
    out = []
    for lo, hi, name in BANDS:
        b = m[(m["score"] >= lo) & (m["score"] < hi)]
        dec = b[b["label"].isin(["normal", *FLAGGED])]
        k = int(dec["label"].isin(FLAGGED).sum())
        ci = wilson(k, len(dec))
        row = {"score_from": lo, "score_to": min(hi, 1.0), "band": name,
               "cells": len(dec), "unsure": int(len(b) - len(dec)),
               "interference_share": round(k / len(dec), 3) if len(dec) else None,
               "ci95": [ci[0], ci[1]]}
        if pools is not None and len(b):
            pk = pools[pools["band"] == names[lo]].set_index("key")["pool"]
            sampled = b.groupby("key").size()
            w_key = (pk.reindex(sampled.index).fillna(0) / sampled).rename("w")
            bw = b.join(w_key, on="key")

            def share(frame, unsure_as=None):
                f = frame.copy()
                y = f["label"].isin(FLAGGED).astype(float)
                keep = f["label"].isin(["normal", *FLAGGED])
                if unsure_as is not None:
                    y = y.where(keep, float(unsure_as))
                    keep = pd.Series(True, index=f.index)
                w = f["w"].where(keep, 0.0)
                return float((w * y).sum() / w.sum()) if w.sum() > 0 else None

            keys = bw["key"].unique()
            reps = []
            for _ in range(boot):
                pick = rng.choice(keys, len(keys), replace=True)
                r = share(pd.concat([bw[bw["key"] == kk] for kk in pick]))
                if r is not None:
                    reps.append(r)
            ps = share(bw)
            row.update({
                "population_share": None if ps is None else round(ps, 3),
                "population_ci95": [round(float(np.quantile(reps, 0.025)), 3),
                                    round(float(np.quantile(reps, 0.975)), 3)] if reps else None,
                "if_unsure_normal": round(share(bw, 0), 3),
                "if_unsure_interference": round(share(bw, 1), 3),
                "cells_represented": int(pk.sum())})
        out.append(row)
    res = {"source": source, "bands": out,
           "meaning": "Share of hand-reviewed cells in each score band that the reviewer "
                      "called interference (jamming, spoofing or both)."}
    if pools is not None:
        res["population_meaning"] = (
            "population_share: the same, weighted to all judged cells of the calibration "
            "days by band and region-day; its interval resamples whole region-days. "
            "if_unsure_*: the weighted share if the cells marked unsure had all been "
            "normal or all interference.")
    return res
