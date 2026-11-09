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
from gnssguard.labels.review import _spread, inclusion
from gnssguard.labels.rules import MIN_USABLE
from gnssguard.model.dataset import borderline

CAL_SIZES = {"C_low": 40, "C_mid": 70, "C_high": 40}
TEST_SIZES = {"T_random": 100, "T_flag": 60, "T_border": 50, "T_sparse": 40}
GROUP_COLUMNS = ["gold_id", "group", "weight", "pool", "score", "rule_label", "model_label"]


# frozen with each sealed-test cell, so scoring needs only the committed files
SEALED_EXTRA = ["pi", "in_flag", "in_border", "n_ac_usable", "p_jamming", "p_spoofing",
                "nic_share", "gpsjam_window", "gpsjam_day"]


def _write(cells, picks, meta, prefix, rng, score, rules, model, extra=None):
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
    cols = list(GROUP_COLUMNS)
    if extra is not None:
        for col in extra.columns:
            g[col] = extra.loc[idx, col].to_numpy()
        cols += list(extra.columns)
    return rows[GOLD_COLUMNS].reset_index(drop=True), g[cols]


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


def sealed_populations(cells, rules, model) -> dict:
    """The populations the sealed test estimates over (masks on cells).

    judged: 3+ usable aircraft. flag: judged cells the model flags. border: judged cells
    both the model and the rules leave unflagged, with some jamming evidence (one
    degraded aircraft or a hot neighbour). sparse: cells the model flags from 1-2
    usable aircraft. flag and border are disjoint parts of judged; sparse is outside it.
    """
    usable = cells["n_ac_usable"].fillna(0)
    dense = usable >= MIN_USABLE
    return {"judged": dense,
            "flag": dense & model.isin(FLAGGED),
            "border": dense & (model == "normal") & ~rules.isin(FLAGGED) & borderline(cells),
            "sparse": (usable >= 1) & (usable < MIN_USABLE) & model.isin(FLAGGED)}


GROUP_POPULATION = {"T_random": "judged", "T_flag": "flag", "T_border": "border",
                    "T_sparse": "sparse"}


def sealed_test_sample(cells, score, rules, model, seed=20261119, sizes=None,
                       extra=None, n_sim=20000):
    """The sealed test sample, with what scoring needs to estimate population values.

    Groups are drawn in order (T_random, T_flag, T_border, T_sparse), each from its
    population minus the cells already taken, spread over region-days (_spread). For
    every drawn cell the groups file keeps its inclusion probability `pi` within its
    own group (simulated from the same rule, `inclusion`), whether it lies in the flag
    or border population, and the frozen predictor values in `extra`. A group whose
    pool is smaller than its size takes every cell (pi = 1) and is reported short; the
    shortfall is not moved to another group.

    Returns (labeling file, groups file, frames): frames has one row per group and
    region-day with the pool after earlier groups, the cells taken from it by earlier
    groups, the expected and the actual number drawn.
    """
    sizes = {**TEST_SIZES, **(sizes or {})}
    rng = np.random.default_rng(seed)
    pops = sealed_populations(cells, rules, model)
    key = _key(cells)
    taken: set = set()
    picks: list = []
    meta: list = []
    pi_of: dict = {}
    frames = []
    for name, pop in GROUP_POPULATION.items():
        member = pops[pop]
        before = cells.index[member & cells.index.isin(list(taken))]
        pool = cells.index[member & ~cells.index.isin(list(taken))]
        ch = _spread(pool, key, sizes[name], rng)
        counts = key.loc[pool].value_counts().to_dict()
        pk = inclusion(counts, sizes[name], n_sim=n_sim, seed=seed)
        got = key.loc[ch].value_counts().to_dict()
        prior = key.loc[before].value_counts().to_dict()
        for k_ in sorted(set(counts) | set(prior)):
            reg, date = k_.split("|", 1)
            frames.append({"group": name, "region": reg, "date": date,
                           "pool": int(counts.get(k_, 0)), "taken_before": int(prior.get(k_, 0)),
                           "expected": round(pk.get(k_, 0.0) * counts.get(k_, 0), 4),
                           "drawn": int(got.get(k_, 0))})
        for i in ch:
            pi_of[i] = pk[key.loc[i]]
        taken.update(ch)
        picks.extend(ch)
        meta.extend((name, 1.0 / pk[key.loc[i]], len(pool)) for i in ch)
    for k_, n in key[pops["judged"]].value_counts().items():
        reg, date = k_.split("|", 1)
        frames.append({"group": "judged", "region": reg, "date": date, "pool": int(n),
                       "taken_before": 0, "expected": 0.0, "drawn": 0})
    x = pd.DataFrame(index=cells.index)
    x["pi"] = pd.Series(pi_of, dtype=float).reindex(cells.index)
    x["in_flag"] = pops["flag"].astype(int)
    x["in_border"] = pops["border"].astype(int)
    if extra is not None:
        x = x.join(extra)
    gold, groups = _write(cells, picks, meta, "S", rng, score, rules, model, extra=x)
    return gold, groups, pd.DataFrame(frames)


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
