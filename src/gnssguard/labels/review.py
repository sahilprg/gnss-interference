"""Review set 1 (docs/evaluation_protocol.md): false alarms.

A blind, stratified sample of cell-windows the system flags where little interference is
expected (calm Central Europe), flags in the hotspot regions, and calm cells it calls
normal, plus the pre-declared analysis. The labeling file looks like the gold sets;
groups, weights and predictions live in a separate file the labeler does not open.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gnssguard.features.cells import CELL_KEYS
from gnssguard.labels.gold import GOLD_COLUMNS
from gnssguard.labels.holdout import FLAGGED, wilson
from gnssguard.labels.rules import MIN_USABLE

CALM = "central_europe"
SIZES = {"R1_calm_flag": 50, "R2_hotspot_flag": 30, "R3_calm_normal": 20}
GROUP_COLUMNS = ["gold_id", "group", "weight", "pool", "rule_label", "model_label"]


KEY_ORDERS = ("shuffled", "alphabetical")


def _interleave(keys: list, rng) -> list:
    """Random order of keys that takes the strata ("region" in "region|date") in turn."""
    if not keys:
        return []
    strata: dict = {}
    for k in keys:
        strata.setdefault(str(k).split("|", 1)[0], []).append(k)
    names = sorted(strata)
    names = [names[i] for i in rng.permutation(len(names))]
    lists = [[strata[n][i] for i in rng.permutation(len(strata[n]))] for n in names]
    out = []
    for r in range(max(len(x) for x in lists)):
        out += [x[r] for x in lists if r < len(x)]
    return out


def _spread(idx: pd.Index, by: pd.Series, k: int, rng, key_order: str = "shuffled") -> list:
    """k of idx, spread as evenly as possible over the values of `by`: one per value per
    round. With "region|date" keys the values are visited region by region in turn (a
    random day of each region, then the next random day of each, ...), so every region
    gets its share even when there are more region-days than k; other keys are visited
    in a random order. The order is seeded and fixed for all rounds.

    key_order="alphabetical" is the order used before 8 Nov 2026, kept only to reproduce
    the samples drawn with it. It is biased when there are more values than k: only the
    first k values in alphabetical order (with region|date keys: the first region and
    its earliest days) are ever reached (evaluation protocol, Correction 2).
    """
    if key_order not in KEY_ORDERS:
        raise ValueError(f"key_order must be one of {KEY_ORDERS}")
    if k <= 0 or len(idx) == 0:
        return []  # an empty pool gives an empty pick; the caller reports the shortfall
    groups = {v: list(rng.permutation(np.asarray(g))) for v, g in idx.groupby(by.loc[idx]).items()}
    keys = sorted(groups)
    if key_order == "shuffled":
        keys = _interleave(keys, rng)
    out: list = []
    while len(out) < k and any(groups.values()):
        for v in keys:
            if groups[v] and len(out) < k:
                out.append(groups[v].pop())
    return out


def _picks_per_key(counts: dict, k: int, rng) -> dict:
    """How many cells _spread takes from each key, from the pool sizes alone. The same
    rule as _spread (same key order, one per key per round), so its distribution of
    picks per key is the sampler's."""
    left = {v: int(n) for v, n in counts.items() if n > 0}
    order = _interleave(sorted(left), rng)
    out = dict.fromkeys(order, 0)
    total = 0
    while total < k and any(left.values()):
        for v in order:
            if left[v] and total < k:
                left[v] -= 1
                out[v] += 1
                total += 1
    return out


def inclusion(counts: dict, k: int, n_sim: int = 20000, seed: int = 0) -> dict:
    """Probability that a given cell of each key is drawn by _spread (shuffled order).

    Within a key the cells are a simple random sample, so a cell's probability is the
    key's expected number of picks over its pool size. The expectation is simulated
    (n_sim draws; relative error about 1% for keys reached half the time). Keys whose
    every cell is always taken get exactly 1.
    """
    counts = {v: int(n) for v, n in counts.items() if n > 0}
    if not counts or k <= 0:
        return {}
    if sum(counts.values()) <= k:
        return dict.fromkeys(counts, 1.0)
    rng = np.random.default_rng(seed)
    acc = dict.fromkeys(counts, 0)
    for _ in range(n_sim):
        for v, n in _picks_per_key(counts, k, rng).items():
            acc[v] += n
    return {v: acc[v] / n_sim / counts[v] for v in counts}


def review_sample(cells: pd.DataFrame, rule_label: pd.Series, model_label: pd.Series,
                  seed: int = 20261104, sizes: dict | None = None,
                  key_order: str = "shuffled") -> tuple[pd.DataFrame, pd.DataFrame]:
    """review_1 (27 Oct-3 Nov) was drawn with key_order="alphabetical"."""
    sizes = {**SIZES, **(sizes or {})}
    rng = np.random.default_rng(seed)
    dense = cells["n_ac_usable"].fillna(0) >= MIN_USABLE
    calm = cells["region"].astype(str) == CALM
    flag = model_label.isin(FLAGGED)
    pools = {"R1_calm_flag": cells.index[dense & calm & flag],
             "R2_hotspot_flag": cells.index[dense & ~calm & flag],
             "R3_calm_normal": cells.index[dense & calm & (model_label == "normal")]}
    spread_by = {"R1_calm_flag": cells["date"].astype(str),
                 "R2_hotspot_flag": cells["region"].astype(str),
                 "R3_calm_normal": cells["date"].astype(str)}
    picks, meta = [], []
    short = 0
    for name in ("R1_calm_flag", "R2_hotspot_flag", "R3_calm_normal"):
        want = sizes[name] + (short if name == "R3_calm_normal" else 0)
        chosen = _spread(pools[name], spread_by[name], want, rng, key_order)
        short += sizes[name] - len(chosen) if name != "R3_calm_normal" else 0
        if name == "R2_hotspot_flag":
            # weight by region: each region's pool size over its sampled count
            reg = cells.loc[chosen, "region"].astype(str)
            pool_reg = cells.loc[pools[name], "region"].astype(str).value_counts()
            n_reg = reg.value_counts()
            w = [pool_reg[r] / n_reg[r] for r in reg]
        else:
            w = [len(pools[name]) / max(len(chosen), 1)] * len(chosen)
        picks += chosen
        meta += [(name, wi, len(pools[name])) for wi in w]
    order = rng.permutation(len(picks))
    rows = cells.loc[[picks[i] for i in order], ["region", "date", *CELL_KEYS]].copy()
    rows["window"] = pd.to_datetime(rows["window"], utc=True)
    rows.insert(0, "gold_id", [f"F{i:04d}" for i in range(1, len(rows) + 1)])
    rows["label"], rows["labeler"], rows["notes"] = "", "", ""
    groups = pd.DataFrame([meta[i] for i in order], columns=["group", "weight", "pool"])
    groups.insert(0, "gold_id", rows["gold_id"].to_numpy())
    groups["rule_label"] = rule_label.loc[rows.index].astype(str).to_numpy()
    groups["model_label"] = model_label.loc[rows.index].astype(str).to_numpy()
    return rows[GOLD_COLUMNS].reset_index(drop=True), groups[GROUP_COLUMNS]


def review_eval(gold: pd.DataFrame, groups: pd.DataFrame, calm_flagged_share: float | None
                = None) -> dict:
    """The pre-declared analysis."""
    m = gold[["gold_id", "label"]].merge(groups, on="gold_id", how="inner")
    m = m[m["label"].fillna("") != ""]
    out: dict[str, Any] = {"labeled": int(len(m)), "unsure": int((m["label"] == "unsure").sum())}
    sure = m[m["label"] != "unsure"].copy()
    sure["h"] = sure["label"].isin(FLAGGED)

    r1 = sure[sure["group"] == "R1_calm_flag"]
    k = int((~r1["h"]).sum())
    lo, hi = wilson(k, len(r1))
    res = {"decided": len(r1), "human_normal": k,
           "false_alarm_share": round(k / len(r1), 3) if len(r1) else None, "ci95": [lo, hi],
           "unsure": int(((m["group"] == "R1_calm_flag") & (m["label"] == "unsure")).sum())}
    if calm_flagged_share is not None and len(r1):
        res["calm_flagged_share"] = calm_flagged_share
        res["estimated_real_false_alarm_rate"] = [round(calm_flagged_share * lo, 5),
                                                   round(calm_flagged_share * k / len(r1), 5),
                                                   round(calm_flagged_share * hi, 5)]
    out["primary_calm_flags"] = res

    r2 = sure[sure["group"] == "R2_hotspot_flag"]
    if len(r2):
        k2 = int(r2["h"].sum())
        w = r2["weight"]
        out["R2_hotspot_precision"] = {
            "decided": len(r2), "human_interference": k2, "ci95": list(wilson(k2, len(r2))),
            "precision": round(k2 / len(r2), 3),
            "weighted": round(float((r2["h"] * w).sum() / w.sum()), 3)}
    r3 = sure[sure["group"] == "R3_calm_normal"]
    if len(r3):
        k3 = int((~r3["h"]).sum())
        out["R3_calm_normal"] = {"decided": len(r3), "human_normal": k3,
                                 "ci95": list(wilson(k3, len(r3)))}
    both = sure[sure["group"].isin(["R1_calm_flag", "R2_hotspot_flag"]) & sure["h"]]
    out["type_agreement_on_confirmed_flags"] = {
        "cells": len(both),
        "same_type": int((both["label"] == both["model_label"]).sum())}
    out["rules_agree_with_model"] = round(float((m["rule_label"].isin(FLAGGED)
                                                == m["model_label"].isin(FLAGGED)).mean()), 3)
    return out
