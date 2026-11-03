"""Holdout test 2 (docs/evaluation_protocol.md): a stratified, blind sample of one fresh
day, and its pre-declared evaluation.

The sample is built to answer one question the first gold set could not: where the model
and the rules disagree on real data, who is right? It therefore over-samples the cells
that matter (disagreements, the model's flags on sparse cells, borderline cells both
call normal) and adds random cells so that day-wide agreement can still be estimated by
weighting each stratum by its size on the day.

The labeling file looks exactly like the first gold set. Strata, weights and the
predictors' answers are kept in a separate file the labeler does not open.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from gnssguard.features.cells import CELL_KEYS
from gnssguard.labels.gold import GOLD_COLUMNS
from gnssguard.labels.rules import MIN_USABLE
from gnssguard.model.dataset import borderline

FLAGGED = ("jamming", "spoofing", "both")
CAPS = {"S1_disagree": 40, "S2_sparse": 50, "S3_borderline": 35}  # S2 raised 3 Nov
STRATA_COLUMNS = ["gold_id", "stratum", "weight", "pool"]


def classify(cells: pd.DataFrame, rule_label: pd.Series, model_label: pd.Series) -> pd.Series:
    """Stratum of every cell-window of the day ('' = not eligible)."""
    usable = cells["n_ac_usable"].fillna(0)
    dense = usable >= MIN_USABLE
    judged = rule_label != "unknown"
    r_flag, m_flag = rule_label.isin(FLAGGED), model_label.isin(FLAGGED)
    s = pd.Series("", index=cells.index)
    s[dense & judged] = "S4_random"
    s[dense & judged & ~r_flag & ~m_flag & borderline(cells)] = "S3_borderline"
    s[dense & judged & (r_flag != m_flag)] = "S1_disagree"
    s[(usable >= 1) & (usable < MIN_USABLE) & m_flag] = "S2_sparse"
    return s


def holdout_sample(cells: pd.DataFrame, rule_label: pd.Series, model_label: pd.Series,
                   n: int = 150, seed: int = 20261101, caps: dict | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (labeling file rows, strata rows)."""
    caps = {**CAPS, **(caps or {})}
    rng = np.random.default_rng(seed)
    stratum = classify(cells, rule_label, model_label)
    picks, meta = [], []

    def take(idx: pd.Index, k: int, name: str, pool: int) -> None:
        k = min(k, len(idx))
        chosen = idx[rng.permutation(len(idx))[:k]] if k else idx[:0]
        picks.extend(chosen)
        meta.extend([(name, pool / k if k else 0.0, pool)] * k)

    for name in ("S1_disagree", "S2_sparse", "S3_borderline"):
        idx = stratum.index[stratum == name]
        take(idx, caps[name], name, len(idx))
    rest = n - len(picks)
    pool = stratum[stratum == "S4_random"]
    regions = sorted(cells.loc[pool.index, "region"].unique())
    per = [rest // len(regions) + (1 if i < rest % len(regions) else 0)
           for i in range(len(regions))] if regions else []
    for region, k in zip(regions, per, strict=True):
        idx = pool.index[cells.loc[pool.index, "region"] == region]
        take(idx, k, "S4_random", len(idx))

    order = rng.permutation(len(picks))
    rows = cells.loc[[picks[i] for i in order], ["region", "date", *CELL_KEYS]].copy()
    rows["window"] = pd.to_datetime(rows["window"], utc=True)
    rows.insert(0, "gold_id", [f"H{i:04d}" for i in range(1, len(rows) + 1)])
    rows["label"], rows["labeler"], rows["notes"] = "", "", ""
    strata = pd.DataFrame([meta[i] for i in order], columns=["stratum", "weight", "pool"])
    strata.insert(0, "gold_id", rows["gold_id"].to_numpy())
    strata["rule_label"] = rule_label.loc[rows.index].astype(str).to_numpy()
    strata["model_label"] = model_label.loc[rows.index].astype(str).to_numpy()
    return rows[GOLD_COLUMNS].reset_index(drop=True), strata


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test against p = 0.5."""
    if n == 0:
        return float("nan")
    pk = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return float(min(1.0, sum(p for p in pk if p <= pk[k] * (1 + 1e-9))))


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def holdout_eval(gold: pd.DataFrame, strata: pd.DataFrame) -> dict:
    """The pre-declared analysis. `strata` carries the frozen predictors' labels."""
    gold = gold if "region" in gold else gold.assign(region="")
    m = gold[["gold_id", "region", "label"]].merge(strata, on="gold_id", how="inner")
    m = m[m["label"] != ""]
    sure = m[m["label"] != "unsure"].copy()
    sure["h"] = sure["label"].isin(FLAGGED)
    sure["r"] = sure["rule_label"].isin(FLAGGED)
    sure["m"] = sure["model_label"].isin(FLAGGED)
    out = {"labeled": int(len(m)), "decided": int(len(sure)),
           "unsure": int((m["label"] == "unsure").sum())}

    s1 = sure[sure["stratum"] == "S1_disagree"]
    model_right = int((s1["m"] == s1["h"]).sum())
    rules_right = int((s1["r"] == s1["h"]).sum())
    n1 = len(s1)
    p = binom_two_sided(model_right, n1) if n1 else float("nan")
    verdict = ("inconclusive (fewer than 10 decided disagreements)" if n1 < 10 else
               "model better" if p < 0.05 and model_right > rules_right else
               "rules better" if p < 0.05 and rules_right > model_right else
               "no significant difference")
    out["primary"] = {"decided_disagreements": n1, "human_sides_with_model": model_right,
                      "human_sides_with_rules": rules_right, "p_value": round(p, 4)
                      if p == p else None, "verdict": verdict}

    s3 = sure[sure["stratum"] == "S3_borderline"]
    out["S3_borderline_missed_by_both"] = {"decided": len(s3), "human_interference":
                                           int(s3["h"].sum()),
                                           "share": round(float(s3["h"].mean()), 3)
                                           if len(s3) else None}
    s2 = sure[sure["stratum"] == "S2_sparse"]
    ci = wilson(int(s2["h"].sum()), len(s2))
    out["primary_2_sparse_precision"] = {
        "decided": len(s2), "human_agrees": int(s2["h"].sum()),
        "precision": round(float(s2["h"].mean()), 3) if len(s2) else None, "ci95": ci,
        "verdict": ("inconclusive" if not len(s2) or ci[0] != ci[0] else
                    "supported" if ci[0] > 0.5 else
                    "not supported" if ci[1] < 0.5 else "inconclusive")}
    dense = sure[sure["stratum"] != "S2_sparse"].copy()
    # Day estimate: agreement within each sampling group (S4 is drawn per region), times
    # the group's size on the day. Unsure cells drop out within their own group, so a
    # group with many unsure cells keeps its full weight (fixed 3 Nov 2026: the first
    # run weighted single cells, which under-weighted groups with many unsure cells).
    dense["group"] = dense["stratum"] + "|" + dense["region"].where(
        dense["stratum"] == "S4_random", "")
    for name, col in (("rules", "r"), ("model", "m")):
        ok = (dense[col] == dense["h"]).astype(float)
        k_raw, n_raw = int(ok.sum()), len(ok)
        by_group = ok.groupby(dense["group"]).mean()
        size = dense.groupby("group")["pool"].first()
        out[f"{name}_agreement"] = {
            "sample": round(k_raw / n_raw, 3) if n_raw else None,
            "sample_ci95": wilson(k_raw, n_raw),
            "weighted_to_day": round(float((by_group * size).sum() / size.sum()), 4)
            if size.sum() else None,
        }
    exact = {}
    for name, col in (("rules", "rule_label"), ("model", "model_label")):
        exact[name] = round(float((sure[col] == sure["label"]).mean()), 3) if len(sure) else None
    out["exact_label_agreement_sample"] = exact
    by = sure.groupby("stratum").apply(
        lambda g: pd.Series({"decided": len(g), "rules_agree": int((g["r"] == g["h"]).sum()),
                             "model_agree": int((g["m"] == g["h"]).sum())}),
        include_groups=False)
    out["by_stratum"] = by.reset_index().to_dict(orient="records")
    return out
