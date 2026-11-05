"""Blind relabel (docs/evaluation_protocol.md, "Blind relabel"): the same labeler labels
again a random share of earlier hand-labeled cells, without seeing the earlier labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gnssguard.labels.gold import GOLD_COLUMNS
from gnssguard.labels.holdout import FLAGGED, wilson

SETS = {"gold_2026-10-26": ("labels/gold.csv", 30),
        "holdout_2026-11-01": ("labels/gold_2026-11-01.csv", 15),
        "review_1": ("labels/review_1.csv", 10)}


def relabel_sample(seed: int = 20261105, sets: dict = SETS) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    parts = []
    for name, (path, k) in sets.items():
        g = pd.read_csv(path, dtype=str, keep_default_na=False)
        g = g[g["label"] != ""]
        pick = g.iloc[np.sort(rng.choice(len(g), size=k, replace=False))].copy()
        pick["source_set"], pick["source_id"] = name, pick["gold_id"]
        parts.append(pick)
    allp = pd.concat(parts, ignore_index=True)
    allp = allp.iloc[rng.permutation(len(allp))].reset_index(drop=True)
    allp["gold_id"] = [f"B{i:04d}" for i in range(1, len(allp) + 1)]
    key = allp[["gold_id", "source_set", "source_id"]].copy()
    key["first_label"] = allp["label"]
    out = allp.copy()
    out["label"], out["labeler"], out["notes"] = "", "", ""
    return out[GOLD_COLUMNS], key


def cohen_kappa(a: pd.Series, b: pd.Series) -> float | None:
    if len(a) == 0:
        return None
    cats = sorted(set(a) | set(b))
    po = float((a.to_numpy() == b.to_numpy()).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def relabel_eval(second: pd.DataFrame, key: pd.DataFrame) -> dict:
    m = key.merge(second[["gold_id", "label"]], on="gold_id")
    m = m[m["label"].fillna("") != ""]
    unsure = (m["first_label"] == "unsure") | (m["label"] == "unsure")
    d = m[~unsure]
    a, b = d["first_label"].isin(FLAGGED), d["label"].isin(FLAGGED)
    k_bin, k_ex = int((a == b).sum()), int((d["first_label"] == d["label"]).sum())
    return {"relabeled": len(m), "unsure_in_either": int(unsure.sum()), "decided_both": len(d),
            "binary_agreement": round(k_bin / len(d), 3) if len(d) else None,
            "binary_ci95": list(wilson(k_bin, len(d))),
            "binary_kappa": cohen_kappa(a.astype(int), b.astype(int)),
            "exact_agreement": round(k_ex / len(d), 3) if len(d) else None,
            "exact_ci95": list(wilson(k_ex, len(d))),
            "exact_kappa": cohen_kappa(d["first_label"], d["label"]),
            "changed": d.loc[d["first_label"] != d["label"],
                             ["source_set", "first_label", "label"]].value_counts()
            .rename("cells").reset_index().to_dict(orient="records")}
