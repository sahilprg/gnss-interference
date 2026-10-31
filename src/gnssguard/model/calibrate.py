"""Make probabilities honest: isotonic calibration and reliability measures.

A probability of 0.8 should mean that 8 in 10 such cells really are affected. Boosted
trees trained on mixed weak and synthetic labels do not guarantee that, and the public
map shows these numbers, so they are checked and, if needed, corrected.

* `fit_isotonic`: pool-adjacent-violators (no scikit-learn dependency). The result is a
  monotone step map from raw score to observed frequency; it is applied with linear
  interpolation between block centres, so it never reorders cells.
* `reliability` / `ece`: the reliability table and expected calibration error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Calibrator:
    x: list[float]  # raw score at each block centre, increasing
    y: list[float]  # observed frequency of each block, non-decreasing
    n: int = 0

    def __call__(self, p) -> np.ndarray:
        p = np.asarray(p, dtype="float64")
        if not self.x:
            return p
        return np.interp(p, self.x, self.y)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "n": self.n}

    @classmethod
    def from_dict(cls, d: dict) -> Calibrator:
        return cls(list(d["x"]), list(d["y"]), int(d.get("n", 0)))


def fit_isotonic(p, y, w=None, *, min_block_weight: float = 20.0) -> Calibrator:
    """Isotonic regression of y on p (pool adjacent violators).

    Blocks lighter than `min_block_weight` are merged with a neighbour, so a handful of
    cells cannot pin the map to 0 or 1. The map is anchored at (0, 0) and (1, 1) only if
    the data reach there; otherwise the end blocks hold flat.
    """
    p = np.asarray(p, dtype="float64")
    y = np.asarray(y, dtype="float64")
    w = np.ones_like(p) if w is None else np.asarray(w, dtype="float64")
    ok = np.isfinite(p) & np.isfinite(y) & (w > 0)
    p, y, w = p[ok], y[ok], w[ok]
    if len(p) == 0:
        return Calibrator([], [], 0)
    order = np.argsort(p, kind="mergesort")
    p, y, w = p[order], y[order], w[order]

    # blocks as [sum_w, sum_wy, sum_wp]
    sw, swy, swp = [], [], []
    for pi, yi, wi in zip(p, y, w, strict=True):
        sw.append(wi)
        swy.append(wi * yi)
        swp.append(wi * pi)
        while len(sw) > 1 and swy[-2] / sw[-2] >= swy[-1] / sw[-1]:
            _merge_last(sw, swy, swp)
    # merge light blocks into the neighbour whose mean is closest
    i = 0
    while len(sw) > 1 and i < len(sw):
        if sw[i] >= min_block_weight:
            i += 1
            continue
        j = i - 1 if i == len(sw) - 1 else (
            i + 1 if i == 0 else
            (i - 1 if abs(swy[i - 1] / sw[i - 1] - swy[i] / sw[i])
             <= abs(swy[i + 1] / sw[i + 1] - swy[i] / sw[i]) else i + 1))
        a, b = min(i, j), max(i, j)
        sw[a] += sw[b]
        swy[a] += swy[b]
        swp[a] += swp[b]
        del sw[b], swy[b], swp[b]
        i = max(a - 1, 0)
    xs = [float(a / b) for a, b in zip(swp, sw, strict=True)]
    ys = [float(a / b) for a, b in zip(swy, sw, strict=True)]
    return Calibrator(xs, ys, int(len(p)))


def _merge_last(*blocks: list) -> None:
    for b in blocks:
        last = b.pop()  # pop first: `b[-2] += b.pop()` would add into the wrong element
        b[-1] += last


def reliability(p, y, bins: int = 10) -> pd.DataFrame:
    """Per probability bin: cells, mean predicted probability, observed frequency."""
    p = np.asarray(p, dtype="float64")
    y = np.asarray(y, dtype="float64")
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for k in range(bins):
        m = b == k
        rows.append({"bin": f"{edges[k]:.1f}-{edges[k + 1]:.1f}", "cells": int(m.sum()),
                     "mean_p": round(float(p[m].mean()), 4) if m.any() else None,
                     "observed": round(float(y[m].mean()), 4) if m.any() else None})
    return pd.DataFrame(rows)


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error: cell-weighted mean |predicted - observed| over bins."""
    t = reliability(p, y, bins)
    t = t[t["cells"] > 0]
    if t.empty:
        return float("nan")
    return float((t["cells"] * (t["mean_p"] - t["observed"]).abs()).sum() / t["cells"].sum())


def mid_ece(p, y, lo: float = 0.05, hi: float = 0.95, bins: int = 10) -> float:
    """ECE over the uncertain cells only (lo <= p < hi).

    Almost all cells sit near 0 or 1, which makes plain ECE tiny even when every
    uncertain cell is badly off; this is the number that matters for a map.
    """
    p = np.asarray(p, dtype="float64")
    y = np.asarray(y, dtype="float64")
    m = (p >= lo) & (p < hi)
    return ece(p[m], y[m], bins) if m.any() else float("nan")
