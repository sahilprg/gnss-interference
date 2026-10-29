"""Train the jamming and spoofing detectors (two LightGBM binary models).

Training rows come from real days (weak labels) and synthetic attacks (exact truth).
The validation day is used only for early stopping. Saved models are plain LightGBM
text files plus a JSON with everything needed to reproduce and score.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from gnssguard.features.cells import GridSpec, load_cells
from gnssguard.labels.labelmodel import DawidSkene
from gnssguard.labels.pipeline import LABELS
from gnssguard.model.dataset import (
    FEATURES,
    fit_label_model,
    real_targets,
    synthetic_rows,
    with_context,
)

log = logging.getLogger(__name__)

PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 40,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "verbose": -1, "seed": 0, "deterministic": True, "num_threads": 4,
}
TARGETS = {"y_jam": "p_jamming", "y_spoof": "p_spoofing"}


def build_table(processed: str | Path, cells_dir: str | Path, train_dates: list[str],
                val_dates: list[str], *, synth_region: str = "central_europe",
                synth_seeds: tuple[int, ...] = (0, 1), spec: GridSpec = GridSpec(),
                regions: list[str] | None = None) -> tuple[pd.DataFrame, DawidSkene]:
    """Real (weak labels) + synthetic (exact truth) rows for the given days."""
    days = train_dates + val_dates
    cells = load_cells(cells_dir, days, regions)
    if cells.empty:
        raise ValueError(f"no cell tables in {cells_dir} for {days}")
    lm = fit_label_model(cells, train_dates)
    real = with_context(cells)
    real = pd.concat([real, real_targets(real, lm)], axis=1)
    real["source"] = "real"

    synth, _ = synthetic_rows(processed, synth_region, days, list(synth_seeds), spec=spec)
    synth["source"] = "synthetic"
    table = pd.concat([real, synth], ignore_index=True)
    table["split"] = np.where(table["date"].isin(train_dates), "train", "val")
    table["weight"] = 1.0
    log.info("training table: %d real + %d synthetic rows", len(real), len(synth))
    return table, lm


@dataclass
class Detectors:
    models: dict[str, lgb.Booster]
    features: list[str]
    label_model: DawidSkene | None = None
    meta: dict = field(default_factory=dict)

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        x = df[self.features]
        for target, col in TARGETS.items():
            out[col] = self.models[target].predict(x)
        jam = out["p_jamming"] >= threshold
        spoof = out["p_spoofing"] >= threshold
        nothing = df["n_ac_usable"].fillna(0) < 1
        label = np.select([nothing, jam & spoof, jam, spoof],
                          ["unknown", "both", "jamming", "spoofing"], default="normal")
        out["label"] = pd.Categorical(label, categories=LABELS)
        return out

    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for target, m in self.models.items():
            m.save_model(str(folder / f"{target}.txt"))
        lm = self.label_model
        meta = {**self.meta, "features": self.features,
                "label_model": None if lm is None else {
                    "rules": lm.rules_, "one_sided": lm.one_sided_, "prior": lm.prior_,
                    "conf": lm.conf_.tolist()}}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return folder

    @classmethod
    def load(cls, folder: str | Path) -> Detectors:
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        models = {t: lgb.Booster(model_file=str(folder / f"{t}.txt")) for t in TARGETS}
        lm = None
        if meta.get("label_model"):
            d = meta["label_model"]
            lm = DawidSkene()
            lm.rules_, lm.one_sided_, lm.prior_ = d["rules"], d["one_sided"], d["prior"]
            lm.conf_ = np.array(d["conf"])
        return cls(models, meta["features"], lm, meta)


def train(table: pd.DataFrame, features: list[str] = FEATURES, *, sources=("real", "synthetic"),
          rounds: int = 3000, early_stop: int = 100, params: dict | None = None,
          label_model: DawidSkene | None = None) -> Detectors:
    params = {**PARAMS, **(params or {})}
    t = table[table["source"].isin(sources)]
    models, info = {}, {}
    for target in TARGETS:
        tr = t[(t["split"] == "train") & t[target].notna()]
        va = t[(t["split"] == "val") & t[target].notna()]
        if tr[target].nunique() < 2:
            raise ValueError(f"{target}: training rows have a single class")
        dtr = lgb.Dataset(tr[features], tr[target], weight=tr["weight"], free_raw_data=False)
        dva = lgb.Dataset(va[features], va[target], weight=va["weight"], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(early_stop, verbose=False)])
        models[target] = m
        info[target] = {"best_iteration": m.best_iteration, "train_rows": len(tr),
                        "train_positives": int(tr[target].sum()), "val_rows": len(va),
                        "val_logloss": round(float(m.best_score["valid_0"]["binary_logloss"]), 5)}
        log.info("%s: %s", target, info[target])
    meta = {"sources": list(sources), "targets": info, "params": params,
            "train_dates": sorted(t.loc[t["split"] == "train", "date"].unique()),
            "val_dates": sorted(t.loc[t["split"] == "val", "date"].unique())}
    return Detectors(models, list(features), label_model, meta)


def importance(det: Detectors) -> pd.DataFrame:
    rows = []
    for target, m in det.models.items():
        gain = m.feature_importance("gain")
        tot = gain.sum() or 1.0
        rows += [{"target": target, "feature": f, "gain_share": round(float(g / tot), 4)}
                 for f, g in zip(det.features, gain, strict=True)]
    return (pd.DataFrame(rows).sort_values(["target", "gain_share"], ascending=[True, False])
            .reset_index(drop=True))
