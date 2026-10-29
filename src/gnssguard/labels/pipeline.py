"""Label every stored cell-window: rules -> label model -> events.

Two outputs per cell-window, because jamming and spoofing leave different evidence:

* `p_jamming`: label-model probability from the integrity rules (NIC, NACp, MLAT
  fallback).
* `spoof_votes`: how many motion rules fire (impossible jumps, position vs ground
  speed, GPS vs baro altitude). These rules are rare and one-sided, so they are
  counted rather than modelled.

`label` combines them: jamming | spoofing | both | normal | unknown (no rule could judge).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gnssguard.features.cells import load_cells
from gnssguard.labels.advisories import advisory_check, load_advisories
from gnssguard.labels.events import load_events, match_events
from gnssguard.labels.labelmodel import DawidSkene, majority_vote
from gnssguard.labels.rules import INTEGRITY_RULES, MOTION_RULES, apply_rules, rule_summary

log = logging.getLogger(__name__)

KEEP = ["region", "date", "cell_lat", "cell_lon", "window", "cell_deg", "window_min",
        "n_aircraft", "n_ac_usable", "n_ac_degraded", "share_ac_degraded"]
LABELS = ("jamming", "spoofing", "both", "normal", "unknown")


def combine(p_jamming: pd.Series, spoof_votes: pd.Series, any_vote: pd.Series) -> pd.Series:
    jam = p_jamming.fillna(0) >= 0.5
    spoof = spoof_votes >= 1
    label = np.select([jam & spoof, jam, spoof, any_vote], ["both", "jamming", "spoofing",
                                                             "normal"], default="unknown")
    return pd.Series(pd.Categorical(label, categories=LABELS), index=p_jamming.index)


def score(votes: pd.DataFrame, model: DawidSkene) -> pd.DataFrame:
    """Label columns for a vote matrix, given a label model fitted on integrity rules."""
    out = pd.DataFrame(index=votes.index)
    out["p_jamming"] = model.predict_proba(votes[INTEGRITY_RULES])
    out["majority_vote_jamming"] = majority_vote(votes[INTEGRITY_RULES])
    out["spoof_votes"] = (votes[MOTION_RULES] == 1).sum(axis=1).astype("int8")
    out["label"] = combine(out["p_jamming"], out["spoof_votes"], (votes != -1).any(axis=1))
    return out


def label_cells(cells: pd.DataFrame, events: pd.DataFrame | None = None
                ) -> tuple[pd.DataFrame, DawidSkene, pd.DataFrame]:
    votes = apply_rules(cells)
    model = DawidSkene().fit(votes[INTEGRITY_RULES])
    out = pd.concat([cells[KEEP], votes, score(votes, model)], axis=1)
    ev = match_events(cells, events if events is not None else pd.DataFrame())
    return pd.concat([out, ev], axis=1), model, rule_summary(votes)


def run(cells_dir: str | Path, out_file: str | Path, events_csv: str | Path | None,
        report_dir: str | Path, dates: list[str] | None = None,
        regions: list[str] | None = None,
        advisories_csv: str | Path | None = None) -> pd.DataFrame:
    cells = load_cells(cells_dir, dates, regions)
    if cells.empty:
        raise ValueError(f"no cell tables in {cells_dir}; run `gnssguard cells` first")
    events = load_events(events_csv) if events_csv else None
    labels, model, summary = label_cells(cells, events)

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(out_file, index=False, compression="zstd")

    rep = Path(report_dir)
    rep.mkdir(parents=True, exist_ok=True)
    summary.to_csv(rep / "rule_summary.csv", index=False)
    model.rule_table().to_csv(rep / "label_model_rules.csv", index=False)
    by_region = (labels.groupby(["region", "label"], observed=False).size()
                 .unstack(fill_value=0).reset_index())
    by_region.to_csv(rep / "labels_by_region.csv", index=False)
    if labels["event_id"].notna().any():
        ev = labels[labels["event_id"].notna()]
        ev.groupby("event_id").agg(
            cell_windows=("label", "size"),
            detected=("label", lambda s: round(float(s.isin(["jamming", "spoofing",
                                                               "both"]).mean()), 3)),
        ).reset_index().to_csv(rep / "event_recall.csv", index=False)
    if advisories_csv:
        adv = load_advisories(advisories_csv)
        if not adv.empty:
            advisory_check(labels, adv).to_csv(rep / "advisory_check.csv", index=False)
    (rep / "label_model.json").write_text(json.dumps({
        "prior_jamming": model.prior_, "iterations": model.n_iter_,
        "one_sided_rules": model.one_sided_, "cell_windows": len(labels),
    }, indent=2), encoding="utf-8")
    log.info("labeled %d cell-windows -> %s", len(labels), out_file)
    return labels
