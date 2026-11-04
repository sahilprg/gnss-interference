"""Check the detectors against the public event catalog (`configs/events.csv`).

The rule is fixed in docs/evaluation_protocol.md ("Event check 1"), before any event
day was scored. Per event and predictor:

* cell-windows of the event's region overlapping its box and period, with 3+ usable
  aircraft, are judged;
* a judged cell-window matches if its label fits the event type (jamming: jamming or
  both; spoofing: spoofing or both; unknown: any interference);
* found: MIN_FOUND or more matches; not found: MIN_JUDGED or more judged and fewer than
  MIN_FOUND matches; otherwise too few cells.

Also: the calm-control flagged share on each event day (a placebo).
"""

from __future__ import annotations

import pandas as pd

from gnssguard.labels.events import load_events
from gnssguard.labels.holdout import wilson

MIN_USABLE = 3
MIN_FOUND = 3
MIN_JUDGED = 10
FLAGGED = ("jamming", "spoofing", "both")
MATCH = {"jamming": ("jamming", "both"), "spoofing": ("spoofing", "both"),
         "unknown": FLAGGED}
CONTROL = "central_europe"


def _in_event(cells: pd.DataFrame, ev) -> pd.Series:
    deg = 1.0
    w0 = pd.to_datetime(cells["window"], utc=True)
    w1 = w0 + pd.Timedelta(minutes=15)
    return ((cells["region"].astype(str) == ev.region)
            & (cells["cell_lat"] < ev.lat_max) & (cells["cell_lat"] + deg > ev.lat_min)
            & (cells["cell_lon"] < ev.lon_max) & (cells["cell_lon"] + deg > ev.lon_min)
            & (w0 < ev.end_utc) & (w1 > ev.start_utc))


def verdict(n_judged: int, n_match: int) -> str:
    if n_match >= MIN_FOUND:
        return "found"
    if n_judged >= MIN_JUDGED:
        return "not found"
    return "too few cells"


def check(cells: pd.DataFrame, labels: dict[str, pd.Series], events: pd.DataFrame
          ) -> pd.DataFrame:
    """One row per event and predictor. labels: predictor name -> label per cell row."""
    dense = cells["n_ac_usable"].fillna(0) >= MIN_USABLE
    rows = []
    for ev in events.itertuples(index=False):
        region = ev.region if isinstance(ev.region, str) and ev.region else None
        inside = _in_event(cells, ev) & dense if region else pd.Series(False, cells.index)
        for name, lab in labels.items():
            n_j = int(inside.sum())
            n_m = int((inside & lab.isin(MATCH[ev.type])).sum())
            rows.append({"event_id": ev.event_id, "type": ev.type,
                         "evidence_level": ev.evidence_level, "region": region or "",
                         "date": ev.start_utc.date().isoformat(), "predictor": name,
                         "judged": n_j, "matching": n_m,
                         "share": round(n_m / n_j, 3) if n_j else None,
                         "verdict": verdict(n_j, n_m) if region else "not checked"})
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    decided = res[res["verdict"].isin(["found", "not found"])]
    out = []
    for keys, g in [(("all",), decided), *[((lv,), g) for lv, g in
                                         decided.groupby("evidence_level")],
                    (("spoofing events",), decided[decided["type"] == "spoofing"])]:
        for name, h in g.groupby("predictor"):
            k, n = int((h["verdict"] == "found").sum()), len(h)
            lo, hi = wilson(k, n)
            out.append({"group": keys[0], "predictor": name, "found": k, "decided": n,
                        "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(out)


def placebo(cells: pd.DataFrame, labels: dict[str, pd.Series]) -> pd.DataFrame:
    dense = (cells["n_ac_usable"].fillna(0) >= MIN_USABLE) & (cells["region"].astype(str)
                                                              == CONTROL)
    rows = []
    for name, lab in labels.items():
        for day, idx in cells[dense].groupby("date").groups.items():
            rows.append({"date": str(day), "predictor": name, "judged": len(idx),
                         "flagged_share": round(float(lab.loc[idx].isin(FLAGGED).mean()), 4)})
    return pd.DataFrame(rows)


def run(cells_dir: str, model_dir: str, events_csv: str) -> dict[str, pd.DataFrame]:
    from gnssguard.features.cells import load_cells
    from gnssguard.model.dataset import with_context
    from gnssguard.model.evaluate import rule_frame
    from gnssguard.model.train import Detectors

    events = load_events(events_csv)
    days = sorted({d.date().isoformat() for s, e in zip(events["start_utc"],
                                                         events["end_utc"], strict=True)
                   for d in pd.date_range(s.floor("D"), e - pd.Timedelta(seconds=1),
                                          freq="D")})
    det = Detectors.load(model_dir)
    cells = load_cells(cells_dir, days, None)
    if cells.empty:
        raise ValueError("no cell tables for the event days; ingest them and run `cells`")
    cells = with_context(cells).reset_index(drop=True)
    labels = {"rules": rule_frame(cells, det)["label"].astype(str),
              "model": det.predict(cells)["label"].astype(str)}
    res = check(cells, labels, events)
    return {"events": res, "summary": summarize(res), "placebo": placebo(cells, labels)}
