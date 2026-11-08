"""`gnssguard benchmark`: regenerate every published result from stored data, labels
and the frozen model, and write them to results/ with their provenance.

Each section reruns a check declared in docs/evaluation_protocol.md, with the same code
as when it was first scored. A section whose inputs are missing is reported as
"not run" with the reason, never silently skipped. Nothing here changes a model, rule or
threshold.

Sections: 26 Oct gold set, 1 Nov holdout, review set 1 (false alarms), calibration set
(score bands), event checks, score vs hand labels, and (if a rolling run is given) the
rolling synthetic and real results with day intervals.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_SETS = {
    "gold_2026-10-26": "labels/gold.csv",
    "holdout_2026-11-01": ("labels/gold_2026-11-01.csv", "labels/holdout_2026-11-01_strata.csv"),
    "review_1": ("labels/review_1.csv", "labels/review_1_groups.csv"),
    "calibration_1": ("labels/calibration_1.csv", "labels/calibration_1_groups.csv"),
}
REVIEW_DAYS = ["2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30", "2026-10-31",
               "2026-11-02", "2026-11-03"]


def file_sha(path: str | Path) -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else None


def _cells(cells_dir, days, regions=None):
    from gnssguard.features.cells import load_cells
    from gnssguard.model.dataset import with_context

    c = load_cells(cells_dir, days, regions)
    return with_context(c).reset_index(drop=True) if not c.empty else c


def _missing(cells: pd.DataFrame, days: list[str]) -> list[str]:
    have = set(cells["date"].astype(str)) if not cells.empty else set()
    return sorted(set(days) - have)


def run(model_dir: str, cells_dir: str = "data/cells",
        labels_table: str = "data/labels/labels.parquet", events_csv: str = "configs/events.csv",
        rolling_dir: str | None = None, out: str = "results") -> dict:
    from gnssguard.labels.holdout import holdout_eval
    from gnssguard.labels.review import review_eval
    from gnssguard.labels.sealed import score_bands
    from gnssguard.model import eventcheck, trust
    from gnssguard.model.evaluate import gold_eval, model_frame, rule_frame
    from gnssguard.model.train import Detectors
    from gnssguard.product.publish import model_hash

    det = Detectors.load(model_dir)
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    res: dict = {}

    def section(name, fn):
        try:
            res[name] = fn()
        except Exception as e:  # recorded, not hidden
            res[name] = {"status": "not run", "reason": f"{type(e).__name__}: {e}"}

    def gold():
        c = _cells(cells_dir, ["2026-10-26"])
        if _missing(c, ["2026-10-26"]):
            return {"status": "not run", "reason": "no cells for 2026-10-26"}
        pop = pd.read_parquet(labels_table) if Path(labels_table).exists() else None
        preds = {"rules": lambda x: rule_frame(x, det), "model": lambda x: model_frame(x, det)}
        t = gold_eval(LABEL_SETS["gold_2026-10-26"], c, preds, population=pop)
        t.to_csv(outp / "gold_2026-10-26.csv", index=False)
        return {"status": "ok", "rows": t.to_dict(orient="records")}

    def holdout():
        g, s = LABEL_SETS["holdout_2026-11-01"]
        gold_ = pd.read_csv(g, dtype=str, keep_default_na=False)
        strata = pd.read_csv(s, keep_default_na=False)
        strata["weight"] = strata["weight"].astype(float)
        strata["pool"] = strata["pool"].astype(int)
        return {"status": "ok", **holdout_eval(gold_, strata)}

    def review():
        g, s = LABEL_SETS["review_1"]
        c = _cells(cells_dir, REVIEW_DAYS, ["central_europe"])
        miss = _missing(c, REVIEW_DAYS)
        if miss:
            return {"status": "not run", "reason": f"no cells for {miss}"}
        calm = c["n_ac_usable"].fillna(0) >= 3
        share = float(det.predict(c[calm])["label"].astype(str)
                      .isin(["jamming", "spoofing", "both"]).mean())
        groups = pd.read_csv(s, keep_default_na=False)
        groups["weight"] = groups["weight"].astype(float)
        r = review_eval(pd.read_csv(g, dtype=str, keep_default_na=False), groups,
                        round(share, 4))
        return {"status": "ok", **r}

    def calibration():
        g, s = LABEL_SETS["calibration_1"]
        return {"status": "ok", **score_bands(pd.read_csv(g, dtype={"label": str}),
                                              pd.read_csv(s), Path(g).name)}

    def events():
        r = eventcheck.run(cells_dir, model_dir, events_csv)
        for k, df in r.items():
            df.to_csv(outp / f"events_{k}.csv", index=False)
        return {"status": "ok", "summary": r["summary"].to_dict(orient="records"),
                "verdicts": r["events"].pivot_table(
                    index="event_id", columns="predictor", values="verdict",
                    aggfunc="first").to_dict(orient="index")}

    def scores_vs_labels():
        files = [LABEL_SETS["gold_2026-10-26"], LABEL_SETS["holdout_2026-11-01"][0],
                 LABEL_SETS["review_1"][0]]
        days = sorted({str(d) for f in files for d in pd.read_csv(f)["date"].unique()})
        c = _cells(cells_dir, days)
        m = trust.human_scored(files, c, det)
        t = trust.reliability_by(m, None)
        t.to_csv(outp / "scores_vs_hand_labels.csv", index=False)
        return {"status": "ok", "decided_cells": len(m), "bands": t.to_dict(orient="records")}

    def rolling():
        if not rolling_dir or not Path(rolling_dir, "rolling_synthetic.csv").exists():
            return {"status": "not run", "reason": "no rolling results given (--rolling); "
                    "run `gnssguard rolling` first (about 30 min)"}
        t = trust.rolling_intervals(rolling_dir)
        t.to_csv(outp / "rolling_intervals.csv", index=False)
        return {"status": "ok", "rows": t[t["predictor"] == "model"].to_dict(orient="records")}

    for name, fn in (("gold_2026-10-26", gold), ("holdout_2026-11-01", holdout),
                     ("review_1", review), ("calibration_1", calibration),
                     ("events", events), ("scores_vs_hand_labels", scores_vs_labels),
                     ("rolling", rolling)):
        section(name, fn)

    from gnssguard import __version__

    inputs = {"events.csv": file_sha(events_csv)}
    for v in LABEL_SETS.values():
        for f in (v if isinstance(v, tuple) else (v,)):
            inputs[Path(f).name] = file_sha(f)
    prov = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_version": __version__, "model_hash": model_hash(model_dir),
            "model_train_dates": [det.meta["train_dates"][0], det.meta["train_dates"][-1]],
            "model_fingerprints": det.meta.get("fingerprints"),
            "score_bands": file_sha(Path(model_dir).parent / "score_bands.json"),
            "inputs": inputs}
    doc = {"provenance": prov, "results": res}
    (outp / "benchmark.json").write_text(json.dumps(doc, indent=1, default=_json) + "\n",
                                         encoding="utf-8")
    (outp / "README.md").write_text(summary_md(doc), encoding="utf-8")
    return doc


def _json(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return str(x)


def _pct(x, d=1):
    return "-" if x is None else f"{100 * x:.{d}f}%"


def summary_md(doc: dict) -> str:
    p, r = doc["provenance"], doc["results"]
    lines = ["# Results", "",
             "Regenerated by `gnssguard benchmark` from stored data, the hand labels and the "
             "frozen model. Every check was declared in docs/evaluation_protocol.md before it "
             "was first scored; this file is overwritten on each run, and its history is the "
             "git history of `results/`.", "",
             f"* Generated: {p['generated_at']}, code {p['code_version']}",
             f"* Model hash {p['model_hash']}, trained {p['model_train_dates'][0]} to "
             f"{p['model_train_dates'][1]}",
             "* Inputs (sha256, 12 characters): " + ", ".join(
                 f"`{k}` {v}" for k, v in p["inputs"].items()),
             "* Incident pages, one per catalogued event (`gnssguard incidents`, not part of "
             "this run): `results/incidents/`", ""]

    def status(name):
        s = r.get(name, {})
        return None if s.get("status") == "ok" else s.get("reason", "missing")

    lines += ["| Check | Result |", "| --- | --- |"]
    if not status("gold_2026-10-26"):
        for row in r["gold_2026-10-26"]["rows"]:
            lines.append(f"| Gold set 26 Oct, {row['predictor']} | "
                         f"{_pct(row['binary_agreement'])} of {row['decided']} decided; "
                         f"weighted to the day {_pct(row.get('weighted_binary_agreement'))} |")
    if not status("holdout_2026-11-01"):
        h = r["holdout_2026-11-01"]
        p2 = h["primary_2_sparse_precision"]
        lines.append(f"| Holdout 1 Nov, flags from 1-2 aircraft | {p2['human_agrees']} of "
                     f"{p2['decided']} right ({_pct(p2['ci95'][0], 0)}-{_pct(p2['ci95'][1], 0)})"
                     f" |")
        lines.append(f"| Holdout 1 Nov, day-weighted agreement | rules "
                     f"{_pct(h['rules_agreement']['weighted_to_day'])}, model "
                     f"{_pct(h['model_agreement']['weighted_to_day'])} |")
    if not status("review_1"):
        v = r["review_1"]["primary_calm_flags"]
        est = v.get("estimated_real_false_alarm_rate") or [None] * 3
        lines.append(f"| False alarms among calm-region flags | {v['human_normal']} of "
                     f"{v['decided']} ({_pct(v['ci95'][0], 0)}-{_pct(v['ci95'][1], 0)}); "
                     f"about {_pct(est[1], 2)} of judged cells |")
        r2 = r["review_1"].get("R2_hotspot_precision")
        if r2:
            lines.append(f"| Hotspot flags confirmed | {r2['human_interference']} of "
                         f"{r2['decided']} |")
    if not status("calibration_1"):
        lines.append("| Reviewer share by score | " + "; ".join(
            f"{b['band']}: {_pct(b['interference_share'], 0)}"
            for b in r["calibration_1"]["bands"]) + " |")
    if not status("events"):
        for s in r["events"]["summary"]:
            if s["predictor"] == "model":
                lines.append(f"| Events found, {s['group']} | {s['found']} of {s['decided']} "
                             f"({_pct(s['ci95_low'], 0)}-{_pct(s['ci95_high'], 0)}) |")
    if not status("rolling"):
        for row in r["rolling"]["rows"]:
            if row["source"] == "synthetic" and row["metric"] == "recall_spoofing" \
                    and row["region"] == "black_sea":
                lines.append(f"| Simulated spoofing caught, Black Sea, {row['profile']} | "
                             f"{_pct(row['mean'], 0)} ({_pct(row['ci95_low'], 0)}-"
                             f"{_pct(row['ci95_high'], 0)}) over {row['days']} days |")
    not_run = [f"* `{k}`: {status(k)}" for k in r if status(k)]
    if not_run:
        lines += ["", "Not run here (inputs missing on this machine):", *not_run]
    return "\n".join(lines) + "\n"
