"""Retrain alarm: is the published model still fit for the data it sees?

Run after each nightly publish. Reads the published day summaries and returns
"ok" or "retrain_recommended" with the reasons. The nightly workflow opens a GitHub
issue (which emails the owner) the first time the status turns to retrain_recommended.

Signals (thresholds are module constants, fixed before the first alarm):

* **Age:** more than MAX_AGE_DAYS between the last training day and the newest published
  day. Traffic, receivers and interference patterns change; retrain regularly anyway.
* **Calm control:** over the last WINDOW published days, Central Europe flags more than
  CONTROL_LIMIT of its judged cells on average (0.4-1.3% on every study day). A sustained
  rise there is model or data drift, not new jamming.
* **Model vs rules:** in any region the model and the rules disagree on more than
  DISAGREE_LIMIT of judged cells on average over the window (about 0.1-1% so far). The
  model then reads the evidence differently from the data it was trained on.
* **Coverage:** in any region the number of judged cells over the window has a median
  below COVERAGE_LOW or above COVERAGE_HIGH times its median over all earlier published
  days (receivers joined or left).

Needs at least WINDOW published days; until then the status is "ok" with a note.
"""

from __future__ import annotations

import datetime as dt
import json
import statistics
from pathlib import Path

from gnssguard.product.publish import read_index, read_summary

WINDOW = 7
MAX_AGE_DAYS = 45
CONTROL_REGION = "central_europe"
CONTROL_LIMIT = 0.015
DISAGREE_LIMIT = 0.03
COVERAGE_LOW, COVERAGE_HIGH = 0.5, 2.0


def assess(out: str | Path) -> dict:
    days = [d["date"] for d in read_index(out) if d["published"]]
    summaries = [s for s in (read_summary(out, d) for d in sorted(days)) if s]
    reasons, notes, signals = [], [], {}
    if not summaries:
        return {"status": "ok", "reasons": [], "notes": ["nothing published yet"],
                "signals": {}}
    latest = summaries[-1]
    train_end = (latest.get("model", {}).get("train_dates") or [None, None])[-1]
    if train_end:
        age = (dt.date.fromisoformat(latest["date"]) - dt.date.fromisoformat(train_end)).days
        signals["model_age_days"] = age
        if age > MAX_AGE_DAYS:
            reasons.append(f"the model's newest training day ({train_end}) is {age} days older "
                           f"than the newest published day (limit {MAX_AGE_DAYS})")
    recent, earlier = summaries[-WINDOW:], summaries[:-WINDOW]
    if len(recent) < WINDOW:
        notes.append(f"only {len(recent)} published days; drift checks need {WINDOW}")
    else:
        def series(rows, region, key):
            return [s["regions"][region][key] for s in rows
                    if s["regions"].get(region, {}).get("status") == "published"
                    and s["regions"][region].get(key) is not None]

        ctrl = series(recent, CONTROL_REGION, "flagged_share")
        if ctrl:
            m = statistics.mean(ctrl)
            signals["control_flagged_share_7d"] = round(m, 4)
            if m > CONTROL_LIMIT:
                reasons.append(f"calm {CONTROL_REGION} flagged {m:.2%} of judged cells on "
                               f"average over {len(ctrl)} days (limit {CONTROL_LIMIT:.1%})")
        regions = sorted({r for s in recent for r in s["regions"]})
        for region in regions:
            dis = series(recent, region, "rule_disagreement")
            if dis:
                m = statistics.mean(dis)
                signals[f"{region}_rule_disagreement_7d"] = round(m, 4)
                if m > DISAGREE_LIMIT:
                    reasons.append(f"{region}: model and rules disagree on {m:.1%} of judged "
                                   f"cells over {len(dis)} days (limit {DISAGREE_LIMIT:.0%})")
            now, before = series(recent, region, "judged_cells"), series(earlier, region,
                                                                          "judged_cells")
            if now and len(before) >= WINDOW:
                ratio = statistics.median(now) / max(statistics.median(before), 1)
                signals[f"{region}_coverage_ratio"] = round(ratio, 2)
                if not COVERAGE_LOW <= ratio <= COVERAGE_HIGH:
                    reasons.append(f"{region}: judged cells changed {ratio:.2f}x against "
                                   "earlier published days (receiver coverage changed)")
    return {"status": "retrain_recommended" if reasons else "ok", "checked_through":
            latest["date"], "reasons": reasons, "notes": notes, "signals": signals}


def write_health(out: str | Path) -> dict:
    res = assess(out)
    data = Path(out) / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "health.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    lines = [f"Status after {res.get('checked_through', 'no day')}: **{res['status']}**", ""]
    lines += [f"- {r}" for r in res["reasons"]] or ["- no retraining signal"]
    lines += ["", "How to retrain: docs/product.md, section \"Retraining\"."]
    (data / "health.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return res
