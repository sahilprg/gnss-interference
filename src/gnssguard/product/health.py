"""Health: is the pipeline running, and is the published model still fit for the data?

Run after each nightly publish. Two separate answers in `data/health.json`:

* `freshness` ("fresh" or "stale"): days between today and the newest published day.
  A day's archive is normally on adsb.lol within a day, so more than STALE_DAYS means
  the pipeline or its input has stopped. Measured against today, not against the last
  published day, so a stopped pipeline cannot report itself as fine. The map also
  computes this in the reader's browser (a stale banner), which works even when the
  nightly job does not run at all.
* `status` ("ok" or "retrain_recommended"), the retrain alarm below.

The nightly workflow opens a GitHub issue (which emails the owner) the first time either
turns bad.

Signals of the retrain alarm (thresholds are module constants, fixed before the first
alarm):

* **Age:** more than MAX_AGE_DAYS between the last training day and today. Traffic,
  receivers and interference patterns change; retrain regularly anyway.
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

from gnssguard.product.publish import read_index, read_summary, write_atomic

WINDOW = 7
MAX_AGE_DAYS = 45
CONTROL_REGION = "central_europe"
CONTROL_LIMIT = 0.015
DISAGREE_LIMIT = 0.03
COVERAGE_LOW, COVERAGE_HIGH = 0.5, 2.0
STALE_DAYS = 3


def freshness(newest: str | None, today: dt.date, backlog: int = 0) -> dict:
    if newest is None:
        return {"status": "stale", "newest_published_day": None, "days_behind": None,
                "backlog_days": backlog, "limit_days": STALE_DAYS}
    behind = (today - dt.date.fromisoformat(newest)).days
    return {"status": "stale" if behind > STALE_DAYS else "fresh",
            "newest_published_day": newest, "days_behind": behind, "backlog_days": backlog,
            "limit_days": STALE_DAYS}


def assess(out: str | Path, today: dt.date | None = None, backlog: int = 0) -> dict:
    today = today or dt.datetime.now(dt.timezone.utc).date()
    days = [d["date"] for d in read_index(out) if d["published"]]
    summaries = [s for s in (read_summary(out, d) for d in sorted(days)) if s]
    reasons, notes, signals = [], [], {}
    stamp = {"checked_on": today.isoformat(),
             "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    if not summaries:
        return {"status": "ok", "freshness": freshness(None, today, backlog), **stamp,
                "reasons": [], "notes": ["nothing published yet"], "signals": {}}
    latest = summaries[-1]
    train_end = (latest.get("model", {}).get("train_dates") or [None, None])[-1]
    if train_end:
        age = (today - dt.date.fromisoformat(train_end)).days
        signals["model_age_days"] = age
        if age > MAX_AGE_DAYS:
            reasons.append(f"the model's newest training day ({train_end}) is {age} days "
                           f"before today (limit {MAX_AGE_DAYS})")
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
    return {"status": "retrain_recommended" if reasons else "ok",
            "freshness": freshness(latest["date"], today, backlog), **stamp,
            "checked_through": latest["date"], "reasons": reasons, "notes": notes,
            "signals": signals}


def write_health(out: str | Path, today: dt.date | None = None, backlog: int = 0) -> dict:
    res = assess(out, today, backlog)
    data = Path(out) / "data"
    data.mkdir(parents=True, exist_ok=True)
    write_atomic(data / "health.json", json.dumps(res, indent=1))
    f = res["freshness"]
    lines = [f"Data: **{f['status']}** (newest published day {f['newest_published_day']}, "
             f"{f['days_behind']} days before {res['checked_on']}; "
             f"{f['backlog_days']} days waiting)", "",
             f"Model: **{res['status']}** after {res.get('checked_through', 'no day')}", ""]
    lines += [f"- {r}" for r in res["reasons"]] or ["- no retraining signal"]
    lines += ["", "How to retrain: docs/product.md, section \"Retraining\"."]
    write_atomic(data / "health.md", "\n".join(lines) + "\n")
    return res
