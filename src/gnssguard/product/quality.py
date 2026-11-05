"""Data-quality gates: a day, or a region of it, is published only if it looks sane.

A detector is only as good as its input. A truncated download, a receiver outage or a
software change upstream can make a region look jammed (or clean) when nothing changed
in the air. These checks compare each region-day with the days already published and
withhold what looks broken, with the reason, instead of publishing it.

Gates (all thresholds are module constants):

* **complete**: the ingest read the whole archive day (manifest `complete_day`).
* **traffic**: the region saw at least TRAFFIC_MIN_RATIO of its median number of
  aircraft over the last HISTORY_DAYS published days.
* **coverage**: at least COVERAGE_MIN_RATIO of its median number of judged cells
  (cells with 3+ usable aircraft).
* **control**: the calm control region flags at most CONTROL_MAX_SHARE of its judged
  cells. Above that, either something systematic is wrong (the model or the data) or a
  real event happened where the model expects none. The day is still published, marked
  "needs review" with the reason, so a real event is not hidden (changed 4 Nov 2026;
  before, the whole day was withheld).

* **hourly gaps**: an hour of the day with under GAP_RATIO of its usual number of
  cell-windows with aircraft (median of the same hour over earlier published days, if
  that median is at least GAP_MIN_CELLS) marks the region "needs review": a receiver or
  archive outage for part of the day, not a reason to hide the rest of it.

The first three test the input and withhold; the control alarm and hourly gaps only
mark the output.

Traffic and coverage gates need at least MIN_HISTORY earlier days; before that they
pass with a note.
"""

from __future__ import annotations

import statistics

HISTORY_DAYS = 14
MIN_HISTORY = 3
TRAFFIC_MIN_RATIO = 0.5
COVERAGE_MIN_RATIO = 0.3
CONTROL_MAX_SHARE = 0.03  # calm Central Europe: 0.4-1.3% on the 28 study days
CONTROL_REGION = "central_europe"


def _median(history: list[dict], region: str, key: str) -> float | None:
    vals = [d["regions"][region][key] for d in history[-HISTORY_DAYS:]
            if d.get("regions", {}).get(region, {}).get("status") == "published"
            and d["regions"][region].get(key) is not None]
    return statistics.median(vals) if len(vals) >= MIN_HISTORY else None


GAP_RATIO = 0.3
GAP_MIN_CELLS = 20


def hourly_gaps(stats: dict, history: list[dict], region: str) -> list[int]:
    today = stats.get("hourly_cells")
    past = [h["regions"][region]["hourly_cells"] for h in history[-14:]
            if h.get("regions", {}).get(region, {}).get("status") == "published"
            and h["regions"][region].get("hourly_cells")]
    if not today or len(past) < MIN_HISTORY:
        return []
    med = [sorted(p[i] for p in past)[len(past) // 2] for i in range(24)]
    return [i for i in range(24) if med[i] >= GAP_MIN_CELLS and today[i] < GAP_RATIO * med[i]]


def check_day(stats: dict[str, dict], history: list[dict], complete_day: bool,
              control: str = CONTROL_REGION) -> tuple[dict[str, dict], list[str]]:
    """Return per-region {"status", "reasons", "notes", "review"} and the day-level
    reasons that withhold everything. "review" lists output warnings (published anyway).

    stats: per region {"aircraft", "judged_cells", "flagged_share"}.
    history: earlier published day summaries, oldest first.
    """
    day_reasons = []
    if not complete_day:
        day_reasons.append("incomplete download: the archive day was not read to the end")
    review = []
    c = stats.get(control)
    if c and c.get("judged_cells") and c["flagged_share"] > CONTROL_MAX_SHARE:
        review.append(
            f"control alarm: calm {control} flagged {c['flagged_share']:.1%} of judged cells "
            f"(limit {CONTROL_MAX_SHARE:.0%}); check before relying on this day")
    out = {}
    for region, s in stats.items():
        reasons, notes = [], []
        for key, ratio, what in (("aircraft", TRAFFIC_MIN_RATIO, "aircraft"),
                                 ("judged_cells", COVERAGE_MIN_RATIO, "judged cells")):
            med = _median(history, region, key)
            if med is None:
                notes.append(f"{what}: fewer than {MIN_HISTORY} earlier days to compare with")
            elif s.get(key, 0) < ratio * med:
                reasons.append(f"low {what}: {s.get(key, 0):,} vs a recent median of "
                               f"{med:,.0f}")
        if s.get("aircraft", 0) == 0:
            reasons.append("no aircraft")
        status = "withheld" if (reasons or day_reasons) else "published"
        gaps = hourly_gaps(s, history, region)
        own = [f"data gap: hours {', '.join(f'{h:02d}' for h in gaps)} UTC had under "
               f"{GAP_RATIO:.0%} of their usual cell-windows"] if gaps else []
        out[region] = {"status": status, "reasons": reasons + day_reasons, "notes": notes,
                       "review": review + own if status == "published" else []}
    return out, day_reasons
