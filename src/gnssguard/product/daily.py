"""The nightly job: ingest the days not yet published, build cells, publish, build site.

Runs the same code as the study (ingest -> cells -> model), one day at a time. A day
whose archive is not on adsb.lol yet is skipped and tried again on the next run (the
look-back window covers a few days of upstream delay). Published days are never
redone, so a nightly run only downloads what is new.
"""

from __future__ import annotations

import datetime as dt
import logging
import shutil
from pathlib import Path

from gnssguard.config import load_regions
from gnssguard.features.cells import GridSpec, build_all
from gnssguard.ingest import ingest_day
from gnssguard.product.publish import publish_day, read_index
from gnssguard.product.site import build_site

log = logging.getLogger(__name__)


def days_to_run(out: str | Path, today: dt.date, lookback: int) -> list[dt.date]:
    """Days in the look-back window that are not published yet, oldest first.

    A withheld day is retried only if it was withheld for an incomplete download; other
    reasons (low traffic, control alarm) would come out the same again.
    """
    known = {d["date"]: d for d in read_index(out)}
    days = []
    for k in range(lookback, 0, -1):
        day = today - dt.timedelta(days=k)
        d = known.get(day.isoformat())
        if d is None:
            days.append(day)
            continue
        if not d["published"]:
            reasons = ""
            p = Path(out) / "data" / day.isoformat() / "summary.json"
            if p.exists():
                reasons = p.read_text(encoding="utf-8")
            if "incomplete download" in reasons:
                days.append(day)
    return days


def run(out: str | Path, *, processed: str | Path, cells_dir: str | Path,
        model_dir: str | Path, config: str | Path, regions: list[str],
        dates: list[dt.date] | None = None, lookback: int = 3, cleanup: bool = False,
        today: dt.date | None = None, spec: GridSpec = GridSpec()) -> list[dict]:
    today = today or dt.datetime.now(dt.timezone.utc).date()
    todo = sorted(dates) if dates else days_to_run(out, today, lookback)
    todo = [d for d in todo if d < today]  # a day is complete only once it is over
    region_objs = load_regions(config, regions)
    done = []
    for day in todo:
        iso = day.isoformat()
        try:
            ingest_day(day, region_objs, processed)
        except LookupError as e:
            log.warning("%s: not available upstream yet (%s); will retry", iso, e)
            continue
        build_all(processed, cells_dir, spec, [iso], regions, force=True)
        summary = publish_day(iso, regions, processed, cells_dir, model_dir, out, spec=spec)
        done.append(summary)
        log.info("%s: published=%s %s", iso, summary["published"],
                 {r: v["status"] for r, v in summary["regions"].items()})
        if cleanup:
            for root in (Path(processed), Path(cells_dir)):
                for folder in root.glob(f"region=*/date={iso}"):
                    shutil.rmtree(folder, ignore_errors=True)
    build_site(out, config, regions)
    return done
