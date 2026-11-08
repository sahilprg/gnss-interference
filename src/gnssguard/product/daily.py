"""The nightly job: ingest the days not yet published, build cells, publish, build site.

Runs the same code as the study (ingest -> cells -> model), one day at a time. A day
whose archive is not on adsb.lol yet is skipped and tried again on the next run.
Published days are never redone, so a nightly run only downloads what is new.

The backlog is persistent: every day from the first published day (or the last
`lookback` days, before anything is published) to yesterday that is not published, or
was withheld for an incomplete download, stays in the queue until it is done, so an
outage longer than a few days leaves no permanent gap. At most `max_days` are done per
run (the job has a time limit): the newest missing day first, then the oldest.
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


def _retry(out: str | Path, day: dt.date, entry: dict | None) -> bool:
    if entry is None:
        return True
    if entry["published"]:
        return False
    p = Path(out) / "data" / day.isoformat() / "summary.json"
    return p.exists() and "incomplete download" in p.read_text(encoding="utf-8")


def backlog(out: str | Path, today: dt.date, lookback: int) -> list[dt.date]:
    """Every day still to do, oldest first (see the module docstring)."""
    known = {d["date"]: d for d in read_index(out)}
    start = today - dt.timedelta(days=lookback)
    if known:
        start = min(start, dt.date.fromisoformat(min(known)))
    days, day = [], start
    while day < today:
        if _retry(out, day, known.get(day.isoformat())):
            days.append(day)
        day += dt.timedelta(days=1)
    return days


def days_to_run(out: str | Path, today: dt.date, lookback: int,
                max_days: int = 4) -> list[dt.date]:
    """This run's share of the backlog: the newest missing day, then the oldest ones."""
    todo = backlog(out, today, lookback)
    if len(todo) <= max_days:
        return todo
    return sorted(todo[:max_days - 1] + [todo[-1]])


def run(out: str | Path, *, processed: str | Path, cells_dir: str | Path,
        model_dir: str | Path, config: str | Path, regions: list[str],
        dates: list[dt.date] | None = None, lookback: int = 3, cleanup: bool = False,
        today: dt.date | None = None, spec: GridSpec = GridSpec(),
        max_days: int = 4) -> list[dict]:
    today = today or dt.datetime.now(dt.timezone.utc).date()
    todo = sorted(dates) if dates else days_to_run(out, today, lookback, max_days)
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
        summary = publish_day(iso, regions, processed, cells_dir, model_dir, out, spec=spec,
                              config=config)
        done.append(summary)
        log.info("%s: published=%s %s", iso, summary["published"],
                 {r: v["status"] for r, v in summary["regions"].items()})
        if cleanup:
            for root in (Path(processed), Path(cells_dir)):
                for folder in root.glob(f"region=*/date={iso}"):
                    shutil.rmtree(folder, ignore_errors=True)
    build_site(out, config, regions)
    return done
