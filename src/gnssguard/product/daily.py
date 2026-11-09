"""The nightly job: ingest the days not yet published, build cells, publish, build site.

Runs the same code as the study (ingest -> cells -> model), one day at a time.

The backlog is persistent: every day from the first published day (or the last
`lookback` days, before anything is published) to yesterday that is not complete stays
in the queue, so an outage longer than a few days leaves no permanent gap. At most
`max_days` are done per run (the job has a time limit): the newest missing day first,
so a timeout loses an old day rather than yesterday, then the oldest.

Each day has a state in `data/queue.json`:

* unavailable: its archive is not on adsb.lol yet; tried again on every run;
* failed: processing raised an error; retried, at most MAX_ATTEMPTS times in all;
* withheld: published with some regions withheld (or the whole day, e.g. after an
  incomplete download); retried, at most MAX_ATTEMPTS times in all, since a withholding
  can come from a passing problem;
* complete: every region published.

A day that used up its attempts is given up and listed by the health check, so a
broken day neither blocks the queue nor disappears silently.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
from pathlib import Path

from gnssguard.config import load_regions
from gnssguard.features.cells import GridSpec, build_all
from gnssguard.ingest import ingest_day
from gnssguard.product.publish import publish_day, read_index
from gnssguard.product.site import build_site

log = logging.getLogger(__name__)


MAX_ATTEMPTS = 3
QUEUE = "queue.json"


def read_queue(out: str | Path) -> dict:
    p = Path(out) / "data" / QUEUE
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("%s is unreadable; starting a new queue", p)
        return {}


def _note(out: str | Path, iso: str, state: str, **info) -> None:
    from gnssguard.product.publish import write_atomic

    q = read_queue(out)
    e = q.get(iso, {"attempts": 0})
    if state != "unavailable":
        e["attempts"] = e.get("attempts", 0) + 1
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    e.update(state=state, updated=now, **info)
    q[iso] = e
    (Path(out) / "data").mkdir(parents=True, exist_ok=True)
    write_atomic(Path(out) / "data" / QUEUE, json.dumps(dict(sorted(q.items())), indent=1))


def _state(entry: dict | None) -> str:
    """complete / withheld / missing, from the day's index entry."""
    if entry is None:
        return "missing"
    regions = entry.get("regions", {})
    if entry.get("published") and regions and all(v == "published" for v in regions.values()):
        return "complete"
    return "withheld"


def given_up(out: str | Path) -> dict:
    """Days that used up their attempts and are no longer tried."""
    return {d: e for d, e in read_queue(out).items()
            if e.get("state") in ("failed", "withheld")
            and e.get("attempts", 0) >= MAX_ATTEMPTS}


def backlog(out: str | Path, today: dt.date, lookback: int) -> list[dt.date]:
    """Every day still to do, oldest first (see the module docstring)."""
    known = {d["date"]: d for d in read_index(out)}
    queue = read_queue(out)
    start = today - dt.timedelta(days=lookback)
    if known:
        start = min(start, dt.date.fromisoformat(min(known)))
    days, day = [], start
    while day < today:
        iso = day.isoformat()
        state = _state(known.get(iso))
        tries = queue.get(iso, {}).get("attempts", 0)
        if state != "complete" and tries < MAX_ATTEMPTS:
            days.append(day)
        day += dt.timedelta(days=1)
    return days


def days_to_run(out: str | Path, today: dt.date, lookback: int,
                max_days: int = 4) -> list[dt.date]:
    """This run's share of the backlog, in the order to run it: the newest missing day
    first, then the oldest ones."""
    todo = backlog(out, today, lookback)
    if not todo:
        return []
    newest, older = todo[-1], todo[:-1]
    return [newest] + older[:max_days - 1]


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
            _note(out, iso, "unavailable", last_error=str(e)[:300])
            continue
        except Exception as e:  # a network or disk error: this day failed, go on
            log.exception("%s: ingest failed; will retry (at most %d attempts)", iso,
                          MAX_ATTEMPTS)
            _note(out, iso, "failed", last_error=f"{type(e).__name__}: {e}"[:300])
            continue
        try:
            build_all(processed, cells_dir, spec, [iso], regions, force=True)
            summary = publish_day(iso, regions, processed, cells_dir, model_dir, out,
                                  spec=spec, config=config)
        except Exception as e:  # one broken day must not stop the others or the site
            log.exception("%s: failed; will retry (at most %d attempts)", iso, MAX_ATTEMPTS)
            _note(out, iso, "failed", last_error=f"{type(e).__name__}: {e}"[:300])
            continue
        withheld = sorted(r for r, v in summary["regions"].items()
                          if v["status"] != "published")
        state = "complete" if summary["published"] and not withheld else "withheld"
        _note(out, iso, state, regions_withheld=withheld if summary["published"] else regions)
        done.append(summary)
        log.info("%s: published=%s %s", iso, summary["published"],
                 {r: v["status"] for r, v in summary["regions"].items()})
        if cleanup:
            for root in (Path(processed), Path(cells_dir)):
                for folder in root.glob(f"region=*/date={iso}"):
                    shutil.rmtree(folder, ignore_errors=True)
    build_site(out, config, regions)
    return done
