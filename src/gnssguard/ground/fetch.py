"""Download daily RINEX 3 observation files of EUREF stations (`gnssguard ground-fetch`).

The EUREF data centres publish one directory per day, e.g.
https://igs.bkg.bund.de/root_ftp/EUREF/obs/2026/305/, with files named
STATION_R_YYYYDDD0000_01D_30S_MO.crx.gz (Hatanaka-compressed RINEX 3, 30 s). The first
data centre that has a file is used; a missing station-day is reported, not guessed.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path

import requests

log = logging.getLogger(__name__)

MIRRORS = ("https://igs.bkg.bund.de/root_ftp/EUREF/obs/{y}/{doy:03d}/",
           "https://datos-geodesia.ign.es/euref/obs/{y}/{doy:03d}/")
HEADERS = {"User-Agent": "gnssguard ground-fetch (research)"}


def file_name(station: str, day: dt.date) -> re.Pattern:
    doy = day.timetuple().tm_yday
    return re.compile(rf"{station}_[RS]_{day.year}{doy:03d}0000_01D_30S_MO\.(crx|rnx)\.gz",
                      re.IGNORECASE)


def fetch_day(stations: list[str], day: dt.date, out: Path, session=None) -> dict[str, str]:
    """Download what exists for one day; returns station -> saved path or 'missing'."""
    s = session or requests.Session()
    doy = day.timetuple().tm_yday
    result = {st: "missing" for st in stations}
    for mirror in MIRRORS:
        todo = [st for st in stations if result[st] == "missing"]
        if not todo:
            break
        url = mirror.format(y=day.year, doy=doy)
        try:
            r = s.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
        except requests.RequestException as e:
            log.warning("%s: %s", url, e)
            continue
        for st in todo:
            m = file_name(st, day).search(r.text)
            if not m:
                continue
            target = out / str(day) / m.group(0)
            if not target.exists():
                f = s.get(url + m.group(0), headers=HEADERS, timeout=300)
                f.raise_for_status()
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(f.content)
            result[st] = str(target)
    return result
