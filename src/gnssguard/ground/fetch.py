"""Download daily RINEX 3 observation files of EUREF stations (`gnssguard ground-fetch`).

The EUREF data centres publish one directory per day (EUREF/obs/YYYY/DDD/) with files
named STATION_R_YYYYDDD0000_01D_30S_MO.crx.gz (Hatanaka-compressed RINEX 3, 30 s). BKG,
the regional data centre that holds all EPN stations, serves downloads only by FTP since
its June 2026 outage (IGSMAIL-8709), so FTP is tried first and the HTTPS addresses after.
The first data centre that has a file is used; a missing station-day is reported, not
guessed.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path

import requests

log = logging.getLogger(__name__)

FTP_MIRRORS = (("igs-ftp.bkg.bund.de", "/EUREF/obs/{y}/{doy:03d}/"),)
MIRRORS = ("https://igs.bkg.bund.de/root_ftp/EUREF/obs/{y}/{doy:03d}/",
           "https://epncb.oma.be/pub/obs/{y}/{doy:03d}/",
           "https://datos-geodesia.ign.es/euref/obs/{y}/{doy:03d}/")
HEADERS = {"User-Agent": "gnssguard ground-fetch (research)"}


def file_name(station: str, day: dt.date) -> re.Pattern:
    doy = day.timetuple().tm_yday
    return re.compile(rf"{station}_[RS]_{day.year}{doy:03d}0000_01D_30S_MO\.(crx|rnx)\.gz",
                      re.IGNORECASE)


def _ftp_day(stations, day, out, result, ftp_factory):
    import ftplib

    doy = day.timetuple().tm_yday
    for host, path in FTP_MIRRORS:
        todo = [st for st in stations if result[st] == "missing"]
        if not todo:
            return
        d = path.format(y=day.year, doy=doy)
        try:
            ftp = ftp_factory(host, timeout=60)
            ftp.login()
            names = ftp.nlst(d)
        except (OSError, ftplib.Error) as e:
            log.warning("ftp://%s%s: %s", host, d, e)
            continue
        listing = " ".join(n.rsplit("/", 1)[-1] for n in names)
        try:
            for st in todo:
                m = file_name(st, day).search(listing)
                if not m:
                    continue
                target = out / str(day) / m.group(0)
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    part = target.with_suffix(target.suffix + ".part")
                    with open(part, "wb") as fh:
                        ftp.retrbinary(f"RETR {d}{m.group(0)}", fh.write)
                    part.replace(target)
                result[st] = str(target)
        finally:
            try:
                ftp.quit()
            except (OSError, ftplib.Error):
                pass


def fetch_day(stations: list[str], day: dt.date, out: Path, session=None,
              ftp_factory=None) -> dict[str, str]:
    """Download what exists for one day; returns station -> saved path or 'missing'."""
    import ftplib

    s = session or requests.Session()
    doy = day.timetuple().tm_yday
    result = {st: "missing" for st in stations}
    _ftp_day(stations, day, out, result, ftp_factory or ftplib.FTP)
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
