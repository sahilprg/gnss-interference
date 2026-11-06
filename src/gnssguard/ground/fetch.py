"""Download daily RINEX 3 observation files of EUREF stations (`gnssguard ground-fetch`).

Routes, in order; the first that has a station-day is used, and a missing one is
reported, not guessed:

1. the EUREF Historical Data Centre's persistent identifiers, one HTTPS address per
   station and day that redirects to the file:
   https://w3id.org/EUREF-HDC/YYYY/DDD/STATION_YYYYDDD_01D_30S_R3 (RINEX 3, 30 s);
2. BKG by FTP (its HTTPS downloads have been off since a June 2026 outage,
   IGSMAIL-8709); skipped quietly if the server or the network refuses FTP;
3. the data centres' HTTPS day directories (EUREF/obs/YYYY/DDD/).
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
PID = "https://w3id.org/EUREF-HDC/{y}/{doy:03d}/{st}_{y}{doy:03d}_01D_30S_R3"


def _pid_day(stations, day, out, result, s):
    doy = day.timetuple().tm_yday
    for st in stations:
        url = PID.format(y=day.year, doy=doy, st=st)
        try:
            r = s.get(url, headers=HEADERS, timeout=300, allow_redirects=True)
        except requests.RequestException as e:
            log.warning("%s: %s", url, e)
            continue
        if r.status_code != 200 or len(r.content) < 1000 or r.content[:15].lower() \
                .startswith(b"<!doctype html"):
            continue
        name = r.url.rstrip("/").rsplit("/", 1)[-1]
        if not name.upper().startswith(st):
            name = f"{st}_{day.year}{doy:03d}_01D_30S_R3"
        target = out / str(day) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(r.content)
        result[st] = str(target)


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
        except (OSError, EOFError, ftplib.Error) as e:
            log.warning("ftp://%s%s: %s (skipped)", host, d, type(e).__name__)
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
            except (OSError, EOFError, ftplib.Error):
                pass


def fetch_day(stations: list[str], day: dt.date, out: Path, session=None,
              ftp_factory=None) -> dict[str, str]:
    """Download what exists for one day; returns station -> saved path or 'missing'."""
    import ftplib

    s = session or requests.Session()
    doy = day.timetuple().tm_yday
    result = {st: "missing" for st in stations}
    _pid_day(stations, day, out, result, s)
    todo = [st for st in stations if result[st] == "missing"]
    if todo:
        _ftp_day(todo, day, out, result, ftp_factory or ftplib.FTP)
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
