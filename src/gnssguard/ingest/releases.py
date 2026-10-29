"""Find the download URLs for one day of the adsb.lol archive.

Each year lives in its own repository (adsblol/globe_history_<year>). The repository's
PREFERRED_RELEASES.txt lists, per day, the comma-separated part URLs of the best
release (prod or staging). Early days of a year can appear in the next year's list.
"""

from __future__ import annotations

import datetime as dt
import logging
import re

import requests

log = logging.getLogger(__name__)

RAW_LIST = "https://raw.githubusercontent.com/adsblol/globe_history_{year}/main/PREFERRED_RELEASES.txt"
RELEASE_URL = (
    "https://github.com/adsblol/globe_history_{year}/releases/download/"
    "v{tag}-planes-readsb-{flavor}/v{tag}-planes-readsb-{flavor}.tar{suffix}"
)
FLAVORS = ("prod-0", "staging-0", "prod-1")
_TAG = re.compile(r"/v(\d{4})\.(\d{2})\.(\d{2})-planes-readsb-([a-z0-9-]+)/")


def parse_preferred(text: str) -> dict[dt.date, list[str]]:
    """Map date -> part URLs from a PREFERRED_RELEASES.txt body."""
    out: dict[dt.date, list[str]] = {}
    for line in text.splitlines():
        urls = [u.strip() for u in line.split(",") if u.strip()]
        if not urls:
            continue
        m = _TAG.search(urls[0])
        if not m:
            continue
        day = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if "mlatonly" in m.group(4):
            continue  # MLAT-only dumps lack ADS-B quality fields
        out.setdefault(day, sorted(urls))
    return out


def _probe(session: requests.Session, day: dt.date) -> list[str]:
    """Fallback: guess release names and find parts .aa, .ab, ... by HEAD requests."""
    tag = day.strftime("%Y.%m.%d")
    for year in (day.year, day.year + 1):
        for flavor in FLAVORS:
            parts = []
            for suffix in ("", *(f".a{c}" for c in "abcdefghij")):
                url = RELEASE_URL.format(year=year, tag=tag, flavor=flavor, suffix=suffix)
                r = session.head(url, allow_redirects=True, timeout=30)
                if r.status_code == 200:
                    parts.append(url)
                    if suffix == "":
                        return parts  # unsplit archive
                elif suffix not in ("", ".aa"):
                    break
            if parts:
                return parts
    return []


def resolve_day(day: dt.date, session: requests.Session | None = None) -> list[str]:
    """Return ordered part URLs for `day`, or raise LookupError."""
    session = session or requests.Session()
    for year in (day.year, day.year + 1):
        try:
            r = session.get(RAW_LIST.format(year=year), timeout=30)
            if r.status_code == 200:
                found = parse_preferred(r.text).get(day)
                if found:
                    return found
        except requests.RequestException as e:
            log.warning("could not read release list for %d: %s", year, e)
    parts = _probe(session, day)
    if not parts:
        raise LookupError(f"no adsb.lol release found for {day}")
    return parts
