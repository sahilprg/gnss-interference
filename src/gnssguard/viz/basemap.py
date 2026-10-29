"""Offline background for the maps: coastlines and country borders.

Natural Earth 1:50m land polygons and country borders (public domain), shipped with the
package as a small gzipped file and clipped to the area around each map. The maps then
need no tile server: online tile servers such as OpenStreetMap's refuse requests from
pages opened as local files, and are volunteer-run besides.
"""

from __future__ import annotations

import gzip
import json
from functools import lru_cache
from pathlib import Path

_DATA = Path(__file__).parent / "data" / "ne_50m.json.gz"
SOURCE = "Natural Earth 1:50m (public domain)"

Box = tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max


@lru_cache(maxsize=1)
def _load() -> dict:
    with gzip.open(_DATA, "rt", encoding="utf-8") as f:
        return json.load(f)


def _clip_ring(ring: list, box: Box) -> list:
    """Sutherland-Hodgman clip of one ring ([lon, lat] points) to a lat/lon box."""
    lat0, lat1, lon0, lon1 = box
    edges = (
        (lambda p: p[0] >= lon0, lambda a, b: _cut_x(a, b, lon0)),
        (lambda p: p[0] <= lon1, lambda a, b: _cut_x(a, b, lon1)),
        (lambda p: p[1] >= lat0, lambda a, b: _cut_y(a, b, lat0)),
        (lambda p: p[1] <= lat1, lambda a, b: _cut_y(a, b, lat1)),
    )
    pts = ring
    for inside, cut in edges:
        if not pts:
            break
        out = []
        prev = pts[-1]
        for cur in pts:
            if inside(cur):
                if not inside(prev):
                    out.append(cut(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cut(prev, cur))
            prev = cur
        pts = out
    return pts if len(pts) >= 3 else []


def _cut_x(a, b, x):
    t = (x - a[0]) / (b[0] - a[0])
    return [x, round(a[1] + t * (b[1] - a[1]), 3)]


def _cut_y(a, b, y):
    t = (y - a[1]) / (b[1] - a[1])
    return [round(a[0] + t * (b[0] - a[0]), 3), y]


def _bbox(points) -> Box:
    lats = [p[1] for p in points]
    lons = [p[0] for p in points]
    return min(lats), max(lats), min(lons), max(lons)


def _overlaps(a: Box, b: Box) -> bool:
    return a[0] <= b[1] and a[1] >= b[0] and a[2] <= b[3] and a[3] >= b[2]


def background(box: Box, margin_lat: float = 8.0, margin_lon: float = 20.0) -> dict:
    """Land polygons and border lines around `box`, as [lon, lat] coordinates.

    The margin is wider east-west because map windows are wider than they are tall.
    """
    data = _load()
    clip = (max(box[0] - margin_lat, -85), min(box[1] + margin_lat, 85),
            max(box[2] - margin_lon, -180), min(box[3] + margin_lon, 180))
    land = []
    for poly in data["land"]:
        if not _overlaps(_bbox(poly[0]), clip):
            continue
        rings = [r for r in (_clip_ring(ring, clip) for ring in poly) if r]
        if rings:
            land.append(rings)
    borders = []
    lat0, lat1, lon0, lon1 = clip
    for line in data["borders"]:
        run: list = []
        for p in line:
            if lat0 <= p[1] <= lat1 and lon0 <= p[0] <= lon1:
                run.append(p)
            elif run:
                if len(run) > 1:
                    borders.append(run)
                run = []
        if len(run) > 1:
            borders.append(run)
    return {"land": land, "borders": borders, "source": SOURCE,
            "extent": [[clip[0], clip[2]], [clip[1], clip[3]]]}
