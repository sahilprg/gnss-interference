"""Decode readsb trace files into flat position rows.

Trace format (readsb README-json.md, "trace jsons"): the file holds aircraft metadata and
`trace`, a list of points. Each point is an array:

    0 seconds after file `timestamp`   7 vertical rate (fpm) or None
    1 latitude                          8 details dict or None (holds nic, nac_p, sil, ...)
    2 longitude                         9 position source type ("adsb_icao", "mlat", ...)
    3 altitude ft, "ground" or None    10 geometric altitude ft or None
    4 ground speed kt or None          11 geometric vertical rate or None
    5 track deg or None                12 indicated airspeed or None
    6 flags bitfield                   13 roll or None

Flags: 1 = stale position, 2 = start of new leg, 4 = vrate geometric, 8 = altitude geometric.

The details dict appears only on some points, so quality fields (NIC, NACp, ...) are carried
forward from the last report and `q_age_s` records how old they are.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pyarrow as pa

from gnssguard.config import Region

QUALITY_KEYS = ("nic", "nac_p", "sil", "nic_baro", "rc", "gva", "sda", "version")

SCHEMA = pa.schema(
    [
        ("icao", pa.string()),
        ("ts", pa.timestamp("ms", tz="UTC")),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        ("alt_baro", pa.int32()),
        ("alt_geom", pa.int32()),
        ("on_ground", pa.bool_()),
        ("gs", pa.float32()),
        ("track", pa.float32()),
        ("vrate", pa.int32()),
        ("src", pa.string()),
        ("stale", pa.bool_()),
        ("new_leg", pa.bool_()),
        ("nic", pa.int8()),
        ("nac_p", pa.int8()),
        ("sil", pa.int8()),
        ("nic_baro", pa.int8()),
        ("rc", pa.int32()),
        ("gva", pa.int8()),
        ("sda", pa.int8()),
        ("adsb_version", pa.int8()),
        ("q_age_s", pa.float32()),
        ("aircraft_type", pa.string()),
        ("db_flags", pa.int16()),
    ]
)
COLUMNS = SCHEMA.names

# Box enclosing all regions, for a cheap first rejection test.
Bounds = tuple[float, float, float, float]


def bounds_of(regions: Sequence[Region]) -> Bounds:
    return (
        min(r.lat_min for r in regions),
        max(r.lat_max for r in regions),
        min(r.lon_min for r in regions),
        max(r.lon_max for r in regions),
    )


def _int(v):
    if v is None or isinstance(v, str):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float(v):
    if v is None or isinstance(v, str):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def iter_rows(
    obj: dict, regions: Sequence[Region], bounds: Bounds | None = None,
    stats: dict | None = None,
) -> Iterator[tuple[str, tuple]]:
    """Yield (region_name, row) for every trace point inside a region.

    A point inside two overlapping regions is yielded once per region.
    Positions whose timestamp does not move forward (exact duplicates or out of order)
    are dropped; their quality fields are still carried forward. On 2026-10-17 one
    parked aircraft produced 2.4 million points in 6 hours, 2.1 million of them exact
    duplicates. Normal traces do have points under a second apart; those are kept.
    `stats["points_not_forward"]` counts the drops when a dict is passed.
    `row` follows COLUMNS order; `ts` is epoch milliseconds.
    """
    trace = obj.get("trace")
    if not trace:
        return
    base = float(obj["timestamp"])
    icao = str(obj.get("icao", "")).lower()
    ac_type = obj.get("t")
    db_flags = _int(obj.get("dbFlags"))
    blat0, blat1, blon0, blon1 = bounds or bounds_of(regions)
    boxes = [(r.name, r.lat_min, r.lat_max, r.lon_min, r.lon_max) for r in regions]

    q = dict.fromkeys(QUALITY_KEYS)
    q_t = None
    last_t = None

    for p in trace:
        n = len(p)
        if n < 3:
            continue
        t = base + float(p[0])
        det = p[8] if n > 8 else None

        # Update carried-forward quality fields before the region test,
        # so a point entering a region inherits the latest values.
        if det and ("nic" in det or "nac_p" in det):
            for k in QUALITY_KEYS:
                if k in det:
                    q[k] = det[k]
            q_t = t

        lat, lon = p[1], p[2]
        if lat is None or lon is None:
            continue
        if last_t is not None and t <= last_t:
            if stats is not None:
                stats["points_not_forward"] = stats.get("points_not_forward", 0) + 1
            continue
        last_t = t
        if not (blat0 <= lat <= blat1 and blon0 <= lon <= blon1):
            continue
        hits = [b[0] for b in boxes if b[1] <= lat <= b[2] and b[3] <= lon <= b[4]]
        if not hits:
            continue

        flags = (_int(p[6]) or 0) if n > 6 else 0
        alt = p[3] if n > 3 else None
        on_ground = alt == "ground"
        alt_baro = None
        alt_geom = _int(p[10]) if n > 10 else None
        if not on_ground and alt is not None:
            if flags & 8:
                alt_geom = alt_geom if alt_geom is not None else _int(alt)
            else:
                alt_baro = _int(alt)
        if alt_geom is None and det:
            alt_geom = _int(det.get("alt_geom"))
        src = p[9] if n > 9 and p[9] is not None else (det.get("type") if det else None)

        row = (
            icao,
            int(round(t * 1000)),
            float(lat),
            float(lon),
            alt_baro,
            alt_geom,
            on_ground,
            _float(p[4]) if n > 4 else None,
            _float(p[5]) if n > 5 else None,
            _int(p[7]) if n > 7 else None,
            src,
            bool(flags & 1),
            bool(flags & 2),
            _int(q["nic"]),
            _int(q["nac_p"]),
            _int(q["sil"]),
            _int(q["nic_baro"]),
            _int(q["rc"]),
            _int(q["gva"]),
            _int(q["sda"]),
            _int(q["version"]),
            None if q_t is None else t - q_t,
            ac_type,
            db_flags,
        )
        for name in hits:
            yield name, row
