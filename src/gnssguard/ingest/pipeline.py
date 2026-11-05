"""Ingest one day: stream the archive, keep points inside study regions, write Parquet.

Raw data is never written to disk. Output goes to a temporary folder and is moved into
place only when the whole day succeeds, so a crash never leaves half a day behind.
The manifest file `_manifests/<date>.json` marks a day as complete.
"""

from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import logging
import shutil
import tarfile
import time
from collections.abc import Sequence
from pathlib import Path

from gnssguard import __version__
from gnssguard.config import Region
from gnssguard.ingest.releases import resolve_day
from gnssguard.ingest.stream import MultiPartStream
from gnssguard.ingest.trace import bounds_of, iter_rows
from gnssguard.ingest.writer import RegionWriter

try:
    import orjson

    _loads = orjson.loads
except ImportError:  # pragma: no cover - optional speedup
    _loads = json.loads

log = logging.getLogger(__name__)


def manifest_path(out_root: Path, day: dt.date) -> Path:
    return Path(out_root) / "_manifests" / f"{day.isoformat()}.json"


POINTS_SCHEMA = "points/2"  # stored point columns (ingest/trace.py SCHEMA)
MAX_DECODE_ERROR_SHARE = 0.001  # more unreadable trace files than this: day incomplete


def field_presence(folder: Path, max_age_s: float = 30.0) -> dict:
    """Share of airborne points carrying each points/2 field (air data only when fresher
    than max_age_s): how far the new spoofing checks can reach in each region."""
    import pyarrow.parquet as pq

    from gnssguard.ingest.trace import NEW_IN_POINTS_2

    names = set(pq.read_schema(next(folder.glob("*.parquet"))).names) if any(
        folder.glob("*.parquet")) else set()
    cols = [c for c in NEW_IN_POINTS_2 if c in names]
    if not cols:
        return {}
    t = pq.read_table(folder, columns=[*cols, "on_ground"]).to_pandas()
    t = t[~t["on_ground"].astype(bool)]
    if t.empty:
        return {}
    fresh = t["air_age_s"].le(max_age_s) if "air_age_s" in t else True
    air = {"tas", "mach", "mag_heading", "true_heading", "wind_dir", "wind_speed", "oat"}
    return {c: round(float((t[c].notna() & (fresh if c in air else True)).mean()), 3)
            for c in cols if c != "air_age_s"}


def is_done(out_root: Path, day: dt.date, regions: Sequence[Region] | None = None) -> bool:
    """True only if a full (not sampled) run of this day finished with the current stored
    point format, and, when regions are given, it covered each of them with the same box
    (a changed box or format needs a new run). Code changes that keep the format do not."""
    path = manifest_path(out_root, day)
    if not path.exists():
        return False
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not m.get("complete_day"):
        return False
    if m.get("points_schema", "points/1") != POINTS_SCHEMA:
        return False
    stored = m.get("regions", {})
    return all(stored.get(r.name) == [r.lat_min, r.lat_max, r.lon_min, r.lon_max]
               for r in regions or [])


def _decode(raw: bytes) -> dict:
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return _loads(raw)


def ingest_day(
    day: dt.date,
    regions: Sequence[Region],
    out_root: str | Path,
    *,
    sources: Sequence[str] | None = None,
    max_traces: int | None = None,
    force: bool = False,
    log_every: int = 25_000,
    max_decode_error_share: float = MAX_DECODE_ERROR_SHARE,
) -> dict:
    """Process one day. Returns the manifest dict.

    sources: part URLs or local paths in order; resolved from adsb.lol when omitted.
    max_traces: stop after this many trace files (sampling / smoke tests).
    """
    out_root = Path(out_root)
    if is_done(out_root, day, regions) and not force:
        log.info("%s already ingested; use --force to redo", day)
        return json.loads(manifest_path(out_root, day).read_text(encoding="utf-8"))

    sources = list(sources) if sources else resolve_day(day)
    tmp_root = out_root / f".tmp-{day.isoformat()}"
    shutil.rmtree(tmp_root, ignore_errors=True)
    writer = RegionWriter(tmp_root, day.isoformat())
    bounds = bounds_of(regions)

    stats = {"trace_files": 0, "traces_with_hits": 0, "decode_errors": 0, "points_kept": 0,
             "points_not_forward": 0}
    t0 = time.time()
    stream = MultiPartStream(sources)
    complete = True
    try:
        buffered = io.BufferedReader(stream, buffer_size=4 << 20)
        with tarfile.open(fileobj=buffered, mode="r|") as tf:
            for member in tf:
                if not member.isfile() or "/traces/" not in member.name:
                    continue
                if "trace_full_" not in member.name:
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                stats["trace_files"] += 1
                try:
                    obj = _decode(fh.read())
                except Exception:  # noqa: BLE001 - count and skip corrupt files
                    stats["decode_errors"] += 1
                    continue
                hit = False
                for region, row in iter_rows(obj, regions, bounds, stats):
                    writer.add(region, row)
                    stats["points_kept"] += 1
                    hit = True
                stats["traces_with_hits"] += hit
                if stats["trace_files"] % log_every == 0:
                    mb = stream.total_read / 1e6
                    rate = mb / max(time.time() - t0, 1e-9)
                    log.info("%s: %d traces, %d points kept, %.0f MB read (%.1f MB/s)",
                             day, stats["trace_files"], stats["points_kept"], mb, rate)
                if max_traces and stats["trace_files"] >= max_traces:
                    complete = False
                    break
    finally:
        rows = writer.close()
        stream.close()
    if stats["decode_errors"] > max_decode_error_share * max(stats["trace_files"], 1):
        log.warning("%s: %d of %d trace files unreadable; day marked incomplete", day,
                    stats["decode_errors"], stats["trace_files"])
        complete = False

    manifest = {
        "date": day.isoformat(),
        "sources": sources,
        "regions": {r.name: [r.lat_min, r.lat_max, r.lon_min, r.lon_max] for r in regions},
        "rows_per_region": rows,
        "bytes_read": stream.total_read,
        "seconds": round(time.time() - t0, 1),
        "complete_day": complete,
        "code_version": __version__,
        "points_schema": POINTS_SCHEMA,
        "ingested_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        **stats,
    }

    # Move finished files into place, replacing any previous version of this day.
    for region in rows:
        final_dir = out_root / f"region={region}" / f"date={day.isoformat()}"
        shutil.rmtree(final_dir, ignore_errors=True)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(writer.path_for(region).parent), str(final_dir))
    shutil.rmtree(tmp_root, ignore_errors=True)
    manifest["field_presence"] = {
        region: field_presence(out_root / f"region={region}" / f"date={day.isoformat()}")
        for region in rows}

    mpath = manifest_path(out_root, day)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("%s done: %s rows in %.0fs", day, sum(rows.values()), manifest["seconds"])
    return manifest
