# Data

## Source

One day of the [adsb.lol archive](https://www.adsb.lol/docs/open-data/historical/) is a tar
archive of gzip JSON files written by [readsb](https://github.com/wiedehopf/readsb/blob/dev/README-json.md#trace-jsons),
split into 2 GB parts (`.tar.aa`, `.tar.ab`, ...). Each year is a separate repository
(`adsblol/globe_history_<year>`); `PREFERRED_RELEASES.txt` in it lists the best release per day.

Inside the archive, `traces/<xx>/trace_full_<icao>.json` holds one aircraft's whole day.
`heatmap/` and `acas/` are skipped.

## Stream, never store

`gnssguard ingest` reads the parts as one continuous HTTP stream and parses the tar on the
fly. Raw data never touches disk: only points inside the study regions are kept and written
to Parquet. A full day reads ~4-6 GB from the network and writes a few tens of MB.

Output layout (Hive partitioning, readable with pandas, pyarrow, DuckDB or Polars):

```
data/processed/
  region=baltic/date=2026-10-26/part-0.parquet
  ...
  _manifests/2026-10-26.json    # sources, row counts, bytes read, duration
```

A day counts as done only when its manifest says `"complete_day": true`. Re-running skips
done days unless `--force` is given. Sampled runs (`--max-traces`) are never marked done.

## Columns

| Column | Type | Meaning |
| --- | --- | --- |
| icao | string | 24-bit ICAO address, lowercase hex |
| ts | timestamp (ms, UTC) | Time of the position |
| lat, lon | float64 | Position in degrees |
| alt_baro | int32 | Barometric altitude, ft; null on ground or when only geometric is known |
| alt_geom | int32 | GPS (geometric) altitude, ft |
| on_ground | bool | Aircraft reported on ground |
| gs, track | float32 | Ground speed (kt), track (deg) |
| vrate | int32 | Vertical rate (ft/min) |
| src | string | Position source: `adsb_icao`, `mlat`, `tisb_*`, `adsc`, ... |
| stale | bool | No position received in the 20 s before this point |
| new_leg | bool | readsb's estimate of a new flight leg |
| nic, nac_p, sil, nic_baro, rc, gva, sda, adsb_version | int | GPS integrity and accuracy fields (DO-260B) |
| q_age_s | float32 | Seconds since the quality fields above were last reported |
| aircraft_type | string | ICAO type code, e.g. A320 |
| db_flags | int16 | readsb database flags (bit 1 = military) |

### Why `q_age_s` exists

The quality fields (NIC, NACp, ...) come in a details object that readsb attaches to only
some trace points, about one in four in our checks. They are carried forward from the last
report, and `q_age_s` says how old they are. Feature code should ignore values older than
a chosen limit (30 s is the default in the coverage report).

**Are the details fresh measurements or a cached snapshot?** Checked on 4 Nov 2026 on
the first 20,000 raw traces of 2 Nov (32.2 million points, 8.1 million details objects):

* A details object is a **snapshot of the whole aircraft state**, not a single message:
  `nic` and `rc` are in 100% of them, `nac_p` and `sil` in 98%, along with callsign,
  squawk, track and, for about 37%, airspeed, heading, roll and wind. So a NACp-only
  update that refreshes an old NIC (finding 1 of the improvement plan) cannot happen:
  every snapshot carries both, as readsb holds them at that moment.
* readsb drops fields one by one when they expire: in 0.7% of ADS-B snapshots `nac_p`
  is missing while `nic` is present. A value in a snapshot is therefore one readsb still
  considers current.
* For MLAT, TIS-B trackfile and most ADS-R snapshots readsb reports `nic` = 0 (MLAT:
  20,349 of 20,349) and usually no `nac_p`: it is not the last ADS-B value carried over.
  The features count an aircraft's MLAT points as degraded only for aircraft also seen
  on ADS-B that day (`mlat_fallback`), which is intended.
* Snapshots come every 17 s (median; 5-95%: 2-79 s), and the NIC changes between
  consecutive snapshots 1.5% of the time.

So `q_age_s` means "seconds since readsb's last state snapshot", and the 30 s limit
keeps values readsb itself still held; a per-field age is not needed.

**The `stale` flag** (bit 1 of the trace flags) does not mark an old position. It marks
the first position after 20 s or more without one: on 1 Nov the gap before a flagged
point is 30 s (median; 5%: 16 s, the rest after points we drop outside the region
boxes), against 4 s before other points. It is set on 2-6% of ADS-B points (Black Sea
6.1%, where flagged points are twice as often degraded: jamming causes the gaps). The
position and its NIC are fresh, so these points stay usable. Leaving them out of the
jump and speed checks was tested and rejected (docs/evaluation_protocol.md, round 3);
the features record them as `after_gap` only.

### Why `src` matters

`mlat` positions are computed from signal arrival times at ground receivers, not from the
aircraft's own GPS. Where an aircraft has both ADS-B and MLAT positions close in time, a
large disagreement is direct evidence of spoofing. Positions from `adsc` arrive via
satellite and are much sparser.

## Coverage check

`gnssguard coverage` reports per region: aircraft, points, median aircraft per hour,
share of ADS-B vs MLAT positions, share of points with fresh quality fields, and a
first-look share of airborne ADS-B points above 10,000 ft with NIC below 7. The last
figure is a quick indicator, not a label.

## Coverage drift

adsb.lol is fed by volunteers, so coverage changes when a receiver joins or leaves.
In Punjab a new feeder appeared on 8 Oct 2026: judged cell-windows per day went from
about 180 to about 6,000, roughly ten times more. Central Europe stayed stable over the
whole period. Consequences:

- **A flagged share is only comparable between days with similar coverage.** A region
  can look "quieter" or "busier" simply because more aircraft are visible. The public
  map therefore shows coverage (aircraft judged per cell) next to every flag, and a cell
  with no usable aircraft is "no data", never "normal".
- **Rolling evaluation** (`gnssguard rolling`, docs/model.md) reports results day by day
  for exactly this reason: one test day can hide a coverage jump.
- `gnssguard coverage` per day is the first check when a region's numbers move suddenly.
