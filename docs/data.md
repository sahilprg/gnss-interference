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
