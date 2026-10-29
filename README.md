# gnssguard: GNSS interference detection from open ADS-B data

Detects GPS jamming and spoofing that affects aircraft, using only free, open flight data
from [adsb.lol](https://www.adsb.lol). Goals: flag affected areas, tell jamming from
spoofing, estimate where the interference comes from, and publish a daily public map.
Total running cost: zero.

> Research project. Not for navigation or operational decisions.

## Status

| Phase | State |
| --- | --- |
| 1. Data foundation: streaming ingest, regional Parquet store, coverage check | **in progress** |
| 2. Labels: weak rules, official events, synthetic attacks, gold set | planned |
| 3. Models: area detector, jamming vs spoofing, source localization | planned |
| 4. Evaluation: time and region splits, ablations, failure analysis | planned |
| 5. Product: daily pipeline, live map, static API | planned |

## Quickstart

```bash
git clone https://github.com/<your-user>/gnss-interference
cd gnss-interference
pip install -e ".[dev,fast]"

make test                      # unit tests, no network
make sample DATE=2026-10-26    # first 5,000 aircraft of one day, ~2 min
make ingest DATE=2026-10-26    # one full day (streams ~4-6 GB, stores tens of MB)
make coverage                  # traffic and data-quality report per region
```

Several days:

```bash
gnssguard ingest --start 2026-10-01 --end 2026-10-07
gnssguard ingest --date 2026-10-26 --regions baltic,central_europe
```

## How the data pipeline works

Each day of the adsb.lol archive is a 4-6 GB tar split into 2 GB parts. `gnssguard ingest`
streams the parts straight from GitHub releases and parses the archive on the fly, so the
raw data is **never written to disk**. Only positions inside the study regions in
`configs/regions.yaml` are kept, with GPS quality fields (NIC, NACp, SIL) attached, and
written as compressed Parquet. Interrupted downloads resume automatically, and a day is
moved into place only when it finishes, so a crash never leaves partial data.

See [docs/data.md](docs/data.md) for the archive format, the output columns, and why
quality fields are carried forward with an age.

## Running on a laptop

Measured on one full day (2026-10-26, all six regions):

| Resource | Ingest | Coverage report |
| --- | --- | --- |
| Peak RAM | ~0.9 GB | ~1.0-1.2 GB, flat as days are added |
| Disk written | ~130 MB per day | none |
| Download | ~3.8 GB per day, streamed, never saved | none |

Disk plan for 20 GB free:

| Item | Size |
| --- | --- |
| Python dependencies for Phase 1 | ~0.3 GB |
| 40 days, all six regions | ~5.5 GB |
| 40 days without `us_midwest` (half the volume) | ~2.7 GB |
| PyTorch with GPU support, needed from Phase 3 (approximate) | ~5 GB |

Tips:

- Install with `pip install --no-cache-dir ...` so pip does not keep a second copy.
- Choose regions per run: `--regions baltic,east_med,black_sea,punjab,central_europe`.
- Put data on another drive with `--out D:/gnss-data` (and `--data` for coverage).
- On Windows without `make`, run the `gnssguard ...` commands from the Makefile directly.
- Download volume is the real cost: 40 days is ~150 GB. On a capped connection, run the
  ingest in a free Google Colab notebook and download only the Parquet output.

## Repository layout

```
configs/regions.yaml        study regions (hotspots and calm controls)
src/gnssguard/ingest/       release lookup, multi-part stream, trace decoder, Parquet writer
src/gnssguard/analysis/     coverage report
tests/                      unit tests with a synthetic split archive
docs/                       data documentation
```

## Data license

Code: MIT. Input data: adsb.lol, [ODbL 1.0](http://opendatacommons.org/licenses/odbl/1.0/).
Derived datasets are published under ODbL with attribution. See [DATA_LICENSE.md](DATA_LICENSE.md).
