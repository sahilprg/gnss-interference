# Development guide

Installing, running the pipeline on a laptop, and the repository layout (moved here from the README on 8 Nov 2026).

## Quickstart

```bash
git clone https://github.com/sahilprg/gnss-interference
cd gnss-interference
pip install -e ".[dev,fast]" -c constraints.txt   # exact tested versions

make test                      # unit tests, no network
make check                     # as in CI: ruff, mypy, tests with coverage (at least 72%)
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

See [docs/data.md](data.md) for the archive format, the output columns, and why
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
| Models (LightGBM, CPU only; no GPU needed) | ~0.1 GB |

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
configs/events.csv          officially reported interference events (filled by hand)
configs/advisories.csv      official area-wide warnings: NOTAMs, EASA bulletins
labels/gold.csv             300 hand-labeled gold cells (26 Oct 2026)
labels/gold_2026-11-01.csv  150 hand-labeled holdout cells (1 Nov 2026), strata in holdout_2026-11-01_strata.csv
src/gnssguard/cli/          the `gnssguard` commands: pipeline, labels, checks, product
src/gnssguard/ingest/       release lookup, multi-part stream, trace decoder, Parquet writer
src/gnssguard/analysis/     coverage report
src/gnssguard/features/     grid-cell x time-window features
src/gnssguard/labels/       rules, label model, events, synthetic attacks, gold sampler
src/gnssguard/model/        detector dataset, training, evaluation, source localization
src/gnssguard/viz/          HTML map viewer
src/gnssguard/product/      daily publishing, quality gates, health, incident pages, website
src/gnssguard/ground/       EUREF station files (RINEX) and the ground-station check
src/gnssguard/network/      OpenSky, the second receiver network
models/detector/            the model the nightly job uses (commit after training)
scripts/                    protocol decisions, localization fitting and checks
tests/                      unit tests (no network), about 170
docs/                       data, labels, model, model card, evaluation protocol, product, roadmap
```

