# gnssguard: GNSS interference detection from open ADS-B data

Detects GPS jamming and spoofing that affects aircraft, using only free, open flight data
from [adsb.lol](https://www.adsb.lol). Goals: flag affected areas, tell jamming from
spoofing, estimate where the interference comes from, and publish a daily public map.
Total running cost: zero.

> Research project. Not for navigation or operational decisions.

## Status

| Phase | State |
| --- | --- |
| 1. Data foundation: streaming ingest, regional Parquet store, coverage check | done |
| 2. Labels: weak rules, label model, synthetic attacks, map viewer, gold-set sampler | **done; gold set and event list to fill by hand** |
| 3. Models: jamming and spoofing detectors, source localization | **first version done; gold set in progress** |
| 4. Evaluation: time and region splits, ablations, failure analysis | planned |
| 5. Product: daily pipeline, live map, static API | planned |

## Quickstart

```bash
git clone https://github.com/sahilprg/gnss-interference
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

## Labels (Phase 2)

No labeled dataset of GNSS interference over aircraft exists, so labels are built from
weak rules, a label model, synthetic attacks and a hand-checked gold set.

```bash
gnssguard cells          # 1° x 15-min grid-cell features for every stored day
gnssguard label          # 7 rules -> label model -> jamming / spoofing / both / normal
gnssguard map --region baltic --date 2026-10-26        # interactive HTML map
gnssguard synth --region central_europe --date 2026-10-26 --seeds 0,1,2,3,4
gnssguard gold-sample --dates 2026-10-26 --n 300       # cells for hand labeling
gnssguard gold-tool                                    # labeling page (keyboard, autosave)
```

`configs/advisories.csv` holds official area-wide warnings (NOTAMs, EASA bulletins);
`gnssguard label` compares flagged rates inside and outside each warned area.

First results on one real day (2026-10-26):

| | Result |
| --- | --- |
| Judged cell-windows flagged, Baltic | 46% |
| Judged cell-windows flagged, Central Europe / US Midwest (calm controls) | 0.65% / 0% |
| Synthetic jamming detected (5 runs) | 76% of affected cell-windows |
| Synthetic spoofing detected (5 runs) | 77% (range 54-97%) |
| False alarms away from synthetic events | 1.1% |

Key findings, the open questions and the gold-set protocol are in
[docs/labels.md](docs/labels.md).

## Detection model (Phase 3)

Two LightGBM models (jamming, spoofing) trained on the rules' weak labels plus synthetic
attacks, using each cell's neighbours and the windows before and after. No location
features. Test day kept sealed for the gold set.

```bash
gnssguard train      # train 20-24 Oct, validate 25 Oct, test 26 Oct (the gold-set day)
gnssguard evaluate   # gold, synthetic (standard + hard), real flag rates, ablations
```

| Test (26 Oct) | Rules | Model |
| --- | --- | --- |
| Gold labels, interference vs normal (45 cells so far) | 100% | 97.8% |
| Hard synthetic attacks: jamming caught | 51% | **95%** |
| Hard synthetic attacks: footprint edges caught | 14% | **92%** |
| Hard synthetic attacks: spoofing caught | 74% | **91%** |
| False alarms, calm Central Europe (real data) | 0.65% | **0.57%** |
| Jammer location error, median (synthetic) | | **13 km** (centroid baseline: 84 km) |

Ablations, limits and the full tables: [docs/model.md](docs/model.md).

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
configs/events.csv          officially reported interference events (filled by hand)
configs/advisories.csv      official area-wide warnings: NOTAMs, EASA bulletins
labels/gold.csv             gold test set for hand labeling (commit your labels)
src/gnssguard/ingest/       release lookup, multi-part stream, trace decoder, Parquet writer
src/gnssguard/analysis/     coverage report
src/gnssguard/features/     grid-cell x time-window features
src/gnssguard/labels/       rules, label model, events, synthetic attacks, gold sampler
src/gnssguard/model/        detector dataset, training, evaluation, source localization
src/gnssguard/viz/          HTML map viewer
tests/                      unit tests (no network)
docs/                       data and labeling documentation
```

## Data license

Code: MIT. Input data: adsb.lol, [ODbL 1.0](http://opendatacommons.org/licenses/odbl/1.0/).
Derived datasets are published under ODbL with attribution. See [DATA_LICENSE.md](DATA_LICENSE.md).
