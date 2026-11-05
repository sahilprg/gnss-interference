# gnssguard: GNSS interference detection from open ADS-B data

Detects GPS jamming and spoofing that affects aircraft, using only free, open flight data
from [adsb.lol](https://www.adsb.lol). Goals: flag affected areas, tell jamming from
spoofing, estimate where the interference comes from, and publish a daily public map.
Total running cost: zero.

**Live map and open data: https://sahilprg.github.io/gnss-interference/** (updated
every night).

> Research project. Not for navigation or operational decisions.

## How far you can trust it

**Evidence so far** (every number: [docs/evaluation_protocol.md](docs/evaluation_protocol.md),
each check declared before it was run; all of them regenerated in about a minute by
`gnssguard benchmark` into [results/](results/README.md), with the model hash and the
hash of every input):

* **Public reports:** 14 of 15 checkable events found (95% interval 70-99%), including
  all 4 backed by ground radio measurements and Delhi spoofing in 2025-2026; the first
  Kaliningrad spoofing (Nov 2024) was missed ([configs/events.csv](configs/events.csv)).
* **False alarms:** on hand review, 13% of flags in calm Central Europe were false, about
  0.1% of judged cells; flags in the hotspot regions were right 26 of 26 times.
* **Hand-labeled test days never used for any decision:** 92% agreement over the whole
  day (1 Nov), and flags from cells with only 1-2 aircraft right 47 of 47 times.
* **Scores, as checked by a reviewer:** cells scored below 0.1 were interference 10% of
  the time, 0.1-0.9 60%, 0.9 or more 92% (shown on the map).

**Limits to keep in mind:**

* **One labeler.** All 700 hand labels come from one person reading the same ADS-B
  evidence the system uses; there is no second annotator yet.
* **Aircraft-altitude view only.** It sees interference where aircraft fly and receivers
  hear them, not on the ground or at sea, and nothing where coverage is thin.
* **It misses weak interference** at the edges of hotspots (an estimated 4-13% of judged
  cells on 1 Nov, mostly over the Black Sea): published shares are lower bounds.
* **Spoofing is the weak part:** 45% of hard simulated spoofing over the Black Sea is
  caught (35-56%).
* **Jammer locations are experimental:** tested on simulated jammers only.
* **The model mostly agrees with its own rules** on real data; it has not yet been shown
  to beat them there.

## Status

| Phase | State |
| --- | --- |
| 1. Data foundation: streaming ingest, regional Parquet store, coverage check | done |
| 2. Labels: weak rules, label model, synthetic attacks, map viewer, 300 hand-labeled gold cells | done |
| 3. Models: jamming and spoofing detectors, source localization | done |
| 4. Evaluation: gold labels, rolling 7-day test, stress attacks, ablations, calibration, pre-declared protocol | done |
| 5. Product: nightly pipeline, live map website, open data files, outside-report checks, retrain alarm | **live** since 1 Nov 2026 ([site](https://sahilprg.github.io/gnss-interference/), [docs/product.md](docs/product.md)) |

Next: a correctness pass and independent evidence before any bigger model
([docs/roadmap.md](docs/roadmap.md), with the status of each item).

## Quickstart

```bash
git clone https://github.com/sahilprg/gnss-interference
cd gnss-interference
pip install -e ".[dev,fast]" -c constraints.txt   # exact tested versions

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
features. The test day (26 Oct) carries the 300 hand labels and was never used for
training.

```bash
gnssguard train      # train on all earlier days, validate on the day before the test day
gnssguard evaluate   # gold, synthetic (standard, hard, stress), real flag rates, ablations
gnssguard rolling    # retrain and test separately on each of the last 7 days
```

| Test (26 Oct; trained on 26 days) | Rules | Model |
| --- | --- | --- |
| 300 hand-labeled cells: interference vs normal (253 decided) | 96.8% | 96.8% |
| Same, weighted to all cells of the day | 94.7% | 94.7% |
| Hand-labeled normal cells wrongly flagged (of 112; 95% upper bound 2.7%) | 0 | 0 |
| Hard synthetic attacks: jamming caught | 51% | **95%** |
| Hard synthetic attacks: footprint edges caught | 14% | **92%** |
| Hard synthetic attacks: spoofing caught | 74% | **96%** |
| Stress attacks (intermittent, moving, paired jammers): jamming caught | 74% | **96%** |
| Hard spoofing over the Black Sea (7 test days) | 33% | 47% |
| Flagged share, calm Central Europe (real data) | 0.65% | 0.65% |
| ... of which false alarms on hand review (46 cells, 27 Oct-3 Nov) | | 13% (6-26%), about 0.1% of cells |
| Jammer location error, median (synthetic) | | **13 km** (centroid baseline: 84 km) |
| Stated location uncertainty that contains the truth (confident estimates) | | **90-100%** |

**Holdout day (1 Nov)**, 150 hand labels drawn and committed with the frozen predictions
before labeling, never used for any decision:

| Test | Rules | Model |
| --- | --- | --- |
| Cells with 3+ aircraft, weighted to the whole day | 92.0% | 92.0% |
| Model flags on cells with only 1-2 aircraft (47 decided) | | **47 of 47** (95% interval 92-100%) |
| Borderline cells both call normal that were interference (18 decided) | 7 | 7 |

So the map now also shows flags from cells with 1-2 aircraft, and the published shares
are lower bounds: both predictors miss weak interference at hotspot edges.

On real data the model agrees with the rules wherever both judge; its advantage at
footprint edges is shown on synthetic attacks, not yet on real data, and spoofing over
busy hotspots remains weak. Every change in the last round was accepted or rejected by
rules written before any result ([docs/evaluation_protocol.md](docs/evaluation_protocol.md)).
Intended use and limits: [docs/model_card.md](docs/model_card.md).

Ablations, limits and the full tables: [docs/model.md](docs/model.md).

## Daily map and open data (Phase 5)

A nightly GitHub Actions job processes the previous day and publishes a map and data
files on GitHub Pages: https://sahilprg.github.io/gnss-interference/ (how it runs, the
file formats and the setup: [docs/product.md](docs/product.md)). Only cell-level aggregates are
published, cells with fewer than 3 aircraft get no count and show only a model flag
(dashed) or low coverage, jammer areas are
coarse, experimental circles, days whose input fails the quality checks are withheld,
and an alarm in the calm control region marks a day for review instead of hiding it.

```bash
gnssguard publish --dates 2026-10-26   # public files + website for a stored day -> public/
python -m http.server -d public 8000 --bind 127.0.0.1   # preview at http://127.0.0.1:8000
```

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
src/gnssguard/ingest/       release lookup, multi-part stream, trace decoder, Parquet writer
src/gnssguard/analysis/     coverage report
src/gnssguard/features/     grid-cell x time-window features
src/gnssguard/labels/       rules, label model, events, synthetic attacks, gold sampler
src/gnssguard/model/        detector dataset, training, evaluation, source localization
src/gnssguard/viz/          HTML map viewer
src/gnssguard/product/      daily publishing, quality gates, website
models/detector/            the model the nightly job uses (commit after training)
scripts/                    protocol decisions, localization fitting and checks
tests/                      unit tests (no network)
docs/                       data, labels, model, model card, evaluation protocol, product, roadmap
```

## How to cite

If you use the code, the method or the published data, please cite the repository
(GitHub's "Cite this repository" button uses `CITATION.cff`):

> sahilprg. *gnssguard: GNSS interference detection from open ADS-B data* (2026).
> https://github.com/sahilprg/gnss-interference. Daily data:
> https://sahilprg.github.io/gnss-interference/, derived from adsb.lol under ODbL 1.0.

Derived data must keep the ODbL attribution ([DATA_LICENSE.md](DATA_LICENSE.md)).

## Data license

Code: MIT. Input data: adsb.lol, [ODbL 1.0](http://opendatacommons.org/licenses/odbl/1.0/).
Derived datasets are published under ODbL with attribution. See [DATA_LICENSE.md](DATA_LICENSE.md).
