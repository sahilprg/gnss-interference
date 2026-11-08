# Evidence and limits

Everything the project has measured about how far its map can be trusted, moved here from the README on 8 Nov 2026 (the README keeps a summary). Every check was declared before it was run: [evaluation_protocol.md](evaluation_protocol.md).

## Summary of the evidence

**Evidence so far** (every number: [docs/evaluation_protocol.md](evaluation_protocol.md),
each check declared before it was run; all of them regenerated in about a minute by
`gnssguard benchmark` into [results/](../results/README.md), with the model hash and the
hash of every input):

* **Public reports:** 14 of 15 checkable events found (95% interval 70-99%), including
  all 4 backed by ground radio measurements and Delhi spoofing in 2025-2026; the first
  Kaliningrad spoofing (Nov 2024) was missed ([configs/events.csv](../configs/events.csv)).
* **False alarms:** on hand review, 13% of flags in calm Central Europe were false, about
  0.1% of judged cells; flags in the hotspot regions were right 26 of 26 times.
* **Hand-labeled test days never used for any decision:** 92% agreement over the whole
  day (1 Nov), and flags from cells with only 1-2 aircraft right 47 of 47 times.
* **Against a simple baseline** (degraded-NIC share, set to about the same calm-region
  alarm rate): the model finds 86% of hand-labeled interference against 76%, and 56 of
  80 spoofing cells against none (development sets; the sealed test decides).
* **Ground stations:** six EUREF stations around Kaliningrad show no agreement with the
  map beyond a shifted-days control (expected and stated before the data: they are
  outside the jammers' line of sight).
* **Second receiver network (OpenSky):** where the map says normal, OpenSky agrees
  (about 2-5% flagged). Where it says jamming, OpenSky confirms 70-84% outside the
  Baltic but 45% in the Baltic, short of the declared 70%: there most jammed aircraft
  have no GNSS position at all and are seen only by multilateration. Counting the same
  aircraft in both, the Baltic reaches 73% (evaluation protocol, second network check 1).
  In calm Central Europe, more than half of the few jamming flags rest on MLAT-only
  aircraft; counted without them, OpenSky confirms 88%.
* **Incident pages** for every catalogued event: the map, the evidence and whether the
  day stands out against ordinary days ([site](https://sahilprg.github.io/gnss-interference/incidents/index.html)).
* **Scores, as checked by a reviewer** (150 cells, weighted to all cells of the review
  days): below 0.1 about 4% interference (0-9%), 0.1-0.9 about 58% (44-86%), 0.9 or more
  about 99% (95-100%), shown on the map. Raw sample shares: 10%, 60%, 92%.

**Limits to keep in mind:**

* **One labeler.** All 700 hand labels come from one person reading the same ADS-B
  evidence the system uses; there is no second annotator. A blind relabel of 55 of them
  matched every time a decision was made twice (39 of 39), so the labels are stable,
  but not checked by anyone else.
* **Aircraft-altitude view only.** It sees interference where aircraft fly and receivers
  hear them, not on the ground or at sea, and nothing where coverage is thin.
* **It misses weak interference** at the edges of hotspots (an estimated 4-13% of judged
  cells on 1 Nov, mostly over the Black Sea): published shares are lower bounds.
* **Spoofing is the weak part:** 45% of hard simulated spoofing over the Black Sea is
  caught (35-56%).
* **Jammer locations are experimental:** tested on simulated jammers only.
* **Where the model and the rules disagree, the model is not shown to be better**
  (blind review of 69 such cells, 8 Nov): on its extra flags the reviewer sided with it
  15 of 27 times (56%, 37-72%); its extra jamming flags were right 8 of 8, its extra
  spoofing flags carried spoofing 4 of 19 times. Where only one aircraft of 3-9 is
  degraded the rules give no verdict and the model says normal; in the Baltic the
  reviewer saw jamming in 17 of 18 such cells, so the map shows some jamming edges as
  normal (about 2% of the Baltic's jammed area). Outside the Baltic the reviewer called
  9 of 12 decided such cells jamming (75%, 47-91%): not enough to call the miss general;
  the sealed test decides. Evaluation
  protocol, "Disagreement set 1".
* **The model mostly agrees with its own rules** on real data; it has not yet been shown
  to beat them there.

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
[docs/labels.md](labels.md).

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
| Hard synthetic attacks: jamming caught | 51% | **99%** |
| Hard synthetic attacks: footprint edges caught | 13% | **98%** |
| Hard synthetic attacks: spoofing caught | 70% | **95%** |
| Stress attacks (intermittent, moving, paired jammers): jamming caught | 74% | **98%** |
| Hard spoofing over the Black Sea (7 test days) | 33% | 47% |
| Flagged share, calm Central Europe (real data) | 0.65% | 0.65% |
| ... share of those alerts judged false on hand review (46 cells, 27 Oct-3 Nov) | | about 10% weighted by day (2-25%; 13% unweighted); about 0.1% of all judged cells |
| Jammer location error, median (synthetic) | | **15 km** (centroid baseline: 89 km) |
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
rules written before any result ([docs/evaluation_protocol.md](evaluation_protocol.md)).
Intended use and limits: [docs/model_card.md](model_card.md).

Ablations, limits and the full tables: [docs/model.md](model.md).

## Daily map and open data (Phase 5)

A nightly GitHub Actions job processes the previous day and publishes a map and data
files on GitHub Pages: https://sahilprg.github.io/gnss-interference/ (how it runs, the
file formats and the setup: [docs/product.md](product.md)). Only cell-level aggregates are
published, cells with fewer than 3 aircraft get no count and show only a model flag
(dashed) or low coverage, jammer areas are
coarse, experimental circles, days whose input fails the quality checks are withheld,
and an alarm in the calm control region marks a day for review instead of hiding it.

```bash
gnssguard publish --dates 2026-10-26   # public files + website for a stored day -> public/
python -m http.server -d public 8000 --bind 127.0.0.1   # preview at http://127.0.0.1:8000
```

## Status

| Phase | State |
| --- | --- |
| 1. Data foundation: streaming ingest, regional Parquet store, coverage check | done |
| 2. Labels: weak rules, label model, synthetic attacks, map viewer, 300 hand-labeled gold cells | done |
| 3. Models: jamming and spoofing detectors, source localization | done |
| 4. Evaluation: gold labels, rolling 7-day test, stress attacks, ablations, calibration, pre-declared protocol | done |
| 5. Product: nightly pipeline, live map website, open data files, outside-report checks, retrain alarm | **live** since 1 Nov 2026 ([site](https://sahilprg.github.io/gnss-interference/), [docs/product.md](product.md)) |

Since then: a correctness pass, independent evidence, a ground-station check,
incident pages, and a correction after an external review (8 Nov 2026: three bugs fixed,
figures re-measured; evaluation protocol, "Correction 1") ([docs/roadmap.md](roadmap.md), with the status of each item).
Release 0.2.0 is archived before the sealed test (19 Nov 2026; [doi:10.5281/zenodo.24015201](https://doi.org/10.5281/zenodo.24015201)), which decides whether
the model beats the integrity-only baseline ([CHANGELOG.md](../CHANGELOG.md)).

