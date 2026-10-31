# Detection model (Phase 3)

The detector beats the Phase 2 rules where they are weakest, at the edges of jamming
footprints and on slow spoofing, without raising false alarms in calm airspace. On real
data it matches the rules on the gold labels so far, and it estimates jammer locations
to a median of 13 km on synthetic events. The advantage holds on realistic "stress"
attacks the model never trained on, and on each of seven separate test days. All numbers: train 29 Sep-24 Oct 2026
(26 days), validate 25 Oct, test 26 Oct (sealed), five regions.

```bash
gnssguard train       # data/models/detector/ + data/model/table.parquet
gnssguard evaluate    # reports/model/*.csv, includes the ablations
gnssguard rolling     # retrain and test day by day over the last 7 days
```

Resources for 28 stored days (2 CPU cores): `train` peaks at 2.5 GB of memory and takes
about 4-8 minutes; `evaluate` peaks at about 2.5 GB and takes 4-8 minutes; `rolling`
retrains seven times and takes about 10-15 minutes.

## How it works

| Part | Choice | Why |
| --- | --- | --- |
| Unit | 1° cell × 15-minute window, as in Phase 2 | same unit the labels use |
| Models | two LightGBM classifiers: `p_jamming`, `p_spoofing` | jamming and spoofing leave different evidence (docs/labels.md) |
| Features | 18 own, 11 from the 8 neighbouring cells, 7 from the window before, 7 from the window after | the context a human labeler uses and the rules ignore |
| Not used | latitude, longitude, region, time of day | the model must learn evidence, not memorise hotspots |
| Real targets | Phase 2 weak labels; label model fitted on training days only | 302,328 training rows, 61,451 jamming positives |
| Synthetic targets | attacks injected into calm Central Europe, 2 seeds per training day | exact truth, including footprint edges the rules miss |
| Split | by day: train 29 Sep-24 Oct, validation 25 Oct (early stopping and calibration), test 26 Oct | no leakage between neighbouring windows |
| Probabilities | isotonic calibration fitted on the validation day; labels still decided on the raw score at 0.5 | section 6 |

Most important features (share of total gain): jamming: own degraded share 0.76,
own degraded aircraft 0.06, MLAT fallback 0.05, neighbours' degraded aircraft 0.04.
Spoofing: own speed mismatch 0.54, neighbours' speed mismatch 0.17, neighbours' jumps
0.15. Full list: `reports/model/importance.csv`.

**The test day stays sealed.** `gnssguard evaluate` writes only aggregate numbers for
26 Oct, never per-cell predictions, so gold labeling can continue blind.

## Results

### 1. Gold labels (hand-checked, 26 Oct)

51 labeled so far; 45 decided (6 unsure). Wilson 95% intervals in brackets.

| Predictor | Interference vs normal | Exact label | Recall |
| --- | --- | --- | --- |
| Rules | 100% [92-100] | 97.8% [88-100] | 1.00 |
| Model | 97.8% [88-100] | 95.6% [85-99] | 0.96 |
| Model, synthetic training only | 55.6% [41-69] | 55.6% [41-69] | 0.20 |

Rules and model are statistically indistinguishable on 45 cells. The synthetic-only
model fails on real data: simulated attacks alone do not transfer.

### 2. Synthetic attacks on the test day

Central Europe, 26 Oct, 5 runs × (4 jamming + 4 spoofing) per profile, seeds never used
in training. Training only ever sees the "standard" generator. The two test-only
profiles change the attacks in different ways:

* **hard**: 15-45 minute events, 60-140 km jammers that mostly degrade aircraft
  partially, slow 60-150 kt spoofing drifts that keep integrity high.
* **stress**: what real jammers do and the generator never does. Half the jammers
  switch on and off (5-15 minutes on, 5-20 off), a third move at 15-60 kt (a vehicle
  or ship), half come with a second jammer 60-200 km away at the same time, and half of
  the aircraft that lose GPS in the core fall back to MLAT positions (as in real Baltic
  data). Spoofing mixes slow and fast drifts and jumps.

| Predictor | Standard: jamming | Standard: spoofing | Hard: jamming | Hard: spoofing | Stress: jamming | Stress: spoofing |
| --- | --- | --- | --- | --- | --- | --- |
| Rules | 0.82 | 0.82 | 0.51 | 0.74 | 0.74 | 0.84 |
| **Model** | **1.00** | **0.94** | **0.95** | **0.95** | **0.97** | **0.95** |

False alarms away from events: rules 1.1% / 0.7% / 0.8%, model 0.9% / 0.6% / 0.7%
(standard / hard / stress).

Jamming recall at the edge of footprints (cells where fewer than half the aircraft are
affected) vs the core:

| Predictor | Standard: edge | Hard: edge | Stress: edge | Core, all profiles |
| --- | --- | --- | --- | --- |
| Rules | 0.35 | 0.14 | 0.49 | 0.99-1.00 |
| **Model** | **1.00** | **0.91** | **0.95** | 1.00 |

### 3. Real data: flagged share of judged cell-windows, 26 Oct

| Region | Rules | Model (28 days) | Model (7 days, first run) |
| --- | --- | --- | --- |
| Central Europe (calm control) | 0.65% | **0.53%** | 0.57% |
| Baltic | 46% | 44% | 47% |
| Black Sea | 20% | 27% | 34% |
| East Med | 2.0% | 1.8% | 3.9% |
| Punjab | 12% | 7.3% | 11% |

No extra false alarms in calm airspace. With four times more training days the model
became more conservative on real data: its extra Black Sea and East Med flags from the
7-day run largely disappeared, while its synthetic edge recall stayed. The remaining
Black Sea gap (27% vs 20%) is for the completed gold set to judge. (The model also
judges sparse cells with 1-2 usable aircraft, which the rules skip, so the denominators
differ.)

### 4. Ablations

Each row retrains the model without one ingredient:

| Variant | Gold agreement | Hard: jamming | Hard: edge | Hard: spoofing | Stress: jamming | Stress: spoofing |
| --- | --- | --- | --- | --- | --- | --- |
| Full model | 97.8% | 0.95 | 0.91 | 0.95 | 0.97 | 0.95 |
| No context (own features only) | 97.8% | 0.96 | 0.93 | **0.74** | 0.98 | **0.85** |
| Causal (no "after" window) | 97.8% | 0.95 | 0.92 | 0.94 | 0.97 | 0.94 |
| Real weak labels only | 100% | **0.53** | **0.15** | 0.74 | **0.76** | 0.84 |
| Synthetic only | **55.6%** | 0.81 | 0.91 | 0.95 | 0.94 | 0.96 |

- **Context matters for spoofing:** without neighbours and adjacent windows, spoofing
  recall drops from 0.95 to 0.74 (hard) and 0.85 (stress).
- **Synthetic training teaches the edges:** with real weak labels only, the model
  copies the rules and misses most edge cells.
- **Real labels keep it grounded:** synthetic-only training fails on real data.
- **The causal model is as good:** the live product can run without waiting for the
  next window.

### 5. Source localization

`model/localize.py`: flagged cells are joined into incidents, and each incident is fitted
with the radio-horizon model (jammer position and power range that best separate
degraded from healthy aircraft, given each aircraft's altitude). Synthetic jamming
events on the test day (a moving jammer is scored against the middle of its track):

| Profile | Events | Radio-horizon fit, median | 90th percentile | Centroid baseline, median |
| --- | --- | --- | --- | --- |
| Standard | 20 | **13 km** | 101 km | 84 km |
| Hard | 20 (19 found) | **29 km** | 97 km | 53 km |
| Stress | 32 | **44 km** | 164 km | 82 km |

**Making the uncertainty honest.** The raw uncertainty (spread of near-best fits)
contained the truth for only 75% of standard, 42% of hard and 31% of stress events. Two
fixes, both fitted on 168 synthetic events injected on 22-25 Oct (never the test day;
`scripts/fit_localization_uncertainty.py`):

1. The stated uncertainty is the raw spread × 3.75, the smallest factor that covered
   90% of confident estimates on those days.
2. A fit that separates degraded from healthy aircraft poorly (score below 0.85) is
   marked low confidence. Most such incidents contain several jammers that merged into
   one: on the fitting days, confident estimates had a median error of 16 km,
   low-confidence ones 68 km.

On the test day:

| Profile | Confident | Median error, confident | Truth inside stated uncertainty (confident) | Median stated uncertainty |
| --- | --- | --- | --- | --- |
| Standard | 18 of 20 | 13 km | **94%** | 107 km |
| Hard | 16 of 19 | 24 km | **94%** | 109 km |
| Stress | 10 of 32 | 40 km | 80% | 101 km |

The stress profile still falls short of 90%: paired jammers that are both fitted as one
source pull the estimate between them. The single-source fit is the limit here; a
multi-source fit is future work. The public map will show low-confidence incidents as
an area only, and confident ones as a circle of the stated radius, never a point.

### 6. Probability calibration

A probability of 0.8 should mean 8 in 10 such cells are affected. On exact synthetic
truth it did not: in the hard test, 51 cells scored 0.7-0.9 for jamming and only 6 of
them were really jammed. Almost all cells sit near 0 or 1, so the usual expected
calibration error (ECE) looks tiny; the number that matters is the error on the cells
the model is unsure about (score 0.05-0.95, about 100 per 10,000 cells).

`gnssguard train` now fits an isotonic map (pool adjacent violators, `model/calibrate.py`)
on the validation day and stores it with the model. It never reorders cells, and labels
are still decided on the raw score at 0.5, so every detection result above is
unchanged; the calibrated probability is what the map and API publish.

| Test day, synthetic truth | Unsure cells | ECE on unsure cells: raw | calibrated | Brier score: raw | calibrated |
| --- | --- | --- | --- | --- | --- |
| Jamming, standard | 89 | 0.57 | **0.32** | 0.0048 | **0.0025** |
| Jamming, hard | 96 | 0.55 | **0.27** | 0.0047 | **0.0028** |
| Jamming, stress | 110 | 0.49 | **0.24** | 0.0058 | **0.0041** |
| Spoofing, all profiles | 15-30 | 0.29-0.39 | unchanged | 0.0012-0.0014 | unchanged |

Calibration halves the error on uncertain jamming cells. A raw score of 0.5 now
reads as about 0.28, which is honest: half-flagged cells at the edge of a footprint are
often clean. Spoofing has too few uncertain cells on one validation day to learn a
correction (the map stays close to the identity); more validation days will fix that.

### 7. Rolling evaluation: seven test days, not one

One test day can be lucky. `gnssguard rolling` trains a fresh model for each of
20-26 Oct on all earlier days only, validates on the day before, and tests on the day
itself (3 synthetic runs per profile per day). Mean, with the worst and best day:

| Profile | Predictor | Jamming recall | Spoofing recall | False alarms |
| --- | --- | --- | --- | --- |
| Standard | Rules | 0.78 (0.72-0.81) | 0.81 (0.76-0.90) | 1.0% |
| Standard | **Model** | **1.00** (1.00-1.00) | **0.92** (0.88-0.97) | 0.6% |
| Hard | Rules | 0.50 (0.46-0.54) | 0.75 (0.68-0.83) | 0.8% |
| Hard | **Model** | **0.93** (0.85-0.99) | **0.94** (0.85-1.00) | 0.5% |
| Stress | Rules | 0.70 (0.67-0.74) | 0.78 (0.63-0.87) | 0.9% |
| Stress | **Model** | **0.99** (0.97-1.00) | **0.90** (0.82-0.96) | 0.6% |

Real data, flagged share of judged cell-windows per day:

| Region | Predictor | 20 Oct | 21 | 22 | 23 | 24 | 25 | 26 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Central Europe (calm) | Rules | 0.52% | 0.63% | 0.41% | 0.82% | 0.49% | 1.24% | 0.65% |
| Central Europe (calm) | **Model** | **0.21%** | **0.39%** | **0.25%** | **0.49%** | **0.32%** | **1.07%** | **0.53%** |
| Baltic | Rules | 63% | 53% | 62% | 57% | 58% | 53% | 46% |
| Baltic | Model | 55% | 45% | 53% | 49% | 47% | 45% | 44% |
| Black Sea | Rules | 17% | 13% | 16% | 23% | 21% | 23% | 20% |
| Black Sea | Model | 25% | 17% | 22% | 31% | 25% | 32% | 27% |
| East Med | Rules | 3.1% | 2.6% | 3.8% | 1.8% | 2.4% | 3.4% | 2.0% |
| East Med | Model | 2.3% | 2.3% | 6.2% | 2.4% | 2.0% | 3.7% | 1.8% |
| Punjab | Rules | 5.8% | 7.9% | 8.3% | 8.4% | 4.7% | 12% | 12% |
| Punjab | Model | 3.2% | 4.4% | 5.5% | 4.4% | 3.9% | 5.3% | 7.3% |

- The model beats the rules on every synthetic profile on **every** one of the seven
  days, and its worst day is better than the rules' best day for jamming.
- In calm Central Europe the model raises fewer flags than the rules on all seven days.
- The Black Sea pattern (model above rules) holds every day, so it is systematic, not
  noise; the gold set decides who is right there. In the Baltic and Punjab the model
  is consistently more conservative than the rules.

## Data quality guards found along the way

- **Duplicate points.** On 17 Oct one parked aircraft (a PC-24, ICAO 4d223c) produced
  2.4 million points in 6 hours, 2.1 million of them exact duplicates, doubling Central
  Europe's volume. Ingest now drops points whose timestamp does not move forward
  (`points_not_forward` in the day manifest). On a normal day this removes about 0.07%
  of rows. It never affected labels: counts are per aircraft and readings below
  3,000 ft are not used as evidence.
- **Memory.** Synthetic training no longer keeps the injected points of every run in
  memory (about 300 MB each; four weeks of training days would have needed ~15 GB).

## Limits

- **Synthetic results are an upper bound.** All synthetic tests use the same injection
  mechanics as the training attacks, even with different settings. The stress profile
  adds intermittent, moving and paired jammers and MLAT fallback; directional antennas,
  terrain shadowing and real receiver gaps are still not modelled.
- **Several jammers in one incident** are fitted as one source. Such estimates are
  marked low confidence, and their uncertainty is still too small in the stress test
  (80% coverage instead of 90%).
- **The gold set is small so far** (45 decided cells). It cannot yet separate rules
  from model, or say whether the extra Black Sea flags are right.
- **Localization on real data** should be published only as coarse areas with
  uncertainty (THREAT_MODEL, Phase 5). Precise numbers here are synthetic only.
- **Calibration uses one validation day.** It is enough for jamming, not yet for
  spoofing.
- Days ingested before the duplicate guard can differ from re-ingested ones by about
  0.1-1% of rows, which can move results very slightly. Results may also differ very
  slightly across machines (thread scheduling in LightGBM).
