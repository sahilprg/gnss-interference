# Detection model (Phase 3) and evaluation (Phase 4)

On 300 hand-labeled cells the model and the Phase 2 rules agree with the human labels
equally often (96.8%; 94.7% weighted to the whole day), and neither ever flagged a cell
the human called normal. On synthetic attacks with exact truth the model is far
stronger: it catches the edges of jamming footprints and slow spoofing that the rules
miss, on attack types it never trained on and on each of seven separate test days. It
places synthetic jammers to a median of 13 km. Two things are **not** shown yet: that
its edge advantage carries over to real data, and that spoofing detection holds up over
a busy hotspot such as the Black Sea (section 7).

All decisions in the last improvement round followed a protocol written before any
result (docs/evaluation_protocol.md): changes were accepted or rejected on data without
hand labels, and the gold set was scored once at the end.

Data: train 29 Sep-24 Oct 2026 (26 days), validate 25 Oct, test 26 Oct, five regions.

```bash
gnssguard train       # data/models/detector/ + data/model/table.parquet
gnssguard evaluate    # reports/model/*.csv, includes the ablations
gnssguard rolling     # 7 test days, attacks on 2 backgrounds
```

Resources for 28 stored days on 2 CPU cores: `train` about 8 minutes (peak about 3 GB
of memory), `evaluate` about 5 minutes, `rolling` with two test backgrounds about 12
minutes. A laptop may take two to three times as long. Install with `pip install -e ".[dev,fast]" -c constraints.txt` to get the exact
package versions the results were produced with (they are also recorded in the saved
model's `meta.json`).

## How it works

| Part | Choice | Why |
| --- | --- | --- |
| Unit | 1° cell × 15-minute window, as in Phase 2 | same unit the labels use |
| Models | two LightGBM classifiers: `p_jamming`, `p_spoofing` | jamming and spoofing leave different evidence (docs/labels.md) |
| Features | 18 own, 11 from the 8 neighbouring cells, 7 from the window before, 7 from the window after | the context a human labeler uses and the rules ignore |
| Not used | latitude, longitude, region, time of day | the model must learn evidence, not memorise hotspots |
| Real targets | Phase 2 weak labels; label model fitted on training days only | real cells of all five regions |
| Synthetic targets | attacks injected into Central Europe and East Med, 2 seeds per training day; the untouched background keeps its real weak label | exact truth for the attacks, including footprint edges the rules miss |
| Size | 422,914 training rows; 74,383 jamming and 12,363 spoofing positives | |
| Split | by day: train 29 Sep-24 Oct, validation 25 Oct (early stopping and calibration), test 26 Oct | no leakage between neighbouring windows |
| Probabilities | isotonic calibration fitted on the validation day; labels decided on the raw score at 0.5 | section 6 |

Most important features (share of total gain): jamming: degraded aircraft 0.57,
degraded share 0.22, MLAT fallback 0.08, neighbours' degraded share 0.04. Spoofing: own
speed mismatch 0.63, neighbours' speed mismatch 0.13, neighbours' jumps 0.06. Full list:
`reports/model/importance.csv`.

`gnssguard evaluate` writes only aggregate numbers for the test day, never per-cell
predictions, so labels can be re-checked blind later.

## Results

### 1. Gold labels (300 hand-labeled cells, 26 Oct)

Labeled by hand in the blind labeling tool (docs/labels.md): 112 normal, 87 jamming, 32
both, 22 spoofing, 47 unsure. Drawn from cells with at least 3 usable aircraft,
stratified by region and by the rules' jamming probability, so the sample deliberately
over-represents uncertain and affected cells. The weighted figure re-weights each
stratum by its size on the day, estimating agreement over all such cells.

| Predictor | Interference vs normal | Weighted to the day | Exact label | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| Rules | 96.8% [93.9-98.4] | 94.7% | 91.7% [87.6-94.5] | 100% | 94.3% |
| **Model** | **96.8%** [93.9-98.4] | **94.7%** | **91.7%** [87.6-94.5] | 100% | 94.3% |
| First model, before the background fix | 95.3% [91.9-97.3] | | 90.1% [85.8-93.2] | 100% | 91.5% |

253 decided cells; Wilson 95% intervals in brackets. The weighted figure is lower
because the large "likely normal" strata hold the misses described below.

- **No false alarms.** Neither rules nor model flagged any of the 112 cells labeled
  normal.
- **Eight cells were missed by both.** Each has weak evidence: 0-2 degraded aircraft
  among 3-26, which the labeler judged from neighbouring cells and adjacent windows.
  This is the "edge" situation where the model beats the rules on synthetic data, yet
  it misses these too.
- **Mixed cells.** 13 cells the labeler called "both" were given a single type (9
  jamming, 4 spoofing) by rules and model alike.
- **Unsure cells are quiet cells.** 43 of the 47 unsure cells are normal for both
  predictors.

**A bug the gold set found.** The first model missed four Central Europe cells that the
labeler and the rules called jamming. Synthetic attacks are injected into real traffic,
and every cell away from an attack was labeled "clean", including real cells the weak
labels call jamming: each such cell appeared once as jamming and twice as clean, and the
model learned to ignore weak real interference. The background now keeps its real weak
label. Because this was found by looking at gold disagreements, the gold numbers of
later models are not a fully independent test; everything after that point was decided
by the protocol without the gold set.

### 2. Synthetic attacks on the test day

Central Europe, 26 Oct, 5 runs × (4 jamming + 4 spoofing) per profile, seeds never used
in training. Training only ever sees the "standard" generator; the other two profiles
are test-only:

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
| **Model** | **1.00** | **0.94** | **0.95** | **0.96** | **0.96** | **0.96** |

False alarms away from events: rules 1.1% / 0.7% / 0.8%, model 0.9% / 0.7% / 0.7%
(standard / hard / stress). They include real weak interference in the background
traffic, which counts against both equally.

Jamming recall at the edge of footprints (cells where fewer than half the aircraft are
affected):

| Predictor | Standard: edge | Hard: edge | Stress: edge | Core, all profiles |
| --- | --- | --- | --- | --- |
| Rules | 0.35 | 0.14 | 0.49 | 0.99-1.00 |
| **Model** | **1.00** | **0.92** | **0.93** | 1.00 |

### 3. Real data: flagged share of judged cell-windows, 26 Oct

Split by usable aircraft per cell. With 3 or more, both predictors judge the cell and
the gold set covers it. With 1-2, only the model judges, and no hand label exists.

| Region | Rules, 3+ aircraft | Model, 3+ aircraft | Model, 1-2 aircraft (cells) |
| --- | --- | --- | --- |
| Central Europe (calm control) | 0.65% | 0.65% | 0% (209) |
| Baltic | 37% | 36% | 11% (2,559) |
| Black Sea | 20% | 20% | 3% (573) |
| East Med | 1.9% | 1.9% | 0.1% (1,438) |
| Punjab | 12% | 12% | 0.2% (600) |

- **Where both judge, they agree** within one point in every region on every rolling
  day (section 7).
- **The earlier "Black Sea gap"** (model flagging more than the rules) was entirely in
  cells with 1-2 aircraft.
- **Sparse cells were the open question.** The model's sparse-cell flag rate moved a lot
  between versions (Baltic 52% in the first model, 11-18% now). The second hand-labeled
  day (1 Nov, docs/evaluation_protocol.md) tested them: 47 of 47 decided flags were
  right (95% interval 92-100%), mostly in the Baltic. The map now shows these flags,
  marked as based on 1-2 aircraft; sparse cells called normal stay "low coverage".

### 4. Ablations

Each row retrains the model without one ingredient (test day, Central Europe):

| Variant | Gold agreement | Hard: jamming | Hard: edge | Hard: spoofing | Stress: jamming | Stress: spoofing |
| --- | --- | --- | --- | --- | --- | --- |
| Full model | 96.8% | 0.95 | 0.92 | 0.96 | 0.96 | 0.96 |
| No context (own features only) | 96.8% | 0.96 | 0.93 | **0.74** | 0.99 | **0.86** |
| Causal (no "after" window) | 96.8% | 0.95 | 0.92 | 0.93 | 0.96 | 0.95 |
| Real weak labels only | 96.8% | **0.53** | **0.15** | 0.74 | **0.76** | 0.84 |
| Synthetic rows only (attacks + weakly labeled background) | 96.8% | 0.97 | 0.95 | 0.96 | 0.98 | 0.96 |

- **Context matters for spoofing:** without neighbours and adjacent windows, spoofing
  recall drops from 0.96 to 0.74 (hard) and 0.86 (stress).
- **Synthetic attacks teach the edges:** with real weak labels only, the model copies
  the rules and misses most edge cells.
- **The weakly labeled synthetic background already carries the real labels:** since
  the background fix, synthetic rows alone match the full model on gold. (Purely
  synthetic training, with the background treated as clean, reached only 55.6%.)
- **The causal model is as good:** the live product can run without waiting for the
  next window.

### 5. Source localization

`model/localize.py`: flagged cells are joined into incidents, and each incident is fitted
with the radio-horizon model (jammer position and power range that best separate
degraded from healthy aircraft, given each aircraft's altitude). Synthetic jamming
events on the test day; a moving jammer is scored against the middle of its track:

| Profile | Events | Radio-horizon fit, median | 90th percentile | Centroid baseline, median |
| --- | --- | --- | --- | --- |
| Standard | 20 | **13 km** | 101 km | 84 km |
| Hard | 20 (19 found) | **29 km** | 97 km | 53 km |
| Stress | 32 | **44 km** | 164 km | 82 km |

**Honest uncertainty.** The raw uncertainty (spread of near-best fits) contained the
truth for only 75% (standard), 42% (hard) and 31% (stress) of events. Two fixes, fitted
on 168 synthetic events injected on 22-25 Oct, never the test day
(`scripts/fit_localization_uncertainty.py`):

1. The stated uncertainty is the raw spread × 4, the smallest factor that covered 90%
   of confident estimates on those days (91.6%).
2. A fit that separates degraded from healthy aircraft poorly (score below 0.85) is
   marked low confidence: on the fitting days confident estimates had a median error of
   16 km, low-confidence ones 49 km. Most low-confidence incidents hold several jammers.

On the test day:

| Profile | Confident | Median error, confident | Truth inside stated uncertainty (confident) | Median stated uncertainty |
| --- | --- | --- | --- | --- |
| Standard | 18 of 20 | 13 km | **100%** | 114 km |
| Hard | 16 of 19 | 24 km | **94%** | 116 km |
| Stress | 10 of 32 | 40 km | **90%** | 107 km |

A two-jammer fit was tried and rejected by the protocol: it changed 4 of 168 events and
never fired on paired stress jammers (the one-jammer fit chooses a range wide enough to
cover both). It is in the code, off by default.

**Real data check (no labels).** Public reporting most often names the Kaliningrad region
as a source of jamming over the southern Baltic. On 20-25 Oct, 20 of the 24 confident
estimates for real Baltic incidents lie within about 200 km of Kaliningrad (median 129
km), closer than the centroids of the degraded aircraft (median 160 km); 9 lie within
100 km. The other 4 sit at the far eastern edge of the region, where aircraft coverage
of the area is poor. This is a plausibility check, not attribution: published output
shows only coarse areas with their uncertainty.

### 6. Probability calibration

A probability of 0.8 should mean 8 in 10 such cells are affected. The first model was
overconfident in the middle (in the hard test, 51 cells scored 0.7-0.9 for jamming and
only 6 were jammed), so `gnssguard train` fits an isotonic map (pool adjacent violators,
`model/calibrate.py`) on the validation day and stores it with the model. It never
reorders cells; labels are still decided on the raw score at 0.5; the calibrated
probability is what the map and API publish.

On the first model it halved the error on uncertain jamming cells (ECE on cells scored
0.05-0.95: 0.57 / 0.55 / 0.49 raw, 0.32 / 0.27 / 0.24 calibrated, standard / hard /
stress). The background fix removed most of the contradictory labels that produced
uncertain scores: the current model is unsure about at most 22 jamming cells in 10,000
on the test day, too few to measure calibration on, and its map is close to the
identity. For spoofing (14-30 unsure cells) calibration lowers the error slightly
(0.16 to 0.12, 0.21 to 0.18, 0.37 to 0.33). The step stays in the pipeline and is
re-fitted with every training.

### 7. Rolling evaluation: seven test days, two backgrounds

`gnssguard rolling` trains a fresh model for each of 20-26 Oct on all earlier days,
validates on the day before and tests on the day itself (3 synthetic runs per profile
per day). Attacks are injected into calm Central Europe and into the Black Sea, a
hotspot region never used for synthetic training. Mean over the 7 days, with the worst
and best day:

| Background | Profile | Predictor | Jamming recall | Spoofing recall | False alarms |
| --- | --- | --- | --- | --- | --- |
| Central Europe | Standard | Rules | 0.78 (0.72-0.81) | 0.81 (0.76-0.90) | 1.0% |
| Central Europe | Standard | **Model** | **1.00** (1.00-1.00) | **0.92** (0.88-0.98) | 0.9% |
| Central Europe | Hard | Rules | 0.50 (0.46-0.54) | 0.75 (0.68-0.83) | 0.8% |
| Central Europe | Hard | **Model** | **0.93** (0.87-0.98) | **0.95** (0.83-1.00) | 0.7% |
| Central Europe | Stress | Rules | 0.70 (0.67-0.74) | 0.78 (0.63-0.87) | 0.9% |
| Central Europe | Stress | **Model** | **0.98** (0.96-1.00) | **0.91** (0.81-0.96) | 0.8% |
| Black Sea | Standard | Rules | 0.84 (0.78-0.90) | 0.70 (0.57-0.80) | 22% |
| Black Sea | Standard | **Model** | **0.98** (0.96-1.00) | **0.84** (0.69-0.93) | 13% |
| Black Sea | Hard | Rules | 0.77 (0.69-0.85) | 0.33 (0.22-0.45) | 20% |
| Black Sea | Hard | **Model** | **0.88** (0.78-0.98) | **0.47** (0.25-0.66) | 13% |
| Black Sea | Stress | Rules | 0.84 (0.78-0.89) | 0.51 (0.37-0.67) | 21% |
| Black Sea | Stress | **Model** | **0.96** (0.94-0.97) | **0.64** (0.52-0.84) | 13% |

Black Sea false alarms include the region's real interference (about 20% of its cells
are flagged on a normal day), so they are not a clean false-alarm rate.

- The model beats the rules on every profile, on both backgrounds, on every day.
- **Spoofing over the Black Sea is the weakest result in the project:** slow, high-
  integrity spoofing (hard) is caught 47% of the time by the model and 33% by the rules.
  Injecting training attacks into a second region (East Med) raised it by 5-6 points
  (protocol decision B) but did not close the gap. Results from calm Central Europe
  should not be read as valid for hotspot regions.

Real data, flagged share of cells with 3+ usable aircraft:

| Region | Predictor | 20 Oct | 21 | 22 | 23 | 24 | 25 | 26 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Central Europe (calm) | Rules | 0.52% | 0.63% | 0.41% | 0.82% | 0.49% | 1.24% | 0.65% |
| Central Europe (calm) | Model | 0.52% | 0.63% | 0.45% | 0.82% | 0.49% | 1.27% | 0.65% |
| Baltic | Rules | 55% | 42% | 53% | 48% | 48% | 43% | 37% |
| Baltic | Model | 54% | 41% | 52% | 47% | 47% | 42% | 36% |
| Black Sea | Rules | 17% | 13% | 16% | 23% | 21% | 23% | 20% |
| Black Sea | Model | 17% | 13% | 15% | 23% | 20% | 23% | 20% |
| East Med | Rules | 2.9% | 2.5% | 3.8% | 1.8% | 2.2% | 3.3% | 1.9% |
| East Med | Model | 2.9% | 2.5% | 3.7% | 1.8% | 2.3% | 3.3% | 1.9% |
| Punjab | Rules | 5.5% | 7.9% | 8.3% | 8.4% | 4.7% | 12% | 12% |
| Punjab | Model | 5.4% | 7.7% | 7.8% | 8.2% | 4.6% | 11% | 12% |

In calm Central Europe the model is never more than 0.04 points above the rules.
Sparse cells (1-2 aircraft), model only: Baltic 11-18% flagged per day, Black Sea
2-7%, the other regions under 0.5%.

### 8. What the last improvement round tried

Full record in docs/evaluation_protocol.md. Decided on 20-25 Oct only:

| Change | Result | Decision |
| --- | --- | --- |
| A. Borderline weak negatives left unlabeled | recall up to +0.12, but false alarms +4 to +44 points, calm Central Europe +3.7 points | rejected |
| B. Training attacks also injected into East Med | Black Sea spoofing +0.05-0.06, Black Sea false alarms -2 to -3 points, Central Europe unchanged | **accepted** |
| C. A and B together | as A | rejected |
| D. MLAT vs ADS-B position check for spoofing | no MLAT positions at all in the Black Sea or Punjab; where they exist, MLAT and ADS-B positions of the same aircraft within 10 s differ by a median of 41 km (East Med) and 116 km (calm Central Europe), so a large difference does not indicate spoofing | not usable with this data |
| E. Two-jammer localization | 4 of 168 events changed; uncertainty coverage 91.6% to 90.7% | rejected |
| F. Real Baltic localization check | see section 5 | reported |

## Data quality guards found along the way

- **Duplicate points.** On 17 Oct one parked aircraft (a PC-24, ICAO 4d223c) produced
  2.4 million points in 6 hours, 2.1 million of them exact duplicates, doubling Central
  Europe's volume. Ingest now drops points whose timestamp does not move forward
  (`points_not_forward` in the day manifest). On a normal day this removes about 0.07%
  of rows. It never affected labels: counts are per aircraft and readings below
  3,000 ft are not used as evidence.
- **Memory.** Synthetic training no longer keeps the injected points of every run in
  memory (about 300 MB each; four weeks of training days would have needed ~15 GB).
- **Contradictory labels.** See section 1.

## Limits

- **Synthetic results are an upper bound.** All synthetic tests use the same injection
  mechanics as the training attacks, even with different settings. Directional
  antennas, terrain shadowing and real receiver gaps are not modelled.
- **Spoofing over busy hotspot regions is weak** (Black Sea, section 7).
- **The model's edge advantage is shown on synthetic data only.** On real cells with 3+
  aircraft it agrees with the rules and misses the same eight weak cells the labeler
  flagged. Doing better on real data probably needs real labels used for training,
  not only testing.
- **Sparse cells (1-2 aircraft):** the model's flags there are validated on one day,
  mostly Baltic (47 of 47 right); its "normal" calls there are not validated.
- **Both predictors miss weak interference** at the edges of hotspots: on 1 Nov about
  4-13% of judged cells (mostly Black Sea) that both called normal were interference
  to the labeler. Published shares are lower bounds.
- **The gold set is one day and one labeler**, and its use to find the background bug
  means later gold numbers are not fully independent.
- **Spoofing labels are not independently confirmed** (docs/labels.md): the labeler sees
  the same evidence as the rules, and the MLAT cross-check is not available here.
- **Several jammers in one incident** are fitted as one source and marked low
  confidence.
- **Localization on real data** is published only as coarse areas with uncertainty
  (docs/model_card.md). Precise numbers here are synthetic only.
- **Calibration uses one validation day.**
- Results can differ very slightly across machines (thread scheduling in LightGBM) and
  with package versions other than those in `constraints.txt`.
