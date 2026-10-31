# Detection model (Phase 3) and evaluation (Phase 4)

On the 300 hand-labeled cells the model and the Phase 2 rules agree with the human
labels equally often (96.8%), and neither ever flagged a cell the human called normal.
On synthetic attacks the model is far stronger: it catches the edges of jamming
footprints and slow spoofing that the rules miss, including on realistic "stress"
attacks it never trained on, and on each of seven separate test days. It estimates
jammer locations to a median of 13 km on synthetic events. Whether its synthetic edge
advantage also shows on real data is not yet proven (section 1). All numbers: train
29 Sep-24 Oct 2026 (26 days), validate 25 Oct, test 26 Oct, five regions.

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
| Real targets | Phase 2 weak labels; label model fitted on training days only | 302,328 training rows, 62,059 jamming positives (real and synthetic) |
| Synthetic targets | attacks injected into calm Central Europe, 2 seeds per training day; the untouched background keeps its real weak label | exact truth for the attacks, including footprint edges the rules miss |
| Split | by day: train 29 Sep-24 Oct, validation 25 Oct (early stopping and calibration), test 26 Oct | no leakage between neighbouring windows |
| Probabilities | isotonic calibration fitted on the validation day; labels still decided on the raw score at 0.5 | section 6 |

Most important features (share of total gain): jamming: own degraded share 0.76,
MLAT fallback 0.06, own degraded aircraft 0.06, neighbours' degraded aircraft 0.04.
Spoofing: own speed mismatch 0.55, neighbours' speed mismatch 0.16, neighbours' jumps
0.15. Full list: `reports/model/importance.csv`.

`gnssguard evaluate` writes only aggregate numbers for the test day, never per-cell
predictions, so labels can be checked blind (for example a later re-label by the same
or a second labeler).

## Results

### 1. Gold labels (300 hand-labeled cells, 26 Oct)

The gold set was labeled by hand in the blind labeling tool (docs/labels.md): 112
normal, 87 jamming, 32 both, 22 spoofing, 47 unsure. It was drawn from cells with at
least 3 usable aircraft, stratified by region and by the rules' jamming probability, so
it over-represents uncertain and affected cells on purpose. Agreement on the 253
decided cells, with Wilson 95% intervals:

| Predictor | Interference vs normal | Exact label | Precision | Recall |
| --- | --- | --- | --- | --- |
| Rules | 96.8% [93.9-98.4] | 91.7% [87.6-94.5] | 100% | 94.3% |
| **Model** | **96.8%** [93.9-98.4] | **91.7%** [87.6-94.5] | 100% | 94.3% |
| Model before the background fix below | 95.3% [91.9-97.3] | 90.1% [85.8-93.2] | 100% | 91.5% |

What the labels show:

- **No false alarms.** Neither rules nor model flagged any of the 112 cells labeled
  normal.
- **Eight cells were missed by both.** Each has weak evidence: 0-2 degraded aircraft
  among 3-26, which the labeler judged from neighbouring cells and adjacent windows.
  This is exactly the "edge" situation where the model beats the rules on synthetic
  data, yet it misses these too. So the synthetic edge advantage is not yet shown on
  real data.
- **Mixed cells.** 13 cells the labeler called "both" were given a single type (9
  jamming, 4 spoofing) by rules and model alike.
- **Unsure cells are quiet cells.** 43 of the 47 unsure cells are normal for both
  predictors (median degraded share 0%).

**A bug the gold set found.** The first version of the model missed four Central
Europe cells that the labeler and the rules called jamming (two aircraft at NIC 0 that
fell back to MLAT, repeating across windows). Cause: synthetic attacks are injected
into real Central Europe traffic, and every cell away from an attack was labeled
"clean", including real cells the weak labels call jamming. Each such cell appeared once
as jamming (real rows) and twice as clean (two synthetic copies), and the model learned
to ignore weak real interference. Now the background keeps its real weak label
(`synthetic_rows(label_model=...)`). This fixed those four cells and left the synthetic
results essentially unchanged. Because the fix was found by looking at gold disagreements, the
gold numbers for the fixed model are no longer a fully independent test; the change is
a correction of contradictory labels, not a tuning step, and the rolling evaluation
(section 7) confirms it on six other days.

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
| **Model** | **1.00** | **0.93** | **0.95** | **0.96** | **0.98** | **0.95** |

False alarms away from events: rules 1.1% / 0.7% / 0.8%, model 1.0% / 0.7% / 0.7%
(standard / hard / stress). These include real weak interference in the background
traffic, which counts against both equally.

Jamming recall at the edge of footprints (cells where fewer than half the aircraft are
affected) vs the core:

| Predictor | Standard: edge | Hard: edge | Stress: edge | Core, all profiles |
| --- | --- | --- | --- | --- |
| Rules | 0.35 | 0.14 | 0.49 | 0.99-1.00 |
| **Model** | **1.00** | **0.92** | **0.96** | 0.99-1.00 |

### 3. Real data: flagged share of judged cell-windows, 26 Oct

Split by how many usable aircraft a cell has. With 3 or more, both predictors judge
the cell and the gold set covers it. With 1-2, only the model judges, and no hand label
exists.

| Region | Rules, 3+ aircraft | Model, 3+ aircraft | Model, 1-2 aircraft (cells) |
| --- | --- | --- | --- |
| Central Europe (calm control) | 0.65% | 0.65% | 0% (209) |
| Baltic | 37% | 36% | 13% (2,559) |
| Black Sea | 20% | 20% | 7% (573) |
| East Med | 1.9% | 1.9% | 0.6% (1,438) |
| Punjab | 12% | 12% | 0.2% (600) |

- **Where both judge, they agree.** On cells with 3+ aircraft the model's flagged
  share is within one point of the rules in every region on every rolling day.
- **The "Black Sea gap" of earlier runs is answered:** every extra model flag there
  was in a cell with 1-2 aircraft. On cells both judge there is no gap.
- **Sparse cells are the open question.** The model's sparse-cell flag rate moved a
  lot between versions (Baltic 52% before the fix, 13-20% after), and no gold label
  covers it. The public map will show these cells as "low coverage", with the
  probability but not as a confident flag.

### 4. Ablations

Each row retrains the model without one ingredient:

| Variant | Gold agreement | Hard: jamming | Hard: edge | Hard: spoofing | Stress: jamming | Stress: spoofing |
| --- | --- | --- | --- | --- | --- | --- |
| Full model | 96.8% | 0.95 | 0.92 | 0.96 | 0.98 | 0.95 |
| No context (own features only) | 96.8% | 0.95 | 0.92 | **0.74** | 0.98 | **0.85** |
| Causal (no "after" window) | 96.8% | 0.95 | 0.92 | 0.93 | 0.97 | 0.95 |
| Real weak labels only | 96.8% | **0.53** | **0.15** | 0.74 | **0.76** | 0.84 |
| Synthetic attacks only (background still weakly labeled) | **88.5%** | 0.98 | 0.97 | 0.96 | 0.97 | 0.96 |

- **Context matters for spoofing:** without neighbours and adjacent windows, spoofing
  recall drops from 0.96 to 0.74 (hard) and 0.95 to 0.85 (stress).
- **Synthetic training teaches the edges:** with real weak labels only, the model
  copies the rules and misses most edge cells.
- **Real labels keep it grounded:** without them gold agreement falls to 88.5%.
  (Before the background fix, purely synthetic training reached only 55.6%.)
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

A probability of 0.8 should mean 8 in 10 such cells are affected. The first model was
overconfident in the middle: in the hard test, 51 cells scored 0.7-0.9 for jamming and
only 6 were really jammed. `gnssguard train` therefore fits an isotonic map (pool
adjacent violators, `model/calibrate.py`) on the validation day and stores it with the
model. It never reorders cells, and labels are still decided on the raw score at 0.5;
the calibrated probability is what the map and API publish.

The usual expected calibration error (ECE) hides the problem, because almost all cells
sit near 0 or 1. The useful number is the error on the cells the model is unsure about
(score 0.05-0.95):

| Synthetic truth, test day | Unsure jamming cells | ECE on unsure cells: raw | calibrated |
| --- | --- | --- | --- |
| First model: standard / hard / stress | 89 / 96 / 110 | 0.57 / 0.55 / 0.49 | **0.32 / 0.27 / 0.24** |
| Current model: standard / hard / stress | 10 / 10 / 24 | 0.16 / 0.31 / 0.32 | unchanged |

The background fix removed the contradictory labels that produced most uncertain
scores: the current model is unsure about one cell in a thousand instead of one in a
hundred, and its calibration map is close to the identity. The step stays in the
pipeline as a check that is re-fitted with every training. Spoofing has too few unsure
cells (13-24) to learn a correction from one validation day.

### 7. Rolling evaluation: seven test days, not one

One test day can be lucky. `gnssguard rolling` trains a fresh model for each of
20-26 Oct on all earlier days only, validates on the day before, and tests on the day
itself (3 synthetic runs per profile per day). Mean, with the worst and best day:

| Profile | Predictor | Jamming recall | Spoofing recall | False alarms |
| --- | --- | --- | --- | --- |
| Standard | Rules | 0.78 (0.72-0.81) | 0.81 (0.76-0.90) | 1.0% |
| Standard | **Model** | **1.00** (1.00-1.00) | **0.92** (0.88-0.97) | 0.9% |
| Hard | Rules | 0.50 (0.46-0.54) | 0.75 (0.68-0.83) | 0.8% |
| Hard | **Model** | **0.93** (0.85-0.98) | **0.94** (0.85-1.00) | 0.7% |
| Stress | Rules | 0.70 (0.67-0.74) | 0.78 (0.63-0.87) | 0.9% |
| Stress | **Model** | **0.99** (0.97-1.00) | **0.90** (0.81-0.96) | 0.8% |

Real data, flagged share of cells with 3+ usable aircraft:

| Region | Predictor | 20 Oct | 21 | 22 | 23 | 24 | 25 | 26 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Central Europe (calm) | Rules | 0.52% | 0.63% | 0.41% | 0.82% | 0.49% | 1.24% | 0.65% |
| Central Europe (calm) | Model | 0.52% | 0.63% | 0.45% | 0.82% | 0.49% | 1.31% | 0.65% |
| Baltic | Rules | 55% | 42% | 53% | 48% | 48% | 43% | 37% |
| Baltic | Model | 54% | 41% | 52% | 47% | 47% | 42% | 36% |
| Black Sea | Rules | 17% | 13% | 16% | 23% | 21% | 23% | 20% |
| Black Sea | Model | 17% | 13% | 15% | 23% | 20% | 23% | 20% |
| East Med | Rules | 2.9% | 2.5% | 3.8% | 1.8% | 2.2% | 3.3% | 1.9% |
| East Med | Model | 2.9% | 2.5% | 3.7% | 1.8% | 2.2% | 3.3% | 1.9% |
| Punjab | Rules | 5.5% | 7.9% | 8.3% | 8.4% | 4.7% | 12% | 12% |
| Punjab | Model | 5.4% | 7.7% | 7.8% | 8.2% | 4.6% | 11% | 12% |

- The model beats the rules on every synthetic profile on **every** one of the seven
  days, and for jamming its worst day is better than the rules' best day.
- On real cells both judge, the two stay within about one point every day. In calm
  Central Europe the model is never more than 0.07 points above the rules, so the fix
  did not bring back false alarms.
- Sparse cells (1-2 aircraft), model only: Baltic 13-20% flagged per day, Black Sea
  5-13%, the other regions under 1.5%.

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
- **The model's edge advantage is shown on synthetic data only.** On real cells with 3+
  aircraft it agrees with the rules, and it misses the same eight weak cells the
  labeler flagged. Learning more from real context probably needs real labels beyond
  the rules (more gold-style labels, used for training, not only testing).
- **Sparse cells (1-2 aircraft) are not validated.** No gold label covers them.
- **The gold set is one day, one labeler, and drawn with stratification.** Its
  agreement figures describe the strata it was drawn from, not every cell of the day.
- **Localization on real data** should be published only as coarse areas with
  uncertainty (THREAT_MODEL, Phase 5). Precise numbers here are synthetic only.
- **Calibration uses one validation day.** Too few uncertain spoofing cells to fit a
  correction yet.
- Days ingested before the duplicate guard can differ from re-ingested ones by about
  0.1-1% of rows, which can move results very slightly. Results may also differ very
  slightly across machines (thread scheduling in LightGBM).
