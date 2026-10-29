# Detection model (Phase 3)

The detector beats the Phase 2 rules where they are weakest, at the edges of jamming
footprints, without raising false alarms in calm airspace. On real data it matches the
rules on the gold labels so far, and it estimates jammer locations to a median of
13 km on synthetic events. All numbers: train 20-24 Oct 2026, validate 25 Oct, test
26 Oct (sealed), five regions.

```bash
gnssguard train       # data/models/detector/ + data/model/table.parquet (~1 min)
gnssguard evaluate    # reports/model/*.csv (~2-3 min, includes the ablations)
```

## How it works

| Part | Choice | Why |
| --- | --- | --- |
| Unit | 1° cell × 15-minute window, as in Phase 2 | same unit the labels use |
| Models | two LightGBM classifiers: `p_jamming`, `p_spoofing` | jamming and spoofing leave different evidence (docs/labels.md) |
| Features | 18 own, 11 from the 8 neighbouring cells, 7 from the window before, 7 from the window after | the context a human labeler uses and the rules ignore |
| Not used | latitude, longitude, region, time of day | the model must learn evidence, not memorise hotspots |
| Real targets | Phase 2 weak labels; label model fitted on training days only | 56,920 training rows, 10,909 jamming positives |
| Synthetic targets | attacks injected into calm Central Europe, 2 seeds per training day | exact truth, including footprint edges the rules miss |
| Split | by day: train 20-24, validation 25 (early stopping only), test 26 | no leakage between neighbouring windows |

Most important features (share of total gain): jamming: own degraded share 0.77,
own degraded aircraft 0.05, neighbours' degraded aircraft 0.05, MLAT fallback 0.05.
Spoofing: own speed mismatch 0.50, neighbours' speed mismatch 0.22, neighbours' jumps
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
| Model, synthetic training only | 57.8% [43-71] | 55.6% [41-69] | 0.24 |

Rules and model are statistically indistinguishable on 45 cells. The synthetic-only
model fails on real data: simulated attacks alone do not transfer.

### 2. Synthetic attacks on the test day

Central Europe, 26 Oct, 5 runs × (4 jamming + 4 spoofing), seeds never used in
training. "Hard" uses a deliberately different attack profile: 15-45 minute events,
60-140 km jammers that mostly degrade aircraft partially, and slow 60-150 kt spoofing
drifts that keep integrity high.

| Predictor | Jamming recall | Spoofing recall | False alarms | Hard: jamming | Hard: spoofing | Hard: false alarms |
| --- | --- | --- | --- | --- | --- | --- |
| Rules | 0.82 | 0.82 | 1.1% | 0.51 | 0.74 | 0.7% |
| **Model** | **1.00** | **0.93** | 1.0% | **0.95** | **0.91** | 0.6% |

Jamming recall at the edge of footprints (cells where fewer than half the aircraft are
affected) vs the core:

| Predictor | Standard: edge | Standard: core | Hard: edge | Hard: core |
| --- | --- | --- | --- | --- |
| Rules | 0.35 | 1.00 | 0.14 | 0.99 |
| **Model** | **1.00** | 1.00 | **0.92** | 1.00 |

### 3. Real data: flagged share of judged cell-windows, 26 Oct

| Region | Rules | Model |
| --- | --- | --- |
| Central Europe (calm control) | 0.65% | **0.57%** |
| Baltic | 46% | 47% |
| Black Sea | 20% | 34% |
| East Med | 2.0% | 3.9% |
| Punjab | 12% | 11% |

No extra false alarms in calm airspace. The model flags more in the Black Sea and the
East Med. That is the expected effect of catching footprint edges, but it could also
be overreach: **the completed gold set will decide**. (The model also judges sparse
cells with 1-2 usable aircraft, which the rules skip, so the denominators differ.)

### 4. Ablations

Each row retrains the model without one ingredient:

| Variant | Gold agreement | Hard: jamming | Hard: edge | Hard: spoofing |
| --- | --- | --- | --- | --- |
| Full model | 97.8% | 0.95 | 0.92 | 0.91 |
| No context (own features only) | 97.8% | 0.95 | 0.91 | **0.74** |
| Causal (no "after" window) | 97.8% | 0.95 | 0.91 | 0.92 |
| Real weak labels only | 97.8% | **0.55** | **0.18** | 0.74 |
| Synthetic only | **57.8%** | 0.82 | 0.92 | 0.91 |

- **Context matters for spoofing:** without neighbours and adjacent windows, hard
  spoofing recall drops from 0.91 to 0.74.
- **Synthetic training teaches the edges:** with real weak labels only, the model
  copies the rules and misses most edge cells.
- **Real labels keep it grounded:** synthetic-only training fails on real data.
- **The causal model is as good:** the live product can run without waiting for the
  next window.

### 5. Source localization

`model/localize.py`: flagged cells are joined into incidents, and each incident is fitted
with the radio-horizon model (jammer position and power range that best separate
degraded from healthy aircraft, given each aircraft's altitude). 20 synthetic jamming
events on the test day:

| Method | Median error | 90th percentile |
| --- | --- | --- |
| Centroid of degraded aircraft | 84 km | 129 km |
| **Radio-horizon fit** | **13 km** | 101 km |

16 of 20 events were placed within 40 km; four missed by 86-423 km. The stated
uncertainty contained the true location only 75% of the time, and it was far too small
for the failures, so it is overconfident. Rules and model give the same localization:
they find the same incidents, so the fit sees the same aircraft.

## Limits

- **Synthetic results are an upper bound.** Both synthetic tests use the same
  injection mechanics as the training attacks, even with different settings. Real
  jammers can be directional, intermittent, mobile or several at once.
- **The gold set is small so far** (45 decided cells). It cannot yet separate rules
  from model, or say whether the extra Black Sea flags are right.
- **One week of data.** More days, especially for the intermittent East Med and
  Punjab, would help most.
- **Localization on real data** should be published only as coarse areas with
  uncertainty (THREAT_MODEL, Phase 5). Precise numbers here are synthetic only.
- Results may differ very slightly across machines (thread scheduling in LightGBM).
