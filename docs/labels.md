# Labels (Phase 2)

There is no labeled dataset of GNSS interference over aircraft, so labels are built
from four sources: weak rules, officially reported events, synthetic attacks and a
small hand-checked gold set. All numbers below come from one real day, 2026-10-26,
five regions, unless stated otherwise.

```
processed points ──> cells (features) ──> rules (votes) ──> label model ──> labels
                                              │                               │
                    official events ──────────┘          gold set (by hand) ──┘ evaluation
                    synthetic attacks ──> same pipeline ──> recall / false alarms
```

```bash
gnssguard cells                       # data/cells/region=*/date=*/part-0.parquet
gnssguard label                       # data/labels/labels.parquet + reports/labels/
gnssguard synth --region central_europe --date 2026-10-26 --seeds 0,1,2,3,4
gnssguard gold-sample --dates 2026-10-26 --n 300   # labels/gold.csv (commit this)
gnssguard map --region baltic --date 2026-10-26 --gold labels/gold.csv
```

## 1. Cell-window features

The unit of labeling is one grid cell in one time window. Counts are of distinct
aircraft, so one aircraft sending many messages cannot dominate a cell.

**Grid: 1° cells, 15-minute windows.** Finer grids leave most cells with too few
aircraft to judge. Share of usable readings that fall in cell-windows with at least
3 usable aircraft:

| Grid | Baltic | Black Sea | Punjab | Central Europe |
| --- | --- | --- | --- | --- |
| 0.5° / 10 min | 43% | 82% | 34% | 96% |
| 1.0° / 15 min (default) | 75% | 95% | 64% | 100% |
| 1.0° / 30 min | 89% | 97% | 82% | 100% |

A 1° cell is about 110 × 60 km, still far smaller than a jammer's footprint of hundreds
of km. Change it with `--cell-deg` and `--window-min`.

**A reading is usable** when it is an ADS-B position (or an MLAT fallback, below) with
quality fields reported in the last 30 s, at or above 3,000 ft.

| Feature | Definition | Why |
| --- | --- | --- |
| `n_ac_degraded` | aircraft with NIC < 7 in at least half their readings | lost position integrity |
| `n_ac_nic0` | aircraft with NIC 0 | integrity completely lost |
| `n_ac_nacp_low` | aircraft with NACp < 8 | lost position accuracy |
| `n_ac_mlat_fallback` | ADS-B aircraft positioned by ground MLAT instead | tracker stopped trusting the aircraft's GPS |
| `n_ac_jump` | consecutive ADS-B positions implying > 1,000 kt | impossible motion |
| `n_ac_speed_mismatch` | position change vs reported ground speed off by > 150 kt (pairs 5-30 s apart) | spoofed position, true speed |
| `n_ac_altdev` | GPS minus baro altitude > 1,000 ft from the local norm | spoofed or lost GPS altitude |
| `n_chronic` | aircraft judged to have faulty equipment (excluded from quality counts) | not interference |

### What the real data showed, and how it shaped the features

- **Faulty transponders.** Some aircraft report NIC 0 everywhere. In calm Central
  Europe, 15 such aircraft produced 53% of all degraded readings. An aircraft is
  marked chronic when, in at least 3 cell-windows where the other aircraft are
  healthy, it is degraded 80% of the time. It still counts as traffic, not as evidence.
- **MLAT fallback.** In the Baltic, 17% of positions from ADS-B-equipped aircraft came
  from MLAT (Central Europe: 0.8%). 99% of these follow the aircraft's own report of
  NIC 0: the tracker switches to ground multilateration when the aircraft's GPS is
  unusable. Early versions dropped these points as "not ADS-B", which hid the most
  strongly jammed aircraft. They now count as quality readings.
- **The normal GPS-baro difference depends on weather.** Its median that day was
  -275 ft in the Baltic and +625 ft in Central Europe, so altitude deviation is
  measured against healthy aircraft at the same hour and 5,000 ft band, and only when
  at least 5 healthy aircraft form that baseline.
- **Speed mismatch is a clean signal.** In Central Europe none of 738,000 point pairs
  5-30 s apart disagreed with the reported ground speed by more than 150 kt; in the
  Baltic and Punjab about 1 in 1,000 did.

## 2. Rules

Each rule votes interference (1), normal (0) or abstain (-1). Rules abstain when a
cell has too little evidence, so a quiet cell is never called normal.

| Rule | Family | Votes interference when | Votes normal when |
| --- | --- | --- | --- |
| `nic_cluster` | integrity | >= 3 usable, >= 2 degraded, share >= 50% | >= 3 usable, share <= 10% |
| `nic_zero` | integrity | >= 3 usable, >= 2 at NIC 0 and >= 30% | >= 3 usable, none degraded |
| `nacp_cluster` | integrity | >= 3 usable, >= 2 low NACp, share >= 50% | >= 3 usable, share <= 10% |
| `mlat_fallback` | integrity | >= 2 MLAT-fallback aircraft | never (one-sided) |
| `jumps` | motion | >= 2 aircraft with impossible jumps | never (one-sided) |
| `speed_mismatch` | motion | >= 2 aircraft with speed mismatch | never (one-sided) |
| `alt_mismatch` | motion | >= 3 with altitude pairs, >= 2 deviating, share >= 30% | >= 3 pairs, none deviating |

**Two families, two outputs.** Jamming removes integrity; spoofing keeps integrity
high but moves positions wrongly. A first version combined all rules into one
"interference" label: the label model learned "interference = lost integrity", gave
the motion rules no weight, and detected **0%** of synthetic spoofing. Now:

- `p_jamming`: label-model probability from the integrity rules.
- `spoof_votes`: number of motion rules that fire (0-3).
- `label`: `jamming` (p_jamming >= 0.5), `spoofing` (spoof_votes >= 1), `both`,
  `normal`, or `unknown` (no rule could judge).

## 3. Label model

`labelmodel.DawidSkene` is the Dawid & Skene (1979) model fitted by EM, the idea
behind data-programming tools such as Snorkel, in plain numpy. It learns each rule's
sensitivity and specificity from how the rules agree, with no ground truth.

Two design decisions, both found by testing on the real day:

1. **Abstentions of two-sided rules carry no evidence.** When abstentions were
   modelled, the model learned "the NIC rules abstain, so it is interference", because
   sparse cells were where MLAT fallback fired. Rules abstain because cells are
   sparse, and sparseness is not a symptom.
2. **One-sided rules are scored as fired / not fired.** Conditioned on voting, a rule
   that only ever says "yes" carries no information.

Learned on 2026-10-26 (`reports/labels/label_model_rules.csv`):

| Rule | Kind | Sensitivity | Specificity |
| --- | --- | --- | --- |
| nic_cluster | two-sided | 0.998 | 1.000 |
| nic_zero | two-sided | 0.998 | 1.000 |
| nacp_cluster | two-sided | 0.574 | 1.000 |
| mlat_fallback | one-sided | 0.753 | 0.989 |

`nic_cluster` and `nic_zero` are strongly correlated, which breaks the model's
independence assumption and makes it overconfident: almost no cell gets a probability
between 0.2 and 0.8. Calibration is a Phase 3 task.

## 4. Results on the real day

Cell-windows per label, 2026-10-26:

| Region | Jamming | Spoofing | Both | Normal | Unknown |
| --- | --- | --- | --- | --- | --- |
| Baltic | 869 | 1 | 3 | 1,009 | 2,708 |
| Black Sea | 139 | 12 | 5 | 630 | 710 |
| East Med | 30 | 2 | 0 | 1,546 | 1,547 |
| Punjab | 11 | 21 | 9 | 296 | 648 |
| Central Europe (control) | 15 | 1 | 1 | 2,591 | 225 |
| US Midwest (control) | 0 | 0 | 0 | 4,172 | 536 |

Of judged cell-windows, 46% are flagged in the Baltic against 0.65% in Central Europe
and 0% in the US Midwest. Baltic hotspots sit around Kaliningrad (the cells at
54°N 18°E and 53°N 22°E are flagged in 59 windows each) and toward the Estonia-Russia
border.

**Open questions for the gold set** (not conclusions):

- **Punjab spoofing candidates** come almost entirely from `alt_mismatch` (30 of 32),
  mostly at night (17:00-02:00 UTC), concentrated around 31°N 74°E. Weak baselines
  explain only 8% of the flagged readings. It could be spoofing near the border or a
  weather effect; the gold set and more days must decide.
- **Black Sea spoofing candidates** come from `speed_mismatch`, clustered near
  Istanbul (41°N 28-29°E), a busy approach area.
- `speed_mismatch` agrees with the other rules only 21% of the time: expected for a
  spoofing signal, but unverified.

## 5. Synthetic attacks

`gnssguard synth` injects attacks into a calm region-day and scores the labels.
Every event records its true location, period and type (`data/synthetic/.../events.json`).

- **Jamming:** a ground transmitter with a power range of 120-300 km and a mast of
  10-100 m. Aircraft are affected only within radio line of sight,
  `horizon_km = 4.12 × (√h_aircraft_m + √h_jammer_m)`, so high aircraft are hit from
  farther away. In the core, NIC and NACp go to 0, GPS altitude may disappear and 15%
  of messages are lost; in the outer ring the effect is partial and random.
- **Spoofing:** inside a 60-150 km area, reported positions drift toward a target
  80-200 km away at 150-400 kt, or jump straight to it. Ground speed stays true and NIC
  stays high in 70% of readings.

**Truth.** An aircraft counts as affected in a cell-window when at least half its
readings there were affected; a truth positive needs at least 2 affected aircraft.
Spoofing is placed at both true and reported positions, since its evidence appears
where the fake positions are. False alarms are counted only outside a one-cell,
one-window buffer around affected cells.

Central Europe, 2026-10-26, 4 jamming + 4 spoofing events per run, 5 seeds:

| Seed | Truth cell-windows | Jamming recall | Spoofing recall | False alarms away from events |
| --- | --- | --- | --- | --- |
| 0 | 566 | 0.710 | 0.650 | 1.5% |
| 1 | 422 | 0.755 | 0.898 | 0.9% |
| 2 | 423 | 0.815 | 0.971 | 1.0% |
| 3 | 373 | 0.759 | 0.541 | 1.0% |
| 4 | 696 | 0.783 | 0.766 | 1.1% |
| **Mean** | | **0.764** | **0.765** | **1.1%** |

The same day with no attacks has 0.65% flagged cells, so the rules add little noise.

**Where the rules miss**, from the diagnostics:

- Jamming misses sit at the edge of the footprint, where about half the aircraft in a
  cell are affected (median 50%, against 97% in detected cells). The rules need 50%
  degraded; the Phase 3 model should do better here.
- Spoofing recall varies a lot between seeds (0.54-0.97): slow drifts in sparse cells
  are the hardest case.

**Limits of the simulation.** Results are an upper bound: the attacks follow the same
assumptions the rules were written with. MLAT fallback is not simulated by default
(`--p-mlat` turns it on), so `mlat_fallback` cannot help on synthetic data. Real
jammers are often directional, intermittent or mobile; these are not.

## 6. Official events

`configs/events.csv` holds reported events, one row each:

```
event_id,type,lat_min,lat_max,lon_min,lon_max,start_utc,end_utc,source_url,notes
```

`type` is `jamming`, `spoofing` or `unknown`; times are UTC ISO 8601; every row needs a
`source_url` (aviation authority bulletins, NOTAMs, regulator or airline statements).
Only add events that fall inside days you have ingested. `gnssguard label` marks
matching cell-windows and writes `reports/labels/event_recall.csv`. Events check the
rules; they are never used to tune thresholds. The file ships empty: no event has
been entered yet.

## 7. Gold set

`gnssguard gold-sample` picks cell-windows with at least 3 usable aircraft, stratified
by region and by stratum (likely normal, uncertain, likely jamming, spoofing
candidate), topping up at random when a stratum is small. The CSV leaves out rule
votes and probabilities on purpose.

To label:

1. `gnssguard map --region <r> --date <d> --gold labels/gold.csv`, open the HTML. The
   coastlines and borders are built into the page and work offline; "Street map
   (online)" adds a detailed CARTO background when you have internet.
2. Gold cells have black outlines with their ID. Use the "Degraded share" view and
   the hover details. Do not use the Label view while labeling.
3. Fill `label` with `jamming`, `spoofing`, `both`, `normal` or `unsure`, plus your
   initials in `labeler` and anything notable in `notes`.
4. A week later, relabel 10% blind and record agreement.
5. Keep the gold days out of training in Phase 3. Commit `labels/gold.csv`.

## 8. Not done yet

- A satellite-geometry (DOP) rule from the GPS almanac, to rule out poor accuracy
  caused by satellite positions.
- A rule using neighboring cells; this goes into the Phase 3 features.
- Calibrated probabilities (the label model is overconfident).
