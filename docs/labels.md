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
gnssguard gold-tool                                # labeling page: reports/gold_tool.html
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
between 0.2 and 0.8. The detector's published probabilities are calibrated separately
(docs/model.md, section 6).

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

**What the 300 gold labels said** (docs/model.md, section 1). Every spoofing candidate
in the gold set was labeled spoofing or both (Punjab 20 of 20, Black Sea 16 of 16), except
two East Med cells labeled unsure; cells where `alt_mismatch` fired: 25 of 25,
`speed_mismatch`: 16 of 18. This is weaker evidence than it looks. The labeler sees the
same quantities the rules use, so the gold set confirms that the evidence is there and
looks like spoofing in context, not that it was spoofing: a weather-driven GPS-baro
difference over Punjab would look the same. An independent source (pilot reports or
NOTAMs naming spoofing) is still needed. An MLAT vs ADS-B position check was tried and
is not usable here: there are no MLAT positions over the Black Sea or Punjab
(docs/evaluation_protocol.md).

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
  degraded. The Phase 3 model does on synthetic attacks (edge recall 0.92-1.00 vs
  0.14-0.49), though not yet shown on real data (docs/model.md).
- Spoofing recall varies a lot between seeds (0.54-0.97): slow drifts in sparse cells
  are the hardest case.

**Limits of the simulation.** Results are an upper bound: the attacks follow the same
assumptions the rules were written with. MLAT fallback is not simulated by default
(`--p-mlat` turns it on), so `mlat_fallback` cannot help on synthetic data. Real
jammers are often directional, intermittent or mobile; these are not.

## 6. Event catalog

`configs/events.csv` holds reported events, one row each:

```
event_id,type,lat_min,lat_max,lon_min,lon_max,start_utc,end_utc,source_url,notes,evidence_level,region
```

`type` is `jamming`, `spoofing` or `unknown`; times are UTC ISO 8601; every row needs a
`source_url`. Boxes and times are only as precise as the source; the notes say what was
inferred. Events check the system; they are never used to train it or tune thresholds.

**Evidence levels** (how far the report is independent of this project's input):

| Level | Meaning | Examples in the catalog |
| --- | --- | --- |
| `observed_anomaly` | seen in ADS-B data by an outside analyst; not independent of our input | Stanford GPS Lab findings near Kaliningrad |
| `suspected_mechanism` | an authority, airline or crew says it happened, without a measurement | UK and Spanish ministers' jets, Finnair at Tartu, Delhi diversions |
| `independently_corroborated` | a measurement not from ADS-B: ground spectrum monitoring, direction finding, a ground GNSS receiver | Estonian and Lithuanian monitoring (ITU letter), Gdynia ground receiver |
| `unresolved` | reported, cause or occurrence unclear | |

`region` names the study region the box lies in; empty means outside all regions
(listed, not checked). The first 13 events (4 Nov 2026) cover 2024-2025; their days are
downloaded on purpose (`gnssguard ingest --date ...`), then scored with
`gnssguard event-check`, under the rule fixed in docs/evaluation_protocol.md ("Event
check 1"). `gnssguard label` also marks matching cell-windows
(`reports/labels/event_recall.csv`).

Adding an event: one row with a new id, the source link, a box and the UTC period as
the source states them, the evidence level, and in the notes what was inferred. If its
day is not stored yet, ingest it before the next event check.

### Official advisories

Standing warnings (NOTAMs, bulletins, pilot reports) cover whole airspaces for months,
so they cannot label single cell-windows. `configs/advisories.csv` holds them as rough
boxes, and `gnssguard label` writes `reports/labels/advisory_check.csv`: within each
study region, the flagged share inside the warned box against outside it, over the
days the warning was in force.

Shipped entries (all found via [Safe Airspace](https://safeairspace.net/summary/) on
29 Oct 2026; original notices not checked; boxes approximate):

| ID | Warning | In force since | Box |
| --- | --- | --- | --- |
| TR1 | Turkey NOTAM LTAA A4412/26 | 2026-10-02 | Turkish part of the Black Sea region |
| CY1 | Cyprus NOTAM LCCC A0799/26 | 2026-07-28 | Nicosia FIR, rough |
| LB1 | Lebanon NOTAM OLBB A0191/26 (jamming and spoofing) | 2026-08-15 | Beirut FIR, rough |
| PK1 | Pilot reports of spoofing near Lahore (not a NOTAM) | 2023-12 | Pakistani side near Lahore |
| RU1-3 | EASA CZIB 2025-01 R3, western Russia | 2025-01-09 | Kaliningrad; NW Russia slice; Krasnodar coast |

On 2026-10-26: inside vs outside flagged share was 14.4% vs 0.8% for Lebanon (17×),
21.9% vs 7.2% near Lahore (3×), and 100% vs 45% for Kaliningrad (2.2×). Turkey (0.58×)
and Cyprus (0.49×) came out lower inside: both are broad FIR-wide notices, and
"outside" is the rest of the same study region, which holds hotter areas (the northern
Black Sea; Lebanon and Israel). A ratio below 1 does not contradict a warning. The
Krasnodar box has no judgeable cells: almost no civil traffic is tracked there.

## 7. Gold set

Status: all 300 cells labeled by hand on 31 Oct 2026 (`labels/gold.csv`); results in
docs/model.md, section 1.

`gnssguard gold-sample` picks cell-windows with at least 3 usable aircraft, stratified
by region and by stratum (likely normal, uncertain, likely jamming, spoofing
candidate), topping up at random when a stratum is small. The CSV leaves out rule
votes and probabilities on purpose.

To label, use the labeling tool:

```bash
gnssguard gold-tool        # writes reports/gold_tool.html
```

1. Open `reports/gold_tool.html` in your browser and enter your initials.
2. Each screen shows one gold cell (black outline) with its 5 × 5 neighbourhood for the
   window before, the window itself and the window after, and the cell's numbers. No
   rule votes or model probabilities are shown.
3. Press `1`-`5` for jamming, spoofing, both, normal, unsure; arrows move. It jumps to
   the next unlabeled cell. Judge from the context the rules cannot see: neighbours
   and the windows before and after. Use `unsure` freely.
4. Labels save in the browser as you go. To finish (or stop for the day), press
   **Download gold.csv** and replace `labels/gold.csv` with it; all other columns stay
   exactly as they were. Commit it.
5. A week later, relabel 10% blind and record agreement.
6. Keep the gold days out of training in Phase 3.

Edit `labels/gold.csv` in a text editor (VS Code), not Excel: Excel rewrites the
`window` timestamps and the rows stop matching.

The map viewer (`gnssguard map --region <r> --date <d> --gold labels/gold.csv`) also
outlines gold cells. Its coastlines and borders are built in and work offline; "Street
map (online)" adds a CARTO background when you have internet.

## 7b. Holdout day: 1 Nov 2026

A second, smaller gold set on a day never used for anything, for the clean test declared
in docs/evaluation_protocol.md ("Holdout test 2"). Drawn with the frozen published model
by `gnssguard holdout-sample` before any label existed:

* `labels/gold_2026-11-01.csv`: the labeling file (150 cell-windows, same format as the
  first gold set);
* `labels/holdout_2026-11-01_strata.csv`: strata, weights and the frozen model's and
  rules' answers. Committed before labeling so anyone can check the predictions were not
  changed afterwards. **Do not open it until labeling is finished.**

Label with `reports/gold_tool_2026-11-01.html` exactly as before (section 7), then:

```bash
gnssguard holdout-eval --gold labels/gold_2026-11-01.csv --strata labels/holdout_2026-11-01_strata.csv
```

## 7c. Review set 1: false alarms (27 Oct-3 Nov 2026)

100 cell-windows from 7 days no decision has used: 50 model flags in calm Central
Europe, 30 model flags in the hotspot regions, 20 calm cells the model calls normal
(docs/evaluation_protocol.md, "Review set 1"). Label them exactly like the gold sets,
with the same tool; the groups are in `labels/review_1_groups.csv`, which the labeler
does not open before labeling is finished.

```bash
gnssguard review-sample --dates 2026-10-27,2026-10-28,2026-10-29,2026-10-30,2026-10-31,2026-11-02,2026-11-03 \
    --out labels/review_1.csv --groups labels/review_1_groups.csv
gnssguard gold-tool --gold labels/review_1.csv --out reports/review_tool.html
gnssguard review-eval --gold labels/review_1.csv --groups labels/review_1_groups.csv --calm-share <share printed by review-sample>
```

## 7d. Sealed set: calibration set and sealed test set

Declared in docs/evaluation_protocol.md ("Sealed set"), with the published model frozen.

* **Calibration set** (15-21 Sep 2026, 150 cells, by score band): label it like the
  others; it is used only to fit the published score to human labels.
* **Sealed test set** (5-18 Nov 2026, 250 cells): drawn on 19 Nov 2026 or later, labeled
  once and scored once at the Phase 1 gate. Nothing is tuned on it.

```bash
gnssguard sealed-sample --kind calibration --dates 2026-09-15,2026-09-16,2026-09-17,2026-09-18,2026-09-19,2026-09-20,2026-09-21 \
    --out labels/calibration_1.csv --groups labels/calibration_1_groups.csv
gnssguard sealed-sample --kind test --dates <5-18 Nov 2026> \
    --out labels/sealed_test.csv --groups labels/sealed_test_groups.csv
gnssguard gold-tool --gold labels/calibration_1.csv --out reports/calibration_tool_1.html
```

As with the other sets, the groups files are committed with the samples, before any
label, and the labeler does not open them.

## 8. Not done yet

- A satellite-geometry (DOP) rule from the GPS almanac, to rule out poor accuracy
  caused by satellite positions.
- Done since: neighbouring cells became detector features, and the detector's
  probabilities are calibrated (docs/model.md, sections 4 and 6). The label model
  itself is still overconfident.

## Disagreement set (8 Nov 2026)

Cells where the model and the rules disagree, to find out which one is right
(evaluation protocol, "Disagreement set 1"):

```
gnssguard disagreement-sample --dates <15 Sep to 4 Nov 2026, comma-separated> \
    --out labels/disagreement_1.csv --groups labels/disagreement_1_groups.csv
gnssguard gold-tool --gold labels/disagreement_1.csv --out reports/disagreement_tool.html
gnssguard disagreement-eval
```

Commit the labeling file and the groups file before labeling; do not open the groups
file until the labels are committed.

Set 1b (declared 8 Nov, after set 1): only the D4 group, outside the Baltic:

```
gnssguard disagreement-sample --only D4_rules_unsure \
    --regions east_med,black_sea,punjab,central_europe --dates <15 Sep to 4 Nov 2026> \
    --seed 20261109 --out labels/disagreement_1b.csv --groups labels/disagreement_1b_groups.csv
gnssguard gold-tool --gold labels/disagreement_1b.csv --out reports/disagreement_1b_tool.html
gnssguard disagreement-eval --gold labels/disagreement_1b.csv --groups labels/disagreement_1b_groups.csv
```

Samplers spread each group over region-days, region by region in turn with random days
(since 8 Nov 2026; evaluation protocol, Correction 2). `--key-order alphabetical`
reproduces the sets drawn before then (calibration_1, review_1, disagreement_1).
