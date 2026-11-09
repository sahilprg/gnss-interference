# Evaluation protocol

Written on 1 Nov 2026, **before** the improvement round it governs, so that no result
in that round can be chosen by looking at the test labels. Changes to this file after
that date are listed at the bottom.

## Why a protocol

The 300 hand labels from 26 Oct have already been looked at once: the disagreement
analysis found the synthetic-background bug (docs/model.md, section 1). Using them
again to choose between model versions would make the gold numbers optimistic in a way
no reader could detect. No second labeled day is available. So every decision below is
made on data that carries no hand labels, by rules fixed in advance, and the gold set is
scored once, at the end, for the final model and for the baseline side by side.

## Baseline

The model trained on 1 Nov 2026 (commit f4799f5, "Final gold evaluation"): gold
agreement 96.8%, saved as the baseline before any change.

## Selection data (used to accept or reject a change)

* **Rolling days 20-25 Oct only.** Each day's model is trained on earlier days and
  tested on that day (`gnssguard rolling`). 26 Oct, the gold day, is excluded from
  every selection decision.
* **Synthetic attacks** (exact truth), three profiles: standard, hard, stress.
  Injected into calm Central Europe as before, and, new in this round, into the Black
  Sea, a hotspot region never used for synthetic training, so that attacks are also
  tested on a background that looks like the real problem. (Its false-alarm figure
  includes real interference in the background; it counts against every model alike.)
* **Real false alarms:** flagged share of cells with 3+ usable aircraft in calm Central
  Europe.

## Acceptance rules (all must hold, means over the six selection days)

1. **No lost recall:** synthetic jamming and spoofing recall do not fall by more than
   0.01 in any profile.
2. **No new false alarms on synthetic data:** the false-alarm rate away from events
   rises by at most 0.2 percentage points in any profile.
3. **No new false alarms on real data:** the calm Central Europe flagged share rises by
   at most 0.2 points on average and never exceeds the rules by more than 0.5 points on
   any single day.
4. **A real gain:** at least one recall figure rises by 0.02 or more, **or** the change
   has a stated purpose other than the numbers (reproducibility, a new independent
   evidence source, a safer output) and does not break rules 1-3.

A change that fails is reverted and reported as tried and rejected, with its numbers.

## What is reported at the end

* All selection-data results for the baseline and the final model.
* Gold agreement for both, computed once, with Wilson 95% intervals, plus a
  population-weighted estimate (the gold sample was stratified, so its raw agreement
  over-represents uncertain cells; weighting by stratum size estimates agreement over
  all judged cells of the day).
* Every change tried, accepted or rejected.

## Known limits of this protocol

* Synthetic recall can only show what the injection model can express.
* Without new labels, a real-data recall gain cannot be proven; the gold check at the
  end is the only real-data recall measurement, and it is not fully independent.

## Candidate changes, declared before any result

* **A. Borderline negatives become unlabeled.** A real cell the weak labels call "not
  jamming" although it has at least one degraded aircraft or a hot neighbour (a
  neighbouring cell with half or more of its aircraft degraded) is left out of the
  jamming target instead of counted as clean, in real rows and in the synthetic
  background alike. About a quarter of real negatives qualify.
* **B. Synthetic attacks also injected into East Med** for training (2 seeds per
  training day, as for Central Europe), for a second, different traffic background.
* **C. A and B together.**
* **D. MLAT vs ADS-B position check** as independent spoofing evidence, if the data
  supports it.
* **E. Two-jammer localization.** When the one-jammer fit scores below 0.85, fit a
  second jammer to the degraded aircraft the first does not explain, and keep two
  sources if that raises the fit score by 0.03 or more. Judged on synthetic events on
  22-25 Oct (never the test day): accepted if the median location error over all events
  falls and the 90% uncertainty coverage, refitted the same way as before, is not worse.
* **F. Real-data localization check (no change to the model).** Confident estimates
  for real Baltic incidents are compared with Kaliningrad, the source most often named
  in public reporting; reported as a sanity check, not as attribution.

## Changes to this file

* 1 Nov 2026, before any result: "quiet hours of a hotspot region" replaced by "the
  Black Sea" (concrete choice of the new test background).

## Results of the 1 Nov 2026 round

Applied mechanically by `scripts/protocol_decide.py` (rolling days 20-25 Oct, model
minus baseline; raw results in `reports/protocol/`, not committed).

| Change | Rule 1: worst recall change | Rule 2: worst synthetic false-alarm change | Rule 3: calm Central Europe, mean change / worst day above rules | Rule 4: best recall gain | Decision |
| --- | --- | --- | --- | --- | --- |
| A. Borderline negatives unlabeled | 0.000 | **+0.442** | **+3.7 / 5.1 points** | +0.116 | rejected |
| B. East Med training attacks | -0.004 | 0.000 | -0.01 / 0.04 points | +0.058 | **accepted** |
| C. A and B | +0.001 | **+0.346** | **+3.9 / 5.2 points** | +0.116 | rejected |

* **D. MLAT vs ADS-B check:** not usable. The Black Sea and Punjab data hold no MLAT
  positions at all. Where both exist, MLAT and ADS-B positions of the same aircraft
  within 10 s differ by a median of 1 km in the Baltic but 41 km in East Med and 116 km
  in calm Central Europe: MLAT here appears exactly when tracking is poor, so a
  disagreement is not evidence of spoofing.
* **E. Two-jammer localization:** rejected. On 168 synthetic events (22-25 Oct) it changed
  4; median error 24.7 to 23.6 km, but coverage of the refitted uncertainty fell from
  91.6% to 90.7% (rule: not worse). It never fired on paired stress jammers.
* **F. Real Baltic check:** reported in docs/model.md, section 5.

Final model = baseline + B. Gold set, scored once at the end (253 decided cells):

| Model | Interference vs normal | Weighted to the day | Exact label |
| --- | --- | --- | --- |
| Baseline (1 Nov, before this round) | 96.8% | 94.7% | 91.7% |
| Final | 96.8% | 94.7% | 91.7% |

The round did not change agreement with the hand labels; its gains are on the new
hotspot test background (spoofing), and in trust: a declared protocol, pinned versions,
a weighted gold estimate, and a record of what failed.

## Holdout test 2: 1 Nov 2026 (declared 3 Nov 2026, before sampling and labeling)

The 26 Oct gold set is spent (it found a bug and was scored once). This second test is
the first clean real-data comparison of model and rules. Nothing below may change after
the labels exist; if the model, the rules or any threshold is changed because of these
labels, the holdout is spent too, and that is reported.

**Frozen predictors.** The published model `models/detector` at commit 05709be (trained
29 Sep-24 Oct, validated 25 Oct), and the Phase 2 rules with the label model stored in
that model. 1 Nov was never used for training, validation or any decision.

**Sample.** 150 cell-windows from 1 Nov, five regions, drawn with a fixed seed by
`gnssguard holdout-sample` before any label exists, in four strata:

| Stratum | Cells | What it tests |
| --- | --- | --- |
| S1 disagree | all, up to 40 | 3+ usable aircraft; model and rules disagree on interference vs normal |
| S2 sparse | 25 | 1-2 usable aircraft; the model flags interference (the rules cannot judge) |
| S3 borderline | 35 | 3+ aircraft; both say normal, but 1+ degraded aircraft or a hot neighbour |
| S4 random | the rest | 3+ aircraft; random, stratified by region |

The strata are stored apart from the labeling file, and the labeler sees neither the
strata nor any model or rule output, only the data panels, as for the first gold set.

**Primary question.** On decided S1 cells (unsure excluded), does the human side with
the model or with the rules more often? Exact two-sided binomial test against 50%:
p < 0.05 means one predictor is better on real disagreements. With fewer than 10
decided S1 cells the answer is "inconclusive".

**Secondary results**, reported whatever they show:

* S3: share of borderline cells the human calls interference (misses by both).
* S2: precision of the model's flags on sparse cells.
* S4, weighted by stratum size on the day: agreement of rules and model with the human
  over all judged cells, with Wilson 95% intervals; exact-label agreement.

**Amendment, 3 Nov 2026, after drawing the sample and before any label.** The draw found
that on 1 Nov the model and the rules disagree on only 1 of 6,542 judged cells (S1 = 1;
the first version of this text said 1,761, a miscount corrected on 3 Nov),
so the primary question above cannot be answered on this day; it is reported as
"inconclusive" as declared. The one place the two really differ is cells with 1-2
aircraft, which the model judges and the rules skip (246 such cells flagged by the
model). The sample is therefore redrawn with the same seed and S2 raised from 25 to 50
(S4 falls to 64), and a second primary question is added:

* **Primary 2: are the model's flags on sparse cells right?** Precision of S2 (human
  calls the cell jamming, spoofing or both; unsure excluded), Wilson 95% interval:
  "supported" if the lower bound is above 50%, "not supported" if the upper bound is
  below 50%, otherwise "inconclusive". If supported, sparse cells may be shown on the
  map with their label instead of "low coverage"; if not, they stay as they are.

### Result (labeled 3 Nov 2026, scored once)

The 150 labels (`labels/gold_2026-11-01.csv`, one labeler, 36 unsure) were committed
after the strata file (commit 9311d0d, before labeling) and scored once with
`gnssguard holdout-eval`. Nothing was changed because of them; the one later code
change, below, is a correction to the scoring and both numbers are given.

| Question | Result | Verdict |
| --- | --- | --- |
| Primary: model or rules on real disagreements | 1 decided S1 cell, the human sided with the model | inconclusive, as declared |
| **Primary 2: model flags on 1-2 aircraft cells** | **47 of 47 decided right** (95% interval 92-100%) | **supported** |
| S3: borderline cells both called normal | 7 of 18 decided were interference (20-61%) | reported |
| Day agreement with the human, 3+ aircraft | rules 92.0%, model 92.0% (roughly 87-96%) | the same |
| Exact-label agreement on the sample | model 92.1%, rules 82.5% | (sample, not day) |
| S4 random cells | rules and model 48 of 48 decided | |

What it means:

* **Sparse cells can be shown.** The model's flags on cells with 1-2 aircraft were right
  every time the labeler could decide (47 of 50; 3 unsure). As declared, the map now
  shows those flags, marked as based on 1-2 aircraft (docs/product.md). The evidence
  is mostly Baltic (46 of the 50; 4 Black Sea), because that is where the model makes
  these flags (229 of 246 on 1 Nov); elsewhere they are rare and less tested.
  Sparse cells the model calls normal stay "low coverage": nothing tested them.
* **Both miss weak interference.** Among cells with 3+ aircraft that both called normal
  but that showed some evidence (1,347 of 6,542 judged cells), the labeler called 7 of
  18 decided cells interference, 6 of them in the Black Sea. Estimated for the whole day,
  both predictors miss about 4-13% of judged cells, mostly at the edges of Black Sea
  interference. The flagged shares on the map are therefore lower bounds. Fixing this
  needs real labels used for training, not only testing.
* **The model does not beat the rules on dense real cells**, confirming the first gold
  set: the two agree on all but 1 of 6,542 judged cells, so they score the same.
* The day agreement (92%) is within the uncertainty of the first gold set's 94.7%.

**Corrections found while reporting, 3 Nov 2026** (not changes to predictors):

* *Day weighting.* The first run gave 94.6% (rules) and 94.6% (model) because it
  weighted single cells, so groups with many unsure cells (S3: 17 of 35 unsure) lost
  weight. The declared method weights each sampling group by its size on the day; the
  fixed code (`holdout_eval`, with a test) gives 92.0% for both. The comparison between
  rules and model is unchanged.
* *The rules on sparse cells.* The S2 description above says the rules cannot judge
  cells with 1-2 aircraft. In fact the rules' label model gives a label to most of them
  (it flagged 214 of the 246); the map simply never showed them. Of the sampled S2 cells
  the rules flagged 38 (37 decided, all right); the other 12, which only the model flags,
  were 10 decided, all jamming. So both predictors are right on these cells; what is
  new is publishing them.

## Round 3: Phase 0 data-handling fixes (declared 4 Nov 2026, before any result)

The improvement plan of 3 Nov lists data-handling findings. The check of 4 Nov
(docs/data.md) showed that readsb's `stale` flag does not mark an old position: it marks
the first position after a gap of 20 s or more without one (median gap 30 s on 1 Nov;
non-flagged points follow a gap of 20 s or more only 0.4-0.7% of the time). The
position and its NIC are fresh, so these points stay usable. What a gap does break is
comparing that point with the one before it: on 1 Nov in calm Central Europe, points
after a gap show impossible jumps 200 times and speed mismatches 25 times more often than
other points, and they carry about half of that region's jumps and 10% of its speed
mismatches. Over the Black Sea and the Baltic they carry 36-45% of all speed
mismatches, the main spoofing evidence; how much of that is real cannot be told
without independent evidence.

**Change S (one change, tested alone):**

* A point counts as "after a gap" if readsb flags it stale or if it follows the same
  aircraft's previous point by 20 s or more in our data (the second part makes the
  synthetic attacks, whose dropped points leave gaps without a flag, behave like real
  data).
* Impossible jumps and speed mismatches are not computed on a pair ending at a point
  after a gap or at the start of a new leg (`new_leg`). Source-switch jumps are not
  computed across a new leg.
* Nothing else changes: the same points are usable, and NIC, NACp and altitude
  evidence are unchanged.

**Baseline:** the published model's configuration (round 2, change B), rebuilt and
rolled with the current code, so that only change S differs.

**Decision:** the acceptance rules above, on 20-25 Oct, with rule 4 met by the stated
purpose (correct data handling) if rules 1-3 hold. Also reported, whatever they show:
the real flag-rate change per region and predictor on the selection days, and the change
in Black Sea spoofing flags. The hand labels of 26 Oct and 1 Nov are not looked at for
this decision; if S is accepted, both holdout days are rescored once, with the new and
the old model side by side, and reported.

### Result of round 3 (4 Nov 2026): change S rejected

Run with `gnssguard rolling` on 20-25 Oct, the baseline on the unchanged code (it
reproduced round 2's change B exactly), and decided by `scripts/protocol_decide.py`
(`reports/protocol3/`).

| Rule | Result | |
| --- | --- | --- |
| 1. No lost recall (at most -0.01) | Black Sea hard spoofing -0.028, stress -0.013 | **fails** |
| 2. No new synthetic false alarms | at most -0.001 | holds |
| 3. No new real false alarms | calm Central Europe -0.01 points | holds |
| 4. Gain or stated purpose | purpose given, but needs rules 1-3 | not met |

Real flagged shares changed by at most 0.16 points in any region (Black Sea 18.6% to
18.5% of judged cells). Impossible-jump and speed-mismatch counts fell by 10-50% per
region-day, while usable aircraft did not change.

S is reverted: features are again identical to the baseline (checked on a stored day),
and the published model is unchanged. Kept, because they change no result: the
`stale` and `new_leg` columns are now loaded (`read_points`) and recorded as the point
flag `after_gap`, and the decoder and docs/data.md describe the flag correctly.

What it means: on synthetic attacks over a busy background, pairs across a reception
gap carry part of the spoofing evidence the model uses, which is also how injected
spoofing looks (its dropped points leave gaps). In calm traffic the same pairs produce
most false jumps, but the model already discounts them: calm false alarms did not move.
Whether real spoofing looks the same cannot be decided without independent evidence
(Phase 1). No variant of S is tried on the same selection days.

## Event check 1: public reports, 2024-2025 (declared 4 Nov 2026, before any event day was scored)

The first events in `configs/events.csv` (13 rows, 12 in study regions) come from
public sources, each with an evidence level (docs/labels.md). Their days were
downloaded after the catalog was written; nothing below was chosen by looking at them.

**Predictors:** the published model (`models/detector`, trained 29 Sep-24 Oct 2026) and
the Phase 2 rules with its label model, unchanged. The events are 1-2.5 years older
than the training data, with different receiver coverage: this is a test of how well
the system carries over in time, and not used for any change.

**Rule, per event and predictor:** take the cell-windows of the event's region that
overlap its box and period and have 3 or more usable aircraft. A cell-window matches if
its label fits the event type (jamming: jamming or both; spoofing: spoofing or both;
unknown: any interference).

* **found:** 3 or more matching cell-windows;
* **not found:** 10 or more judged cell-windows and fewer than 3 matching;
* **too few cells** otherwise (also when the region has no data that day).

**Reported:** the verdict and matching share for every event; found / (found + not
found) per predictor and per evidence level, with Wilson 95% intervals; for spoofing
events separately, because spoofing is the weaker detector.

**Placebo (calm control):** on each event day, the flagged share of judged cells in
calm Central Europe for both predictors, which should stay in its usual 0.4-1.3%. A
high share there would mean the old days are read wrongly altogether.

**Known limit, stated now:** around Kaliningrad interference is close to constant, so a
"found" there shows that the system sees what was reported, not that it can tell
reported days from others. Telling them apart needs reviewed negative days (Phase 1,
sealed set).

### Result of event check 1 (4 Nov 2026, scored once)

`gnssguard event-check` on the 14 event days (all downloads complete, 0 decode errors).
Model and rules gave the same verdict on every event.

| Event | Type | Evidence | Judged | Matching (model) | Verdict |
| --- | --- | --- | --- | --- | --- |
| EV01 UK minister's jet, Kaliningrad, 13 Mar 2024 | jamming | suspected mechanism | 48 | 6 (12%) | found |
| EV02 Finnair at Tartu, 25 Apr 2024 | jamming | suspected mechanism | 0 | 0 | too few cells |
| EV03 Lithuanian monitoring, 16 Oct 2024 | unknown | corroborated | 711 | 101 (14%) | found |
| EV04 Lithuanian monitoring, 4 Nov 2024 | unknown | corroborated | 683 | 47 (7%) | found |
| EV05 first Kaliningrad spoofing, 9 Nov 2024 | spoofing | observed anomaly | 69 | 1 (1%) | **not found** |
| EV06 wideband emission, 27 Jan 2025 | jamming | corroborated | 371 | 196 (53%) | found |
| EV07 Narva monitoring, 7-9 Feb 2025 | unknown | corroborated | 12 | 12 (100%) | found |
| EV08 Kaliningrad 250 km jamming, 5 Mar 2025 | jamming | observed anomaly | 274 | 39 (14%) | found |
| EV09 Kaliningrad spoofing, 9 Mar 2025 | spoofing | observed anomaly | 13 | 5 (38%) | found |
| EV10 Gdynia ground receiver, 18 May 2025 | jamming | corroborated | 2 | 2 | too few cells |
| EV11 Spanish minister's jet, 24 Sep 2025 | jamming | suspected mechanism | 271 | 31 (11%) | found |
| EV12 Plovdiv, Bulgaria, 31 Aug 2025 | jamming | suspected mechanism | - | - | not checked (no region) |
| EV13 Delhi spoofing, 4 Nov 2025 | spoofing | suspected mechanism | 251 | 16 (6%; rules 15) | found |

| Group | Found / decided (model = rules) | 95% interval |
| --- | --- | --- |
| All events | 9 / 10 | 60-98% |
| Independently corroborated | 4 / 4 | 51-100% |
| Spoofing events | 2 / 3 | 21-94% |

Calm Central Europe on the event days: 0.04-0.83% flagged on 12 days, 1.27% on 18 May
2025 and **1.82% on 9 Nov 2024** (model; rules 1.78%), above the 0.4-1.3% of the study
days but below the 3% alarm.

What it means:

* The system flags what independent ground monitoring measured on every checkable
  corroborated day (4 of 4), on data 1-2 years older than its training data. With 4
  events the interval is wide (51-100%), and around Kaliningrad interference is near
  constant, so this shows the system sees reported interference, not that it tells
  reported days from others (stated in advance).
* **Delhi, 4 Nov 2025:** never trained on, outside every study region: 16 of 251 judged
  cell-windows flagged as spoofing, all between 10:00 and 14:00 UTC (15:30-19:30 IST);
  none the rest of the day. The reported diversions were that night (local time): the
  timing is close but not shown to match. This is the first India check, and worth a
  Delhi region once its coverage is checked (Phase 2).
* **The first Kaliningrad spoofing (9 Nov 2024) was missed**: 1 of 69 judged cells. The
  spoofing detector is the weak part, as on the synthetic Black Sea tests.
* Coverage limits what can be checked: Tartu (Apr 2024) had no judged cells, and adsb.lol
  had no Punjab data on any of these days, so India-Pakistan border events before 2026
  cannot be checked from this archive.
* Model and rules agree on every event, as on both hand-labeled days: the model adds
  nothing over the rules on real data yet. Showing a difference needs independent labels
  (Phase 1, sealed set).

## Review set 1: false alarms (declared 4 Nov 2026, before sampling and labeling)

The event check counts what the system finds; it cannot count false alarms. This set
does: hand review of the cells the system flags where little or no interference is
expected, plus control cells.

**Days:** 27-31 Oct and 2-3 Nov 2026 (7 days). None was used for training (29 Sep-24
Oct), validation (25 Oct), selection (20-25 Oct) or hand labels (26 Oct, 1 Nov).

**Frozen predictors:** the published model `models/detector` and the rules with its
label model, as in the event check.

**Sample:** 100 cell-windows with 3+ usable aircraft, drawn with a fixed seed by
`gnssguard review-sample` before any label exists, in three groups:

| Group | Cells | What it answers |
| --- | --- | --- |
| R1 calm flags | 50 | model flags in calm Central Europe, spread over the 7 days: are they real? |
| R2 hotspot flags | 30 | model flags in the four hotspot regions, spread by region: how often is a flag right where interference is common? |
| R3 calm normal | 20 | random cells the model calls normal in Central Europe: does the labeler also see nothing there? |

If a group has fewer cells than its size, all are taken and the rest goes to R3. The
groups, weights and predictions are stored apart from the labeling file; the labeler
sees only the data panels, as before.

**Primary question: what share of calm-region flags are false alarms?** Share of
decided R1 cells the labeler calls normal, with a Wilson 95% interval. The real calm
false-alarm rate is then estimated as the calm flagged share on these days times that
share (with its interval). Reported whatever it shows.

**Secondary:** R2 precision (share the labeler calls interference), weighted by region
size; R3 share called normal; for R1 and R2, type agreement (jamming vs spoofing).
Unsure cells are counted and shown.

**Known limit:** the labeler reads the same ADS-B evidence as the predictors, so this
measures agreement with a careful human reading, not physical truth.

**Sample drawn 4 Nov 2026, before any label** (seed 20261104, published model): R1 50 of
151 calm flags, R2 30 of 5,281 hotspot flags, R3 20 of 18,011 calm normal cells. The
calm flagged share on these 7 days is 0.83% (model), the value used in the primary
estimate. Labeling file `labels/review_1.csv`; groups and predictions
`labels/review_1_groups.csv`, committed before labeling.

### Result of review set 1 (labeled 4 Nov 2026, scored once)

Labels committed at 16:10 (commit da0e383), after the sample and predictions at 15:35
(298ede4); 100 labeled, 8 unsure. Scored with `gnssguard review-eval`.

| Question | Result |
| --- | --- |
| **Primary: calm-region flags the labeler calls normal** | **6 of 46 decided: 13% (95% interval 6-26%)**; 4 unsure |
| Estimated real false-alarm rate, calm Central Europe | 0.83% flagged x 13% = **0.11% of judged cells (0.05-0.21%)** |
| Hotspot flags the labeler confirms (R2) | 26 of 26 (87-100%); 4 unsure |
| Calm cells the model calls normal, also normal to the labeler (R3) | 20 of 20 (84-100%) |
| Same interference type, on confirmed flags | 63 of 66 |
| Rules agree with the model (interference or not), all 100 | 98% |

What it means:

* **Most flags in the "calm" control region are not false alarms to a careful human
  reading.** 40 of the 48 jamming flags there were called jamming: on average 2.7 of
  9 usable aircraft degraded, against 1.2 of 21 in the flags called normal. They are
  spread over many cells and all 7 days, not one place. Both spoofing flags in the
  calm region were called normal: spoofing is where the calm false alarms are.
* So the calm flagged share (0.4-1.3%) is not a false-alarm rate, as earlier
  documents treated it: by this review the false-alarm rate is about 0.1%, and Central
  Europe has some low-level GNSS degradation of its own. Whether that is real
  interference (local sources, tests) or an aircraft-side effect that a human also reads
  as interference cannot be told from ADS-B alone (the known limit stated above).
  Ground stations in Germany (EUREF) could tell (Phase 3).
* Hotspot flags were right every time the labeler could decide.

## Trust report (declared 4 Nov 2026; reporting only, no decision)

`gnssguard trust-report` adds the uncertainty and calibration checks the improvement plan
asks for (Phase 1, step 4). It changes no model, rule or threshold, and none of its
numbers decides anything.

1. **Intervals grouped by test day.** Neighbouring cell-windows are not independent,
   so rolling results get 95% intervals from a bootstrap over the six selection days
   (cluster = day), not over cells.
2. **Calibration against hand labels.** For the three hand-labeled sets (26 Oct gold,
   1 Nov holdout, review set 1; decided cells with 3+ aircraft), the share the labeler
   called interference in each band of the published score (max of `p_jamming`,
   `p_spoofing`), overall, by region and by usable aircraft (3-5, 6-10, 11+). The sets
   were sampled by strata, so this is a diagnostic: it shows whether high scores are
   right more often than low ones, not a population calibration curve.
3. **Aircraft overlap.** For each test day, the share of its aircraft (ICAO addresses)
   also seen on the training days. The model never sees aircraft identities; a high
   overlap only means "the same fleet", and is reported so a reader can judge it.

### Result of the trust report (4 Nov 2026)

Run on the round-3 baseline rolling results (identical to the published configuration),
the three hand-labeled sets and the published model (`reports/trust/`).

**1. Intervals over days.** Most results stay as reported; the wide ones are spoofing
over the Black Sea: hard 0.45 (95% interval 0.35-0.56), stress 0.65 (0.56-0.74),
standard 0.84 (0.77-0.90). Calm Central Europe flagged share 0.70% (0.50-0.95%).

**2. The published scores are effectively a second copy of the label.** On all judged
cells of 2-3 Nov, 99.9% score below 0.1 or at least 0.9 (raw and calibrated alike), so
there is almost nothing to calibrate in between. Against 412 decided hand labels:

| Score | Cells | Labeler called interference |
| --- | --- | --- |
| below 0.1 | 194 | 7% (4-12%); Black Sea 35% (17-59%), Baltic 20% (9-39%) |
| 0.1-0.9 | 3 | 3 of 3 |
| 0.9 or more | 215 | 97% (94-99%); calm Central Europe 90% (81-96%) |

(The sets over-sample uncertain cells, so the shares are diagnostic, not day-wide.) A
score of 0.001 over the Black Sea is wrong about a third of the time in these samples:
the misses at hotspot edges found on 1 Nov. The scores are confident because the
training labels are near-deterministic functions of the same features, and calibrating
against them, on any split, cannot make them more honest. A separate calibration day
(planned as round 4) would therefore not help and is not run; what would is
calibration against human labels, which needs a labeled calibration set of its own
(Phase 1, sealed set).

**3. Aircraft overlap.** 94.5-96.2% of the aircraft on each test day were also seen on
the training days: the same fleet. The model does not see identities, but test days
are not "new aircraft".

## Event check 2: six more events (declared 4 Nov 2026, before their days were scored)

Rows EV14-EV19 of `configs/events.csv`, added from public sources on 4 Nov 2026: a
diversion at Vilnius (Jan 2025), the Pakistan CAA's February 2025 peak (listed only:
adsb.lol has no Punjab data in 2025), Delhi spoofing in Jan-Feb 2026 (checked on three
days fixed now: 15 Jan, 1 Feb, 15 Feb 2026), Constanta (4-5 Jun 2026), and the
western Black Sea and Kaliningrad on 3 Nov 2026. Same predictors, rule and placebo as
event check 1; the summary is reported for the new events alone and for all events.
The 3 Nov 2026 day was published on the map before; no decision used it.

## Sealed set (declared 4 Nov 2026, before any of its days was looked at)

Two hand-labeled sets for the Phase 1 gate, from blocks of days that no decision has
used, drawn with the published model frozen at commit 0509328 (`models/detector`).

**Calibration set: 15-21 Sep 2026** (before the training period), 150 cell-windows with
3+ usable aircraft. Its only use: fitting the published scores to human labels
(isotonic, per type), so that a score means "share of such cells a careful reviewer calls
interference". Drawn by score band so every band is filled: 40 from scores below 0.1, 40
from 0.9 or more, all cells (up to 70) between 0.1 and 0.9 (if fewer, the rest is split evenly between
the other two bands), each band spread over regions and days; weights stored.

**Sealed test set: 5-18 Nov 2026**, 250 cell-windows with 3+ usable aircraft, drawn on 19
Nov 2026 or later, with the same frozen model. Groups (each spread over regions and
days): 100 random judged cells (the day-weighted estimate), 60 model flags, 50
borderline cells both predictors call normal (some evidence, as in holdout S3), 40
cells the model flags with 1-2 aircraft. Labeled once, scored once at the Phase 1
gate, with these questions fixed now:

1. day-weighted agreement of model and rules with the labeler, with day-clustered
   intervals;
2. false-alarm share of model flags, and the share of borderline normal cells that are
   interference (misses);
3. precision of the flags from 1-2 aircraft;
4. calibration of the published score against the labels, before and after the
   human-label calibration fitted on the calibration set.

Nothing may be tuned on the sealed set. Its files stay unopened (`labels/sealed_*`)
until the gate; the groups file is committed with the sample, before labeling.

### Result of event check 2 (4 Nov 2026, scored once)

All 13 downloaded days were complete with 0 decode errors.

| Event | Judged | Matching, model (rules) | Verdict |
| --- | --- | --- | --- |
| EV14 Vilnius diversion, 16 Jan 2025 | 17 | 3 (3) | found, at the threshold |
| EV15 Lahore FIR, Feb 2025 | 0 | - | too few cells (no Punjab data in 2025) |
| EV16 Delhi spoofing, Jan-Feb 2026 (3 days) | 1,311 | **135 (rules 92)** | found |
| EV17 Constanta, 4-5 Jun 2026 | 187 | 59 (59) | found |
| EV18 western Black Sea, 3 Nov 2026 | 249 | 81 (81) | found |
| EV19 Kaliningrad and Lithuania, 3 Nov 2026 | 364 | 252 (252) | found |

New events: 5 of 5 decided found. All events so far: **14 of 15** for model and rules
alike (95% interval 70-99%); spoofing 3 of 4 (30-95%); independently corroborated 4 of 4.

* **Delhi, by day:** 127 of 501 judged cell-windows flagged as spoofing on 15 Jan, 1 of
  339 on 1 Feb, 7 of 471 on 15 Feb 2026. The reported 623 incidents in two months
  (about 10 a day) are compatible with spoofing on some days and almost none on others;
  the map would show which.
* **The first event where the model and the rules differ:** in Delhi the model flags 135
  spoofing cell-windows, the rules 92. Which is right needs a hand check; it is not
  scored here.
* Calm Central Europe on the new event days: 0.30-1.65% flagged (1.65% on 3 Nov 2026).

**Calibration set drawn 4 Nov 2026, before any label** (seed 20261104, published model):
C_low 40 of 39,329 cells scored below 0.1, C_mid 70 of 92 between 0.1 and 0.9, C_high
40 of 8,459 scored 0.9 or more, from 15-21 Sep 2026 (East Med 62, Baltic 30, Black Sea
26, Central Europe 22, Punjab 10). Labeling file `labels/calibration_1.csv`, groups
`labels/calibration_1_groups.csv`, committed before labeling.

### Result of the calibration set (labeled 5 Nov 2026)

Labels committed at 18:36 on 5 Nov (80887dd), after the sample at 01:09 (a00d0f8); 150
labeled, 29 unsure (17 of them in the middle band).

| Published score | Decided | Reviewer called interference | 95% interval |
| --- | --- | --- | --- |
| below 0.1 | 31 | 10% | 3-25% |
| 0.1 to 0.9 | 53 | 60% | 47-72% |
| 0.9 or more | 37 | 92% | 79-97% |

* **What is published:** this three-band table (`models/score_bands.json`, made by
  `gnssguard score-bands`), in every summary and map file as `reviewed_score_bands`,
  and on the map as "On hand review, cells scored like this were interference X% of the
  time". The score, the label and the model are unchanged (same model hash).
* **Why a table and not a fitted curve:** almost all scores sit at the ends, and an
  isotonic curve through 121 decided cells came out jagged (0.83 just below 0.999, 0.97
  at 1.0), which is noise. Three bands with intervals say only what the labels support.
* **The middle band is where the misses are.** 70 of the 92 middle-band cells of the week
  were sampled; the reviewer called 60% of the decided ones interference, and the model
  labels every cell under 0.5 normal. Across the set, 25 cells the model labels normal
  were interference to the reviewer (18 jamming, 5 spoofing, 2 both), 22 of them in the
  middle band (East Med 12, Black Sea 7, Baltic 5); every decided middle-band cell in
  the Black Sea and the Baltic (13) was interference. These cells are rare (92 of
  47,880 judged cells in the week, 0.2%), so the effect on the shares is small.
* Consistent with the trust report (412 earlier hand labels: 7% below 0.1, 97% at 0.9 or
  more).

**Added to the sealed test before it is drawn (question 5):** would a label threshold
of 0.1 instead of 0.5 agree better with the reviewer? The calibration set suggests it;
it may not decide it (its declared use is the score table). Both thresholds are scored
once on the sealed test, which then decides; the 0.5 threshold stays until then.

## Blind relabel (declared 5 Nov 2026, before sampling)

There is no second annotator. The plan's 10% blind relabel is the closest substitute:
the same labeler labels again, without seeing the earlier labels, a random 10% of the
cells already labeled. It measures how stable the labels are, not whether a different
person would agree, and is reported as such.

**Sample:** 55 cells, 10% of the 550 cells labeled before the calibration set (26 Oct
gold 300, 1 Nov holdout 150, review set 1 100), drawn in proportion to each set (30, 15,
10) with seed 20261105, new ids (`B0001`...), shuffled, in `labels/relabel_1.csv`; which
earlier cell each one is lives in `labels/relabel_1_key.csv`, committed with the sample
before labeling. The earlier labels are not shown.

**Reported:** agreement on interference or not, and on the exact label, with Wilson 95%
intervals and Cohen's kappa; cells unsure in either round are counted separately. The
earlier labels and every result based on them stay as they are, whatever this shows.

### Result of the blind relabel (labeled 5 Nov 2026, scored once)

Sample and key committed at 20:33 (9eae95e), labels at 22:55 (3092e56).

| | Result |
| --- | --- |
| Cells relabeled | 55 (30 from 26 Oct, 15 from 1 Nov, 10 from review set 1) |
| Unsure in either round | 16 (8 decided before and unsure now, 4 unsure before and normal now, 4 unsure both times) |
| **Interference or not, decided both times** | **39 of 39 (95% interval 91-100%), kappa 1.0** |
| Exact label | 38 of 39 (87-99%), kappa 0.96; one "both" became "jamming" |

The labels are stable: whenever the labeler decided twice, the decision matched. The
inconsistency is in deciding at all: 12 cells moved between a decision and "unsure".
Unsure cells are already kept out of every result, so the earlier results stand as
they are. This is the same person twice, not a second annotator: it shows the labels are
reproducible, not that someone else would read the evidence the same way.

## Baseline for the Phase 2 gate (declared 6 Nov 2026, before the sealed test is drawn)

The plan's Phase 2 gate: the model must beat the integrity-only baseline on the sealed
test at a fixed false-alert rate. Fixed now:

* **Baseline:** a cell-window with 3+ usable aircraft is interference if the share of
  them with degraded NIC is at least s. s is the loosest multiple of 0.05 whose flagged
  share in calm Central Europe over the model's training days (29 Sep-24 Oct) is not
  above the model's: s = 0.30 (0.30% against the model's 0.56%); also reported with the
  next looser s = 0.25 (0.65%), which raises slightly more calm alarms than the model.
* **Added to the sealed test as question 6:** recall of interference and false-alarm
  share for model, rules and both baselines, day-weighted, with day-clustered
  intervals; and spoofing found. "Beats the baseline" means a higher recall than the
  s = 0.25 baseline with a false-alarm share not higher by more than 2 points.

**Development check (6 Nov, not the gate):** `scripts/baseline_check.py` on the four
hand-labeled sets so far (533 decided cells with 3+ aircraft, stratified samples):

| Predictor | Calm alarms, training days | Interference found | False alarms | Spoofing found |
| --- | --- | --- | --- | --- |
| Model | 0.56% | 255/295 (86%; 82-90%) | 12/238 (5%) | 56/80 |
| Rules | | 247/295 (84%; 79-88%) | 8/238 (3%) | 48/80 |
| Integrity-only, s = 0.25 | 0.65% | 225/295 (76%; 71-81%) | 4/238 (2%) | 0/80 |
| Integrity-only, s = 0.30 | 0.30% | 208/295 (71%; 65-75%) | 2/238 (1%) | 0/80 |

At about the same calm-region alarm rate, the model finds 10 points more of what the
labeler called interference, and spoofing the integrity rule cannot see at all, at the
cost of a few more false alarms on these hard samples. Some of these sets informed
earlier decisions (the 26 Oct gold set most), so the sealed test decides.

## Ground check 1: EUREF reference stations around the Baltic (declared 6 Nov 2026, before any station file was downloaded)

The plan's Phase 3 step 2: a sensor that shares nothing with this pipeline. Permanent
geodetic GNSS stations of the EUREF network record the signal strength (C/N0) of every
satellite every 30 s; jamming within their radio horizon lowers it on all satellites at
once and can make them lose satellites.

**Stations and days:** the six EPN stations listed in `configs/ground_stations.csv`
(chosen for being within about 350 km of Kaliningrad, before any data was seen; one replaced, see the amendment below; their
exact positions are taken from the files' own headers), on 26 Oct-4 Nov
2026 (10 days, all published on the map). Files from the EUREF data centres
(`gnssguard ground-fetch`); a station-day that is missing is reported as missing.

**Ground signal, per station and 15-minute window** (`gnssguard ground-check`): the
median C/N0 of GPS L1 C/A (S1C) over all satellites with a reading, and the number of
GPS satellites tracked. Its usual value is the median for the same window of the day over
the 10 days (the GPS sky repeats daily, about 4 minutes earlier, close enough at 15
minutes). A window is **ground-affected** if the median C/N0 is at least 3 dB below
usual or the satellites tracked are at least 30% below usual.

**Aircraft signal:** the share of judged cell-windows (3+ aircraft) within 150 km of the
station, in the same window, that the published model labels jamming or both.

**Reported, per station and pooled:**

1. how often the aircraft flag nearby when the ground is affected, against when it is
   not (a ratio above 1 means the two agree in time);
2. the same with the aircraft series shifted by whole days (1-9), as a control for
   time-of-day patterns: agreement should be lower than unshifted;
3. the correlation, over windows, between the C/N0 drop and the aircraft flagged share.

**Known limits, stated now:** a ground antenna sees a jammer only within a few tens of
kilometres and with clear line of sight, while aircraft at altitude see it from hundreds
of kilometres; a nine-month ground monitor in Gdynia found no correlation with ADS-B
events (Inside GNSS, 2025). So a weak or absent match is expected and does not refute
the map; a clear match is support, not proof. Spoofing is not checked here.

**Amendment (6 Nov 2026, after the download and before any ground-check result was
seen):** KLPD00LTU is not in the EPN archive (not listed in the data centre's index of
26 Oct), so it is missing on every day, as the station list
allowed. VAIN00LVA (Vaiņode, Latvia, about 210 km north-east of Kaliningrad), the
EPN station closest to the gap it leaves, is added in its place. The thresholds, days and
reported quantities are unchanged.

### Ground check 1: result (6 Nov 2026)

51 of 60 station-days downloaded (4 Nov not yet published; IRBE 2-3 Nov and BYDG 3 Nov
not available); all six stations readable (S1C). Reports: `reports/ground/`.

| Station | Windows | Ground-affected | Aircraft flag when affected | otherwise | Ratio | Ratio, days shifted (median) |
| --- | --- | --- | --- | --- | --- | --- |
| LAMA00POL | 549 | 9 | 0.889 | 0.867 | 1.03 | 1.03 |
| VAIN00LVA | 551 | 5 | 1.000 | 0.603 | 1.66 | 0.91 |
| BYDG00POL | 378 | 2 | 0.500 | 0.612 | 0.82 | 0.86 |
| VIS000SWE | 554 | 2 | 0.000 | 0.370 | 0.00 | 0.00 |
| VLNS00LTU | 522 | 1 | 1.000 | 0.762 | 1.31 | 1.28 |
| IRBE00LVA | 413 | 0 | - | 0.361 | - | - |
| **Pooled** | 2,967 | 19 | 0.789 (15/19) | 0.603 | **1.31** | **1.10** (max 1.39) |

Spearman correlation between C/N0 drop and aircraft jamming share: 0.03 pooled (-0.03
to 0.07 per station).

**Reading, against what was declared:**

1. Ground-affected windows are rare (0.6%). When they occur, nearby aircraft flag
   jamming in 15 of 19 (79%, about 57-92%) against 60% of other windows.
2. The control does not separate: the unshifted pooled ratio (1.31) lies within the
   ratios with the aircraft series shifted by 1-9 days (median 1.10, up to 1.39). At
   the two stations with most affected windows, LAMA is equal shifted and unshifted, and
   only VAIN (5 windows) is above its control.
3. No correlation between the size of the ground drop and the aircraft signal.

**Conclusion:** no agreement in time between these ground stations and the map beyond
what time-of-day patterns produce. As stated before the data, this was the expected
outcome and does not refute the map: aircraft near Kaliningrad report jamming in most
windows (60-87% here), while ground antennas 90-350 km away, outside the jammer's line
of sight, rarely see a drop, so the two series cannot line up in time. It also gives no
support. A useful ground test needs a receiver within line of sight of a known jammer
(e.g. published spectrum-monitoring data from Estonia or Lithuania, catalog events with
ground evidence) or a second receiver network (Phase 3 step 1).

## Note: ordinary-day comparison on the incident pages (6 Nov 2026)

The incident pages (docs/product.md) compare each event day with ordinary days of the
same box. This comparison was added after event checks 1 and 2 were scored and is
descriptive only: it changes no verdict and is not used to claim detection. If it is
ever used as a test (e.g. "the event day is above the 90th percentile of ordinary days"),
that will be declared here first and scored on events added after the declaration.

## Jammer circles outside the observed area (7 Nov 2026)

**Found:** a check of all 87 jammer circles published from 26 Oct to 5 Nov 2026 against
publicly reported source areas (Kaliningrad and Baltiysk, St Petersburg, Sevastopol,
Novorossiysk, Beirut, Latakia, Lahore, Amritsar; reported areas, not measured jammer
positions). Black Sea circles cluster on southern Crimea (17 of 31 within 100 km of a
reported area) and Eastern Mediterranean circles on southern Lebanon and the
Lebanon-Syria border (20 of 24). Baltic circles are mostly not near Kaliningrad (11 of
28), and 10 of 28 lie **outside the region box**, e.g. in the sea off western Poland or
east of St Petersburg, where no aircraft are collected: nothing there can confirm or
rule out a jammer, so the fit is unconstrained.

**Rule, from 7 Nov 2026:** a circle is drawn only if its centre lies inside the region
box; otherwise the incident is published with `confidence: outside_coverage` and its
affected area only. Applied to the 87 circles: 13 removed (Baltic 10, Eastern
Mediterranean 1, Black Sea 1, Punjab 1); Baltic 18 left, 10 of them within 100 km of
Kaliningrad or Baltiysk. The remaining Baltic circles are still mixed, and the layer
stays experimental. The reported areas were used only to find the problem, not to tune
the rule, which removes no circle for being far from them.

## Second network check 1: OpenSky (declared 7 Nov 2026, before any OpenSky data was queried)

**Question.** Does a second receiver network with its own receivers and its own decoding
(OpenSky Network, Trino access granted 7 Nov 2026 until 1 Dec 2026) see the same loss of
position integrity in the cells our map flags?

**What it can and cannot show.** Both networks receive the same broadcasts from the same
aircraft: NIC is the aircraft's own report, so a jammed aircraft reports low NIC to both.
The check therefore tests **our data handling and our input** (adsb.lol feeders,
reception, decoding, aggregation, the chronic-aircraft filter), not the physics of
jamming. A clear disagreement would mean a problem on our side (or OpenSky's) that must
be explained before any claim. Some feeders send to both networks; the overlap is not
known. Spoofing is not checked (NIC is usually normal under spoofing).

**Data.** OpenSky `position_data4` (airborne position messages with NIC), queried one
UTC hour at a time with the `hour` partition filter and aggregated on the server to the
map's grid: 1-degree cells, 15-minute windows (by `mintime`), the five study boxes
(baltic, east_med, black_sea, punjab, central_europe as calm control). Days: 26 Oct-4 Nov
2026 (the ground check days). Per aircraft and cell-window, as in our pipeline: only
airborne messages at or above 3,000 ft (914 m) with a NIC value; the aircraft is
degraded if NIC < 7 in at least half of its messages there. Aircraft our pipeline marks
as chronic (faulty transponders) are not removed from the OpenSky side, so they are
counted as a known source of difference and reported.

**Matched cell-windows.** 3 or more usable aircraft in both networks.
**OpenSky flag.** Share of OpenSky aircraft degraded at least 0.30 (the primary
threshold of the integrity-only baseline).

**Reported** (per region and pooled over baltic, east_med, black_sea, punjab):

1. Coverage: share of our judged cell-windows that OpenSky also judges.
2. P(OpenSky flag | our model says jamming or both) and P(OpenSky flag | our model says
   normal), with Wilson 95% intervals; the same for the rules' labels.
3. Spearman correlation of the degraded share between the two networks.
4. Calm control (central_europe): OpenSky flag rate where our model says normal.
5. Disagreements: the 20 matched cell-windows with the largest difference in degraded
   share each way, for inspection (descriptive).

**Agreement is declared** if, pooled over the four hotspot regions, the lower 95% bound
of P(OpenSky flag | model jamming) is at least 0.70 and the upper 95% bound of
P(OpenSky flag | model normal) is at most 0.10. Otherwise the disagreement is
investigated and reported before any further use of the map's numbers. No threshold,
day, region or definition above is changed after the data is seen; a mistake found in
the code is fixed and reported as such.

**First query: a probe** of one hour (26 Oct 2026, 12 UTC) over the Baltic box, returning
only the distribution of NIC and its consistency with HCR (containment radius), to check
the column meanings before the full fetch. It does not look at agreement.

Cite: M. Schäfer, M. Strohmeier, V. Lenders, I. Martinovic, M. Wilhelm, "Bringing Up
OpenSky: A Large-scale ADS-B Sensor Network for Research", IPSN 2014.

### Second network check 1: result (7 Nov 2026)

240 of 240 hours fetched (26 Oct-4 Nov 2026, one query per hour). Reports:
`reports/opensky/`; table in `results/opensky_check_1.csv`.

| Region | Our judged | Also judged by OpenSky | P(OpenSky flag \| model jamming) | P(OpenSky flag \| model normal) | Spearman |
| --- | --- | --- | --- | --- | --- |
| Baltic | 14,271 | 77% | 0.45 (0.43-0.47), n=2,383 | 0.053 (0.048-0.058) | 0.67 |
| Black Sea | 8,094 | 90% | 0.82 (0.80-0.84), n=1,208 | 0.093 (0.086-0.101) | 0.78 |
| East Med | 14,968 | 99% | 0.70 (0.65-0.75), n=354 | 0.023 (0.021-0.026) | 0.73 |
| Punjab | 3,397 | 88% | 0.84 (0.78-0.88), n=226 | 0.073 (0.064-0.084) | 0.77 |
| **Hotspots pooled** | 40,730 | 89% | **0.60 (0.59-0.62)**, n=4,171 | **0.049 (0.046-0.051)** | 0.73 |
| Central Europe (control) | 25,911 | 100% | 0.30 (0.24-0.37), n=201 | 0.002 (0.002-0.003) | 0.50 |

The rules give the same numbers within 0.01 (their jamming cells are the same cells).

**Verdict, by the declared rule: no agreement.** The lower bound for P(OpenSky flag |
model jamming) is 0.59 (needed 0.70); the normal side passes (upper bound 0.05, needed
0.10 or less). The shortfall is the Baltic (0.45); the other hotspots are 0.70-0.84.

**Investigation (after the result; not part of the declared check).** In our Baltic
jamming cells, 93% of the degraded aircraft are positioned only by ground multilateration
(MLAT fallback: no GNSS position of their own, NIC 0), against 8% in the East Med and
0% in the Black Sea and Punjab. OpenSky's `position_data4` holds only aircraft that still
broadcast a position, so it does not count these aircraft, and its degraded share in
the Baltic is lower. The ordering of the regions follows the MLAT share.

**Test of that explanation, declared before it is run** (`scripts/opensky_investigate.py`):
recount our side with only aircraft that broadcast their own position (usable minus
MLAT-fallback aircraft, degraded minus MLAT-fallback aircraft) and flag a cell-window
when that share is at least 0.30 with 3 or more such aircraft, as on the OpenSky side.
The explanation is **supported** if, on matched cell-windows, P(OpenSky flag | our
recount flag) in the Baltic is at least 0.70 (the level of the other hotspots), and
**not supported** otherwise. This does not change the verdict above: the declared check
failed, and the map's Baltic jamming flags rest mostly on aircraft that lost their GNSS
position entirely, which OpenSky's position data cannot confirm.

**Follow-up result (7 Nov 2026): explanation supported.** Counting only aircraft that
broadcast their own position (matched cell-windows with 3 or more such aircraft in both
networks):

| Region | Recount flags | P(OpenSky flag \| recount flag) | P(OpenSky flag \| recount normal) |
| --- | --- | --- | --- |
| Baltic | 992 | **0.727** (0.698-0.754) | 0.019 |
| Black Sea | 1,862 | 0.716 (0.695-0.736) | 0.042 |
| East Med | 776 | 0.648 (0.614-0.681) | 0.005 |
| Punjab | 435 | 0.772 (0.731-0.809) | 0.025 |
| Pooled | 4,065 | 0.712 (0.698-0.726) | 0.017 |

The Baltic reaches the declared 0.70 (point estimate 0.727; the 95% interval just
touches 0.70) and now matches the other hotspots. In the model's Baltic jamming cells
our pipeline sees 5.9 aircraft on average, 3.9 of them with a position of their own, and
OpenSky 4.9.

**Conclusion.** The two networks agree on which cells have aircraft reporting degraded
integrity, at about 70% (and about 2% false flags), once both count the same aircraft.
The declared check failed because, near Kaliningrad, most aircraft that make a cell
"jammed" in our map have lost their GNSS position completely and are seen only through
ground multilateration, which OpenSky's position table does not include. So the Baltic
jamming flags rest largely on aircraft that a second network's position data cannot
confirm or refute; other evidence for them (event catalog, Lithuanian and Estonian
spectrum monitoring) is unaffected. A second network check with OpenSky's
multilateration or state-vector data would be needed to compare those aircraft.

### Second network check 1: follow-up 2, Central Europe (declared 8 Nov 2026, before it is run)

**Why.** In the calm control region, OpenSky confirms only 30% (24-37%) of the model's
201 jamming cell-windows, against a hand-review false-alert share of 13% (about 10%
weighted, below). Raised by the review of 7 Nov 2026. A look at our own cells only (no
OpenSky data) shows two likely causes: 58% of the degraded aircraft in those cells are
MLAT-only (as in the Baltic), and about half of the cells have only 2 degraded aircraft
among about 10, close to the 0.30 share threshold.

**Test** (`scripts/opensky_investigate.py <days> central_europe`, the 240 hours already
fetched, no new queries): the same recount as follow-up 1, counting only aircraft that
broadcast their own position. The MLAT explanation is **supported** if P(OpenSky flag |
recount flag) in central_europe is at least 0.70. Reported beside it, descriptive: the
OpenSky flag rate in model jamming cells with 2 or fewer degraded aircraft and in the
rest. If not supported, the remaining Central Europe flags are treated as possible false
alarms in the next review set.

## Correction 1 (8 Nov 2026): review of 7 Nov, fixes and their effect

An external review of commit 249fccf found three correctness bugs and several reporting
problems. Each fix below has a regression test; each effect was measured, not assumed.
Nothing here changes the sealed test: its model, labels, baseline and questions stay
those archived in release 0.2.0, and the sealed-test days are processed by the same
ingest code throughout (bug 3's fix waits until the test is scored).

**Bug 1, dropped points vanished from synthetic truth** (`dataset.py`, `synthetic.py`).
Truth was rebuilt from the points that survived the attack, so an aircraft whose points
were all dropped did not count as affected. Fixed: truth includes dropped points, and
evaluation reports recall end to end (unobservable truth counted as missed) and on
observable truth, with the unobservable count. **Bug 2, mixed jamming and spoofing in one
cell collapsed to one type.** Fixed: affected aircraft counted per type; a cell with both
is a positive for both detectors.
*Effect,* measured by running `gnssguard evaluate` with the old and the new code on the
same data and model (`results/correction_synthetic_truth.csv`): **no published synthetic
number changes** (standard, hard and stress recall and false alarms identical to 3
decimals); the stress edge breakdown moves by 7 of 791 cells. Why: the generator drops
points one at a time (probability 0.15), so an aircraft almost never loses all its points
in a cell, and separate events rarely overlap. The fixes matter for harsher generators
(planned: a second attack generator).

**Bug 3, an expired NACp looked fresh** (`ingest/trace.py`). When readsb dropped NACp from
a snapshot (it expires; NIC still reported), the old value was kept and its age reset.
The roadmap's Phase 0 verdict "not an issue" on this point was **wrong**. Fixed in a
separate change, point format points/3, merged after the sealed test is scored.
*Effect,* measured by re-ingesting 1 Nov 2026 with the fix (`results/correction_stale_nacp_2026-11-01.csv`):
in the Baltic, points without NACp rise from 1.2% to 11.2% and 559 of 3,702 cell-windows
lose low-NACp aircraft (only in cells that already have degraded NIC: jammed aircraft
kept an expired low NACp). Labels move only towards normal: model 5 of 629 jamming
cell-windows (0.8%), rules 1 of 619 jamming and 15 unknown. Other regions: 0-1 model
labels. So the published Baltic jamming was slightly overstated by stale NACp.

**Model hash.** The laptop's `results/` showed a different hash (a1d3d152259c) for the
published model than the site (81062b5eb365). The hash now covers only the model's own
files with line endings normalized; on the published model it is unchanged
(81062b5eb365). Also found: `gnssguard evaluate` defaults to `data/models/detector`,
which need not be the published model; the README's synthetic figures came from an
earlier evaluation and are replaced by an evaluation of the published model.

**Score bands, weighted.** The calibration sample was spread evenly over region-days
within each score band, but the map showed the raw sample share. Weighted to all judged
cells of 15-21 Sep by band and region-day (scores recomputed: all 150 sampled cells
reproduce exactly), with intervals from resampling region-days:

| Score | Sample share (was shown) | Weighted to all cells (now shown) | Unsure all normal / all interference |
| --- | --- | --- | --- |
| below 0.1 | 10% (3-25%), 31 decided | **3.5%** (0-9%) | 3% / 15% |
| 0.1 to 0.9 | 60% (47-72%), 53 decided | **58%** (44-86%) | 43% / 69% |
| 0.9 or more | 92% (79-97%), 37 decided | **98.6%** (95-100%) | 97% / 99% |

**False alerts in the calm region, weighted and named.** Review set 1 spread calm-region
flags evenly over days; weighted by each day's flags, the share of calm-region alerts the
reviewer judged false is about 10% (2-25%, resampling days) against 13% unweighted
(6 of 46). Named denominators from now on: *share of alerts that were false* (10-13%),
*false alerts among all judged calm cells* (about 0.1%), and the false-positive rate among
truly normal cells, which is about the same 0.1% because almost all calm cells are normal.

**Monitoring.** Health is now measured against today: a "stale" status when the newest
published day is more than 3 days old (with a GitHub issue), the model's age to today, a
stale banner on the map computed in the reader's browser, a persistent backlog of missing
days (at most 4 per run) instead of a 3-day look-back, and a step that re-enables the
nightly workflow on every run (GitHub disables scheduled workflows after 60 days without
repository activity).

**Provenance.** Every published day records the git commit (and whether the working tree
was modified), the region boxes, the point format, a hash of the feature list and of
`constraints.txt`, besides the model hash and settings hash.

**Follow-up 2 result (8 Nov 2026): explanation supported.** Counting only aircraft that
broadcast their own position, OpenSky flags 0.884 of the 86 Central Europe
cell-windows the recount flags (needed 0.70), and 0.001 of the rest (25,771). Of the
model's 199 Central Europe jamming cell-windows, only 86 remain flagged on the recount:
the others rest on MLAT-only aircraft. The share threshold is not the cause: OpenSky flags
0.27 of the model's cells with 2 or fewer degraded aircraft and 0.34 of the rest.

**What this leaves open.** In the Baltic, MLAT-only aircraft at NIC 0 are aircraft that
lost GPS; in calm Central Europe, with dense multilateration, they may also be aircraft
whose ADS-B position is missing for other reasons (equipment, non-GNSS aircraft). So more
than half of the calm region's jamming flags rest on evidence that does not by itself
show GPS loss. Planned, declared before it is looked at: in the next review set, sample
calm-region flags separately by whether they rest on MLAT-only aircraft.

**Model hash on the laptop, resolved.** `models/detector` on the laptop has Windows line
endings in all three files (meta.json 4,414 bytes against 4,197); with line endings
normalized it hashes to 81062b5eb365, the published value.

## Sealed test, addendum 1: a GPSJam-style baseline (declared 8 Nov 2026, after release 0.2.0 and before the sealed test is drawn)

Added after the archived release 0.2.0, so it is marked as an addendum; it changes none
of questions 1-6, the model, the labels or the NIC baseline. Raised by the review of 7
Nov 2026: compare with the best-known public method, not only with our own baseline.

* **GPSJam-style score** (gpsjam.org, FAQ): per cell, 100 x (bad - 1) / all aircraft,
  flagged "red" above 10%. GPSJam names NACp as the field but publishes no cut; here an
  aircraft is bad if its NACp is below 8 (position worse than 93 m, 95% of the time) in at
  least half of its usable points, the same cut as the rules. GPSJam uses H3 hexagons over
  a whole day; here it is computed on the same 1-degree cells, **per 15-minute
  cell-window** (primary, like every other predictor) and **over the cell's whole day**
  (secondary, as GPSJam aggregates; an aircraft seen in several windows counts in each).
* **Threshold:** 10%, GPSJam's own red. Matching the model's calm-region alarm rate on
  the training days (the rule used for the NIC baseline) gives the same 10%: it flags
  0.25% of calm cells against the model's 0.56%, and 5% would flag more than the model.
* **Question 7:** recall of interference, false-alarm share and spoofing found for both
  GPSJam-style variants, day-weighted with day-clustered intervals, beside the model,
  the rules and the NIC baselines of question 6. Reported, not a gate.

**Development check (8 Nov, not the gate),** the 533 decided cells of the four
hand-labeled sets (`scripts/baseline_check.py`, stratified samples, not day-wide rates):

| Predictor | Interference found | False alarms | Spoofing found |
| --- | --- | --- | --- |
| Model | 255/295 (86%; 82-90%) | 12/238 (5%) | 56/80 |
| Rules | 247/295 (84%; 79-88%) | 8/238 (3%) | 48/80 |
| NIC share, s = 0.25 | 225/295 (76%; 71-81%) | 4/238 (2%) | 0/80 |
| GPSJam-style, per window | 132/295 (45%; 39-50%) | 1/238 (0%) | 0/80 |
| GPSJam-style, whole day | 119/295 (40%; 35-46%) | 17/238 (7%) | 0/80 |

A NACp-only score finds about half of what the labeler called interference: NACp often
stays normal while NIC drops. The labeler's labels draw on NIC too, which favours the
NIC-based predictors; the sealed test uses the same labels, so it inherits that bias.

## Disagreement set 1 (declared 8 Nov 2026, before it is drawn)

**Why.** On real data the model and the rules agree on about 99% of judged
cell-windows, so the hand-labeled sets so far say little about which is right when they
differ (the 1 Nov holdout had one decided disagreement). Raised by the review of 7 Nov.

**Sample** (`gnssguard disagreement-sample`, seed 20261108, the published model): every
judged cell-window (3+ usable aircraft) of 15 Sep-4 Nov 2026 where the two differ (never
the sealed days 5-18 Nov), in four groups, each spread over regions and days, all cells
taken where a pool is smaller: D1 the model flags and the rules do not (30), D2 the rules
flag and the model says normal (15), D3 both flag with a different type (15), D4 the rules
give no verdict and the model says normal (20). A trial on 43 days: pools of 79, 4, 7 and
2,847 (D1 is mostly spoofing the rules miss). Labeled blind once with the labeling tool;
the groups file is committed with the sample and not opened before labeling.

**Reported** (`gnssguard disagreement-eval`): per group, how many decided labels side
with the model and with the rules (interference or not; the type in D3), with Wilson
intervals, and the unsure count. **Read as:** the model adds something over the rules if
it is right on most of D1 (its extra flags) and on most of D4 (its calls where the rules
abstain). Small groups (D2, D3) are reported as counts, not rates. Some of these days
trained the model; these are its own training days, so a model "win" here shows that it
learned something the rules miss, not how often it is right on new days.

**Disagreement set 2, Delhi (declared with set 1):** the same sampler on the three Delhi
days of event EV16 (15 Jan, 1 Feb, 15 Feb 2026; region delhi; never training days),
where the model flags 135 spoofing cell-windows and the rules 92, the largest real-data
disagreement so far. Same groups and sizes (`--out labels/disagreement_2.csv`, seed
20261108), labeled blind after set 1, reported separately.

## Correction 2 (8 Nov 2026, before the sealed test is drawn): samplers reached only the first region-days in alphabetical order

**Found** while reading disagreement set 1: all 20 cells of its group D4 came from the
Baltic, on 20 consecutive days (15 Sep-4 Oct), though the group was declared "spread
over regions and days".

**The bug.** The function that spreads a sample (`_spread` in `labels/review.py`) takes
one cell per region-day per round, but visited the region-days in alphabetical order
("baltic|2026-09-15" first) and stopped when the sample was full. When a group has more
region-days than cells to take, only the first ones in that order were ever reached: the
first region, its earliest days.

**Effect on the sets already drawn** (all labeled; kept as they are, results stand with
the limits below):

| Set | Group: cells / region-days available | Effect |
| --- | --- | --- |
| Calibration 1 (15-21 Sep) | 40-70 per band / 35 | every region-day reached; the 5 extra cells per band went to the Baltic. The weighted score bands (Correction 1) weight by region-day pool, so they are unaffected in expectation |
| Review set 1 (27 Oct-3 Nov) | spread by day (7) or by region (4) | every day or region reached; the remainder went to the earliest days |
| Disagreement set 1 | D1 30 of 113 cells, one from each of 30 region-days; D2, D3 all or most cells taken | D1 reached all five regions; if its pool had more than 30 region-days, the last ones alphabetically (Punjab, then late East Med days) were missed: a small effect |
| Disagreement set 1 | **D4 20 / about 240** (204 on the 43 days checked here) | **Baltic only, 15 Sep-4 Oct**: its result describes the Baltic, not the pool |

The 26 Oct gold set and the 1 Nov holdout use other samplers and are unaffected.

**Effect on the sealed test, had it been drawn with this code:** 70 region-days (14 days,
5 regions); the groups of 60, 50 and 40 cells would have reached only the first 60, 50
and 40 region-days: Punjab and most of the East Med would have had no borderline or
sparse cells at all, against the declaration ("each spread over regions and days").

**Fix.** Region-days are now visited region by region in turn, a random day of each
region in each round (seeded with the sample's seed; other keys in a random order), so
every region gets its share and the days are random. The old order is kept as
`--key-order alphabetical`, only to reproduce the three sets above; the sealed test
refuses it. Tests: the old order is shown biased, the new one gives 4 cells to each of 5
regions for 20 seeds, the sealed sampler reaches all five regions in every group, and the
old order reproduces the earlier draws exactly.

**The sealed test** is drawn with the fixed sampler; nothing else about it changes
(seed 20261119, groups, sizes, questions, the frozen model of release 0.2.0). The commit
of the sampler is recorded with the draw.

## Disagreement set 1: result (labeled 8 Nov 2026 by SP, scored once)

| Group | Pool | Sampled | Decided | Unsure | Labeler sides with model | with rules | Model right (95% CI) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D1 model flags, rules do not | 113 | 30 | 27 | 3 | 15 | 11 (+4 no verdict)* | 56% (37-72%) |
| D2 rules flag, model normal | 4 | 4 | 4 | 0 | 0 | 4 | (count) |
| D3 both flag, different type | 20 | 15 | 15 | 0 | 7 | 8 | (count) |
| D4 rules no verdict, model normal (**Baltic only**, Correction 2) | 3,375 | 20 | 18 | 2 | 1 | 0 (18 no verdict)* | 6% (1-26%) |

\* A rule "unknown" is no verdict and never counts as a correct "normal" (corrected 9
Nov 2026 after a review: first reported as D1 12 and D4 1, which scored "unknown" as
"not interference"; the conclusions do not change). In D4 the rules decide no cell; what
counts is that the labeler called 17 of the 18 decided cells jamming.

**Read as declared** ("the model adds something over the rules if it is right on most of
D1 and on most of D4"): **not shown.** On its extra flags the labeler sides with the
model about as often as not (D1, 56%, interval includes 50%); where the rules abstain it
says normal and the labeler disagrees (D4, Baltic).

**What is behind each group:**

* **D1, by type.** The model's 8 extra jamming flags: all 8 jamming. Its 22 extra
  spoofing flags: 12 normal, 3 unsure, 3 jamming, 1 both, 3 spoofing; so 4 of 19 decided
  carry spoofing. The model's spoofing flags that no rule supports are mostly not
  spoofing on real data. The pool is small (113 cells in 51 days; the model flags about
  47,000 judged cells in the 43 days checked here) but this is where the model's real-data spoofing calls differ from the
  rules, and it does not support them.
* **D2:** the four cells where the rules flag and the model does not (all Black Sea) are
  all jamming: four model misses.
* **D3:** the type, where both flag: model 7, rules 8. No difference.
* **D4: single degraded aircraft.** Every D4 cell that could be checked here (13 of 20;
  the other 7 are on days not in this workspace) has exactly 3 usable aircraft with one
  degraded (NIC 0, low NACp). With one degraded aircraft out of 3-9 every integrity rule
  abstains (the cluster rules need 2, "normal" needs none), so the label model has no
  verdict, these cells get no training label, and the model, which learned from busier
  cells where one degraded aircraft is noise, says normal with scores of 0.0001-0.004.
  The labeler called them jamming from the neighbouring cells and windows. On 43 of the
  51 days (those in this workspace), these cells and how often they touch a rule-jamming
  cell (adjacent cell or window):

  | Region | D4 cells | of them one degraded | touching rule jamming | normal cells touching rule jamming (control) | rule jamming cells |
  | --- | --- | --- | --- | --- | --- |
  | Baltic | 1,126 | 1,106 | 89% | 73% | 52,521 |
  | East Med | 734 | 688 | 58% | - | 3,063 |
  | Black Sea | 521 | 368 | 67% | - | 5,122 |
  | Punjab | 269 | 252 | 39% | - | 446 |
  | Central Europe | 197 | 88 | 53% | 5% | 697 |

  In the Baltic, where jamming is widespread, such cells are very likely jamming edges
  and the map shows them as normal: about 2% of the Baltic's jammed area. Elsewhere the
  same cells are a larger share of the flagged area (East Med 734 against 3,063), but no
  labels exist there yet (set 1b below).

**Limits.** One labeler, who also wrote the code. The labeler judges from the context
(neighbours, windows before and after), which the model also sees as features. These are
partly the model's training days: a model loss here means it did not learn something,
not a measured error rate on new days. D4 covers the Baltic only.

**Not changed now.** The model stays frozen until the sealed test. The sealed test's
group T_border (borderline cells both call normal, which include these cells) measures
the miss on new days, now spread over all regions (Correction 2).

## Disagreement set 1b: one-aircraft cells outside the Baltic (declared 8 Nov 2026, before it is drawn)

**Question.** Does the model's "normal" on cells where the rules abstain (D4) also fail
outside the Baltic?

**Sample.** `gnssguard disagreement-sample --only D4_rules_unsure`, regions east_med,
black_sea, punjab and central_europe, days 15 Sep-4 Nov 2026 (never the sealed days),
seed 20261109, shuffled region-days (Correction 2), 20 cells (4-5 per region). Labeled
blind once, as set 1 (`labels/disagreement_1b.csv`, groups file committed with the
sample).

**Reported:** the share of decided cells the labeler calls interference, with a Wilson
interval, overall and per region (counts), and the unsure count. **Read as:** if most
are interference (lower bound above 50%), the miss is general, not a Baltic effect, and a
fix (training labels for one-aircraft cells from their neighbours) is declared as the
first change after the sealed test; if most are normal, the miss is specific to the
Baltic's widespread jamming.

### Set 1b: result (labeled 8 Nov 2026 by SP, scored once)

20 cells from a pool of 2,026 (51 days, four regions), 5 per region on 20 different days.

| Region | Jamming | Normal | Unsure |
| --- | --- | --- | --- |
| Black Sea | 4 | 0 | 1 |
| East Med | 3 | 0 | 2 |
| Central Europe | 2 | 3 | 0 |
| Punjab | 0 | 0 | 5 |
| **All** | **9** | **3** | **8** |

Interference in **9 of 12 decided cells (75%, 95% interval 47-91%)**; 8 unsure.

**Read as declared:** the lower bound (47%) is not above 50%, so **a general miss is not
shown**, narrowly; nor are these cells mostly normal, so a Baltic-only effect is not
shown either. The declared consequence (a fix as the first change after the sealed test)
is therefore not triggered by this set.

**Exploratory, not declared:** in the hotspot regions (Black Sea, East Med) the decided
cells were jamming 7 of 7, as in the Baltic (17 of 18, set 1); in calm Central Europe 2 of
5; in Punjab no cell could be decided (5 unsure: one degraded aircraft with too little
around it to tell). The pattern suggests the miss follows the amount of jamming nearby,
not the region, but these per-region counts are small and were not the declared test.

**What decides it:** the sealed test's T_border group (borderline cells both call normal,
which include these one-aircraft cells), drawn on new days and now spread over all five
regions (Correction 2). If it shows these cells are mostly interference, the fix is the
first change after the sealed test.

## Sealed test, addendum 2: one-aircraft cells (declared 8 Nov 2026, before the draw)

**Question 8.** Within the sealed group T_border, the cells where the rules give no
verdict (rule label "unknown": one degraded aircraft and no other rule evidence) are
reported separately: the share the labeler calls interference, with a Wilson interval,
and the unsure count. Nothing about the draw changes; this only names a subset of a
group already declared. **Read as:** a lower bound above 50% means the model's "normal"
on these cells is mostly wrong on new days, and the fix (training labels for such cells
from their neighbours) becomes the first change after the sealed test; otherwise it is
reported as not shown. The groups file already records each cell's rule label, so the
subset is fixed at the draw.

## Sealed test, addendum 3: scoring code, estimands and one primary endpoint (declared 9 Nov 2026, before the draw)

Raised by the review of 9 Nov 2026: until now only the sample had code; how the labels
become numbers (weights, intervals, unsure cells, rule "unknown") was still open, and
those choices must not be made after the labels exist. This addendum fixes them. The
model (release 0.2.0), the seed (20261119), the groups, their sizes and questions 1-8
are unchanged.

**Scoring code:** `gnssguard sealed-score` (`src/gnssguard/labels/sealed_score.py`), in the
commit that adds this addendum; its run records its own commit and the SHA-256 of the
three input files. It refuses to run while any cell is unlabeled, and refuses to
overwrite its result (the test is scored once). Tested before any label exists on fake
populations with known truth (`tests/test_sealed_score.py`): over 40 draws the estimates
match the true population values within 3 points, and break by 6-7 points when the
weights are removed.

**Primary endpoint (the only pass/fail): question 6, the Phase 2 gate.** The model
passes if its day-weighted recall is higher than the NIC-share baseline's (s = 0.25)
**with the paired 95% interval of the difference above 0** (the declared "higher",
tested on the same cells rather than compared as two numbers), and its false-alarm rate
is not higher by more than 2 points (as declared). Questions 1-5, 7 and 8 are reported,
not gates; with that many numbers some will look good by chance.

**The likely outcome, stated now.** On the development sets (6 Nov, stratified, not
day-weighted) the model raised false alarms on 12 of 238 normal cells (5.0%) against the
baseline's 4 of 238 (1.7%): 3.4 points more, over the 2-point allowance. If the sealed
test looks like that, the model fails the gate, and it is reported as failing. The gate
is not changed.

**Who counts as what.** Interference: the labeler's jamming, spoofing or both; normal:
normal. Unsure: left out of every share and counted; for the primary endpoint the result
is also given with all unsure cells as interference and all as normal. A rule
"unknown" is no flag for recall and false alarms, and is never a correct "normal": in
question 1 it counts as disagreement, and the rules' agreement where they decide and
their no-verdict share are reported beside it.

**Estimands.** Over the cell-windows of 5-18 Nov 2026 in the five regions:

| Question | Estimand | Population | Weighting |
| --- | --- | --- | --- |
| 1 | agreement (interference or not) with the labeler, model and rules | judged cells (3+ usable aircraft) | day-weighted |
| 2a | share of model flags the labeler calls normal | judged cells the model flags | all cell-windows |
| 2b | share of borderline normal cells the labeler calls interference | judged cells both leave unflagged, with jamming evidence | all cell-windows |
| 3 | share of 1-2 aircraft flags that are interference | cells the model flags from 1-2 aircraft | all cell-windows |
| 4 | interference share per published score band, and the Brier score of the raw score | judged cells, by band | all cell-windows |
| 5 | agreement at threshold 0.1 against 0.5, paired | judged cells | day-weighted |
| 6, 7 | recall, false-alarm rate (share of normal cells flagged), spoofing found, for model, rules, NIC 0.25 and 0.30, GPSJam-style per window and per day | judged cells | day-weighted |
| 8 | share of the T_border cells with rule "unknown" that are interference | the sample itself (as declared) | none |

Day-weighted: each of the 14 days counts the same (weights divided by the day's judged
cells). Intervals: 95%, 2000 bootstrap resamples of days; paired differences use the same
resamples for both predictors, and the discordant cell counts (McNemar) are reported.

**Weights.** At the draw, each cell's inclusion probability within its group is
simulated from the sampler's own rule (`inclusion`, 20,000 runs) and frozen in the groups
file, with the predictors' values (model, rules, NIC share, GPSJam-style scores) so that
scoring needs only the three committed files. A third file, `labels/sealed_test_frames.csv`,
holds each group's pool per region-day. A cell stands for 1/pi cells of its group's
pool. T_random cells that lie in the flag or border populations stand for themselves
(weight 1), because they were removed from the later groups' pools; T_flag and T_border
estimate the rest of those populations. This replaces the earlier per-region and
per-group ratios, which over-weighted small region-days.

**Short groups.** A group whose pool is smaller than its size takes every cell (pi = 1)
and is reported short; the shortfall is not moved to another group.

**Disclosures, before the draw:**

* The nightly map publishes every sealed day as it happens; anyone, the labeler
  included, can see the model's verdicts for 5-18 Nov. The labeling tool shows no model
  output, but the labeler has seen the map: the README screenshot was of 7 Nov (now
  replaced by 3 Nov), and on 9 Nov the published flagged shares per region for 7 Nov
  were reported to the labeler in a status check. The labeler will not open the map
  for 5-18 Nov until the labels are committed.
* The sealed days are also refused, by code, for training, evaluation and every other
  sample until the sealed test is scored (`model/splits.py`, `refuse_sealed`).

**Verifiable timing.** Commit dates can be set freely, so each declaration is pinned by
something outside the repository: this addendum and the scoring code are archived on
Zenodo (a GitHub release, archived automatically with a date) before the draw, and the
draw, the labels and the score are each pushed to GitHub before the next step (GitHub
records push times).

**Event test from now on.** "Found" (3 or more matching cells) does not show that a day
was unusual. For new events the test is whether the event day **stands out**: its
matching share is reached or beaten on at most 10% of ordinary days in the same box,
with at least 20 ordinary days (fewer: "too few ordinary days"). For the 14 events found
so far, as a description (not a re-scoring): EV09 (0 of 56 ordinary days) and EV10 (4 of
56) stand out; EV13 and EV16 (0 of 7) have too few ordinary days; EV18 (12 of 67) partly;
the others are matched or beaten on most ordinary days (Kaliningrad-area jamming is
routine).
