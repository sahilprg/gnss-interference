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
