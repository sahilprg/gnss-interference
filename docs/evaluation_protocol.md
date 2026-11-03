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
that on 1 Nov the model and the rules disagree on only 1 of 1,761 judged cells (S1 = 1),
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
