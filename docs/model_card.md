# Model card: gnssguard GNSS interference detector

A short, plain summary for anyone who wants to use or build on this model. Details
and all numbers: docs/model.md. How decisions were made: docs/evaluation_protocol.md.

## What it does

For each 1° × 15-minute grid cell, from open ADS-B flight data (adsb.lol), it gives:

* `p_jamming`, `p_spoofing`: model scores that aircraft in the cell had their GPS jammed
  or spoofed, calibrated against the system's own training labels (weak rules and
  simulated attacks), so not validated probabilities of real interference;
* a label (jamming, spoofing, both, normal, unknown);
* for groups of jammed cells, a coarse estimate of where the jammer was, with an
  uncertainty radius and a confidence flag. **Experimental:** tested only on simulated
  jammers that follow the same physics as the fit; accuracy on real jammers is unknown.
  The affected areas are the main result.

In practice 99.9% of judged cells score below 0.1 or at least 0.9, so the score adds
little to the label. Against hand labels, scores of 0.9 or more were interference 97% of
the time, scores below 0.1 7% of the time (Black Sea 35%, on stratified samples).

On the calibration set (150 hand-reviewed cells, 15-21 Sep 2026), weighted to all
cells of the week (what the map shows), the reviewer called interference 3.5% of cells
scored below 0.1 (0-9%), 58% between 0.1 and 0.9 (44-86%) and 98.6% at 0.9 or more
(95-100%); unweighted, on the sample as drawn, 10%, 60% and 92%. Both are published with
every file (`models/score_bands.json`); the score itself is unchanged.

The label is decided on the uncalibrated score at 0.5. Every published summary and map
file states the threshold, what the scores mean, the model's hash and the run settings.

## Intended use

* Situational awareness and research: where and when GNSS interference affected air
  traffic, published daily with a delay.
* **Not** for navigation, flight planning, air traffic control, or any operational or
  safety-of-life decision. **Not** for attributing interference to a state, unit or
  person.

## Training data

28 days (29 Sep-26 Oct 2026), five regions (Baltic, Black Sea, East Med, Punjab, and
calm Central Europe as a control). Labels: weak labels from seven rules combined by a
label model, plus synthetic attacks injected into Central Europe and East Med with exact
truth. No hand labels are used for training.

## Evaluation (test day 26 Oct unless noted)

| What | Result |
| --- | --- |
| Agreement with 300 hand labels (253 decided), interference vs normal | 96.8% [93.9-98.4]; 94.7% weighted to the whole day; same as the rules |
| Hand-labeled normal cells flagged | 0 of 112 (95% upper bound about 2.7%, rule of three) |
| Holdout day 1 Nov (150 hand labels, never used for any decision), 3+ aircraft | 92.0% weighted to the whole day (roughly 87-96%); same as the rules |
| Holdout 1 Nov, flags on cells with 1-2 aircraft | 47 of 47 right [92-100%] |
| Synthetic jamming caught, calm background (standard / hard / stress) | 1.00 / 0.986 / 0.978 (rules 0.82 / 0.52 / 0.74); `results/synthetic_*_published_model.csv` |
| Synthetic spoofing caught, calm background | 0.92 / 0.945 / 0.952 (rules 0.80 / 0.70 / 0.85); same files |
| Synthetic spoofing caught, Black Sea background, 7 days | 0.84 / 0.47 / 0.64 (rules 0.70 / 0.33 / 0.51) |
| Real flagged share, calm Central Europe, 7 days | 0.45-1.27% (rules 0.41-1.24%) |
| Calm-region flags a hand reviewer called normal (review set 1, 46 decided) | 13% [6-26%]: real false-alarm rate about 0.11% [0.05-0.21%] of judged cells |
| Hotspot flags a hand reviewer confirmed (review set 1, 26 decided) | 26 of 26 [87-100%] |
| Jammer location error, synthetic, median | 15 km (standard), 40 km (hard), 52 km (stress); `results/localization_published_model.csv` |
| Stated location uncertainty contains the truth (confident estimates) | 90-100% |

## Known limitations

* Strong on synthetic footprint edges, but on real cells it agrees with the rules and
  misses the same weak cells; the edge advantage is not proven on real data.
* Spoofing detection is much weaker over a busy hotspot (Black Sea) than over calm
  airspace.
* Cells with only 1-2 aircraft: flags there were right 47 of 47 times on the 1 Nov
  hand-labeled day (mostly Baltic) and are shown, marked as such; "normal" there is not
  validated and is shown as low coverage.
* Both the model and the rules miss weak interference at hotspot edges (an estimated
  4-13% of judged cells on 1 Nov, mostly Black Sea): affected shares are lower bounds.
* Spoofing is never independently confirmed (no usable MLAT cross-check in this data).
* Where the model and the rules disagree (blind review, 8 Nov 2026): its extra jamming
  flags were right 8 of 8, but its extra spoofing flags carried spoofing only 4 of 19
  times. Cells with one degraded aircraft out of 3-9 get no rule verdict and no training
  label; the model calls them normal, and in the Baltic the reviewer saw jamming in 17
  of 18 (evaluation protocol, "Disagreement set 1"); outside the Baltic 9 of 12 decided
  (47-91%, set 1b), not enough to call it general; the sealed test decides.
* Several jammers close together are fitted as one source.
* One labeler, two labeled days (26 Oct, 1 Nov). Coverage changes over time (new receivers) shift flag
  rates without any change in interference (docs/data.md).

## Responsible publication rules (applied by the Phase 5 product)

* Only cell-level aggregates are published. No aircraft identifiers (ICAO addresses,
  callsigns, registrations) and no individual tracks.
* Cells need at least 3 usable aircraft to be shown as normal or counted in the
  affected shares. With 1-2, a jamming or spoofing flag is shown, marked "1-2 aircraft"
  (validated on the 1 Nov holdout), and anything else is "low coverage"; no exact
  aircraft count is given. No aircraft is "no data", never "normal".
* Jammer estimates are shown as areas with their uncertainty radius (at most 200 km),
  never as points; low-confidence or wider estimates as the incident area only.
* Output is published with a delay of at least one day, with this card and the
  not-for-navigation notice.
* Each day is published only if its data-quality checks pass.

## Reproducing

```bash
pip install -e ".[dev,fast]" -c constraints.txt
gnssguard train && gnssguard evaluate && gnssguard rolling
```

The saved model records its package versions and training options in `meta.json`.
Data: adsb.lol under ODbL 1.0 (DATA_LICENSE.md). Code: MIT.
