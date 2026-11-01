# Model card: gnssguard GNSS interference detector

A short, plain summary for anyone who wants to use or build on this model. Details
and all numbers: docs/model.md. How decisions were made: docs/evaluation_protocol.md.

## What it does

For each 1° × 15-minute grid cell, from open ADS-B flight data (adsb.lol), it gives:

* `p_jamming`, `p_spoofing`: calibrated probabilities that aircraft in the cell had their
  GPS jammed or spoofed;
* a label (jamming, spoofing, both, normal, unknown);
* for groups of jammed cells, a coarse estimate of where the jammer was, with an
  uncertainty radius and a confidence flag.

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
| Hand-labeled normal cells flagged | 0 of 112 |
| Synthetic jamming caught, calm background (standard / hard / stress) | 1.00 / 0.95 / 0.96 (rules 0.82 / 0.51 / 0.74) |
| Synthetic spoofing caught, calm background | 0.94 / 0.96 / 0.96 (rules 0.82 / 0.74 / 0.84) |
| Synthetic spoofing caught, Black Sea background, 7 days | 0.84 / 0.47 / 0.64 (rules 0.70 / 0.33 / 0.51) |
| Real flagged share, calm Central Europe, 7 days | 0.45-1.27% (rules 0.41-1.24%) |
| Jammer location error, synthetic, median | 13 km (standard), 29 km (hard), 44 km (stress) |
| Stated location uncertainty contains the truth (confident estimates) | 90-100% |

## Known limitations

* Strong on synthetic footprint edges, but on real cells it agrees with the rules and
  misses the same weak cells; the edge advantage is not proven on real data.
* Spoofing detection is much weaker over a busy hotspot (Black Sea) than over calm
  airspace.
* Cells with only 1-2 aircraft are not validated: treat them as low coverage.
* Spoofing is never independently confirmed (no usable MLAT cross-check in this data).
* Several jammers close together are fitted as one source.
* One labeler, one labeled day. Coverage changes over time (new receivers) shift flag
  rates without any change in interference (docs/data.md).

## Responsible publication rules (applied by the Phase 5 product)

* Only cell-level aggregates are published. No aircraft identifiers (ICAO addresses,
  callsigns, registrations) and no individual tracks.
* Cells need at least 3 usable aircraft to be shown as flagged or normal; fewer is
  shown as "low coverage", none as "no data", never as "normal".
* Jammer estimates are shown as areas with their uncertainty radius, never as points,
  and low-confidence estimates as the incident area only.
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
