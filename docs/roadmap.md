# Roadmap

The next gains come from trustworthy evidence and correct data handling, not a bigger
model. Five phases, each closed by a gate; work in a later phase is not trusted until
the gate before it passes.

| Phase | Goal | Gate |
| --- | --- | --- |
| 0. Correctness | every number comes from correct data handling | regression tests pass; one re-ingested day explained |
| 1. Independent evidence | real-data results stop being agreement with the system's own rules | dozens of corroborated incidents; a sealed two-annotator test set |
| 2. Signals and baselines | catch real interference the rules miss, without more false alerts | beats an integrity-only baseline on the sealed set at a fixed false-alert rate |
| 3. External validation | confirm detections with sensors that share nothing with this pipeline | second receiver network; one ground-sensor comparison |
| 4. Product and users | find out who needs it before building more | an outside analyst reproduces an incident; two pilots give feedback |

## Phase 0 status (4 Nov 2026)

Code findings from the review of commit 05709be, checked against the code and data:

| # | Finding | Status |
| --- | --- | --- |
| 1 | Old NIC looks fresh after a NACp-only update | **Not an issue.** readsb's details are full state snapshots (NIC in 100%, NACp in 98%), and readsb drops expired fields itself (docs/data.md) |
| 2 | `stale` and `new_leg` not loaded | Loaded now. `stale` marks the first position after a 20 s gap, not an old position. Leaving those pairs out of the motion checks was tested and **rejected** (evaluation protocol, round 3) |
| 3 | Calibration on the early-stopping day, on weak labels | Wording fixed: scores are called model scores everywhere, with their meaning and the 0.5 label threshold in every published file. Separate calibration split: Phase 1 |
| 4 | Features use the whole day | Fine for a next-day map; a causal feature path is needed only before any real-time claim |
| 5, 6 | Synthetic truth rebuilt from surviving points; mixed events merged | Open (affects synthetic scores only); needs a protocol round |
| 7 | Decode errors do not mark a day incomplete; changed regions reuse a day | **Fixed**: more than 0.1% unreadable trace files mark the day incomplete (0 on all 32 days so far); a changed region box needs a new run |
| 8 | Control alarm hides a possible real event | **Fixed**: input failures withhold; the control alarm publishes the day marked "needs review" |
| 9 | All regions' points in memory at once while publishing | Open (low: every nightly run has fitted easily) |
| 10 | Label on the raw score, map shows the calibrated one | **Documented** in every published file (`label_threshold`, `score_meaning`) |
| 11 | `nic_cluster` and `nic_zero` near-duplicates | Open; changes labels, needs a protocol round |
| 12 | Airspeed, heading, roll, wind not extracted | Open; needs the one-time re-ingest (present in about 37% of state snapshots) |
| 13 | `scipy` recorded but not declared | Not an issue: a dependency of LightGBM, pinned in constraints.txt |
| 14 | MLAT vs ADS-B difference looks like a matching error | Open |
| 15 | 1° cells are not equal area | Phase 2 |

Also done: model hash and run settings (with a hash) in every summary; the jammer
circles are labelled experimental; "0 of 112 normal cells flagged" is shown with its
95% upper bound (2.7%).

## Phase 1 status (4 Nov 2026)

* **Event catalog started** (`configs/events.csv`, docs/labels.md section 6): 13 public
  events from 2024-2025 with sources and evidence levels, 5 of them backed by ground
  measurements independent of ADS-B (Estonian and Lithuanian spectrum monitoring,
  a ground receiver in Gdynia).
* **Event check 1** (evaluation protocol): model and rules both found 9 of 10 checkable
  events, 4 of 4 with ground measurements; they missed the first Kaliningrad spoofing
  (9 Nov 2024); Delhi spoofing on 4 Nov 2025 was flagged in an afternoon window.
* **Review set 1** (false alarms, 100 hand-labeled cells from 27 Oct-3 Nov): 13% of
  calm-region flags were false alarms to the reviewer, so the real calm false-alarm rate
  is about 0.1% of judged cells; hotspot flags 26 of 26 right; calm spoofing flags were
  the false ones.
* Next: more events, especially India and the Black Sea; a second annotator for the
  review sets; ground-station data (EUREF) for calm Central Europe.
* Not possible from this archive: Punjab before 2026 (no adsb.lol data there on the
  checked days).
* **Trust report** (evaluation protocol): intervals over test days; the published scores
  are almost always below 0.1 or above 0.9, and a low score over the Black Sea is wrong
  about a third of the time on hand-labeled cells. A separate calibration day would not
  fix this (the training labels are as confident as the model); calibrating against a
  labeled set would. 95% of test-day aircraft were also seen in training.
* **Event check 2:** six more events (Vilnius 2025, Delhi Jan-Feb 2026, Constanta Jun
  2026, western Black Sea and Kaliningrad on 3 Nov 2026, Lahore listed only): 5 of 5
  checkable found, 14 of 15 overall. In Delhi the model flags more spoofing than the
  rules (135 vs 92 cell-windows): the first real-data difference between them.
* **Sealed set declared:** calibration set (15-21 Sep 2026, 150 cells, drawn) and sealed
  test set (5-18 Nov 2026, 250 cells, drawn on 19 Nov or later).
* **Calibration set labeled** (5 Nov): published scores now come with a reviewer table
  (below 0.1: 10% interference; 0.1-0.9: 60%; 0.9 or more: 92%), on the map and in the
  files. The middle band holds misses the 0.5 label threshold calls normal; a 0.1
  threshold is pre-declared as question 5 of the sealed test.
