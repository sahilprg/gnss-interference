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
