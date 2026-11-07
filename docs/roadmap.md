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
| 7 | Decode errors do not mark a day incomplete; changed regions reuse a day | **Fixed**: more than 0.1% unreadable trace files mark the day incomplete (0 on all days so far); a changed region box or stored point format needs a new run |
| 8 | Control alarm hides a possible real event | **Fixed**: input failures withhold; the control alarm publishes the day marked "needs review" |
| 9 | All regions' points in memory at once while publishing | **Fixed**: statistics first, then one region's points at a time, only where localization needs them |
| 10 | Label on the raw score, map shows the calibrated one | **Documented** in every published file (`label_threshold`, `score_meaning`) |
| 11 | `nic_cluster` and `nic_zero` near-duplicates | Open; changes labels, needs a protocol round |
| 12 | Airspeed, heading, roll, wind not extracted | **Fixed**: point format points/2 (docs/data.md); wind on 53-69% and true airspeed on 64-75% of airborne points. Re-ingest gate passed on 1 Nov: identical rows and features |
| 13 | `scipy` recorded but not declared | Not an issue: a dependency of LightGBM, pinned in constraints.txt |
| 14 | MLAT vs ADS-B difference looks like a matching error | Open |
| 15 | 1° cells are not equal area | Phase 2 |

Also done: `gnssguard benchmark` regenerates every published result into `results/`
with the model hash, training fingerprints and input hashes; an hourly-gap check marks
partial outages for review; model hash and run settings (with a hash) in every summary; the jammer
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
* **Blind relabel** (5 Nov, substitute for a second annotator): 39 of 39 decisions
  matched on interference or not, 38 of 39 on the exact label (kappa 0.96).

## Phase 2 status (6 Nov 2026)

* **Full re-ingest done** (15 Sep-4 Nov 2026, 51 days, about 196 GB streamed, no errors):
  every stored day is in point format points/2 with the air-data fields.
* **First exploration: air data vs GNSS velocity** (`scripts/air_data_check.py`, no
  decision). On 26 Oct, 1 and 3 Nov (5 regions) and Delhi on 15 Jan 2026:
  * readsb drops implausible winds, so the wind-consistency residual stays low
    everywhere (median 5-12 kt), including in spoofing cells;
  * a heading-track gap over 45 degrees is rare: 119 of 184,817 aircraft-cell-windows,
    mostly at NIC 0, where it repeats the jamming evidence;
  * the spoofing found on these days (Delhi, Punjab, East Med) is **position
    spoofing**: speed mismatches and position jumps with normal NIC and a velocity that
    still matches the heading. Velocity spoofing, which the air-data checks would catch,
    is not seen on these days.
  * So the wind and heading checks are kept as evidence for the incident pages, not as
    model features for now. The spoofing features that fit what is observed are
    position-based (plan, Phase 2 step 3): many aircraft converging on one point,
    circles and loops, a change in one aircraft's GPS-minus-baro offset, and
    jam-then-spoof sequences.
* **Baseline fixed for the gate** (evaluation protocol): integrity-only rule (share of
  aircraft with degraded NIC), set to the model's calm-region alarm rate. Development
  check on 533 hand-labeled cells: model 86% of interference found, baseline 76%;
  spoofing 56 of 80 against 0. The sealed test (after 19 Nov) decides the Phase 2 gate.
* **Second exploration: position-spoofing signatures** (`scripts/position_spoof_check.py`,
  no decision), on 1 Nov (5 regions) and Delhi on 15 Jan 2026:
  * aircraft converging on one point (within 300 m, over 1,000 ft apart) and circling
    tracks are common in spoofing cells over the East Med (50% and 83%) but also fire
    in normal busy airspace (calm Central Europe 4% and 13%: holding stacks);
  * comparing the direction between successive positions with the reported track
    is noisy (16% of normal Black Sea cells);
  * a per-aircraft jump in GPS-minus-baro altitude separates well (East Med spoofing 50%
    of cells with two or more such aircraft, normal 0.2%), but the existing feature
    `n_ac_altdev` already does better (58% vs 0.7%; Punjab 68% vs 6.7%);
  * Delhi's spoofing carries no altitude shift and no convergence: it is found by speed
    mismatches and jumps, which the model already uses.
  * So no feature is added: the model's current spoofing evidence already covers the
    spoofing seen in these regions, and the new signatures add mostly noise. They stay
    available for incident pages (plan, Phase 4).
* Next: Phase 3 ground check (EUREF stations around the Baltic), while the sealed test
  waits for 19 Nov.

## Phase 3 status (6 Nov 2026)

* **Ground check 1 declared and built** (docs/ground.md, evaluation protocol): six EUREF
  stations within about 350 km of Kaliningrad, 26 Oct-4 Nov 2026; `gnssguard
  ground-fetch` and `gnssguard ground-check`, tested on synthetic station files. The
  station archives are not reachable from the development environment, so the run is
  on the laptop.
* **Ground check 1 result** (51 of 60 station-days; KLPD00LTU is not in the archive and
  was replaced by VAIN00LVA before any result was seen): ground stations rarely see a
  signal drop (19 of 2,967 windows); aircraft flag jamming nearby in 15 of those 19
  against 60% otherwise (ratio 1.31), but shifting the aircraft series by whole days
  gives ratios up to 1.39 (median 1.10), so there is **no agreement beyond time-of-day
  patterns**, and no correlation (0.03). Expected and stated before the data: stations
  90-350 km from Kaliningrad are outside the jammer's line of sight, while aircraft see
  it most of the time. Neither support nor refutation.
* **Second network check 1 declared** (7 Nov, before any OpenSky data; docs/opensky.md):
  OpenSky's NIC reports in the same cells and windows, 26 Oct-4 Nov 2026, with a fixed
  agreement rule. Trino access until 1 Dec 2026.
* **Second network check 1 result** (240 hours): **no agreement** by the declared rule.
  Where our model says normal, OpenSky agrees (5% flagged, calm control 0.2%); where it
  says jamming, OpenSky flags 82-84% in the Black Sea and Punjab, 70% in the East Med,
  but only 45% in the Baltic, so 60% pooled (needed 70%). In the Baltic, 93% of the
  degraded aircraft in our jamming cells have no GNSS position at all (MLAT only), which
  OpenSky's position data cannot count. A follow-up declared before it ran supports
  this: counting only aircraft that report their own position, OpenSky confirms 73% of
  Baltic flags (71% pooled, 2% false flags), in line with the other regions.
* Next for Phase 3: a ground source within line of sight of a jammer (published Baltic
  spectrum monitoring); OpenSky's multilateration data for the aircraft without GNSS.

## Phase 4 status (6 Nov 2026)

* **Incident pages** (`gnssguard incidents`, docs/product.md): one page per catalogued
  event, on the site under Incidents. With every event compared with ordinary days of
  the same box (descriptive, not a test):
  * the event day **stands out** for the spoofing events: Baltic 9 Mar 2025 (no ordinary
    day of 56 as high), Delhi 4 Nov 2025 and Jan-Feb 2026 (none of 7), and partly the
    Black Sea on 3 Nov 2026 (12 of 67 ordinary days as high);
  * for the Baltic jamming events, and the Black Sea in June 2026 (40 of 66), the box is
    flagged as much on most ordinary days (e.g. 3 Nov 2026 near Kaliningrad: 38 of 56):
    the map confirms that the reported area was affected, not that the reported day was
    special. Jamming around Kaliningrad is routine in this data.
  * the index agrees with the event checks: 14 of 15 judged events found.
* **Release 0.2.0 archived** on 6 Nov 2026 (doi:10.5281/zenodo.24015201, tag v0.2.0,
  commit bf89f7f; steps in docs/release.md), before the sealed test is drawn, so the
  model, labels, baseline and questions are fixed with a public date.
* **Jammer circles** (7 Nov): a circle whose centre falls outside the region box (no
  aircraft there) is no longer drawn; 13 of 87 published circles affected, 10 of them
  Baltic (evaluation protocol, "Jammer circles outside the observed area").
* Next: the write-up after the sealed test.
