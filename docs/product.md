# The daily product (Phase 5)

A public website with a daily map of GNSS interference over air traffic, and the same
results as downloadable files. Everything runs on free GitHub services: a nightly
GitHub Actions job does the processing, and GitHub Pages serves the result.

Live site: https://sahilprg.github.io/gnss-interference/ (published since 1 Nov 2026,
with days from 26 Oct 2026 onwards).

## How it runs

```
05:17 UTC every night (GitHub Actions, .github/workflows/daily.yml)
  for each of the last 3 days not yet published:
    ingest      stream ~4 GB from adsb.lol, keep the 5 regions   (~5 min)
    cells       1° x 15-min cell features                          (<1 min)
    publish     model -> public files, quality gates, incidents    (<1 min)
  build site    pages + basemaps
  commit        to the gh-pages branch -> GitHub Pages
```

* The model is the one committed in `models/detector/`. Retraining stays a deliberate,
  local step (`gnssguard train`, then copy the folder and commit), so the public numbers
  always come from a model whose evaluation is documented.
* A day whose archive adsb.lol has not published yet is skipped and tried on the next
  night. Published days are never redone. Run the workflow by hand (Actions tab, "daily",
  "Run workflow") with a list of days to backfill or redo them.
* The processed flight data is deleted after each run; only the small public files are
  kept (about 0.5 MB per day, about 180 MB per year).

The same steps run locally:

```bash
gnssguard publish --dates 2026-10-24,2026-10-25,2026-10-26   # days already ingested
python -m http.server -d public 8000 --bind 127.0.0.1         # open http://127.0.0.1:8000
gnssguard daily --out public                                  # what the nightly job runs
```

## What is published

| File | Content |
| --- | --- |
| `data/index.json` | every processed day, newest first, per-region status, latest published day |
| `data/<day>/summary.json` | per region: status, quality-gate reasons, aircraft, judged cells, cells with 1-2 aircraft and how many of them are flagged, affected shares; model versions |
| `data/<day>/<region>.json` | map file: per cell, one label, model-score pair and aircraft count per 15-minute window; the label threshold and what the scores mean |
| `data/<day>/<region>.csv.gz` | the same as a table, one row per cell-window with any aircraft |
| `data/<day>/incidents.json` | groups of jammed cells: time span, affected area, and a source circle for confident fits |

Format version `gnssguard-public/1`; column meanings are on the site's Data page.

**Privacy and responsible-publication rules** (docs/model_card.md), enforced in
`product/publish.py` and covered by tests:

* Cell aggregates only: no aircraft identifiers, callsigns, registrations or tracks.
* Fewer than 3 usable aircraft: no exact count. A model flag there is published as
  `jamming_sparse`, `spoofing_sparse` or `both_sparse` (dashed on the map) and is not
  counted in the shares; anything else is "low coverage". The flags were right 47 of 47
  times on the 1 Nov hand-labeled day, mostly Baltic (docs/evaluation_protocol.md,
  holdout test 2); shown since 4 Nov 2026. No aircraft: the cell-window is left out,
  which the map shows as no data, never normal.
* Jammer estimates: only for confident fits, as a circle (centre rounded to 0.1°, radius
  the stated 90% uncertainty, at least 50 km), described as a statistical estimate and
  not an attribution. A circle wider than 200 km in radius says little and is not drawn
  (about 4 in 10 confident fits on the study days); nor is one whose centre lies
  outside the region box, where no aircraft are collected to confirm or rule it out
  (since 7 Nov 2026). Those incidents, like all others, show only their affected area.
* Results appear the day after, at the earliest.

## Quality gates

`product/quality.py`. Gates on the input withhold: a region-day that fails has no files,
only its status and reasons. The gate on the output (control alarm) only marks the day
for review (`needs_review` in the summary). Since 4 Nov 2026; before, it withheld.

| Gate | Withholds | Rule |
| --- | --- | --- |
| Complete download | the whole day | the ingest must have read the archive day to the end |
| Control alarm | nothing (marks the day "needs review") | calm Central Europe flags at most 3% of judged cells (0.4-1.3% on all study days). Above that the day is still published, with the reason shown on the map, so a real event is not hidden |
| Traffic | the region | at least half its median number of aircraft over the last 14 published days |
| Coverage | the region | at least 30% of its median number of judged cells |
| Hourly gaps | nothing (marks the region "needs review") | an hour with under 30% of its usual cell-windows with aircraft (median of the same hour over the last 14 published days, if at least 20): a partial outage |

The traffic and coverage gates need 3 earlier published days; until then they pass and
say so in the summary. A day withheld for an incomplete download is retried
automatically; other failures would repeat and are not retried.

## Daily check against outside reports

Every published day is compared with the outside reports in force that day
(`configs/advisories.csv`): NOTAMs, aviation-safety bulletins and pilot reports, each
with an approximate box and its start and end dates. For each report and region the job
writes `data/<day>/outside_reports.json`: the affected share of judged cells inside the
box and in the rest of the region, and a verdict ("seen" when inside is higher, "not
seen", or "too few cells"). The map page lists them per region.

This is a consistency check, not a confirmation: warnings cover large areas for months,
and the boxes are rough. On 1 Nov the Lebanon, Lahore and Kaliningrad reports were
"seen"; the Turkey and Cyprus NOTAM boxes showed less interference inside than outside,
because the jamming that day sat on the Russian Black Sea coast and outside the Cyprus
box.

**Adding a report** (for example a pilot report or a news story about jamming on a given
day): add one row to `configs/advisories.csv` with a new id, a source link, a rough box
and its dates (leave `end_utc` empty if it is still in force), commit and push. From the
next night it is checked every day it is in force. Never add a report without a source.

No independent outside source is published in machine-readable form for free: GPSJam
(gpsjam.org) states no data licence or download, and NOTAM services need accounts. If
GPSJam's author agrees to its data being used, a daily comparison with it can be added.

## Is it still running? (freshness, backlog, keep-alive)

Three separate signals, all measured against today's date (review of 7 Nov 2026: health
used to be measured against the newest published day, so a stopped pipeline said "ok"):

* **Freshness** in `data/health.json`: "stale" when the newest published day is more than
  3 days before today. The nightly job then opens a "Published data is stale" issue
  (which emails the owner). The map shows a banner computed in the reader's browser from
  the reader's clock, so it appears even when the nightly job does not run at all.
* **Backlog:** every day since the first published day that is not published (or was
  withheld for an incomplete download) stays queued until done, at most 4 per run (the
  newest first, then the oldest), so a long outage leaves no permanent gap.
* **Model age** to today (the retrain alarm below).

GitHub disables scheduled workflows in a public repository after 60 days without
repository activity; the nightly job re-enables itself on every run, which resets that
clock.

## Retrain alarm

After each publish, `gnssguard health` (`product/health.py`) checks the published days and
writes `data/health.json`; the map page shows the result. If it recommends retraining,
the nightly job opens a GitHub issue titled "Retrain recommended" (GitHub emails you),
once, until you close it. It fires when:

* the model's newest training day is more than 45 days older than the newest published
  day (retrain regularly anyway);
* calm Central Europe is flagged above 1.5% on average over 7 days (0.4-1.3% on all
  study days);
* in a region, the model and the rules disagree on more than 3% of judged cells over
  7 days (about 0-1% so far);
* in a region, the number of judged cells moves below half or above double its earlier
  level (receivers joined or left).

## Retraining

When the issue opens (or every month or so), on the laptop:

```bash
gnssguard ingest --start <first new day> --end <yesterday> --regions baltic,east_med,black_sea,punjab,central_europe
gnssguard cells
gnssguard label
gnssguard train --test-dates 2026-10-26,2026-11-01   # both hand-labeled days stay out
gnssguard evaluate --test-dates 2026-10-26
gnssguard rolling
```

Then run `gnssguard benchmark --model data/models/detector --rolling reports/model
--out reports/benchmark_new` and compare with `results/README.md` (the published model)
before publishing the new model: gold agreement not lower, calm Central Europe not higher,
synthetic recall not lower, events found not fewer. After publishing, run
`gnssguard benchmark --rolling reports/model` again so `results/` matches the new model. Then copy
`data/models/detector` to `models/detector`, record the new training dates in
docs/model_card.md, commit, push, and close the issue. Disk: about 70 MB per new day for
the five regions; old days can be deleted once a newer model is published.

## The website

Static pages (`src/gnssguard/product/site/`), no build step and no tracking. The only
third-party request is the Leaflet map library, pinned by version and integrity hash.
Coastlines and borders are Natural Earth outlines shipped with the site.

* **Map:** a region, a day, and either the whole day (colour strength = share of
  15-minute windows affected) or one window, chosen on the day strip below the map,
  which shows how many cells were jammed or spoofed in each window.
* **How it works:** method, test results, known limits.
* **Data:** file formats, a code example, and the licence (ODbL, from adsb.lol).

## Incident pages

`gnssguard incidents` writes one static page per event of the public catalog
(`configs/events.csv`) to `results/incidents/`, committed and copied to the site's
`incidents/` folder by every site build. A page shows, for an outside reader who runs
nothing:

* **what was reported**: type, box, period, source, evidence level and notes;
* **what the map saw**: model and rules against the fixed event-check rule
  (docs/evaluation_protocol.md, "Event check 1"), a map of the box (cells by the share of
  judged windows flagged), one timeline row per day with data, the evidence in flagged
  cell-windows against normal ones (aircraft with low NIC, NIC 0, low NACp, MLAT only,
  GPS-baro altitude gap, position jumps, speed mismatch) and the strongest flagged
  cell-windows;
* **ordinary days**: the same box and hours on every archive day of the region that
  overlaps no catalogued event there, with the event day's share against them. This is
  descriptive, added after the event checks were scored, not a test: it says whether
  the day stands out or the box is like this most days;
* **reproduce**: ingest, cells and incidents commands, model hash, cell-file hashes.

Cell-level aggregates only, as on the map. Rebuild one page with `--only EV19`; pages
not rebuilt are kept. Events with no data, or outside the study regions, get a page that
says so.

## Setup (done once; kept for forks and rebuilds)

1. Copy your trained model into the repository and push:
   `models/detector/` with `meta.json`, `y_jam.txt`, `y_spoof.txt`.
2. On GitHub, Actions tab, "daily", "Run workflow": enter the days to backfill (for
   example the last week). The first run creates the `gh-pages` branch.
3. Settings, Pages: Source "Deploy from a branch", branch `gh-pages`, folder `/ (root)`.
4. The site appears at https://sahilprg.github.io/gnss-interference/ within a few
   minutes; from then on the nightly job keeps it current.

If the push to `gh-pages` fails with a permissions error: Settings, Actions, General,
Workflow permissions, "Read and write permissions". GitHub pauses scheduled workflows in
repositories without activity for 60 days; the nightly commits count as activity.
