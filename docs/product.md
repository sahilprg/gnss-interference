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
| `data/<day>/summary.json` | per region: status, quality-gate reasons, aircraft, judged cells, affected shares; model versions |
| `data/<day>/<region>.json` | map file: per cell, one label, probability pair and aircraft count per 15-minute window |
| `data/<day>/<region>.csv.gz` | the same as a table, one row per cell-window with any aircraft |
| `data/<day>/incidents.json` | groups of jammed cells: time span, affected area, and a source circle for confident fits |

Format version `gnssguard-public/1`; column meanings are on the site's Data page.

**Privacy and responsible-publication rules** (docs/model_card.md), enforced in
`product/publish.py` and covered by tests:

* Cell aggregates only: no aircraft identifiers, callsigns, registrations or tracks.
* Fewer than 3 usable aircraft: "low coverage", no label, and no exact count. No
  aircraft: the cell-window is left out, which the map shows as no data, never normal.
* Jammer estimates: only for confident fits, as a circle (centre rounded to 0.1°, radius
  the stated 90% uncertainty, at least 50 km), described as a statistical estimate and
  not an attribution. A circle wider than 200 km in radius says little and is not drawn
  (about 4 in 10 confident fits on the study days); those incidents, like all others,
  show only their affected area.
* Results appear the day after, at the earliest.

## Quality gates

`product/quality.py`. A region-day that fails is withheld: its files are not written,
and its status and reasons are published instead.

| Gate | Withholds | Rule |
| --- | --- | --- |
| Complete download | the whole day | the ingest must have read the archive day to the end |
| Control alarm | the whole day | calm Central Europe flags at most 3% of judged cells (0.4-1.3% on all study days) |
| Traffic | the region | at least half its median number of aircraft over the last 14 published days |
| Coverage | the region | at least 30% of its median number of judged cells |

The traffic and coverage gates need 3 earlier published days; until then they pass and
say so in the summary. A day withheld for an incomplete download is retried
automatically; other failures would repeat and are not retried.

## The website

Static pages (`src/gnssguard/product/site/`), no build step and no tracking. The only
third-party request is the Leaflet map library, pinned by version and integrity hash.
Coastlines and borders are Natural Earth outlines shipped with the site.

* **Map:** a region, a day, and either the whole day (colour strength = share of
  15-minute windows affected) or one window, chosen on the day strip below the map,
  which shows how many cells were jammed or spoofed in each window.
* **How it works:** method, test results, known limits.
* **Data:** file formats, a code example, and the licence (ODbL, from adsb.lol).

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
