"""Self-contained HTML map of one region-day: cells, labels and gold-set cells.

Open the file in a browser. It loads Leaflet from unpkg (pinned version, integrity
checked); everything else is embedded, including coastlines and borders from Natural
Earth, so the map needs no tile server. A street-map background (CARTO) can be
switched on but needs internet access.

Views
* Degraded share: share of usable aircraft with NIC < 7 (sequential blue). Cells with
  fewer than 3 usable aircraft are drawn as dashed outlines: too sparse to judge.
* Label: jamming / spoofing / both / normal from `gnssguard label`.
* Jamming probability: the label-model output.
A time slider steps through windows; "whole day" aggregates them. Hovering a cell
shows its counts; the table under the map lists the cells of the current view.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gnssguard.features.cells import CELL_KEYS
from gnssguard.labels.rules import MIN_USABLE
from gnssguard.viz.basemap import background

LABEL_CODES = {"normal": 0, "jamming": 1, "spoofing": 2, "both": 3, "unknown": 4}


def _payload(cells: pd.DataFrame, labels: pd.DataFrame | None,
             gold: pd.DataFrame | None) -> dict:
    c = cells.copy()
    c["window"] = pd.to_datetime(c["window"], utc=True)
    if labels is not None and not labels.empty:
        lab = labels[[*CELL_KEYS, "label", "p_jamming", "spoof_votes"]].copy()
        lab["window"] = pd.to_datetime(lab["window"], utc=True)
        c = c.merge(lab, on=CELL_KEYS, how="left")
    gold_ids: list[str] = []
    if gold is not None and not gold.empty:
        g = gold[[*CELL_KEYS, "gold_id"]].copy()
        g["window"] = pd.to_datetime(g["window"], utc=True)
        c = c.merge(g, on=CELL_KEYS, how="left")
        gold_ids = sorted(c["gold_id"].dropna().unique().tolist())

    cell_list = (c[["cell_lat", "cell_lon"]].drop_duplicates()
                 .sort_values(["cell_lat", "cell_lon"]).reset_index(drop=True))
    cell_idx = {(r.cell_lat, r.cell_lon): i for i, r in enumerate(cell_list.itertuples())}
    windows = sorted(c["window"].unique())
    win_idx = {w: i for i, w in enumerate(windows)}

    has_labels = "label" in c
    rows: list[list[list]] = [[] for _ in windows]
    for r in c.itertuples(index=False):
        label = LABEL_CODES.get(str(r.label), -1) if has_labels else -1
        p = getattr(r, "p_jamming", np.nan) if has_labels else np.nan
        sv = getattr(r, "spoof_votes", np.nan) if has_labels else np.nan
        gid = getattr(r, "gold_id", None)
        rows[win_idx[r.window]].append([
            cell_idx[(r.cell_lat, r.cell_lon)], int(r.n_aircraft), int(r.n_ac_usable),
            int(r.n_ac_degraded), int(r.n_ac_mlat_fallback), int(r.n_ac_speed_mismatch),
            label, -1 if pd.isna(p) else int(round(float(p) * 100)),
            -1 if pd.isna(sv) else int(sv),
            gold_ids.index(gid) if isinstance(gid, str) else -1,
        ])
    deg = float(c["cell_deg"].iloc[0]) if "cell_deg" in c else 1.0
    box = (float(cell_list.cell_lat.min()), float(cell_list.cell_lat.max() + deg),
           float(cell_list.cell_lon.min()), float(cell_list.cell_lon.max() + deg))
    return {
        "basemap": background(box),
        "region": str(c["region"].iloc[0]) if "region" in c else "",
        "date": str(c["date"].iloc[0]) if "date" in c else "",
        "cell_deg": deg,
        "window_min": int(c["window_min"].iloc[0]) if "window_min" in c else 15,
        "min_usable": MIN_USABLE,
        "has_labels": bool(has_labels),
        "cells": cell_list.to_numpy().round(4).tolist(),
        "windows": [pd.Timestamp(w).strftime("%H:%M") for w in windows],
        "rows": rows,
        "gold_ids": gold_ids,
        "bounds": [[float(cell_list.cell_lat.min()), float(cell_list.cell_lon.min())],
                   [float(cell_list.cell_lat.max() + deg), float(cell_list.cell_lon.max() + deg)]],
    }


def build_map(cells: pd.DataFrame, labels: pd.DataFrame | None = None,
              gold: pd.DataFrame | None = None, title: str | None = None) -> str:
    if cells.empty:
        raise ValueError("no cells to draw")
    data = _payload(cells, labels, gold)
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    title = title or f"GNSS interference · {data['region']} · {data['date']}"
    return (_TEMPLATE.replace("__TITLE__", html.escape(title))
            .replace("__DATA__", blob))


def write_map(out: str | Path, cells: pd.DataFrame, labels: pd.DataFrame | None = None,
              gold: pd.DataFrame | None = None) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_map(cells, labels, gold), encoding="utf-8")
    return out


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
  integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
  integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<style>
  :root {
    color-scheme: light;
    --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e;
    --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
    --jam: #2a78d6; --spoof: #eb6834; --both: #1baf7a; --normal: #c3c2b7;
    --border: rgba(11,11,11,0.10);
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--page); color: var(--ink);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { padding: 14px 16px 6px; }
  h1 { font-size: 17px; font-weight: 600; margin: 0 0 2px; }
  .sub { color: var(--ink-2); font-size: 13px; margin: 0; }
  .controls { display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center;
              padding: 8px 16px 10px; }
  .controls label { color: var(--ink-2); font-size: 13px; display: flex; gap: 6px;
                    align-items: center; }
  select, button { font: inherit; font-size: 13px; padding: 4px 8px; border-radius: 6px;
                   border: 1px solid var(--axis); background: var(--surface); color: var(--ink); }
  button { cursor: pointer; min-width: 34px; }
  input[type=range] { width: min(360px, 60vw); }
  #time { font-variant-numeric: tabular-nums; font-weight: 600; min-width: 92px; }
  #map { height: 62vh; min-height: 360px; margin: 0 16px; border-radius: 8px;
         border: 1px solid var(--border); }
  #map.leaflet-container { background: #e3eaf0; }  /* sea */
  .legend { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center;
            padding: 8px 16px; font-size: 12.5px; color: var(--ink-2); }
  .sw { display: inline-block; width: 14px; height: 14px; border-radius: 3px;
        vertical-align: -2px; margin-right: 5px; border: 1px solid var(--border); }
  .sw.dash { border: 1.5px dashed var(--muted); background: transparent; }
  .ramp { display: inline-flex; vertical-align: -2px; margin: 0 6px; }
  .ramp i { width: 14px; height: 12px; display: block; }
  .tip b { font-weight: 600; }
  .tip td { padding: 0 6px 0 0; }
  section { padding: 6px 16px 24px; }
  h2 { font-size: 14px; font-weight: 600; margin: 8px 0; }
  .tablewrap { overflow-x: auto; }
  table.cells { border-collapse: collapse; font-size: 12.5px; min-width: 640px;
                font-variant-numeric: tabular-nums; background: var(--surface); }
  table.cells th, table.cells td { padding: 4px 10px; border-bottom: 1px solid var(--grid);
                                   text-align: right; }
  table.cells th:first-child, table.cells td:first-child,
  table.cells th:nth-last-child(2), table.cells td:nth-last-child(2) { text-align: left; }
  table.cells th { color: var(--ink-2); font-weight: 600; }
  .note { color: var(--muted); font-size: 12px; margin-top: 8px; }
  .goldlbl { background: #0b0b0b; color: #fff; border: 0; font-size: 11px;
             padding: 1px 4px; box-shadow: none; }
  .goldlbl::before { display: none; }
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <p class="sub" id="subtitle"></p>
</header>
<div class="controls">
  <label>View
    <select id="mode">
      <option value="share">Degraded share</option>
      <option value="label">Label</option>
      <option value="pjam">Jamming probability</option>
    </select>
  </label>
  <label><input type="checkbox" id="day"> Whole day</label>
  <label title="Needs internet. The default background (coastlines and borders) works offline.">
    <input type="checkbox" id="street"> Street map (online)</label>
  <button id="prev" aria-label="Previous window">&#9664;</button>
  <input type="range" id="slider" min="0" value="0" aria-label="Time window">
  <button id="next" aria-label="Next window">&#9654;</button>
  <button id="play">Play</button>
  <span id="time"></span>
</div>
<div id="map" role="img" aria-label="Map of grid cells coloured by GNSS interference evidence"></div>
<div class="legend" id="legend"></div>
<section>
  <h2 id="tabletitle">Cells in this view</h2>
  <div class="tablewrap"><table class="cells" id="table"></table></div>
  <p class="note">Data: adsb.lol (ODbL 1.0). Research tool, not for navigation or
  operational decisions. Counts are distinct aircraft; faulty-transponder aircraft are
  excluded from quality counts.</p>
</section>
<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
  const D = JSON.parse(document.getElementById('data').textContent);
  const RAMP = ['#cde2fb','#b7d3f6','#9ec5f4','#86b6ef','#6da7ec','#5598e7','#3987e5',
                '#2a78d6','#256abf','#1c5cab','#184f95','#104281','#0d366b'];
  const LABELS = ['normal','jamming','spoofing','both','unknown'];
  const LCOLOR = {1: 'var(--jam)', 2: 'var(--spoof)', 3: 'var(--both)', 0: 'var(--normal)'};
  const css = getComputedStyle(document.documentElement);
  const col = v => v.startsWith('var(') ? css.getPropertyValue(v.slice(4, -1)).trim() : v;
  const seq = v => RAMP[Math.max(0, Math.min(RAMP.length - 1, Math.round(v * (RAMP.length - 1))))];
  const F = {cell: 0, ac: 1, usable: 2, deg: 3, mlat: 4, speed: 5, label: 6, p: 7, spoof: 8, gold: 9};

  document.getElementById('title').textContent =
    'GNSS interference evidence · ' + D.region + ' · ' + D.date;
  document.getElementById('subtitle').textContent =
    D.cell_deg + '° cells, ' + D.window_min + '-minute windows (UTC). Cells with fewer than ' +
    D.min_usable + ' usable aircraft are dashed outlines: too sparse to judge.';

  const modeSel = document.getElementById('mode');
  if (!D.has_labels) {
    for (const o of modeSel.options) if (o.value !== 'share') o.disabled = true;
  }

  // Whole-day aggregates per cell.
  const day = D.cells.map(() => ({ac: 0, usable: 0, deg: 0, mlat: 0, speed: 0,
                                   judged: 0, jam: 0, spoof: 0, both: 0, pmax: -1, windows: 0}));
  D.rows.forEach(rows => rows.forEach(r => {
    const a = day[r[F.cell]];
    a.windows++; a.ac = Math.max(a.ac, r[F.ac]); a.usable += r[F.usable]; a.deg += r[F.deg];
    a.mlat += r[F.mlat]; a.speed += r[F.speed]; a.pmax = Math.max(a.pmax, r[F.p]);
    if (r[F.label] >= 0 && r[F.label] !== 4) a.judged++;
    if (r[F.label] === 1) a.jam++; if (r[F.label] === 2) a.spoof++; if (r[F.label] === 3) a.both++;
  }));

  const map = L.map('map', {preferCanvas: true, zoomSnap: 0.5}).fitBounds(D.bounds, {padding: [10, 10]});
  // Offline background: Natural Earth land and borders, embedded in this page.
  const ll = ring => ring.map(p => [p[1], p[0]]);
  const LAND = {color: '#b9b7ae', weight: 0.7, opacity: 1, fillColor: '#f3f2ee', fillOpacity: 1, interactive: false};
  const BORDER = {color: '#a9a79f', weight: 0.8, opacity: 0.9, dashArray: '4 3', interactive: false};
  const land = L.polygon(D.basemap.land.map(poly => poly.map(ll)), LAND).addTo(map);
  const borders = L.polyline(D.basemap.borders.map(ll), BORDER).addTo(map);
  // Do not zoom or pan beyond the area the embedded outlines cover.
  map.setMinZoom(Math.max(map.getBoundsZoom(D.basemap.extent, true), map.getZoom() - 1));
  map.setMaxBounds(L.latLngBounds(D.basemap.extent).pad(0.05));
  map.attributionControl.setPrefix(false);
  map.attributionControl.addAttribution('<a href="https://leafletjs.com">Leaflet</a> · coastlines: Natural Earth');
  // Optional street map: needs internet; tiles from CARTO (OpenStreetMap data).
  const street = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    subdomains: 'abcd', maxZoom: 12,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
  });
  document.getElementById('street').onchange = e => {
    if (e.target.checked) {
      street.addTo(map);
      land.setStyle({fillOpacity: 0, opacity: 0}); borders.setStyle({opacity: 0});
    } else {
      map.removeLayer(street);
      land.setStyle(LAND); borders.setStyle(BORDER);
    }
  };
  const goldLayer = L.layerGroup().addTo(map);

  let current = new Map();  // cell index -> record for the view
  const rects = D.cells.map((c, i) => {
    const r = L.rectangle([[c[0], c[1]], [c[0] + D.cell_deg, c[1] + D.cell_deg]],
                          {weight: 0, fillOpacity: 0, interactive: true});
    r.bindTooltip(() => tip(i), {sticky: true, className: 'tip', direction: 'top'});
    r.addTo(map);
    return r;
  });

  const slider = document.getElementById('slider');
  slider.max = Math.max(0, D.windows.length - 1);
  // start at the busiest window
  let best = 0;
  D.rows.forEach((rows, i) => { if (rows.length > D.rows[best].length) best = i; });
  slider.value = best;

  function view() {
    const dayMode = document.getElementById('day').checked;
    current = new Map();
    if (dayMode) {
      day.forEach((a, i) => { if (a.windows) current.set(i, {day: true, ...a}); });
    } else {
      (D.rows[+slider.value] || []).forEach(r => current.set(r[F.cell], {
        day: false, ac: r[F.ac], usable: r[F.usable], deg: r[F.deg], mlat: r[F.mlat],
        speed: r[F.speed], label: r[F.label], p: r[F.p], spoof: r[F.spoof], gold: r[F.gold]}));
    }
    return dayMode;
  }

  function style(rec, mode) {
    const hidden = {weight: 0, fillOpacity: 0, opacity: 0};
    if (!rec) return hidden;
    const sparse = {color: '#898781', weight: 1, dashArray: '3 3', opacity: 0.8, fillOpacity: 0};
    const fill = c => ({color: '#fcfcfb', weight: 1, opacity: 0.9, fillColor: c, fillOpacity: 0.72});
    if (mode === 'share') {
      if (rec.day) return rec.usable >= D.min_usable ? fill(seq(rec.deg / rec.usable)) : sparse;
      return rec.usable >= D.min_usable ? fill(seq(rec.deg / rec.usable)) : sparse;
    }
    if (mode === 'pjam') {
      const p = rec.day ? rec.pmax : rec.p;
      return p >= 0 ? fill(seq(p / 100)) : sparse;
    }
    if (rec.day) {
      // Whole day: how often the cell was flagged (share of judged windows), as a ramp.
      // Spoofing or both in any window adds an orange outline.
      if (!rec.judged) return sparse;
      const flagged = rec.jam + rec.spoof + rec.both;
      if (!flagged) return {...fill(col(LCOLOR[0])), fillOpacity: 0.35};
      const s = fill(seq(flagged / rec.judged));
      if (rec.spoof + rec.both) { s.color = col(LCOLOR[2]); s.weight = 2.5; }
      return s;
    }
    if (rec.label < 0 || rec.label === 4) return sparse;
    const s = fill(col(LCOLOR[rec.label]));
    if (rec.label === 0) s.fillOpacity = 0.35;
    return s;
  }

  function pct(n, d) { return d ? Math.round(100 * n / d) + '%' : '–'; }

  function tip(i) {
    const rec = current.get(i);
    const c = D.cells[i];
    const head = '<b>' + c[0].toFixed(1) + '°, ' + c[1].toFixed(1) + '°</b> ';
    if (!rec) return head + '<br>no aircraft in this view';
    const rows = [];
    if (rec.day) {
      rows.push(['Windows with traffic', rec.windows], ['Max aircraft in a window', rec.ac],
                ['Degraded readings', rec.deg + ' of ' + rec.usable + ' (' + pct(rec.deg, rec.usable) + ')'],
                ['MLAT fallback aircraft', rec.mlat], ['Speed-mismatch aircraft', rec.speed]);
      if (D.has_labels) rows.push(['Windows judged', rec.judged],
        ['Flagged', (rec.jam + rec.spoof + rec.both) + ' (' + pct(rec.jam + rec.spoof + rec.both, rec.judged) + ')'],
        ['Jamming / spoofing / both', rec.jam + ' / ' + rec.spoof + ' / ' + rec.both]);
    } else {
      rows.push(['Aircraft', rec.ac], ['Usable aircraft', rec.usable],
                ['Degraded aircraft', rec.deg + ' (' + pct(rec.deg, rec.usable) + ')'],
                ['MLAT fallback aircraft', rec.mlat], ['Speed-mismatch aircraft', rec.speed]);
      if (D.has_labels) rows.push(['Label', rec.label >= 0 ? LABELS[rec.label] : '–'],
        ['Jamming probability', rec.p >= 0 ? rec.p + '%' : '–'], ['Spoofing votes', rec.spoof >= 0 ? rec.spoof : '–']);
      if (rec.gold >= 0) rows.push(['Gold set', D.gold_ids[rec.gold]]);
    }
    return head + '<table>' + rows.map(r => '<tr><td>' + r[0] + '</td><td><b>' + r[1] + '</b></td></tr>').join('') + '</table>';
  }

  function legend(mode) {
    const el = document.getElementById('legend');
    const sparse = '<span><i class="sw dash"></i>too sparse to judge</span>';
    const ramp = '<span class="ramp">' + RAMP.filter((_, i) => i % 2 === 0).map(c => '<i style="background:' + c + '"></i>').join('') + '</span>';
    if (mode === 'share') el.innerHTML = '<span>Share of usable aircraft with degraded NIC: 0%' + ramp + '100%</span>' + sparse;
    else if (mode === 'pjam') el.innerHTML = '<span>Jamming probability (whole day: highest window): 0%' + ramp + '100%</span>' + sparse;
    else if (document.getElementById('day').checked) el.innerHTML =
      '<span>Share of judged windows flagged (jamming, spoofing or both): 0%' + ramp + '100%</span>' +
      '<span><i class="sw" style="background:transparent;border:2.5px solid ' + col(LCOLOR[2]) + '"></i>spoofing flagged in some window</span>' +
      '<span><i class="sw" style="background:' + col(LCOLOR[0]) + ';opacity:.5"></i>never flagged</span>' + sparse;
    else el.innerHTML = ['jamming', 'spoofing', 'both'].map((n, k) =>
        '<span><i class="sw" style="background:' + col(LCOLOR[k + 1]) + '"></i>' + n + '</span>').join('') +
      '<span><i class="sw" style="background:' + col(LCOLOR[0]) + ';opacity:.5"></i>normal</span>' + sparse;
  }

  function table(mode, dayMode) {
    const t = document.getElementById('table');
    const items = [...current.entries()].filter(([, r]) => r.usable >= D.min_usable || r.judged);
    const key = ([, r]) => dayMode ? (r.jam + r.spoof + r.both) * 1000 + r.deg / Math.max(r.usable, 1)
                                   : (r.label > 0 && r.label < 4 ? 1000 : 0) + r.deg / Math.max(r.usable, 1);
    items.sort((a, b) => key(b) - key(a));
    const top = items.slice(0, 60);
    document.getElementById('tabletitle').textContent = (dayMode ? 'Whole day' : 'Window ' + D.windows[+slider.value] + ' UTC') +
      ': ' + items.length + ' cells with enough aircraft' + (items.length > 60 ? ' (top 60 shown)' : '');
    let head = '<tr><th>Cell (lat, lon)</th><th>Aircraft</th><th>Usable</th><th>Degraded</th><th>Share</th><th>MLAT fb</th><th>Speed mm</th>';
    head += D.has_labels ? (dayMode ? '<th>Jam / spoof / both windows</th><th></th>' : '<th>P(jam)</th><th>Label</th><th>Gold</th>') : '<th></th>';
    t.innerHTML = head + '</tr>' + top.map(([i, r]) => {
      const c = D.cells[i];
      let row = '<tr><td>' + c[0].toFixed(1) + ', ' + c[1].toFixed(1) + '</td><td>' + r.ac + '</td><td>' + r.usable +
        '</td><td>' + r.deg + '</td><td>' + pct(r.deg, r.usable) + '</td><td>' + r.mlat + '</td><td>' + r.speed + '</td>';
      if (D.has_labels) row += dayMode ? '<td>' + r.jam + ' / ' + r.spoof + ' / ' + r.both + '</td><td></td>'
        : '<td>' + (r.p >= 0 ? r.p + '%' : '–') + '</td><td>' + (r.label >= 0 ? LABELS[r.label] : '–') + '</td><td>' + (r.gold >= 0 ? D.gold_ids[r.gold] : '') + '</td>';
      else row += '<td></td>';
      return row + '</tr>';
    }).join('');
  }

  function draw() {
    const mode = modeSel.value;
    const dayMode = view();
    rects.forEach((r, i) => r.setStyle(style(current.get(i), mode)));
    goldLayer.clearLayers();
    if (!dayMode) current.forEach((rec, i) => {
      if (rec.gold < 0) return;
      const c = D.cells[i];
      L.rectangle([[c[0], c[1]], [c[0] + D.cell_deg, c[1] + D.cell_deg]],
                  {color: '#0b0b0b', weight: 2.5, fill: false, interactive: false})
        .bindTooltip(D.gold_ids[rec.gold], {permanent: true, direction: 'center', className: 'goldlbl'})
        .addTo(goldLayer);
    });
    document.getElementById('time').textContent = dayMode ? 'whole day'
      : D.windows[+slider.value] + '–' + nextTime(+slider.value) + ' UTC';
    ['slider', 'prev', 'next', 'play'].forEach(id => document.getElementById(id).disabled = dayMode);
    legend(mode);
    table(mode, dayMode);
  }
  function nextTime(i) {
    const [h, m] = D.windows[i].split(':').map(Number);
    const t = h * 60 + m + D.window_min;
    return String(Math.floor(t / 60) % 24).padStart(2, '0') + ':' + String(t % 60).padStart(2, '0');
  }

  let timer = null;
  document.getElementById('play').onclick = e => {
    if (timer) { clearInterval(timer); timer = null; e.target.textContent = 'Play'; return; }
    e.target.textContent = 'Pause';
    timer = setInterval(() => { slider.value = (+slider.value + 1) % D.windows.length; draw(); }, 700);
  };
  document.getElementById('prev').onclick = () => { slider.value = Math.max(0, +slider.value - 1); draw(); };
  document.getElementById('next').onclick = () => { slider.value = Math.min(+slider.max, +slider.value + 1); draw(); };
  slider.oninput = draw;
  modeSel.onchange = draw;
  document.getElementById('day').onchange = draw;
  draw();
})();
</script>
</body>
</html>
"""
