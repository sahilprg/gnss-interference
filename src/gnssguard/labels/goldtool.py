"""A self-contained HTML page for hand-labeling the gold set, one cell-window at a time.

For each gold cell-window the page shows the 5 x 5 cells around it for the window
before, the window itself and the window after, plus the cell's numbers. Rule votes
and model probabilities are deliberately left out: the gold set must not be biased by
what the model thinks.

Labels are saved in the browser as you go (localStorage) and written back with
"Download gold.csv", which keeps every other column exactly as it was.
"""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gnssguard.features.cells import load_cells
from gnssguard.labels.gold import GOLD_LABELS
from gnssguard.viz.basemap import background

CONTEXT = 2  # cells on each side of the gold cell
METRICS = ["n_aircraft", "n_ac_usable", "n_ac_degraded", "n_ac_nic0", "n_ac_nacp_low",
           "n_ac_mlat_fallback", "n_ac_jump", "n_ac_speed_mismatch", "n_ac_altdev",
           "n_ac_altpair", "n_chronic"]


def _path(rings, k: float, closed: bool) -> str:
    parts = []
    for ring in rings:
        if len(ring) < 2:
            continue
        pts = " ".join(f"{p[0] * k:.3f} {-p[1]:.3f}" for p in ring)
        parts.append("M" + pts.replace(" ", ",", 1) + ("Z" if closed else ""))
    return "".join(parts)


def _region_basemap(cells: pd.DataFrame, deg: float) -> dict:
    box = (float(cells["cell_lat"].min()), float(cells["cell_lat"].max() + deg),
           float(cells["cell_lon"].min()), float(cells["cell_lon"].max() + deg))
    k = float(np.cos(np.radians((box[0] + box[1]) / 2)))
    bg = background(box, margin_lat=3.0, margin_lon=3.0)
    rings = [ring for poly in bg["land"] for ring in poly]
    return {"k": round(k, 5), "land": _path(rings, k, True),
            "borders": _path(bg["borders"], k, False)}


def build_payload(gold_csv: str | Path, cells_dir: str | Path) -> dict:
    raw = pd.read_csv(gold_csv, dtype=str, keep_default_na=False)
    if raw.empty:
        raise ValueError(f"{gold_csv} has no rows")
    g = raw.copy()
    g["cell_lat"] = g["cell_lat"].astype(float)
    g["cell_lon"] = g["cell_lon"].astype(float)
    g["window_ts"] = pd.to_datetime(g["window"], utc=True)

    cells = load_cells(cells_dir, sorted(g["date"].unique()), sorted(g["region"].unique()))
    if cells.empty:
        raise ValueError(f"no cell tables in {cells_dir} for the gold dates; run `gnssguard cells`")
    cells["window"] = pd.to_datetime(cells["window"], utc=True)
    deg = float(cells["cell_deg"].iloc[0])
    win = pd.Timedelta(minutes=int(cells["window_min"].iloc[0]))
    by_rd = {key: grp for key, grp in cells.groupby(["region", "date"])}

    basemaps = {r: _region_basemap(cells[cells["region"] == r], deg)
                for r in sorted(g["region"].unique())}
    items = []
    for row in g.itertuples(index=False):
        c = by_rd.get((row.region, row.date))
        ctx = []
        if c is not None:
            near = c[(c["cell_lat"].sub(row.cell_lat).abs() <= CONTEXT * deg + 1e-6)
                     & (c["cell_lon"].sub(row.cell_lon).abs() <= CONTEXT * deg + 1e-6)
                     & c["window"].isin([row.window_ts - win, row.window_ts,
                                         row.window_ts + win])]
            for r in near.itertuples(index=False):
                ctx.append([int(round((r.cell_lat - row.cell_lat) / deg)),
                            int(round((r.cell_lon - row.cell_lon) / deg)),
                            int((r.window - row.window_ts) / win) + 1,
                            *[int(getattr(r, m)) for m in METRICS]])
        items.append({"id": row.gold_id, "region": row.region, "date": row.date,
                      "lat": row.cell_lat, "lon": row.cell_lon,
                      "time": row.window_ts.strftime("%H:%M"),
                      "end": (row.window_ts + win).strftime("%H:%M"), "ctx": ctx})

    fid = hashlib.sha1("|".join(raw["gold_id"] + raw["window"]).encode()).hexdigest()[:12]
    return {"file_id": fid, "deg": deg, "context": CONTEXT, "metrics": METRICS,
            "labels": list(GOLD_LABELS), "columns": list(raw.columns),
            "rows": raw.to_dict(orient="records"), "items": items, "basemaps": basemaps,
            "window_min": int(win.total_seconds() // 60), "source": str(gold_csv)}


def write_tool(gold_csv: str | Path, cells_dir: str | Path, out: str | Path) -> Path:
    data = build_payload(gold_csv, cells_dir)
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_TEMPLATE.replace("__DATA__", blob)
                   .replace("__SOURCE__", html.escape(str(gold_csv))), encoding="utf-8")
    return out


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gold set labeling</title>
<style>
  :root {
    color-scheme: light;
    --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
    --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --accent: #2a78d6;
    --sea: #e3eaf0; --land: #f3f2ee;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--page); color: var(--ink);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { position: sticky; top: 0; z-index: 2; background: var(--page);
           border-bottom: 1px solid var(--grid); padding: 10px 16px; display: flex;
           flex-wrap: wrap; gap: 8px 18px; align-items: center; }
  h1 { font-size: 16px; font-weight: 600; margin: 0; }
  .progress { display: flex; align-items: center; gap: 8px; color: var(--ink-2); font-size: 13px; }
  .bar { width: 160px; height: 8px; background: var(--grid); border-radius: 4px; overflow: hidden; }
  .bar i { display: block; height: 100%; background: var(--accent); width: 0; }
  header label { font-size: 13px; color: var(--ink-2); display: flex; gap: 6px; align-items: center; }
  input[type=text], textarea { font: inherit; font-size: 13px; padding: 5px 8px; border-radius: 6px;
           border: 1px solid var(--axis); background: var(--surface); color: var(--ink); }
  #labeler { width: 70px; }
  button { font: inherit; font-size: 13px; padding: 6px 10px; border-radius: 6px; cursor: pointer;
           border: 1px solid var(--axis); background: var(--surface); color: var(--ink); }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  main { padding: 14px 16px 40px; max-width: 1280px; }
  .card-head { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: baseline; margin-bottom: 10px; }
  .card-head h2 { font-size: 18px; margin: 0; }
  .meta { color: var(--ink-2); }
  .nav { margin-left: auto; display: flex; gap: 6px; }
  .layout { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 18px; }
  @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
  .panels { display: flex; gap: 10px; flex-wrap: wrap; }
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 8px; }
  .panel h3 { font-size: 12.5px; font-weight: 600; margin: 0 0 6px; color: var(--ink-2); }
  .panel.now h3 { color: var(--ink); }
  svg { display: block; background: var(--sea); border-radius: 4px; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 12px; color: var(--ink-2); margin: 8px 0 14px; }
  .ramp { display: inline-flex; vertical-align: -2px; margin: 0 5px; }
  .ramp i { width: 13px; height: 11px; display: block; }
  .sw { display: inline-block; width: 13px; height: 13px; vertical-align: -2px; margin-right: 4px; }
  table { border-collapse: collapse; font-size: 13px; background: var(--surface);
          font-variant-numeric: tabular-nums; width: 100%; max-width: 620px; }
  th, td { padding: 5px 10px; border-bottom: 1px solid var(--grid); text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--ink-2); font-weight: 600; }
  td.now { font-weight: 600; }
  .side { display: flex; flex-direction: column; gap: 10px; }
  .labels { display: grid; grid-template-columns: 1fr; gap: 6px; }
  .labels button { text-align: left; padding: 8px 10px; display: flex; justify-content: space-between; }
  .labels button[aria-pressed=true] { background: var(--ink); color: #fff; border-color: var(--ink); }
  kbd { font: 11px ui-monospace, monospace; border: 1px solid var(--axis); border-radius: 4px;
        padding: 0 5px; color: var(--ink-2); background: var(--page); }
  .labels button[aria-pressed=true] kbd { color: #fff; background: transparent; border-color: #fff8; }
  textarea { width: 100%; min-height: 56px; resize: vertical; }
  .guide { font-size: 12.5px; color: var(--ink-2); background: var(--surface);
           border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
  .guide b { color: var(--ink); }
  .guide p { margin: 0 0 6px; }
  .notice { font-size: 12.5px; color: var(--ink-2); }
  .empty { padding: 30px; color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Gold set labeling</h1>
  <div class="progress"><div class="bar"><i id="bar"></i></div><span id="count"></span></div>
  <label>Your initials <input type="text" id="labeler" maxlength="8" autocomplete="off"></label>
  <label><input type="checkbox" id="skip" checked> Skip labeled</label>
  <button class="primary" id="download">Download gold.csv</button>
  <span class="notice" id="notice"></span>
</header>
<main>
  <div class="card-head">
    <h2 id="gid"></h2><span class="meta" id="meta"></span>
    <div class="nav">
      <button id="prev" title="Previous (←)">&#9664; Prev</button>
      <button id="next" title="Next (→)">Next &#9654;</button>
    </div>
  </div>
  <div class="layout">
    <section>
      <div class="panels" id="panels"></div>
      <div class="legend">
        <span>Share of usable aircraft with degraded GPS: 0%<span class="ramp" id="ramp"></span>100%</span>
        <span><i class="sw" style="border:1.5px dashed #898781"></i>fewer than 3 usable aircraft</span>
        <span>numbers in cells: degraded / usable aircraft</span>
        <span><i class="sw" style="border:2.5px solid #0b0b0b"></i>the cell to label</span>
      </div>
      <table id="table"></table>
    </section>
    <aside class="side">
      <div class="labels" id="labels"></div>
      <label class="notice" for="notes">Notes (optional)</label>
      <textarea id="notes" placeholder="What made you decide?"></textarea>
      <div class="guide">
        <p><b>jamming:</b> aircraft here lose GPS integrity (degraded, NIC 0, MLAT fallback) and the pattern also shows in neighbouring cells or the windows before/after.</p>
        <p><b>spoofing:</b> integrity looks fine, but aircraft move impossibly (jumps, speed mismatch) or GPS altitude is off, especially next to a degraded area.</p>
        <p><b>both:</b> clear signs of each.</p>
        <p><b>normal:</b> healthy here and around.</p>
        <p><b>unsure:</b> one signal with nothing like it nearby, edge of coverage, or you cannot tell. Use it freely; a forced guess hurts more.</p>
        <p>Judge from the context the rules cannot see: neighbours and the windows before and after.</p>
      </div>
      <p class="notice">Saved in this browser as you go. When done (or to stop for the day), press
        <b>Download gold.csv</b> and replace <code>__SOURCE__</code> with it.
        <a href="#" id="reset">Discard browser copy</a></p>
    </aside>
  </div>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
  const D = JSON.parse(document.getElementById('data').textContent);
  const RAMP = ['#cde2fb','#b7d3f6','#9ec5f4','#86b6ef','#6da7ec','#5598e7','#3987e5',
                '#2a78d6','#256abf','#1c5cab','#184f95','#104281','#0d366b'];
  const seq = v => RAMP[Math.max(0, Math.min(12, Math.round(v * 12)))];
  const M = Object.fromEntries(D.metrics.map((m, i) => [m, i + 3]));
  const KEY = 'gnssguard-gold:' + D.file_id;
  const N = D.items.length;
  document.getElementById('ramp').innerHTML = RAMP.filter((_, i) => i % 2 === 0).map(c => '<i style="background:' + c + '"></i>').join('');

  // state: gold_id -> {label, labeler, notes}; starts from the CSV, browser copy wins
  let state = {};
  D.rows.forEach(r => { state[r.gold_id] = {label: r.label || '', labeler: r.labeler || '', notes: r.notes || ''}; });
  const notice = document.getElementById('notice');
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) || 'null');
    if (saved && saved.state) {
      Object.assign(state, saved.state);
      const n = Object.values(saved.state).filter(s => s.label).length;
      notice.textContent = 'Restored ' + n + ' labels saved in this browser (' + saved.at.slice(0, 16).replace('T', ' ') + ').';
      if (saved.labeler) document.getElementById('labeler').value = saved.labeler;
    }
  } catch (e) { notice.textContent = 'Browser storage unavailable: download often.'; }
  function save() {
    try {
      localStorage.setItem(KEY, JSON.stringify({state, labeler: document.getElementById('labeler').value,
                                                at: new Date().toISOString()}));
    } catch (e) { notice.textContent = 'Could not save in this browser: download now.'; }
  }

  let idx = D.items.findIndex(it => !state[it.id].label);
  if (idx < 0) idx = 0;

  function cellText(rec) { return rec ? rec[M.n_ac_degraded] + '/' + rec[M.n_ac_usable] : ''; }

  function panel(item, w, title) {
    const bm = D.basemaps[item.region], k = bm.k, d = D.deg, c = D.context;
    const x0 = (item.lon - c * d) * k, y0 = -(item.lat + (c + 1) * d);
    const W = (2 * c + 1) * d * k, H = (2 * c + 1) * d;
    const PH = 240, px = PH / H, wpx = Math.round(W * px);
    const recs = new Map(item.ctx.filter(r => r[2] === w).map(r => [r[0] + ',' + r[1], r]));
    let cells = '';
    for (let dy = -c; dy <= c; dy++) for (let dx = -c; dx <= c; dx++) {
      const r = recs.get(dy + ',' + dx);
      const x = (item.lon + dx * d) * k, y = -(item.lat + (dy + 1) * d);
      if (!r) continue;
      const u = r[M.n_ac_usable], g = r[M.n_ac_degraded];
      const judge = u >= 3;
      cells += '<rect x="' + x + '" y="' + y + '" width="' + d * k + '" height="' + d + '" ' +
        (judge ? 'fill="' + seq(g / u) + '" fill-opacity="0.8" stroke="#fcfcfb" stroke-width="' + 0.02 + '"'
               : 'fill="none" stroke="#898781" stroke-width="0.015" stroke-dasharray="0.06 0.05"') + '/>';
      const dark = judge && g / u > 0.55;
      cells += '<text x="' + (x + d * k / 2) + '" y="' + (y + d / 2 + 0.08) + '" font-size="' + 0.22 +
        '" text-anchor="middle" fill="' + (dark ? '#ffffff' : '#0b0b0b') + '">' + cellText(r) + '</text>';
    }
    const fx = item.lon * k, fy = -(item.lat + d);
    const focus = '<rect x="' + fx + '" y="' + fy + '" width="' + d * k + '" height="' + d +
      '" fill="none" stroke="#0b0b0b" stroke-width="0.06"/>';
    const svg = '<svg width="' + wpx + '" height="' + PH + '" viewBox="' + x0 + ' ' + y0 + ' ' + W + ' ' + H +
      '" role="img" aria-label="' + title + '">' +
      '<path d="' + bm.land + '" fill="#f3f2ee" stroke="#b9b7ae" stroke-width="0.02"/>' +
      '<path d="' + bm.borders + '" fill="none" stroke="#a9a79f" stroke-width="0.02" stroke-dasharray="0.08 0.06"/>' +
      cells + focus + '</svg>';
    return '<div class="panel' + (w === 1 ? ' now' : '') + '"><h3>' + title + '</h3>' + svg + '</div>';
  }

  const ROWS = [['Aircraft', 'n_aircraft'], ['Usable (fresh GPS quality)', 'n_ac_usable'],
    ['Degraded (NIC < 7)', 'n_ac_degraded'], ['NIC 0', 'n_ac_nic0'], ['Low NACp', 'n_ac_nacp_low'],
    ['MLAT fallback', 'n_ac_mlat_fallback'], ['Impossible jumps', 'n_ac_jump'],
    ['Speed mismatch', 'n_ac_speed_mismatch'], ['GPS-baro altitude off', 'n_ac_altdev'],
    ['  (aircraft with both altitudes)', 'n_ac_altpair'], ['Faulty transponders (excluded)', 'n_chronic']];

  function addMin(t, m) {
    const [h, mm] = t.split(':').map(Number), x = (h * 60 + mm + m + 1440) % 1440;
    return String(Math.floor(x / 60)).padStart(2, '0') + ':' + String(x % 60).padStart(2, '0');
  }

  function render() {
    const it = D.items[idx], st = state[it.id];
    document.getElementById('gid').textContent = it.id;
    document.getElementById('meta').textContent = it.region + ' · ' + it.date + ' · cell ' +
      it.lat.toFixed(1) + '°N ' + it.lon.toFixed(1) + '°E (to ' + (it.lat + D.deg).toFixed(1) + '°, ' +
      (it.lon + D.deg).toFixed(1) + '°) · ' + it.time + '–' + it.end + ' UTC';
    const before = addMin(it.time, -D.window_min), after = it.end;
    document.getElementById('panels').innerHTML = it.ctx.length
      ? panel(it, 0, 'Before · ' + before) + panel(it, 1, 'This window · ' + it.time) + panel(it, 2, 'After · ' + after)
      : '<div class="empty">No cell data found for this row. Check that `gnssguard cells` ran for ' + it.date + '.</div>';
    const focal = [0, 1, 2].map(w => it.ctx.find(r => r[0] === 0 && r[1] === 0 && r[2] === w));
    document.getElementById('table').innerHTML = '<tr><th>This cell</th><th>Before</th><th>This window</th><th>After</th></tr>' +
      ROWS.map(([name, m]) => '<tr><td>' + name + '</td>' + focal.map((r, w) =>
        '<td' + (w === 1 ? ' class="now"' : '') + '>' + (r ? r[M[m]] : '–') + '</td>').join('') + '</tr>').join('');
    document.getElementById('labels').innerHTML = D.labels.map((l, i) =>
      '<button data-label="' + l + '" aria-pressed="' + (st.label === l) + '"><span>' + l + '</span><kbd>' + (i + 1) + '</kbd></button>').join('');
    document.querySelectorAll('#labels button').forEach(b => b.onclick = () => setLabel(b.dataset.label));
    document.getElementById('notes').value = st.notes;
    const done = D.items.filter(x => state[x.id].label).length;
    document.getElementById('count').textContent = done + ' of ' + N + ' labeled · item ' + (idx + 1);
    document.getElementById('bar').style.width = (100 * done / N) + '%';
  }

  function step(dir) {
    const skip = document.getElementById('skip').checked;
    for (let i = 1; i <= N; i++) {
      const j = (idx + dir * i + N * 2) % N;
      if (!skip || !state[D.items[j].id].label) { idx = j; return render(); }
    }
    idx = (idx + dir + N) % N; render();
  }

  function setLabel(l) {
    const it = D.items[idx], st = state[it.id];
    const who = document.getElementById('labeler').value.trim();
    if (!who) { notice.textContent = 'Enter your initials first.'; document.getElementById('labeler').focus(); return; }
    st.label = l; st.labeler = who; st.notes = document.getElementById('notes').value;
    save(); notice.textContent = it.id + ' → ' + l;
    step(1);
  }

  document.getElementById('notes').oninput = e => { state[D.items[idx].id].notes = e.target.value; save(); };
  document.getElementById('labeler').oninput = save;
  document.getElementById('prev').onclick = () => { idx = (idx - 1 + N) % N; render(); };
  document.getElementById('next').onclick = () => step(1);
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
    if (e.key >= '1' && e.key <= String(D.labels.length)) setLabel(D.labels[+e.key - 1]);
    else if (e.key === 'ArrowRight') step(1);
    else if (e.key === 'ArrowLeft') { idx = (idx - 1 + N) % N; render(); }
  });

  function csvCell(v) {
    v = v == null ? '' : String(v);
    return /[",\r\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
  }
  document.getElementById('download').onclick = () => {
    const lines = [D.columns.map(csvCell).join(',')];
    D.rows.forEach(r => {
      const s = state[r.gold_id];
      lines.push(D.columns.map(c => csvCell(c === 'label' ? s.label : c === 'labeler' ? s.labeler
                                             : c === 'notes' ? s.notes : r[c])).join(','));
    });
    const url = URL.createObjectURL(new Blob([lines.join('\n') + '\n'], {type: 'text/csv'}));
    const a = Object.assign(document.createElement('a'), {href: url, download: 'gold.csv'});
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    notice.textContent = 'Downloaded. Replace ' + D.source + ' with it, then commit.';
  };
  document.getElementById('reset').onclick = e => {
    e.preventDefault();
    if (!window.confirm('Discard the labels saved in this browser and reload from the file?')) return;
    try { localStorage.removeItem(KEY); } catch (err) {}
    location.reload();
  };
  render();
})();
</script>
</body>
</html>
"""
