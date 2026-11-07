/* GNSS interference map: reads the static files under data/ (schema gnssguard-public/1). */
(function () {
  'use strict';
  const COL = {jam: '#a3195b', spoof: '#11707c', both: '#5b1d4f', low: '#8a949b', normal: '#ffffff',
               land: '#f7f6f1', landLine: '#b8bfb9', border: '#a7adb0', src: '#a3195b', area: '#1c2a35'};
  const NAMES = {baltic: 'Baltic', black_sea: 'Black Sea', east_med: 'Eastern Mediterranean',
                 punjab: 'Punjab', central_europe: 'Central Europe'};
  const LABELS = ['no data', 'low coverage (1-2 aircraft)', 'normal', 'jamming', 'spoofing', 'jamming and spoofing',
                  'jamming, from 1-2 aircraft', 'spoofing, from 1-2 aircraft', 'jamming and spoofing, from 1-2 aircraft'];
  // codes 6-8: model flags on cells with 1-2 aircraft (checked on the 1 Nov hand-labeled day)
  const JUDGED = v => v >= 2 && v <= 5, FLAG = v => v >= 3 && v <= 5, SPARSE = v => v >= 6;
  const $ = id => document.getElementById(id);
  const pct = (x, d = 1) => (x == null ? '–' : (100 * x).toFixed(d) + '%');
  const fmt = n => (n == null ? '–' : n.toLocaleString('en-GB'));
  const name = r => NAMES[r] || r.replace(/_/g, ' ');

  const state = {index: null, regions: [], date: null, region: null, w: -1,
                 summary: null, incidents: [], day: null, basemaps: {}};
  const map = L.map('map', {zoomSnap: 0.25, attributionControl: true});
  // Re-fit the region when the window size changes, until the viewer pans or zooms.
  let userMoved = false, fitBox = null;
  ['mousedown', 'wheel', 'touchstart', 'keydown'].forEach(ev =>
    $('map').addEventListener(ev, () => { userMoved = true; }, {passive: true}));
  new ResizeObserver(() => {
    map.invalidateSize();
    if (fitBox && !userMoved) map.fitBounds(fitBox, {padding: [12, 12]});
  }).observe($('map'));
  map.attributionControl.setPrefix(false);
  map.attributionControl.addAttribution('<a href="https://leafletjs.com">Leaflet</a> · outlines: Natural Earth · flight data: <a href="https://www.adsb.lol">adsb.lol</a> (ODbL)');
  const baseLayer = L.layerGroup().addTo(map);
  const cellLayer = L.layerGroup().addTo(map);
  const incLayer = L.layerGroup().addTo(map);
  let rects = [];

  async function getJSON(url) {
    const r = await fetch(url, {cache: 'no-cache'});
    if (!r.ok) throw new Error(url + ': ' + r.status);
    return r.json();
  }

  function readHash() {
    const h = new URLSearchParams(location.hash.slice(1));
    return {date: h.get('date'), region: h.get('region'), w: h.has('w') ? +h.get('w') : -1};
  }
  function writeHash() {
    const h = new URLSearchParams({date: state.date || '', region: state.region || ''});
    if (state.w >= 0) h.set('w', state.w);
    history.replaceState(null, '', '#' + h.toString());
  }

  function message(text) {
    let m = document.querySelector('.overlay-msg');
    if (!text) { if (m) m.remove(); return; }
    if (!m) { m = document.createElement('div'); m.className = 'overlay-msg'; $('map').appendChild(m); }
    m.textContent = text;
  }

  // ---------- start ----------
  async function start() {
    try {
      [state.index, state.regions] = await Promise.all([
        getJSON('data/index.json'), getJSON('data/regions.json').then(d => d.regions)]);
    } catch (e) {
      $('region-name').textContent = 'No data published yet';
      $('status').textContent = 'The daily pipeline has not published a day yet. Check back after its next run.';
      map.setView([45, 25], 4);
      return;
    }
    const want = readHash();
    const days = state.index.days || [];
    const sel = $('date');
    days.forEach(d => {
      const o = document.createElement('option');
      o.value = d.date; o.textContent = d.date + (d.published ? '' : ' (withheld)');
      sel.appendChild(o);
    });
    state.date = days.some(d => d.date === want.date) ? want.date : (state.index.latest || (days[0] || {}).date);
    sel.value = state.date;
    sel.onchange = () => { state.date = sel.value; loadDay(); };

    const tabs = $('regions');
    state.regions.forEach(r => {
      const b = document.createElement('button');
      b.type = 'button'; b.textContent = name(r.name); b.dataset.region = r.name;
      b.onclick = () => { state.region = r.name; loadRegion(); };
      tabs.appendChild(b);
    });
    const names = state.regions.map(r => r.name);
    state.region = names.includes(want.region) ? want.region : (names.includes('baltic') ? 'baltic' : names[0]);
    state.w = want.w;

    $('slider').oninput = e => { state.w = +e.target.value; render(); };
    $('whole-day').onclick = () => { state.w = -1; render(); };
    $('show-inc').onchange = () => drawIncidents();
    $('bars').onclick = e => {
      const i = [...$('bars').children].indexOf(e.target.closest('div'));
      if (i >= 0) { state.w = i; render(); }
    };
    healthLine();
    await loadDay();
  }

  async function loadDay() {
    if (!state.date) return;
    try { state.summary = await getJSON(`data/${state.date}/summary.json`); }
    catch (e) { state.summary = null; }
    try { state.incidents = (await getJSON(`data/${state.date}/incidents.json`)).incidents || []; }
    catch (e) { state.incidents = []; }
    try { state.reports = (await getJSON(`data/${state.date}/outside_reports.json`)).checks || []; }
    catch (e) { state.reports = []; }
    await loadRegion();
  }

  async function loadRegion() {
    [...$('regions').children].forEach(b => b.setAttribute('aria-pressed', b.dataset.region === state.region));
    const meta = state.regions.find(r => r.name === state.region);
    $('region-name').textContent = name(state.region);
    $('region-note').textContent = meta ? meta.note + (meta.role === 'control' ? ' Calm control region.' : '') : '';
    if (!state.basemaps[state.region]) {
      try { state.basemaps[state.region] = await getJSON(`data/basemap/${state.region}.json`); }
      catch (e) { state.basemaps[state.region] = {land: [], borders: []}; }
    }
    drawBase(meta);
    const rs = state.summary && state.summary.regions && state.summary.regions[state.region];
    state.day = null;
    if (rs && rs.status === 'published') {
      try { state.day = await getJSON(`data/${state.date}/${state.region}.json`); } catch (e) { state.day = null; }
    }
    panel(rs);
    reportsPanel();
    buildCells();
    render();
  }

  // ---------- map ----------
  function drawBase(meta) {
    baseLayer.clearLayers();
    const bm = state.basemaps[state.region];
    const ll = ring => ring.map(p => [p[1], p[0]]);
    L.polygon(bm.land.map(poly => poly.map(ll)), {color: COL.landLine, weight: 0.8, fillColor: COL.land,
      fillOpacity: 1, interactive: false}).addTo(baseLayer);
    L.polyline(bm.borders.map(ll), {color: COL.border, weight: 0.8, dashArray: '4 3', interactive: false}).addTo(baseLayer);
    if (meta) {
      const [a, b, c, d] = meta.bbox;
      L.rectangle([[a, c], [b, d]], {color: '#1c2a35', weight: 1, dashArray: '2 4', fill: false,
        interactive: false}).addTo(baseLayer);
      map.invalidateSize();  // the panel layout may have changed the map's size
      // fit the whole cell grid: cells start at whole degrees and can reach past the box
      fitBox = [[Math.floor(a), Math.floor(c)], [Math.ceil(b), Math.ceil(d)]];
      userMoved = false;
      map.fitBounds(fitBox, {padding: [12, 12]});
    }
  }

  function dayStats(c) {
    let judged = 0, jam = 0, spoof = 0, low = 0, sp = 0, spJam = 0, spSpoof = 0, pmax = null;
    c.label.forEach((v, i) => {
      if (JUDGED(v)) judged++;
      if (v === 3 || v === 5) jam++;
      if (v === 4 || v === 5) spoof++;
      if (v === 1) low++;
      if (SPARSE(v)) { sp++; if (v !== 7) spJam++; if (v !== 6) spSpoof++; }
      if (v >= 1 && c.p_jamming[i] != null) pmax = Math.max(pmax ?? 0, c.p_jamming[i]);
    });
    const flagged = c.label.filter(FLAG).length;
    return {judged, jam, spoof, low, sp, spJam, spSpoof, flagged, share: judged ? flagged / judged : 0, pmax};
  }

  function buildCells() {
    cellLayer.clearLayers();
    rects = [];
    if (!state.day) return;
    const deg = state.day.cell_deg;
    state.day.cells.forEach(c => {
      c.stats = dayStats(c);
      const r = L.rectangle([[c.lat, c.lon], [c.lat + deg, c.lon + deg]], {weight: 0, fillOpacity: 0});
      r.bindTooltip(() => tip(c), {sticky: true, className: 'tip', direction: 'top'});
      r.addTo(cellLayer);
      rects.push([r, c]);
    });
  }

  function styleFor(c) {
    const off = {weight: 0, opacity: 0, fillOpacity: 0};
    const lowStyle = {color: COL.low, weight: 1, opacity: 0.9, dashArray: '3 3', fillColor: COL.low, fillOpacity: 0.06};
    const normal = {color: '#c7cdd0', weight: 0.6, opacity: 1, fillColor: COL.normal, fillOpacity: 0.55};
    const fill = (col, a) => ({color: '#ffffff', weight: 0.6, opacity: 0.9, fillColor: col, fillOpacity: a});
    const sparse = col => ({color: col, weight: 1.2, opacity: 0.95, dashArray: '3 3', fillColor: col, fillOpacity: 0.3});
    const sparseCol = s => (s.spJam && s.spSpoof && Math.min(s.spJam, s.spSpoof) / Math.max(s.spJam, s.spSpoof) > 0.5
      ? COL.both : (s.spJam >= s.spSpoof ? COL.jam : COL.spoof));
    if (state.w < 0) {
      const s = c.stats;
      if (!s.flagged && s.sp) return sparse(sparseCol(s));
      if (!s.judged) return s.low ? lowStyle : off;
      if (!s.flagged) return normal;
      const col = s.jam && s.spoof && Math.min(s.jam, s.spoof) / Math.max(s.jam, s.spoof) > 0.5
        ? COL.both : (s.jam >= s.spoof ? COL.jam : COL.spoof);
      return fill(col, 0.18 + 0.72 * s.share);
    }
    const v = c.label[state.w];
    return [off, lowStyle, normal, fill(COL.jam, 0.82), fill(COL.spoof, 0.82), fill(COL.both, 0.85),
      sparse(COL.jam), sparse(COL.spoof), sparse(COL.both)][v] || off;
  }

  function tip(c) {
    const where = `${c.lat.toFixed(0)}–${(c.lat + state.day.cell_deg).toFixed(0)}°N, ${c.lon.toFixed(0)}–${(c.lon + state.day.cell_deg).toFixed(0)}°E`;
    if (state.w < 0) {
      const s = c.stats;
      const sp = s.sp ? `<br>Flagged in ${s.sp} window${s.sp > 1 ? 's' : ''} with only 1-2 aircraft` : '';
      if (!s.judged) return `<b>${where}</b><br>` + (s.sp ? sp.slice(4) + ' (not in the shares)'
        : (s.low ? 'Low coverage all day: 1-2 aircraft at a time' : 'No data'));
      return `<b>${where}</b><br>Affected in ${s.flagged} of ${s.judged} judged windows (${pct(s.share, 0)})` +
        (s.jam ? `<br>Jamming in ${s.jam}` : '') + (s.spoof ? `<br>Spoofing in ${s.spoof}` : '') + sp +
        (s.low ? `<br>${s.low} low-coverage windows not judged` : '');
    }
    const v = c.label[state.w], i = state.w;
    if (!v) return `<b>${where}</b><br>No data in this window`;
    const ac = c.aircraft[i] != null ? `${c.aircraft[i]} aircraft checked`
      : (SPARSE(v) ? 'Only 1-2 aircraft: a model flag, right 47 of 47 times on a hand-labeled day' : '1-2 aircraft (not judged)');
    const score = Math.max(c.p_jamming[i] ?? 0, c.p_spoofing[i] ?? 0);
    const band = ((state.day.reviewed_score_bands || {}).bands || [])
      .find(b => score >= b.score_from && (score < b.score_to || b.score_to >= 1));
    const reviewed = band && band.interference_share != null
      ? `<br>On hand review, cells scored like this were interference ${pct(band.interference_share, 0)} of the time (${pct(band.ci95[0], 0)}-${pct(band.ci95[1], 0)})`
      : '';
    return `<b>${where}</b><br>${LABELS[v][0].toUpperCase() + LABELS[v].slice(1)}<br>${ac}` +
      `<br>Model score: jamming ${pct(c.p_jamming[i], 0)}, spoofing ${pct(c.p_spoofing[i], 0)}` + reviewed;
  }

  function drawIncidents() {
    incLayer.clearLayers();
    if (!$('show-inc').checked || !state.day) return;
    const t = state.w >= 0 ? new Date(Date.parse(state.date + 'T00:00:00Z') + state.w * state.day.window_min * 60000) : null;
    state.incidents.filter(x => x.region === state.region).forEach(x => {
      if (t && !(new Date(x.start) <= t && t < new Date(x.end))) return;
      const when = `${x.start.slice(11, 16)}–${x.end.slice(11, 16)} UTC, ${x.cell_windows} affected cell-windows`;
      if (x.source) {
        L.circle([x.source.lat, x.source.lon], {radius: x.source.radius_km * 1000, color: COL.src, weight: 2,
          dashArray: '6 5', fill: false})
          .bindTooltip(`<b>Possible jammer area (experimental)</b><br>${when}<br>90% circle, radius ${x.source.radius_km} km.<br>Tested on simulated jammers only (median error 13 km); accuracy on real ones is not known.<br>A statistical estimate, not an attribution.`,
                       {sticky: true, className: 'tip'})
          .addTo(incLayer);
      } else {
        const a = x.area;
        L.rectangle([[a.lat_min, a.lon_min], [a.lat_max, a.lon_max]], {color: COL.area, weight: 1, dashArray: '1 4', fill: false})
          .bindTooltip(`<b>Affected area</b><br>${when}<br>` + (x.confidence === 'too_uncertain'
            ? 'A source was fitted, but its 90% circle would be over 200 km wide.'
            : x.confidence === 'outside_coverage'
            ? 'A source was fitted outside the area where aircraft are collected, so nothing there can confirm it; not drawn.'
            : 'Source not located with confidence.'), {sticky: true, className: 'tip'})
          .addTo(incLayer);
      }
    });
  }

  // ---------- panel, strip ----------
  function panel(rs) {
    const st = $('status');
    st.className = 'status';
    $('figures').innerHTML = '';
    $('downloads').innerHTML = '';
    if (!state.summary) { st.textContent = 'Nothing was published for this day.'; message(''); return; }
    if (!rs) { st.textContent = 'This region was not processed on this day.'; return; }
    if (rs.status !== 'published') {
      st.classList.add('withheld');
      st.textContent = 'Withheld by the quality checks: ' + (rs.reasons || []).join('; ') + '.';
      message('No map for this region on this day: the data did not pass the quality checks.');
    } else if ((rs.review || []).length) {
      st.classList.add('withheld');
      st.textContent = 'Published, but marked for review: ' + rs.review.join('; ') + '.';
      message('');
    } else {
      st.textContent = 'Published. Quality checks passed.';
      message('');
    }
    const rows = [['Aircraft seen', fmt(rs.aircraft)], ['Cell-windows judged', fmt(rs.judged_cells)],
      ['With 1-2 aircraft', fmt(rs.low_coverage_cells)],
      ...(rs.sparse_flagged_cells != null ? [['… of them flagged', fmt(rs.sparse_flagged_cells)]] : []),
      ['Judged and affected', pct(rs.flagged_share)],
      ['… by jamming', pct(rs.jamming_share)], ['… by spoofing', pct(rs.spoofing_share)]];
    $('figures').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
    if (rs.status === 'published') {
      const base = `data/${state.date}/`;
      $('downloads').innerHTML = [
        [`${base}${state.region}.csv.gz`, `${name(state.region)}: cell table (CSV, gzip)`],
        [`${base}${state.region}.json`, `${name(state.region)}: map file (JSON)`],
        [`${base}incidents.json`, 'All incidents of the day (JSON)'],
        [`${base}summary.json`, 'Day summary and quality checks (JSON)']]
        .map(([h, t]) => `<a href="${h}" download>${t}</a>`).join('');
    }
  }

  function reportsPanel() {
    const el = $('reports');
    const rows = (state.reports || []).filter(r => r.region === state.region);
    if (!rows.length) { el.innerHTML = '<span class="small">None for this region and day.</span>'; return; }
    const esc = t => String(t).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
    el.innerHTML = rows.map(r => `<div><b>${esc(r.reference)} (${esc(r.report)})</b><b>${esc(r.verdict)}</b>` +
      `<span>${esc(r.issuer)}: affected ${pct(r.share_inside)} inside the box, ${pct(r.share_outside)} outside` +
      ` (${r.judged_inside} cells inside). <a href="${esc(r.source)}" rel="noopener">Source</a></span></div>`).join('');
  }

  async function healthLine() {
    try {
      const h = await getJSON('data/health.json');
      $('health').textContent = h.status === 'ok'
        ? `Model check after ${h.checked_through}: no sign of drift.`
        : `Model check after ${h.checked_through}: retraining recommended (${h.reasons.length} signal${h.reasons.length > 1 ? 's' : ''}).`;
    } catch (e) { $('health').textContent = ''; }
  }

  function strip() {
    const bars = $('bars');
    bars.innerHTML = '';
    const n = state.day ? state.day.windows : 96;
    const jam = new Array(n).fill(0), spoof = new Array(n).fill(0), sparse = new Array(n).fill(0);
    if (state.day) state.day.cells.forEach(c => c.label.forEach((v, i) => {
      if (v === 3 || v === 5) jam[i]++;
      if (v === 4) spoof[i]++;
      if (SPARSE(v)) sparse[i]++;
    }));
    const top = Math.max(1, ...jam.map((j, i) => j + spoof[i]));
    for (let i = 0; i < n; i++) {
      const d = document.createElement('div');
      if (i === state.w) d.className = 'sel';
      d.title = `${clock(i)} UTC: ${jam[i]} jammed, ${spoof[i]} spoofed cells`;
      const j = document.createElement('i'); j.className = 'j'; j.style.height = (100 * jam[i] / top) + '%';
      const s = document.createElement('i'); s.className = 's'; s.style.height = (100 * spoof[i] / top) + '%';
      d.append(j, s);
      bars.appendChild(d);
    }
    return {jam, spoof, sparse};
  }

  function clock(i) {
    const m = i * (state.day ? state.day.window_min : 15);
    return String(Math.floor(m / 60)).padStart(2, '0') + ':' + String(m % 60).padStart(2, '0');
  }

  function render() {
    $('slider').value = state.w;
    rects.forEach(([r, c]) => r.setStyle(styleFor(c)));
    const s = strip();
    if (state.w < 0) {
      $('when').textContent = 'Whole day';
      const flaggedCells = state.day ? state.day.cells.filter(c => c.stats.flagged || c.stats.sp).length : 0;
      $('when-detail').textContent = state.day ? `${state.date}, ${flaggedCells} cells affected at some point` : state.date || '';
      $('whole-day').hidden = true;
    } else {
      $('when').textContent = `${clock(state.w)}–${clock(state.w + 1)} UTC`;
      $('when-detail').textContent = `${s.jam[state.w]} cells jammed, ${s.spoof[state.w]} spoofed` +
        (s.sparse[state.w] ? `, ${s.sparse[state.w]} more flagged from 1-2 aircraft` : '');
      $('whole-day').hidden = false;
    }
    drawIncidents();
    writeHash();
  }

  start();
})();
