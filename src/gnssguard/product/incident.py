"""Incident pages: one static page per event of the public catalog (`gnssguard incidents`).

Each page puts what was reported next to what the map saw, so an outside reader can
check one event without running anything:

* the report: type, box, period, source, evidence level, notes (configs/events.csv);
* the event-check verdict for model and rules (the fixed rule of
  docs/evaluation_protocol.md, "Event check 1");
* a map of the box (cells by the share of judged windows the model flagged) and a
  timeline of flagged cells per 15-minute window;
* the evidence in flagged cell-windows against normal ones in the same box;
* **ordinary days**: the same box and hours on archive days that overlap no catalogued
  event, so a reader sees whether the event day stands out or the box is like this every
  day (as near Kaliningrad);
* how to reproduce it: commands, model hash, cell-file hashes.

Cell-level aggregates only, as on the map: no aircraft identifiers. Pages are written to
`results/incidents/` (committed); the site build copies them to `incidents/`.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import pandas as pd

from gnssguard.model.eventcheck import FLAGGED, MATCH, MIN_USABLE, _in_event, verdict
from gnssguard.product.publish import NOTICE, cell_table, file_hash, model_hash

EVIDENCE = (  # cell column, what it counts (per cell-window: aircraft with ...)
    ("n_ac_degraded", "degraded position integrity (low NIC)"),
    ("n_ac_nic0", "NIC 0 (no integrity at all)"),
    ("n_ac_nacp_low", "low position accuracy (NACp)"),
    ("n_ac_mlat_fallback", "positions only from ground multilateration"),
    ("n_ac_altdev", "GPS altitude far from barometric altitude"),
    ("n_ac_jump", "position jumps"),
    ("n_ac_speed_mismatch", "position change not matching reported speed"),
)
LEVELS = {
    "independently_corroborated": "independently corroborated: a measurement that does "
    "not come from ADS-B (ground spectrum monitoring or a ground receiver)",
    "observed_anomaly": "observed anomaly: interference seen in flight data by others "
    "(e.g. public trackers), not by an independent sensor",
    "suspected_mechanism": "reported: a press or official report of an incident; the "
    "exact place and time are often approximate",
}
SHORT = {
    "independently_corroborated": "independently corroborated",
    "observed_anomaly": "observed anomaly",
    "suspected_mechanism": "reported",
}
COLORS = {"jam": "#a3195b", "spoof": "#11707c", "both": "#5b1d4f", "judged": "#c5cfd4"}
MARGIN_DEG = 1.0
REPO = "https://github.com/sahilprg/gnss-interference"


# -- data --------------------------------------------------------------------------------


def _days(ev) -> list[str]:
    return [
        d.date().isoformat()
        for d in pd.date_range(
            ev.start_utc.floor("D"), ev.end_utc - pd.Timedelta(seconds=1), freq="D"
        )
    ]


def _tod_mask(cells: pd.DataFrame, ev) -> pd.Series:
    """Windows in the event's hours of the day (all, for events of a day or more)."""
    if ev.end_utc - ev.start_utc >= pd.Timedelta(days=1):
        return pd.Series(True, cells.index)
    w = pd.to_datetime(cells["window"], utc=True)
    tod = (w - w.dt.floor("D")).dt.total_seconds()
    s0 = (ev.start_utc - ev.start_utc.floor("D")).total_seconds()
    s1 = s0 + (ev.end_utc - ev.start_utc).total_seconds()
    return (tod + 900 > s0) & (tod < s1)


def _box_mask(cells: pd.DataFrame, ev) -> pd.Series:
    return (
        (cells["cell_lat"] < ev.lat_max)
        & (cells["cell_lat"] + 1 > ev.lat_min)
        & (cells["cell_lon"] < ev.lon_max)
        & (cells["cell_lon"] + 1 > ev.lon_min)
    )


class Scored:
    """Cells of one region scored by model and rules, loaded once per region and day."""

    def __init__(self, cells_dir: str | Path, model_dir: str | Path):
        from gnssguard.model.train import Detectors

        self.cells_dir, self.model_dir = Path(cells_dir), Path(model_dir)
        self.det = Detectors.load(model_dir)
        self._cache: dict[tuple[str, str], pd.DataFrame] = {}

    def available(self, region: str) -> list[str]:
        d = self.cells_dir / f"region={region}"
        return sorted(p.name.split("=", 1)[1] for p in d.glob("date=*")) if d.exists() else []

    def day(self, region: str, date: str) -> pd.DataFrame:
        key = (region, date)
        if key not in self._cache:
            from gnssguard.features.cells import load_cells
            from gnssguard.model.dataset import with_context
            from gnssguard.model.evaluate import rule_frame

            c = load_cells(self.cells_dir, [date], [region])
            if c.empty:
                self._cache[key] = c
            else:
                c = with_context(c).reset_index(drop=True)
                t = cell_table(c, self.det)
                c["p_jamming_cal"], c["p_spoofing_cal"] = t["p_jamming"], t["p_spoofing"]
                c["model"] = self.det.predict(c)["label"].astype(str).to_numpy()
                c["rules"] = rule_frame(c, self.det)["label"].astype(str).to_numpy()
                c["window"] = pd.to_datetime(c["window"], utc=True)
                self._cache[key] = c
        return self._cache[key]

    def cell_hash(self, region: str, date: str) -> str | None:
        d = self.cells_dir / f"region={region}" / f"date={date}"
        return file_hash(d) if d.exists() else None


def _share(c: pd.DataFrame, kinds, col: str = "model") -> tuple[int, int]:
    dense = c["n_ac_usable"].fillna(0) >= MIN_USABLE
    return int(dense.sum()), int((dense & c[col].isin(kinds)).sum())


def ordinary_days(scored: Scored, ev, events: pd.DataFrame) -> pd.DataFrame:
    """Same box and hours on archive days of the region that overlap no catalogued event
    in that region."""
    busy = set()
    for e in events.itertuples(index=False):
        if e.region == ev.region:
            busy.update(_days(e))
    rows = []
    for date in scored.available(ev.region):
        if date in busy:
            continue
        c = scored.day(ev.region, date)
        if c.empty:
            continue
        c = c[_box_mask(c, ev) & _tod_mask(c, ev)]
        n, k = _share(c, MATCH[ev.type])
        if n:
            rows.append({"date": date, "judged": n, "matching": k, "share": k / n})
    return pd.DataFrame(rows, columns=["date", "judged", "matching", "share"])


def incident(scored: Scored, ev, events: pd.DataFrame) -> dict:
    """Everything one page shows, as plain data (also written as JSON)."""
    out: dict = {
        "event": {
            "id": ev.event_id,
            "type": ev.type,
            "region": ev.region if isinstance(ev.region, str) else "",
            "box": [ev.lat_min, ev.lat_max, ev.lon_min, ev.lon_max],
            "start_utc": ev.start_utc.isoformat(),
            "end_utc": ev.end_utc.isoformat(),
            "source_url": ev.source_url,
            "notes": ev.notes,
            "evidence_level": ev.evidence_level,
        }
    }
    region = out["event"]["region"]
    days = _days(ev)
    have = [d for d in days if region and d in scored.available(region)]
    out["days"], out["days_with_data"] = days, have
    if not have:
        out["status"] = "no data" if region else "not checked (outside the study regions)"
        return out
    c = pd.concat([scored.day(region, d) for d in have], ignore_index=True)
    c["region"] = c["region"].astype(str)
    inside = _in_event(c, ev)
    dense = c["n_ac_usable"].fillna(0) >= MIN_USABLE
    kinds = MATCH[ev.type]
    checks = {}
    for col in ("model", "rules"):
        n_j = int((inside & dense).sum())
        n_m = int((inside & dense & c[col].isin(kinds)).sum())
        checks[col] = {
            "judged": n_j,
            "matching": n_m,
            "share": round(n_m / n_j, 3) if n_j else None,
            "verdict": verdict(n_j, n_m),
        }
    out["check"] = checks
    out["status"] = checks["model"]["verdict"]

    # map: the box with a margin, per cell over the event period
    near = (
        (c["cell_lat"] < ev.lat_max + MARGIN_DEG)
        & (c["cell_lat"] + 1 > ev.lat_min - MARGIN_DEG)
        & (c["cell_lon"] < ev.lon_max + MARGIN_DEG)
        & (c["cell_lon"] + 1 > ev.lon_min - MARGIN_DEG)
    )
    period = (c["window"] + pd.Timedelta(minutes=15) > ev.start_utc) & (c["window"] < ev.end_utc)
    m = c[near & period & dense]
    cells = []
    for (lat, lon), g in m.groupby(["cell_lat", "cell_lon"]):
        jam = g["model"].isin(["jamming", "both"]).sum()
        spf = g["model"].isin(["spoofing", "both"]).sum()
        cells.append(
            {
                "lat": float(lat),
                "lon": float(lon),
                "judged": len(g),
                "jamming": int(jam),
                "spoofing": int(spf),
            }
        )
    out["map"] = cells

    # timeline over the event period, inside the box
    ins = c[inside]
    tl = []
    for w, g in ins.groupby("window"):
        d = g[g["n_ac_usable"].fillna(0) >= MIN_USABLE]
        tl.append(
            {
                "window": w.isoformat(),
                "judged": len(d),
                "jamming": int(d["model"].isin(["jamming", "both"]).sum()),
                "spoofing": int(d["model"].isin(["spoofing", "both"]).sum()),
                "rules": int(d["rules"].isin(FLAGGED).sum()),
            }
        )
    out["timeline"] = tl

    # evidence: flagged against normal cell-windows inside the box
    d = ins[ins["n_ac_usable"].fillna(0) >= MIN_USABLE]
    flag, norm = d[d["model"].isin(kinds)], d[d["model"] == "normal"]
    out["evidence"] = [
        {"what": what, "flagged": _mean(flag, col), "normal": _mean(norm, col)}
        for col, what in EVIDENCE
        if col in d
    ]
    out["evidence_n"] = {
        "flagged": len(flag),
        "normal": len(norm),
        "usable_aircraft_flagged": _mean(flag, "n_ac_usable"),
        "usable_aircraft_normal": _mean(norm, "n_ac_usable"),
    }
    top = (
        flag.assign(p=flag[["p_jamming_cal", "p_spoofing_cal"]].max(axis=1))
        .sort_values(["p", "n_ac_degraded"], ascending=False)
        .head(12)
    )
    out["top_cells"] = [
        {
            "window": r.window.isoformat(),
            "cell": [float(r.cell_lat), float(r.cell_lon)],
            "usable_aircraft": int(r.n_ac_usable),
            "degraded": int(r.n_ac_degraded),
            "altdev": int(r.n_ac_altdev),
            "speed_mismatch": int(r.n_ac_speed_mismatch),
            "p_jamming": round(float(r.p_jamming_cal), 3),
            "p_spoofing": round(float(r.p_spoofing_cal), 3),
            "model": r.model,
            "rules": r.rules,
        }
        for r in top.itertuples()
    ]

    # ordinary days in the same box
    od = ordinary_days(scored, ev, events)
    ev_share = checks["model"]["share"]
    base = {"days": len(od), "dates": [od["date"].min(), od["date"].max()] if len(od) else None}
    if len(od) and ev_share is not None:
        base.update(
            {
                "median_share": round(float(od["share"].median()), 3),
                "p90_share": round(float(od["share"].quantile(0.9)), 3),
                "max_share": round(float(od["share"].max()), 3),
                "days_at_or_above_event": int((od["share"] >= ev_share).sum()),
                "days_with_matching_cells": int((od["matching"] >= 3).sum()),
            }
        )
    train = set(scored.det.meta.get("train_dates") or [])
    base["training_days"] = int(od["date"].isin(train).sum()) if len(od) else 0
    out["ordinary_days"] = base
    out["ordinary_series"] = [
        {"date": r.date, "share": round(r.share, 4), "judged": r.judged} for r in od.itertuples()
    ]
    out["reading"] = reading(out)
    out["provenance"] = {
        "model_hash": model_hash(scored.model_dir),
        "model_train_dates": scored.det.meta.get("train_dates"),
        "cell_files": {d: scored.cell_hash(region, d) for d in have},
    }
    return out


def _mean(df: pd.DataFrame, col: str) -> float | None:
    return round(float(df[col].mean()), 2) if len(df) and col in df else None


def reading(p: dict) -> str:
    """One plain paragraph: what the page allows a reader to conclude."""
    ch, base = p["check"]["model"], p["ordinary_days"]
    if ch["verdict"] == "too few cells":
        return (
            "Too few aircraft in the box over the period for a judgement: the map "
            "can say nothing about this event."
        )
    s = []
    if ch["verdict"] == "found":
        s.append(
            f"The map flags {ch['matching']} of {ch['judged']} judged cell-windows in "
            f"the reported box and period ({ch['share']:.0%}) with interference of the "
            "reported type."
        )
    else:
        s.append(
            f"The map does not find the reported interference: {ch['matching']} of "
            f"{ch['judged']} judged cell-windows flagged, below the fixed threshold."
        )
    if base.get("days"):
        k, n = base["days_at_or_above_event"], base["days"]
        if ch["verdict"] == "found" and k == 0:
            s.append(
                f"On none of {n} ordinary days was the same box flagged as much "
                f"(median {base['median_share']:.0%}, highest {base['max_share']:.0%}): "
                "the event day stands out."
            )
        elif ch["verdict"] == "found" and k <= max(1, round(0.1 * n)):
            s.append(
                f"Only {k} of {n} ordinary days reached the same share (median "
                f"{base['median_share']:.0%}): the event day is unusual for this box."
            )
        else:
            s.append(
                f"But {k} of {n} ordinary days reach the same share in this box "
                f"(median {base['median_share']:.0%}): interference here is routine, "
                "so the map shows that the box was affected, not that this day was "
                "special."
            )
    else:
        s.append("No ordinary days of this region are in the archive for comparison.")
    if p["event"]["type"] == "unknown":
        s.append("The report does not say whether it was jamming or spoofing; any flag counts.")
    return " ".join(s)


# -- page --------------------------------------------------------------------------------


def _esc(v) -> str:
    return html.escape("" if v is None else str(v))


def _pct(v) -> str:
    return "–" if v is None else f"{v:.0%}" if v >= 0.01 or v == 0 else f"{v:.1%}"


def svg_map(p: dict, background: dict | None) -> str:
    """Cells shaded by the share of judged windows flagged; reported box dashed."""
    lat0, lat1, lon0, lon1 = p["event"]["box"]
    lat0, lat1 = math.floor(lat0 - MARGIN_DEG), math.ceil(lat1 + MARGIN_DEG)
    lon0, lon1 = math.floor(lon0 - MARGIN_DEG), math.ceil(lon1 + MARGIN_DEG)
    k = math.cos(math.radians((lat0 + lat1) / 2))
    w_deg, h_deg = (lon1 - lon0) * k, lat1 - lat0
    W = 640
    H = max(200, round(W * h_deg / w_deg))
    sx, sy = W / (lon1 - lon0), H / (lat1 - lat0)

    def xy(lon, lat):
        return f"{(lon - lon0) * sx:.1f},{(lat1 - lat) * sy:.1f}"

    parts = [
        f'<svg viewBox="0 0 {W} {H}" role="img" class="imap" aria-label="Map of '
        f'flagged cells around the reported box">',
        f'<rect width="{W}" height="{H}" class="sea"/>',
    ]
    over = []
    if background:
        for poly in background["land"]:
            d = " ".join("M" + " L".join(xy(x, y) for x, y in ring) + "Z" for ring in poly)
            parts.append(f'<path d="{d}" class="land"/>')
            over.append(f'<path d="{d}" class="coast"/>')
        for line in background["borders"]:
            over.append(f'<path d="M{" L".join(xy(x, y) for x, y in line)}" class="border"/>')
    for c in p["map"]:
        share = max(c["jamming"], c["spoofing"]) / c["judged"]
        kind = (
            "both"
            if c["jamming"] and c["spoofing"]
            else "spoof"
            if c["spoofing"] > c["jamming"]
            else "jam"
        )
        x, y = (c["lon"] - lon0) * sx, (lat1 - c["lat"] - 1) * sy
        fill = COLORS[kind] if share > 0 else "none"
        op = 0.12 + 0.68 * share if share > 0 else 0
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{sx:.1f}" height="{sy:.1f}" '
            f'fill="{fill}" fill-opacity="{op:.2f}" class="cell"><title>'
            f"{c['lat']:.0f}–{c['lat'] + 1:.0f}°N {c['lon']:.0f}–{c['lon'] + 1:.0f}°E: "
            f"{max(c['jamming'], c['spoofing'])} of {c['judged']} judged windows "
            f"flagged</title></rect>"
        )
    parts += over
    b = p["event"]["box"]
    parts.append(
        f'<rect x="{(b[2] - lon0) * sx:.1f}" y="{(lat1 - b[1]) * sy:.1f}" '
        f'width="{(b[3] - b[2]) * sx:.1f}" height="{(b[1] - b[0]) * sy:.1f}" '
        'class="evbox"/>'
    )
    for lon in range(lon0, lon1 + 1, 2 if lon1 - lon0 > 8 else 1):
        parts.append(
            f'<text x="{(lon - lon0) * sx + 3:.1f}" y="{H - 4}" class="tick">{lon}°E</text>'
        )
    for lat in range(lat0 + 1, lat1, 1 if lat1 - lat0 <= 6 else 2):
        parts.append(f'<text x="3" y="{(lat1 - lat) * sy - 3:.1f}" class="tick">{lat}°N</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_timeline(p: dict) -> str:
    """One row per day with data (24 h each), judged cells grey, flagged in colour."""
    tl = p["timeline"]
    if not tl:
        return ""
    t0, t1 = pd.Timestamp(p["event"]["start_utc"]), pd.Timestamp(p["event"]["end_utc"])
    by_day: dict[str, list] = {}
    for r in tl:
        by_day.setdefault(r["window"][:10], []).append(r)
    W, row, lab = 640, 70, 0 if len(by_day) == 1 else 52
    H = row * len(by_day) + 18
    top = max(1, max(r["judged"] for r in tl))
    one_day = (t1 - t0) <= pd.Timedelta(days=1)
    h0 = t0 if one_day else t0.floor("D")
    span = (t1 - t0).total_seconds() if one_day else 86400.0
    plot = W - lab - 2
    bw = max(1.0, plot * 900 / span - 0.6)
    parts = [f'<svg viewBox="0 0 {W} {H}" class="itime" role="img" aria-label="Cells judged '
             'and flagged per 15-minute window">',
             f'<text x="{lab + 2}" y="10" class="tick">scale: {top} cells</text>']
    for i, (day, rows) in enumerate(sorted(by_day.items())):
        base = 14 + row * (i + 1) - 6
        start = h0 if one_day else pd.Timestamp(day, tz="UTC")
        if lab:
            parts.append(f'<text x="0" y="{base - row / 2 + 8:.1f}" class="tick">'
                         f'{pd.Timestamp(day):%d %b %Y}</text>')
        for r in rows:
            xi = lab + (pd.Timestamp(r["window"]) - start).total_seconds() / span * plot
            hj = r["judged"] / top * (row - 14)
            parts.append(f'<rect x="{xi:.1f}" y="{base - hj:.1f}" width="{bw:.1f}" '
                         f'height="{hj:.1f}" fill="{COLORS["judged"]}"/>')
            f = max(r["jamming"], r["spoofing"])
            if f:
                col = COLORS["spoof"] if r["spoofing"] > r["jamming"] else COLORS["jam"]
                hf = f / top * (row - 14)
                parts.append(f'<rect x="{xi:.1f}" y="{base - hf:.1f}" width="{bw:.1f}" '
                             f'height="{hf:.1f}" fill="{col}"><title>'
                             f'{pd.Timestamp(r["window"]):%d %b %H:%M} UTC: {f} of '
                             f'{r["judged"]} judged cells flagged</title></rect>')
        parts.append(f'<line x1="{lab}" x2="{W}" y1="{base:.1f}" y2="{base:.1f}" '
                     'stroke="var(--rule)"/>')
    hours = int(span // 3600)
    for hh in range(0, hours + 1, 3 if hours > 6 else 1):
        t = h0 + pd.Timedelta(hours=hh)
        x = lab + hh * 3600 / span * plot
        parts.append(f'<text x="{min(x, W - 34):.1f}" y="{H - 2}" class="tick">'
                     f'{t:%H:%M}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_ordinary(p: dict) -> str:
    s = p.get("ordinary_series") or []
    ev = p["check"]["model"]["share"]
    if not s or ev is None:
        return ""
    W, H = 640, 90
    vals = sorted([r["share"] for r in s])
    top = max(vals + [ev, 0.05])
    bw = (W - 10) / len(vals)
    parts = [
        f'<svg viewBox="0 0 {W} {H}" class="iord" role="img" aria-label="Flagged share '
        'in the box on ordinary days, sorted, with the event day as a line">'
    ]
    for i, v in enumerate(vals):
        hgt = v / top * (H - 16)
        parts.append(
            f'<rect x="{i * bw:.1f}" y="{H - 4 - hgt:.1f}" width="{max(bw - 1, 0.8):.1f}" '
            f'height="{hgt:.1f}" fill="{COLORS["judged"]}"/>'
        )
    y = H - 4 - ev / top * (H - 16)
    parts.append(
        f'<line x1="0" x2="{W}" y1="{y:.1f}" y2="{y:.1f}" class="evline"/>'
        f'<text x="{W - 4}" y="{max(y - 4, 10):.1f}" class="tick" text-anchor="end">'
        f"event day {_pct(ev)}</text></svg>"
    )
    return "".join(parts)


PAGE_CSS = """
.page.incident{max-width:760px}
.facts{display:grid;grid-template-columns:max-content 1fr;gap:2px 16px;margin:8px 0 16px}
.facts dt{color:var(--ink-soft)}.facts dd{margin:0}
.verdict{border-left:4px solid var(--ink);padding:8px 14px;background:var(--panel);margin:12px 0}
.verdict.found{border-color:var(--jam)}.verdict.none{border-color:var(--low)}
figure{margin:12px 0 20px}figure svg{width:100%;height:auto;display:block}
figcaption{color:var(--ink-soft);font-size:13px;margin-top:4px}
.imap .sea{fill:var(--sea)}.imap .land{fill:var(--land)}
.imap .coast{fill:none;stroke:#5f6c74;stroke-width:.7}
.imap .border{fill:none;stroke:#5f6c74;stroke-width:.6;stroke-dasharray:3 2}
.imap .cell{stroke:var(--rule);stroke-width:.4}
.imap .evbox{fill:none;stroke:var(--ink);stroke-width:1.6;stroke-dasharray:6 4}
.tick{font:11px var(--body);fill:var(--ink-soft)}
.evline{stroke:var(--jam);stroke-width:1.5}
table.ev{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0 16px}
table.ev th,table.ev td{border-bottom:1px solid var(--rule);padding:4px 6px;text-align:left}
table.ev td.d{white-space:nowrap}
table.ev td.n,table.ev th.n{text-align:right;font-variant-numeric:tabular-nums}
.wrap{overflow-x:auto}.small{font-size:13px;color:var(--ink-soft)}
code{font-size:13px}pre{background:var(--panel);padding:10px;overflow-x:auto;font-size:13px}
"""


def _head(title: str, prefix: str) -> str:
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{_esc(title)}</title><link rel="stylesheet" href="{prefix}style.css">'
        f'<style>{PAGE_CSS}</style></head><body><header class="bar"><div class="brand">'
        "<b>GNSS interference over air traffic</b><span>Incident pages: reported events "
        "against the map</span></div><nav>"
        f'<a href="{prefix}index.html">Map</a><a href="{prefix}incidents/index.html" '
        f'aria-current="page">Incidents</a><a href="{prefix}about.html">How it works</a>'
        f'<a href="{prefix}data.html">Data</a></nav></header><main class="page incident">'
    )


def page(p: dict, background: dict | None = None) -> str:
    e = p["event"]
    title = f"{e['id']}: {e['type']} event, {e['start_utc'][:10]}"
    out = [
        _head(title, "../"),
        "<p class='small'><a href='index.html'>All incidents</a></p>",
        f"<h1>{_esc(e['id'])}: reported {_esc(e['type'])}, "
        f"{_esc(e['region'] or 'outside the study regions')}</h1>",
        "<h2>What was reported</h2><dl class='facts'>",
        f"<dt>Period (UTC)</dt><dd>{_esc(e['start_utc'][:16].replace('T', ' '))} to "
        f"{_esc(e['end_utc'][:16].replace('T', ' '))}</dd>",
        f"<dt>Box</dt><dd>{e['box'][0]}–{e['box'][1]}°N, {e['box'][2]}–{e['box'][3]}°E</dd>",
        f"<dt>Evidence</dt><dd>{_esc(LEVELS.get(e['evidence_level'], e['evidence_level']))}</dd>",
        f"<dt>Source</dt><dd><a href='{_esc(e['source_url'])}'>{_esc(e['source_url'])}</a></dd>",
        f"<dt>Notes</dt><dd>{_esc(e['notes'])}</dd></dl>",
    ]
    if "check" not in p:
        out.append(
            f"<div class='verdict none'><b>{_esc(p['status'].capitalize())}.</b> "
            f"No cell files for {_esc(', '.join(p['days'][:3]))}"
            f"{' …' if len(p['days']) > 3 else ''} in this region.</div>"
        )
        return "".join(out) + _foot(p)
    ch = p["check"]
    cls = "found" if ch["model"]["verdict"] == "found" else "none"
    out.append(
        f"<h2>What the map saw</h2><div class='verdict {cls}'><b>"
        f"{_esc(ch['model']['verdict'].capitalize())}.</b> {_esc(p['reading'])}</div>"
    )
    if len(p["days_with_data"]) < len(p["days"]):
        out.append(
            f"<p class='small'>Only {len(p['days_with_data'])} of {len(p['days'])} days "
            f"of the reported period are in the archive: "
            f"{_esc(', '.join(p['days_with_data']))}.</p>"
        )
    out.append(
        "<div class='wrap'><table class='ev'><tr><th></th><th class='n'>Judged "
        "cell-windows</th><th class='n'>Flagged as reported</th><th class='n'>Share</th>"
        "<th>Verdict</th></tr>"
    )
    for k, name in (("model", "Model (published map)"), ("rules", "Hand-written rules")):
        r = ch[k]
        out.append(
            f"<tr><td>{name}</td><td class='n'>{r['judged']}</td><td class='n'>"
            f"{r['matching']}</td><td class='n'>{_pct(r['share'])}</td><td>"
            f"{_esc(r['verdict'])}</td></tr>"
        )
    out.append(
        f"</table></div><p class='small'>Judged: 1° cells and 15-minute windows "
        "overlapping the box and period, with 3 or more usable aircraft. Found: 3 or "
        "more matching (rule fixed before any event was scored, "
        f"<a href='{REPO}/blob/main/docs/evaluation_protocol.md'>evaluation protocol, "
        "event check 1</a>).</p>"
    )
    out.append(
        f"<figure>{svg_map(p, background)}<figcaption>Cells over the reported period, "
        "shaded by the share of judged windows the model flagged (darker: more of the "
        "time); dashed: the reported box. Unshaded: judged, never flagged; blank: "
        "fewer than 3 aircraft.</figcaption></figure>"
    )
    if p["timeline"]:
        out.append(
            f"<figure>{svg_timeline(p)}<figcaption>Inside the box, per 15-minute "
            "window: judged cells (grey) and cells flagged by the model (colour)."
            "</figcaption></figure>"
        )
    base = p["ordinary_days"]
    out.append("<h2>Is this day unusual for this box?</h2>")
    if base.get("median_share") is not None:
        out.append(
            f"<figure>{svg_ordinary(p)}<figcaption>Share of judged cell-windows in the same "
            f"box and hours flagged as reported, on {base['days']} ordinary days "
            f"({_esc(base['dates'][0])} to {_esc(base['dates'][1])}, sorted; days of "
            "catalogued events left out), with the event day as a line. Median "
            f"{_pct(base['median_share'])}, 90th percentile {_pct(base['p90_share'])}, "
            f"{base['days_at_or_above_event']} days at or above the event day. "
            f"{base.get('training_days', 0)} of these days are training days of the "
            "model.</figcaption></figure>"
        )
    elif base.get("days"):
        out.append("<p>No comparison: the event day itself has no judged cells in the box.</p>")
    else:
        out.append("<p>No ordinary days of this region are in the archive.</p>")
    out.append(
        "<h2>The evidence in flagged cells</h2><p>Average number of aircraft per "
        "cell-window, inside the box, in cell-windows the model flagged as reported "
        f"({p['evidence_n']['flagged']}) and in those it called normal "
        f"({p['evidence_n']['normal']}).</p><div class='wrap'><table class='ev'>"
        "<tr><th>Aircraft with …</th><th class='n'>Flagged</th><th class='n'>Normal</th></tr>"
        "<tr><td>usable positions (all)</td><td class='n'>"
        f"{_esc(p['evidence_n']['usable_aircraft_flagged'])}"
        f"</td><td class='n'>{_esc(p['evidence_n']['usable_aircraft_normal'])}</td></tr>"
    )
    for r in p["evidence"]:
        out.append(
            f"<tr><td>{_esc(r['what'])}</td><td class='n'>{_esc(r['flagged'])}</td>"
            f"<td class='n'>{_esc(r['normal'])}</td></tr>"
        )
    out.append("</table></div>")
    if p["top_cells"]:
        out.append(
            "<h3>Strongest flagged cell-windows</h3><div class='wrap'><table class='ev'>"
            "<tr><th>Window (UTC)</th><th>Cell</th><th class='n'>Aircraft</th><th class='n'>"
            "Degraded</th><th class='n'>Alt. dev.</th><th class='n'>Speed mism.</th><th "
            "class='n'>p jam</th><th class='n'>p spoof</th><th>Rules</th></tr>"
        )
        for r in p["top_cells"]:
            out.append(
                f"<tr><td>{pd.Timestamp(r['window']):%d %b %H:%M}</td><td>{r['cell'][0]:.0f}°N "
                f"{r['cell'][1]:.0f}°E</td><td class='n'>{r['usable_aircraft']}</td>"
                f"<td class='n'>{r['degraded']}</td><td class='n'>{r['altdev']}</td>"
                f"<td class='n'>{r['speed_mismatch']}</td><td class='n'>{r['p_jamming']:.2f}</td>"
                f"<td class='n'>{r['p_spoofing']:.2f}</td><td>{_esc(r['rules'])}</td></tr>"
            )
        out.append(
            "</table></div><p class='small'>Scores are the published, calibrated model "
            "scores (see the map's score notes). Cell: south-west corner of the 1° cell.</p>"
        )
    return "".join(out) + _foot(p)


def _foot(p: dict) -> str:
    e = p["event"]
    pv = p.get("provenance") or {}
    files = "".join(
        f"<li>{_esc(d)}: <code>{_esc(h)}</code></li>"
        for d, h in (pv.get("cell_files") or {}).items()
    )
    tr = pv.get("model_train_dates") or []
    days = ",".join(p["days_with_data"] or p["days"][:1])
    return (
        "<h2>Reproduce</h2><pre>"
        + "".join(f"gnssguard ingest --date {_esc(d)} --regions {_esc(e['region'] or '…')}\n"
                  for d in days.split(","))
        + f"gnssguard cells --dates {_esc(days)}\n"
        f"gnssguard incidents --events configs/events.csv --only {_esc(e['id'])}</pre>"
        + (
            f"<p class='small'>Model hash <code>{_esc(pv.get('model_hash'))}</code>"
            + (f", trained {_esc(tr[0])} to {_esc(tr[-1])}" if tr else "")
            + f". Cell files (SHA-256, first 12):</p><ul class='small'>{files}</ul>"
            if pv
            else ""
        )
        + f"<p class='small'>Event row: <code>configs/events.csv</code>, {_esc(e['id'])}. "
        f"Code and data handling: <a href='{REPO}'>{REPO}</a>. "
        "Cell-level aggregates only; no aircraft identifiers.</p>"
        f"<p class='small'><b>{_esc(NOTICE)}</b> A reported event is not established by "
        "this page: the box and times in the catalog are often approximate, and a flag "
        "says aircraft reported degraded or inconsistent positions, not who or what "
        "caused it.</p></main></body></html>"
    )


def index_page(pages: list[dict]) -> str:
    rows = []
    for p in sorted(pages, key=lambda q: q["event"]["start_utc"], reverse=True):
        e, ch = p["event"], p.get("check", {}).get("model", {})
        base = p.get("ordinary_days") or {}
        unusual = (
            "–"
            if not base.get("days") or ch.get("share") is None
            else f"{base['days_at_or_above_event']} of {base['days']}"
        )
        rows.append(
            f"<tr><td><a href='{_esc(e['id'])}.html'>{_esc(e['id'])}</a></td>"
            f"<td class='d'>{_esc(e['start_utc'][:10])}</td><td>{_esc(e['region'] or '–')}</td>"
            f"<td>{_esc(e['type'])}</td>"
            f"<td>{_esc(SHORT.get(e['evidence_level'], e['evidence_level']))}</td>"
            f"<td>{_esc(p['status'])}</td><td class='n'>{_pct(ch.get('share'))}</td>"
            f"<td class='n'>{unusual}</td></tr>"
        )
    found = sum(p["status"] == "found" for p in pages)
    decided = sum(p["status"] in ("found", "not found") for p in pages)
    return (
        _head("Incidents: reported events against the map", "../")
        + "<h1>Incidents</h1><p>Every event in the public catalog "
        "(<code>configs/events.csv</code>) with what the map saw at that place and time. "
        f"The map found {found} of {decided} events it could judge. The last column "
        "says on how many ordinary days the same box was flagged at least as much: a "
        "high number means interference there is routine and the event day does not "
        "stand out.</p><div class='wrap'><table class='ev'><tr><th>Event</th><th>Date</th>"
        "<th>Region</th><th>Reported</th><th>Evidence</th><th>Map</th><th class='n'>"
        "Share flagged</th><th class='n'>Ordinary days as high</th></tr>"
        + "".join(rows)
        + "</table></div>"
        + _foot_index()
    )


def _foot_index() -> str:
    return (
        f"<p class='small'>Evidence levels: {' '.join(_esc(v) + '.' for v in LEVELS.values())}</p>"
        f"<p class='small'><b>{_esc(NOTICE)}</b></p></main></body></html>"
    )


def build(
    cells_dir: str | Path,
    model_dir: str | Path,
    events_csv: str | Path,
    out: str | Path,
    only: list[str] | None = None,
    config: str | Path = "configs/regions.yaml",
) -> list[dict]:
    from gnssguard.config import load_regions
    from gnssguard.labels.events import load_events
    from gnssguard.viz.basemap import background

    events = load_events(events_csv)
    scored = Scored(cells_dir, model_dir)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    known = {r.name for r in load_regions(config, None)}
    pages = []
    for ev in events.itertuples(index=False):
        if only and ev.event_id not in only:
            f = out / f"{ev.event_id}.json"
            if f.exists():
                pages.append(json.loads(f.read_text(encoding="utf-8")))
            continue
        p = incident(scored, ev, events)
        if isinstance(ev.region, str) and ev.region not in known:
            p["status"] = "not checked (region not configured)"
        bg = background((ev.lat_min, ev.lat_max, ev.lon_min, ev.lon_max), 3.0, 4.0)
        (out / f"{ev.event_id}.json").write_text(
            json.dumps(p, indent=1, default=str), encoding="utf-8"
        )
        (out / f"{ev.event_id}.html").write_text(page(p, bg), encoding="utf-8")
        pages.append(p)
    (out / "index.html").write_text(index_page(pages), encoding="utf-8")
    return pages
