"""Second receiver network: OpenSky Network (`gnssguard opensky-fetch`, `opensky-check`).

Declared before any data was queried: docs/evaluation_protocol.md, "Second network
check 1". OpenSky's `position_data4` holds airborne position messages with the NIC the
aircraft reported. Each query covers **one UTC hour** (the `hour` partition filter that
OpenSky requires) and is aggregated on the server to the map's grid, so only a few
thousand rows come back per hour. Queries run one at a time; finished hours are kept on
disk and skipped on a re-run.

Usage rules (https://openskynetwork.github.io/opensky-api/trino.html): always filter on
the partition column, at most two concurrent queries, split anything slower than five
minutes, no bulk downloads. Access is for non-commercial research; cite OpenSky
(Schäfer et al., IPSN 2014).
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEGRADED_NIC = 7      # as in features/cells.py
MIN_ALT_M = 914       # 3,000 ft, as in features/cells.py
WINDOW_S = 900
MIN_AIRCRAFT = 3      # matched cell-window: 3 or more usable aircraft in both networks
FLAG_SHARE = 0.30     # OpenSky flag: share of aircraft degraded
HOTSPOTS = ("baltic", "east_med", "black_sea", "punjab")
CONTROL = "central_europe"
SLOW_S = 300

PROBE_HOUR = dt.datetime(2026, 10, 26, 12, tzinfo=dt.timezone.utc)


def hour_ts(t: dt.datetime) -> int:
    return int(t.replace(minute=0, second=0, microsecond=0).timestamp())


def _boxes_sql(boxes: dict[str, tuple[float, float, float, float]]) -> tuple[str, str]:
    case = " ".join(
        f"WHEN lat BETWEEN {b[0]} AND {b[1]} AND lon BETWEEN {b[2]} AND {b[3]} THEN '{r}'"
        for r, b in boxes.items())
    where = " OR ".join(
        f"(lat BETWEEN {b[0]} AND {b[1]} AND lon BETWEEN {b[2]} AND {b[3]})"
        for b in boxes.values())
    return f"CASE {case} END", where


def hour_sql(hour: int, boxes: dict[str, tuple[float, float, float, float]]) -> str:
    """Cell-window counts for one hour: aircraft with NIC, degraded (NIC < 7 in at least
    half of their messages there) and NIC 0, airborne at or above 3,000 ft."""
    region, where = _boxes_sql(boxes)
    return f"""
WITH m AS (
  SELECT icao24, {region} AS region,
         CAST(floor(lat) AS integer) AS cell_lat, CAST(floor(lon) AS integer) AS cell_lon,
         CAST(floor(mintime / {WINDOW_S}) * {WINDOW_S} AS bigint) AS w, nic
  FROM position_data4
  WHERE hour = {int(hour)}
    AND nic IS NOT NULL AND lat IS NOT NULL AND lon IS NOT NULL AND alt >= {MIN_ALT_M}
    AND ({where})
), a AS (
  SELECT region, cell_lat, cell_lon, w, icao24, count(*) AS n_msg,
         sum(CASE WHEN nic < {DEGRADED_NIC} THEN 1 ELSE 0 END) AS deg_msg,
         sum(CASE WHEN nic = 0 THEN 1 ELSE 0 END) AS nic0_msg
  FROM m GROUP BY 1, 2, 3, 4, 5
)
SELECT region, cell_lat, cell_lon, w, count(*) AS os_ac,
       sum(CASE WHEN 2 * deg_msg >= n_msg THEN 1 ELSE 0 END) AS os_degraded,
       sum(CASE WHEN 2 * nic0_msg >= n_msg THEN 1 ELSE 0 END) AS os_nic0,
       sum(n_msg) AS os_msg
FROM a GROUP BY 1, 2, 3, 4
""".strip()


def probe_sql(box: tuple[float, float, float, float], hour: int) -> str:
    """Column meanings only: NIC distribution and the containment radius (HCR) per NIC."""
    return f"""
SELECT nic, count(*) AS n_msg, count(DISTINCT icao24) AS n_ac,
       min(hcr) AS hcr_min, approx_percentile(hcr, 0.5) AS hcr_median, max(hcr) AS hcr_max,
       min(alt) AS alt_min, max(alt) AS alt_max,
       arbitrary(typeof(mintime)) AS t_mintime, arbitrary(typeof(alt)) AS t_alt,
       arbitrary(typeof(nic)) AS t_nic
FROM position_data4
WHERE hour = {int(hour)}
  AND lat BETWEEN {box[0]} AND {box[1]} AND lon BETWEEN {box[2]} AND {box[3]}
GROUP BY nic ORDER BY nic
""".strip()


def _client():
    try:
        from pyopensky.trino import Trino
    except ImportError as e:  # pragma: no cover - optional dependency
        raise SystemExit("pip install 'pyopensky>=2.18' (or: pip install -e .[opensky])") from e
    return Trino()


def run_query(sql: str, client=None) -> pd.DataFrame:
    client = client or _client()
    t0 = time.monotonic()
    df = client.query(sql, cached=False)
    took = time.monotonic() - t0
    if took > SLOW_S:
        log.warning("query took %.0f s (over 5 minutes): use --by-region to split it", took)
    return pd.DataFrame(df)


def fetch(days: list[str], boxes: dict[str, tuple[float, float, float, float]], out: str | Path,
          by_region: bool = False, client=None, sleep_s: float = 1.0) -> list[Path]:
    """One query per hour (or per hour and region), one at a time; finished hours are
    kept and skipped."""
    out = Path(out)
    written = []
    for day in days:
        d0 = dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc)
        for h in range(24):
            f = out / day / f"{h:02d}.parquet"
            if f.exists():
                continue
            hour = hour_ts(d0 + dt.timedelta(hours=h))
            groups = [{r: b} for r, b in boxes.items()] if by_region else [boxes]
            parts = []
            for g in groups:
                parts.append(run_query(hour_sql(hour, g), client))
                time.sleep(sleep_s)
            df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(".part")
            df.to_parquet(tmp, index=False)
            tmp.replace(f)
            written.append(f)
            log.info("%s %02d UTC: %d cell-windows", day, h, len(df))
    return written


def load(root: str | Path, days: list[str]) -> pd.DataFrame:
    parts = [pd.read_parquet(f) for d in days for f in sorted((Path(root) / d).glob("*.parquet"))]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=["region", "cell_lat", "cell_lon", "window", "os_ac",
                                     "os_degraded", "os_nic0", "os_msg"])
    os_ = pd.concat(parts, ignore_index=True)
    os_ = os_[os_["region"].notna()].copy()
    for c in ("cell_lat", "cell_lon", "w", "os_ac", "os_degraded", "os_nic0", "os_msg"):
        os_[c] = pd.to_numeric(os_[c]).astype("int64")
    # an hour boundary never splits a 15-minute window, but sum to be safe
    os_ = os_.groupby(["region", "cell_lat", "cell_lon", "w"], as_index=False)[
        ["os_ac", "os_degraded", "os_nic0", "os_msg"]].sum()
    os_["window"] = pd.to_datetime(os_["w"], unit="s", utc=True)
    os_["region"] = os_["region"].astype(str)
    return os_.drop(columns="w")


# -- comparison ------------------------------------------------------------------------

def _wilson(k: int, n: int) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    from gnssguard.model.evaluate import wilson

    lo, hi = wilson(k, n)
    return round(lo, 3), round(hi, 3)


def matched(ours: pd.DataFrame, os_: pd.DataFrame) -> pd.DataFrame:
    """Our judged cell-windows joined with OpenSky's, keeping ours that OpenSky lacks."""
    o = ours.copy()
    o["window"] = pd.to_datetime(o["window"], utc=True)
    o["region"] = o["region"].astype(str)
    for c in ("cell_lat", "cell_lon"):
        o[c] = o[c].astype("int64")
    o = o[o["n_ac_usable"].fillna(0) >= MIN_AIRCRAFT]
    m = o.merge(os_, on=["region", "cell_lat", "cell_lon", "window"], how="left")
    m["os_ac"] = m["os_ac"].fillna(0).astype(int)
    m["both"] = m["os_ac"] >= MIN_AIRCRAFT
    m["our_share"] = m["n_ac_degraded"] / m["n_ac_usable"]
    m["os_share"] = np.where(m["os_ac"] > 0, m["os_degraded"] / m["os_ac"].clip(lower=1), np.nan)
    m["os_flag"] = m["os_share"] >= FLAG_SHARE
    return m


def _rates(m: pd.DataFrame, col: str) -> dict:
    b = m[m["both"]]
    jam = b[b[col].isin(["jamming", "both"])]
    norm = b[b[col] == "normal"]
    kj, nj = int(jam["os_flag"].sum()), len(jam)
    kn, nn = int(norm["os_flag"].sum()), len(norm)
    return {f"{col}_jam_n": nj, f"{col}_jam_os_flag": round(kj / nj, 3) if nj else None,
            f"{col}_jam_ci": _wilson(kj, nj),
            f"{col}_normal_n": nn, f"{col}_normal_os_flag": round(kn / nn, 3) if nn else None,
            f"{col}_normal_ci": _wilson(kn, nn)}


def summarize(m: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr

    rows = []
    groups = [(r, m[m["region"] == r]) for r in sorted(m["region"].unique())]
    groups.append(("hotspots pooled", m[m["region"].isin(HOTSPOTS)]))
    for name, g in groups:
        b = g[g["both"]]
        rho = (spearmanr(b["our_share"], b["os_share"]).statistic
               if len(b) > 2 and b["our_share"].nunique() > 1 and b["os_share"].nunique() > 1
               else np.nan)
        rows.append({"region": name, "our_judged": len(g), "both_judged": len(b),
                     "coverage": round(len(b) / len(g), 3) if len(g) else None,
                     **_rates(g, "model"), **_rates(g, "rules"),
                     "spearman_degraded_share": round(float(rho), 3) if rho == rho else None})
    return pd.DataFrame(rows)


def verdict(summary: pd.DataFrame) -> str:
    p = summary[summary["region"] == "hotspots pooled"].iloc[0]
    lo = p["model_jam_ci"][0]
    hi = p["model_normal_ci"][1]
    if lo is None or hi is None:
        return "not decided (too few matched cell-windows)"
    ok = lo >= 0.70 and hi <= 0.10
    return (f"{'agreement' if ok else 'no agreement'}: lower bound P(OpenSky flag | model "
            f"jamming) {lo:.2f} (need 0.70), upper bound P(OpenSky flag | model normal) "
            f"{hi:.2f} (need 0.10 or less)")


def disagreements(m: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    b = m[m["both"]].assign(diff=lambda x: x["our_share"] - x["os_share"])
    cols = ["region", "window", "cell_lat", "cell_lon", "n_ac_usable", "n_ac_degraded",
            "os_ac", "os_degraded", "our_share", "os_share", "model", "rules", "diff"]
    hi = b.sort_values("diff", ascending=False).head(k).assign(direction="ours higher")
    lo = b.sort_values("diff").head(k).assign(direction="OpenSky higher")
    return pd.concat([hi, lo])[cols + ["direction"]]


def run(ours: pd.DataFrame, os_: pd.DataFrame, report: str | Path) -> tuple[pd.DataFrame, str]:
    report = Path(report)
    report.mkdir(parents=True, exist_ok=True)
    m = matched(ours, os_)
    s = summarize(m)
    v = verdict(s)
    s.to_csv(report / "opensky_check.csv", index=False)
    disagreements(m).to_csv(report / "opensky_disagreements.csv", index=False)
    (report / "opensky_verdict.txt").write_text(v + "\n", encoding="utf-8")
    return s, v
