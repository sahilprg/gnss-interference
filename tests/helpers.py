"""Small hand-made point tables for Phase 2 tests."""

import numpy as np
import pandas as pd

T0 = pd.Timestamp("2026-10-26T10:00:00Z")


def flight(icao, lat0, lon0, *, n=30, step_s=10, dlat=0.0, dlon=0.01, alt=35000, nic=8,
           nac_p=9, src="adsb_icao", gs=None, t0=T0, alt_geom_offset=500):
    """A straight flight: n points, step_s apart, moving dlat/dlon degrees per point."""
    ts = [t0 + pd.Timedelta(seconds=step_s * i) for i in range(n)]
    lat = lat0 + dlat * np.arange(n)
    lon = lon0 + dlon * np.arange(n)
    if gs is None:  # true ground speed of this motion, in knots
        km = np.hypot(dlat * 111.2, dlon * 111.2 * np.cos(np.radians(lat0)))
        gs = km / (step_s / 3600) / 1.852
    return pd.DataFrame({
        "icao": icao, "ts": ts, "lat": lat, "lon": lon,
        "alt_baro": alt, "alt_geom": alt + alt_geom_offset, "on_ground": False,
        "gs": float(gs), "src": src, "nic": float(nic), "nac_p": float(nac_p), "q_age_s": 0.0,
    })


def points(*flights):
    df = pd.concat(flights, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df
