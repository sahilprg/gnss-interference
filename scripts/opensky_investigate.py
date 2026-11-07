"""Second network check 1, follow-up (declared in docs/evaluation_protocol.md before it
is run): recount our cells with only aircraft that broadcast their own position, as
OpenSky's position table does, and compare again.

Usage: python scripts/opensky_investigate.py 2026-10-26,...,2026-11-04
"""

import sys

import pandas as pd

from gnssguard.features.cells import load_cells
from gnssguard.model.dataset import with_context
from gnssguard.model.train import Detectors
from gnssguard.network.opensky import FLAG_SHARE, MIN_AIRCRAFT, load

REGIONS = ["baltic", "black_sea", "east_med", "punjab"]


def main(days):
    det = Detectors.load("models/detector")
    c = with_context(load_cells("data/cells", days, REGIONS)).reset_index(drop=True)
    c["model"] = det.predict(c)["label"].astype(str).to_numpy()
    c["window"] = pd.to_datetime(c["window"], utc=True)
    c["region"] = c["region"].astype(str)
    for k in ("cell_lat", "cell_lon"):
        c[k] = c[k].astype("int64")
    os_ = load("data/opensky", days)
    m = c.merge(os_, on=["region", "cell_lat", "cell_lon", "window"], how="inner")
    m["own_usable"] = (m["n_ac_usable"] - m["n_ac_mlat_fallback"]).clip(lower=0)
    m["own_degraded"] = (m["n_ac_degraded"] - m["n_ac_mlat_fallback"]).clip(lower=0)
    m = m[(m["own_usable"] >= MIN_AIRCRAFT) & (m["os_ac"] >= MIN_AIRCRAFT)]
    m["own_flag"] = m["own_degraded"] / m["own_usable"] >= FLAG_SHARE
    m["os_flag"] = m["os_degraded"] / m["os_ac"] >= FLAG_SHARE
    rows = []
    for name, g in [(r, m[m["region"] == r]) for r in REGIONS] + [("pooled", m)]:
        f, nf = g[g["own_flag"]], g[~g["own_flag"]]
        jam = g[g["model"].isin(["jamming", "both"])]
        rows.append({
            "region": name, "matched": len(g),
            "own_flag_n": len(f),
            "p_os_flag_given_own_flag": round(f["os_flag"].mean(), 3) if len(f) else None,
            "p_os_flag_given_own_normal": round(nf["os_flag"].mean(), 3) if len(nf) else None,
            "model_jam_cells": len(jam),
            "model_jam_mean_our_aircraft":
                round(jam["n_ac_usable"].mean(), 1) if len(jam) else None,
            "model_jam_mean_own_position": round(jam["own_usable"].mean(), 1) if len(jam) else None,
            "model_jam_mean_opensky": round(jam["os_ac"].mean(), 1) if len(jam) else None,
        })
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    b = out[out["region"] == "baltic"].iloc[0]["p_os_flag_given_own_flag"]
    print(f"\nBaltic P(OpenSky flag | recount flag) = {b}: explanation "
          f"{'supported' if b is not None and b >= 0.70 else 'not supported'} (declared: >= 0.70)")


if __name__ == "__main__":
    main(sys.argv[1].split(","))
