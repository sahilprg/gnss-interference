"""Assemble the static website: pages, script, styles and per-region basemaps.

`build_site(out)` copies the page files shipped with the package into `out` and writes
`data/regions.json` and one offline basemap per region (Natural Earth outlines, public
domain), so the site needs no tile server and makes no third-party requests except for
the pinned Leaflet library. The committed incident pages (`results/incidents`,
`gnssguard incidents`) are copied to `incidents/`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from gnssguard.config import load_regions
from gnssguard.viz.basemap import background

ASSETS = Path(__file__).parent / "site"
PAGES = ("index.html", "about.html", "data.html", "app.js", "style.css")


INCIDENTS = Path("results/incidents")  # committed incident pages (`gnssguard incidents`)


def build_site(out: str | Path, config: str | Path = "configs/regions.yaml",
               regions: list[str] | None = None,
               incidents: str | Path | None = INCIDENTS) -> Path:
    out = Path(out)
    (out / "data" / "basemap").mkdir(parents=True, exist_ok=True)
    for name in PAGES:
        shutil.copyfile(ASSETS / name, out / name)
    (out / ".nojekyll").write_text("", encoding="utf-8")  # serve files as they are
    if incidents and (Path(incidents) / "index.html").exists():
        (out / "incidents").mkdir(exist_ok=True)
        for f in Path(incidents).glob("*"):
            if f.suffix in (".html", ".json"):
                shutil.copyfile(f, out / "incidents" / f.name)
    meta = []
    for r in load_regions(config, regions):
        box = (r.lat_min, r.lat_max, r.lon_min, r.lon_max)
        bg = background(box, margin_lat=6.0, margin_lon=16.0)
        (out / "data" / "basemap" / f"{r.name}.json").write_text(
            json.dumps(bg, separators=(",", ":")), encoding="utf-8")
        meta.append({"name": r.name, "role": r.role, "note": r.note,
                     "bbox": [r.lat_min, r.lat_max, r.lon_min, r.lon_max]})
    (out / "data" / "regions.json").write_text(json.dumps({"regions": meta}, indent=1),
                                                encoding="utf-8")
    return out
