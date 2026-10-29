"""Region configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Region:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    role: str = "hotspot"
    note: str = ""

    def contains(self, lat: float, lon: float) -> bool:
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max


def load_regions(path: str | Path, only: list[str] | None = None) -> list[Region]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)["regions"]
    regions = []
    for name, spec in raw.items():
        if only and name not in only:
            continue
        lat_min, lat_max, lon_min, lon_max = (float(v) for v in spec["bbox"])
        if not (lat_min < lat_max and lon_min < lon_max):
            raise ValueError(f"region {name}: bbox must be [lat_min, lat_max, lon_min, lon_max]")
        regions.append(
            Region(name, lat_min, lat_max, lon_min, lon_max,
                   spec.get("role", "hotspot"), spec.get("note", ""))
        )
    if only:
        missing = set(only) - {r.name for r in regions}
        if missing:
            raise ValueError(f"unknown regions: {sorted(missing)}")
    if not regions:
        raise ValueError("no regions selected")
    return regions
