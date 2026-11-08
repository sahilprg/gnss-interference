"""Helpers shared by the command modules."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import overload

from gnssguard.config import load_regions


def dates(args) -> list[dt.date]:
    if args.date:
        return [dt.date.fromisoformat(d) for d in args.date]
    if args.start and args.end:
        start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
        if end < start:
            raise SystemExit("--end is before --start")
        return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    raise SystemExit("give --date (repeatable) or --start and --end")


@overload
def split(v: str) -> list[str]: ...
@overload
def split(v: None) -> None: ...
def split(v: str | None) -> list[str] | None:
    """Comma-separated text to a list; no text (an unset optional flag) gives None."""
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


def grid(args):
    from gnssguard.features.cells import GridSpec

    return GridSpec(cell_deg=args.cell_deg, window_min=args.window_min)


def split_days(args) -> tuple[list[str], list[str], list[str]]:
    """Train / validation / test days. Default: test = the gold-set days, validation =
    the last stored day that is not a test day, train = every other stored day (also days
    after the test day, so retraining uses new data; the test days never enter)."""
    import pandas as pd

    from gnssguard.analysis.coverage import partitions

    stored = sorted({d for _, d, _ in partitions(args.cells)})
    test: list[str]
    if args.test_dates:
        test = split(args.test_dates)
    elif Path(args.gold).exists():
        test = sorted(pd.read_csv(args.gold, dtype=str)["date"].unique())
    else:
        raise SystemExit("give --test-dates, or a gold set (--gold) to take them from")
    before = [d for d in stored if d not in test]
    val = split(args.val_dates) or before[-1:]
    train = split(args.train_dates) or [d for d in before if d not in val]
    if not train or not val:
        raise SystemExit(f"need stored days before the test day {test}: have {stored}")
    return train, val, test


PRODUCT_REGIONS = "baltic,east_med,black_sea,punjab,central_europe"


def pd_series(values, index):
    import pandas as pd

    return pd.Series(values, index=index)


OPENSKY_REGIONS = ("baltic", "east_med", "black_sea", "punjab", "central_europe")


def opensky_boxes(config):
    return {r.name: (r.lat_min, r.lat_max, r.lon_min, r.lon_max)
            for r in load_regions(config, list(OPENSKY_REGIONS))}


def pd_options():
    import pandas as pd

    return pd.option_context("display.width", 160, "display.max_rows", 200)


def default_model() -> str:
    # the model committed with the repository (models/) is what the nightly job uses
    committed = Path("models/detector")
    return str(committed) if (committed / "meta.json").exists() else "data/models/detector"


def product_args(p) -> None:
    p.add_argument("--out", default="public", help="website folder")
    p.add_argument("--data", default="data/processed")
    p.add_argument("--cells", default="data/cells")
    p.add_argument("--model", help="model folder (default: models/detector if present)")
    p.add_argument("--config", default="configs/regions.yaml")
    p.add_argument("--regions", default=PRODUCT_REGIONS)


def model_args(p) -> None:
    p.add_argument("--data", default="data/processed")
    p.add_argument("--cells", default="data/cells")
    p.add_argument("--gold", default="labels/gold.csv")
    p.add_argument("--model", default="data/models/detector")
    p.add_argument("--table", default="data/model/table.parquet")
    p.add_argument("--report", default="reports/model")
    p.add_argument("--synth-region", default="central_europe")
    p.add_argument("--test-dates")


def grid_args(p) -> None:
    p.add_argument("--cell-deg", type=float, default=1.0, help="grid cell size in degrees")
    p.add_argument("--window-min", type=int, default=15, help="time window in minutes")
