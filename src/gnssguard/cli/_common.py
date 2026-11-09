"""Helpers shared by the command modules."""

from __future__ import annotations

import datetime as dt
import sys
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
    """Train / validation / test days, checked (model/splits.py). Default: test = the
    gold-set days; validation = the last usable stored day; train = the other usable
    days. Usable: before the first test day (--purpose benchmark, the default), or any
    day but the test days (--purpose deployment); never the sealed days before the
    sealed test is scored."""
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
    from gnssguard.model.splits import SEALED_DAYS, SplitError, check_split

    purpose = getattr(args, "purpose", "benchmark")
    sealed_scored = Path("results/sealed_test.json").exists()
    usable = [d for d in stored if d not in test and (sealed_scored or d not in SEALED_DAYS)]
    if purpose == "benchmark":  # only days before the first test day
        usable = [d for d in usable if d < min(test)]
    val = split(args.val_dates) or usable[-1:]
    train = split(args.train_dates) or [d for d in usable if d not in val]
    if not train or not val:
        raise SystemExit(f"need stored days before the test day {test}: have {stored}")
    try:
        check_split(train, val, test, purpose, sealed_scored)
    except SplitError as e:
        raise SystemExit(f"refused: {e}") from None
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
    p.add_argument("--model", help="train: where to write (default data/models/detector, "
                   "never the published model); evaluate: what to load (default "
                   "models/detector, the published model, if present)")
    p.add_argument("--table", default="data/model/table.parquet")
    p.add_argument("--report", default="reports/model")
    p.add_argument("--synth-region", default="central_europe")
    p.add_argument("--test-dates")


def grid_args(p) -> None:
    p.add_argument("--cell-deg", type=float, default=1.0, help="grid cell size in degrees")
    p.add_argument("--window-min", type=int, default=15, help="time window in minutes")


def sealed_refused(days, what: str) -> bool:
    """Print why and return True if `what` would touch the sealed test days too early."""
    from gnssguard.model.splits import SplitError, refuse_sealed

    try:
        refuse_sealed(days, what)
    except SplitError as e:
        print(f"refused: {e}", file=sys.stderr)
        return True
    return False
