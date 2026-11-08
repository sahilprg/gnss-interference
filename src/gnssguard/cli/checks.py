"""Independent checks: public events, incident pages, ground stations, OpenSky, benchmark."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

from gnssguard.cli._common import (
    OPENSKY_REGIONS,
    default_model,
    opensky_boxes,
    pd_options,
    split,
)


def cmd_ground_fetch(args) -> int:
    import csv

    from gnssguard.ground.fetch import fetch_day

    stations = [r["station"] for r in csv.DictReader(open(args.stations, encoding="utf-8"))]
    days = [dt.date.fromisoformat(d) for d in split(args.dates)]
    missing = 0
    for day in days:
        got = fetch_day(stations, day, Path(args.out))
        for st, where in got.items():
            missing += where == "missing"
            print(f"{day} {st:10s} {'missing' if where == 'missing' else 'ok'}")
    print(f"{len(days) * len(stations) - missing} of {len(days) * len(stations)} "
          f"station-days in {args.out}")
    return 0


def cmd_ground_check(args) -> int:
    from gnssguard.features.cells import load_cells
    from gnssguard.ground.check import run
    from gnssguard.model.dataset import with_context
    from gnssguard.model.train import Detectors

    root = Path(args.files)
    days = split(args.dates)
    files: dict[str, list[Path]] = {}
    for day in days:
        for f in sorted((root / day).glob("*")):
            files.setdefault(f.name[:9].upper(), []).append(f)
    if not files:
        print(f"no station files under {root}; run `gnssguard ground-fetch` first",
              file=sys.stderr)
        return 1
    det = Detectors.load(args.model or default_model())
    cells = with_context(load_cells(args.cells, days, ["baltic"])).reset_index(drop=True)
    labels = det.predict(cells)["label"].astype(str)
    with pd_options():
        print(run(files, cells, labels, Path(args.report)).to_string(index=False))
    print(f"\nreports: {args.report}/")
    return 0


def cmd_benchmark(args) -> int:
    from gnssguard.model.benchmark import run

    doc = run(args.model or default_model(), args.cells, args.labels_table, args.events,
              args.rolling, args.out)
    print((Path(args.out) / "README.md").read_text(encoding="utf-8"))
    bad = [k for k, v in doc["results"].items() if v.get("status") != "ok"]
    print(f"results -> {args.out}/  ({len(doc['results']) - len(bad)} checks run, "
          f"{len(bad)} not run)")
    return 0


def cmd_event_check(args) -> int:
    from gnssguard.model.eventcheck import run

    res = run(args.cells, args.model or default_model(), args.events)
    rep = Path(args.report)
    rep.mkdir(parents=True, exist_ok=True)
    for name, df in res.items():
        df.to_csv(rep / f"{name}.csv", index=False)
    with pd_options():
        print(res["events"].pivot_table(index=["event_id", "type", "evidence_level"],
                                        columns="predictor", values="verdict",
                                        aggfunc="first").to_string())
        print("\n" + res["summary"].to_string(index=False))
        print("\ncalm control on the event days\n" + res["placebo"].pivot_table(
            index="date", columns="predictor", values="flagged_share").to_string())
    print(f"\nreports: {rep}/")
    return 0


def cmd_incidents(args) -> int:
    from gnssguard.product.incident import build

    only = args.only.split(",") if args.only else None
    pages = build(args.cells, args.model or default_model(), args.events, args.out, only)
    for pg in pages:
        ch = pg.get("check", {}).get("model", {})
        base = pg.get("ordinary_days") or {}
        extra = (f"  ordinary days as high: {base['days_at_or_above_event']} of {base['days']}"
                 if base.get("days") and ch.get("share") is not None else "")
        print(f"{pg['event']['id']}  {pg['status']:<14} share {ch.get('share')}{extra}")
    print(f"\npages: {args.out}/index.html")
    return 0


def cmd_opensky_probe(args) -> int:
    from gnssguard.network.opensky import PROBE_HOUR, hour_ts, probe_sql, run_query

    box = opensky_boxes(args.config)["baltic"]
    df = run_query(probe_sql(box, hour_ts(PROBE_HOUR)))
    with pd_options():
        print(df.to_string(index=False))
    return 0


def cmd_opensky_fetch(args) -> int:
    from gnssguard.network.opensky import fetch

    days = split(args.dates)
    written = fetch(days, opensky_boxes(args.config), args.out, by_region=args.by_region)
    have = sum(len(list((Path(args.out) / d).glob("*.parquet"))) for d in days)
    print(f"{len(written)} hours fetched now; {have} of {24 * len(days)} hours in {args.out}")
    return 0


def cmd_opensky_check(args) -> int:
    from gnssguard.features.cells import load_cells
    from gnssguard.model.dataset import with_context
    from gnssguard.model.evaluate import rule_frame
    from gnssguard.model.train import Detectors
    from gnssguard.network.opensky import load, run

    days = split(args.dates)
    os_ = load(args.files, days)
    if os_.empty:
        print(f"no OpenSky files under {args.files}; run `gnssguard opensky-fetch` first",
              file=sys.stderr)
        return 1
    det = Detectors.load(args.model or default_model())
    cells = with_context(load_cells(args.cells, days, list(OPENSKY_REGIONS))).reset_index(drop=True)
    cells["model"] = det.predict(cells)["label"].astype(str).to_numpy()
    cells["rules"] = rule_frame(cells, det)["label"].astype(str).to_numpy()
    summary, verdict = run(cells, os_, args.report)
    with pd_options():
        print(summary.to_string(index=False))
    print(f"\n{verdict}\nreports: {args.report}/")
    return 0


def register(sub) -> None:
    pe = sub.add_parser("event-check", help="score the detectors on the public event catalog")
    pe.add_argument("--cells", default="data/cells")
    pe.add_argument("--model", help="model folder (default: models/detector if present)")
    pe.add_argument("--events", default="configs/events.csv")
    pe.add_argument("--report", default="reports/events")
    pe.set_defaults(func=cmd_event_check)

    pi = sub.add_parser("incidents", help="one page per catalogued event: report vs map")
    pi.add_argument("--cells", default="data/cells")
    pi.add_argument("--model", help="model folder (default: models/detector if present)")
    pi.add_argument("--events", default="configs/events.csv")
    pi.add_argument("--out", default="results/incidents")
    pi.add_argument("--only", help="event ids to rebuild (comma-separated); others are kept")
    pi.set_defaults(func=cmd_incidents)

    po = sub.add_parser("opensky-probe", help="one-hour OpenSky probe: NIC column meanings")
    po.add_argument("--config", default="configs/regions.yaml")
    po.set_defaults(func=cmd_opensky_probe)

    pf = sub.add_parser("opensky-fetch", help="OpenSky cell-window counts, one hour per query")
    pf.add_argument("--dates", required=True)
    pf.add_argument("--out", default="data/opensky")
    pf.add_argument("--config", default="configs/regions.yaml")
    pf.add_argument("--by-region", action="store_true",
                    help="one query per hour and region (if an hour takes over 5 minutes)")
    pf.set_defaults(func=cmd_opensky_fetch)

    pk = sub.add_parser("opensky-check", help="second network check 1: OpenSky against the map")
    pk.add_argument("--dates", required=True)
    pk.add_argument("--files", default="data/opensky")
    pk.add_argument("--cells", default="data/cells")
    pk.add_argument("--model", help="model folder (default: models/detector if present)")
    pk.add_argument("--report", default="reports/opensky")
    pk.set_defaults(func=cmd_opensky_check)

    gf = sub.add_parser("ground-fetch", help="download EUREF station files (RINEX 3)")
    gf.add_argument("--stations", default="configs/ground_stations.csv")
    gf.add_argument("--dates", required=True)
    gf.add_argument("--out", default="data/ground")
    gf.set_defaults(func=cmd_ground_fetch)

    gc = sub.add_parser("ground-check", help="station C/N0 against the aircraft map")
    gc.add_argument("--dates", required=True)
    gc.add_argument("--files", default="data/ground")
    gc.add_argument("--cells", default="data/cells")
    gc.add_argument("--model", help="model folder (default: models/detector if present)")
    gc.add_argument("--report", default="reports/ground")
    gc.set_defaults(func=cmd_ground_check)

    pk = sub.add_parser("benchmark", help="regenerate every published result into results/")
    pk.add_argument("--model", help="frozen model (default: models/detector if present)")
    pk.add_argument("--cells", default="data/cells")
    pk.add_argument("--labels-table", default="data/labels/labels.parquet")
    pk.add_argument("--events", default="configs/events.csv")
    pk.add_argument("--rolling", help="folder with rolling_*.csv (optional)")
    pk.add_argument("--out", default="results")
    pk.set_defaults(func=cmd_benchmark)
