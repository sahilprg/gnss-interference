"""Command line interface.

    gnssguard ingest --date 2026-10-26
    gnssguard ingest --start 2026-10-01 --end 2026-10-07 --regions baltic,central_europe
    gnssguard ingest --date 2026-10-26 --max-traces 5000 --out data/sample   # quick sample
    gnssguard coverage --data data/processed

    gnssguard cells                                  # grid-cell features for every stored day
    gnssguard label                                  # rules + label model -> data/labels
    gnssguard map --region baltic --date 2026-10-26  # HTML map in reports/maps/
    gnssguard synth --region central_europe --date 2026-10-26 --seeds 0,1,2
    gnssguard gold-sample --dates 2026-10-26 --n 300
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

from gnssguard.config import load_regions


def _dates(args) -> list[dt.date]:
    if args.date:
        return [dt.date.fromisoformat(d) for d in args.date]
    if args.start and args.end:
        start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
        if end < start:
            raise SystemExit("--end is before --start")
        return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    raise SystemExit("give --date (repeatable) or --start and --end")


def cmd_ingest(args) -> int:
    from gnssguard.ingest import ingest_day

    regions = load_regions(args.config, args.regions.split(",") if args.regions else None)
    failures = 0
    for day in _dates(args):
        try:
            m = ingest_day(day, regions, args.out, sources=args.source or None,
                           max_traces=args.max_traces, force=args.force)
            print(f"{day}: {sum(m['rows_per_region'].values()):,} rows, "
                  f"{m['trace_files']:,} traces, {m['bytes_read'] / 1e9:.2f} GB read, "
                  f"{m['seconds']:.0f}s")
        except Exception as e:  # noqa: BLE001 - keep going with the next day
            failures += 1
            logging.exception("%s failed: %s", day, e)
    return 1 if failures else 0


def cmd_coverage(args) -> int:
    from gnssguard.analysis.coverage import scan

    report, hourly = scan(args.data, args.dates.split(",") if args.dates else None,
                          args.regions.split(",") if args.regions else None)
    if report.empty:
        print("no data found", file=sys.stderr)
        return 1
    print(report.to_string(index=False))
    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        report.to_csv(out / "coverage.csv", index=False)
        hourly.to_csv(out / "hourly_degraded.csv", index=False)
        print(f"saved to {out}/")
    return 0


def _split(v: str | None) -> list[str] | None:
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


def _grid(args):
    from gnssguard.features.cells import GridSpec

    return GridSpec(cell_deg=args.cell_deg, window_min=args.window_min)


def cmd_cells(args) -> int:
    from gnssguard.features.cells import build_all

    done = build_all(args.data, args.out, _grid(args), _split(args.dates),
                     _split(args.regions), force=args.force)
    for s in done:
        print(f"{s['region']:15s} {s['date']}  {s['cell_windows']:6,} cell-windows, "
              f"{s['chronic_aircraft']} faulty-transponder aircraft excluded")
    if not done:
        print("nothing new to build (use --force to rebuild)")
    return 0


def cmd_label(args) -> int:
    import pandas as pd

    from gnssguard.labels.pipeline import run

    labels = run(args.cells, args.out, args.events, args.report, _split(args.dates),
                 _split(args.regions), advisories_csv=args.advisories)
    table = (labels.groupby(["region", "label"], observed=False).size()
             .unstack(fill_value=0))
    with pd.option_context("display.width", 140):
        print(table.to_string())
        adv = Path(args.report) / "advisory_check.csv"
        if args.advisories and adv.exists():
            print("\nOfficial advisories: flagged share inside vs outside the warned area")
            print(pd.read_csv(adv).to_string(index=False))
    print(f"\nlabels: {args.out}\nreports: {args.report}/")
    return 0


def cmd_map(args) -> int:
    import pandas as pd

    from gnssguard.features.cells import load_cells
    from gnssguard.viz.map import write_map

    cells = load_cells(args.cells, [args.date], [args.region])
    if cells.empty:
        print(f"no cells for {args.region} {args.date}; run `gnssguard cells` first",
              file=sys.stderr)
        return 1
    labels = None
    if args.labels and Path(args.labels).exists():
        labels = pd.read_parquet(args.labels)
        labels = labels[(labels["region"] == args.region) & (labels["date"] == args.date)]
    gold = None
    if args.gold and Path(args.gold).exists():
        gold = pd.read_csv(args.gold, parse_dates=["window"])
        gold = gold[(gold["region"] == args.region) & (gold["date"] == args.date)]
    out = args.out or f"reports/maps/{args.region}_{args.date}.html"
    print(f"map written to {write_map(out, cells, labels, gold)}")
    return 0


def cmd_synth(args) -> int:
    from gnssguard.labels.synthetic import run_experiment

    seeds = [int(s) for s in _split(args.seeds) or ["0"]]
    res = run_experiment(args.data, args.region, args.date, n_jamming=args.n_jamming,
                         n_spoofing=args.n_spoofing, seeds=seeds, p_mlat=args.p_mlat,
                         real_cells=args.cells, out_dir=args.out, report_dir=args.report,
                         spec=_grid(args))
    print(res.to_string(index=False))
    print(f"\nreports: {args.report}/   runs: {args.out}/")
    return 0


def cmd_gold(args) -> int:
    import pandas as pd

    from gnssguard.labels.gold import sample_gold

    labels = pd.read_parquet(args.labels)
    gold = sample_gold(labels, n=args.n, seed=args.seed, dates=_split(args.dates))
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} exists; it may hold your hand labels. Use --force to overwrite.",
              file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(out, index=False)
    print(f"{len(gold)} cell-windows written to {out}")
    print("Fill the `label` column with: jamming | spoofing | both | normal | unsure")
    print("Look each one up on the map: gnssguard map --region R --date D --gold " + str(out))
    return 0


def cmd_gold_tool(args) -> int:
    from gnssguard.labels.goldtool import write_tool

    out = write_tool(args.gold, args.cells, args.out)
    print(f"labeling tool written to {out}")
    print("Open it in your browser. Press 1-5 to label, arrows to move.")
    print(f"When done, press 'Download gold.csv' and replace {args.gold} with it.")
    return 0


def _grid_args(p) -> None:
    p.add_argument("--cell-deg", type=float, default=1.0, help="grid cell size in degrees")
    p.add_argument("--window-min", type=int, default=15, help="time window in minutes")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gnssguard")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="stream adsb.lol days into regional Parquet")
    pi.add_argument("--date", action="append", help="YYYY-MM-DD (repeatable)")
    pi.add_argument("--start")
    pi.add_argument("--end")
    pi.add_argument("--config", default="configs/regions.yaml")
    pi.add_argument("--regions", help="comma-separated subset of region names")
    pi.add_argument("--out", default="data/processed")
    pi.add_argument("--source", action="append",
                    help="override: part URL or local path (repeat in order)")
    pi.add_argument("--max-traces", type=int, help="stop early, for samples and tests")
    pi.add_argument("--force", action="store_true", help="redo days already ingested")
    pi.set_defaults(func=cmd_ingest)

    pc = sub.add_parser("coverage", help="traffic and data-quality report per region")
    pc.add_argument("--data", default="data/processed")
    pc.add_argument("--dates", help="comma-separated YYYY-MM-DD")
    pc.add_argument("--regions")
    pc.add_argument("--save", help="folder for coverage.csv and hourly_degraded.csv")
    pc.set_defaults(func=cmd_coverage)

    pl = sub.add_parser("cells", help="build grid-cell x time-window features")
    pl.add_argument("--data", default="data/processed")
    pl.add_argument("--out", default="data/cells")
    pl.add_argument("--dates")
    pl.add_argument("--regions")
    pl.add_argument("--force", action="store_true", help="rebuild existing tables")
    _grid_args(pl)
    pl.set_defaults(func=cmd_cells)

    pb = sub.add_parser("label", help="apply weak rules and the label model")
    pb.add_argument("--cells", default="data/cells")
    pb.add_argument("--out", default="data/labels/labels.parquet")
    pb.add_argument("--events", default="configs/events.csv")
    pb.add_argument("--advisories", default="configs/advisories.csv")
    pb.add_argument("--report", default="reports/labels")
    pb.add_argument("--dates")
    pb.add_argument("--regions")
    pb.set_defaults(func=cmd_label)

    pm = sub.add_parser("map", help="HTML map of one region-day")
    pm.add_argument("--region", required=True)
    pm.add_argument("--date", required=True)
    pm.add_argument("--cells", default="data/cells")
    pm.add_argument("--labels", default="data/labels/labels.parquet")
    pm.add_argument("--gold", help="gold-set CSV to outline on the map")
    pm.add_argument("--out", help="output HTML (default reports/maps/<region>_<date>.html)")
    pm.set_defaults(func=cmd_map)

    ps = sub.add_parser("synth", help="inject synthetic attacks and score the rules")
    ps.add_argument("--region", required=True, help="preferably a calm control region")
    ps.add_argument("--date", required=True)
    ps.add_argument("--data", default="data/processed")
    ps.add_argument("--cells", default="data/cells",
                    help="real cell tables, used when fitting the label model")
    ps.add_argument("--n-jamming", type=int, default=4)
    ps.add_argument("--n-spoofing", type=int, default=4)
    ps.add_argument("--seeds", default="0,1,2")
    ps.add_argument("--p-mlat", type=float, default=0.0,
                    help="share of jammed points the tracker moves to MLAT")
    ps.add_argument("--out", default="data/synthetic")
    ps.add_argument("--report", default="reports/synthetic")
    _grid_args(ps)
    ps.set_defaults(func=cmd_synth)

    pg = sub.add_parser("gold-sample", help="sample cell-windows for hand labeling")
    pg.add_argument("--labels", default="data/labels/labels.parquet")
    pg.add_argument("--dates", help="days reserved for the gold set (keep them out of training)")
    pg.add_argument("--n", type=int, default=300)
    pg.add_argument("--seed", type=int, default=0)
    pg.add_argument("--out", default="labels/gold.csv")
    pg.add_argument("--force", action="store_true")
    pg.set_defaults(func=cmd_gold)

    pt = sub.add_parser("gold-tool", help="HTML page for labeling the gold set")
    pt.add_argument("--gold", default="labels/gold.csv")
    pt.add_argument("--cells", default="data/cells")
    pt.add_argument("--out", default="reports/gold_tool.html")
    pt.set_defaults(func=cmd_gold_tool)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
