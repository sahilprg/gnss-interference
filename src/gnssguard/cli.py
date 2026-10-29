"""Command line interface.

    gnssguard ingest --date 2026-10-26
    gnssguard ingest --start 2026-10-01 --end 2026-10-07 --regions baltic,central_europe
    gnssguard ingest --date 2026-10-26 --max-traces 5000 --out data/sample   # quick sample
    gnssguard coverage --data data/processed
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
    from gnssguard.analysis.coverage import coverage_report, hourly_degraded, load

    df = load(args.data, args.dates.split(",") if args.dates else None,
              args.regions.split(",") if args.regions else None)
    if df.empty:
        print("no data found", file=sys.stderr)
        return 1
    report = coverage_report(df)
    print(report.to_string(index=False))
    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        report.to_csv(out / "coverage.csv", index=False)
        hourly_degraded(df).to_csv(out / "hourly_degraded.csv", index=False)
        print(f"saved to {out}/")
    return 0


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

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
