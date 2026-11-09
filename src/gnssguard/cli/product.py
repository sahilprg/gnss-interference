"""The daily product: publish, nightly run and health."""

from __future__ import annotations

import datetime as dt

from gnssguard.cli._common import (
    default_model,
    grid,
    grid_args,
    product_args,
    split,
)


def cmd_publish(args) -> int:
    from gnssguard.product.publish import publish_day
    from gnssguard.product.site import build_site

    regions = split(args.regions)
    for day in split(args.dates) or []:
        s = publish_day(day, regions, args.data, args.cells, args.model or default_model(),
                        args.out, spec=grid(args), config=args.config)
        print(f"{day}: " + ", ".join(f"{r} {v['status']}" for r, v in s["regions"].items())
              + f"; {s['incidents']} incidents")
    build_site(args.out, args.config, regions)
    print(f"site written to {args.out}/")
    print(f"preview: python -m http.server -d {args.out} 8000 --bind 127.0.0.1, "
          "then open http://127.0.0.1:8000")
    return 0


def cmd_daily(args) -> int:
    import datetime as dt

    from gnssguard.product.daily import run

    dates = [dt.date.fromisoformat(d) for d in split(args.dates)] if args.dates else None
    done = run(args.out, processed=args.data, cells_dir=args.cells,
               model_dir=args.model or default_model(), config=args.config,
               regions=split(args.regions), dates=dates, lookback=args.lookback,
               cleanup=args.cleanup, spec=grid(args))
    for s in done:
        print(f"{s['date']}: " + ", ".join(f"{r} {v['status']}" for r, v in s["regions"].items()))
    if not done:
        print("nothing new to publish")
    return 0


def cmd_health(args) -> int:
    from gnssguard.product.daily import backlog
    from gnssguard.product.health import write_health

    today = dt.datetime.now(dt.timezone.utc).date()
    waiting = len(backlog(args.out, today, lookback=3))
    res = write_health(args.out, today, waiting)
    f = res["freshness"]
    print(f"data: {f['status']} (newest published day {f['newest_published_day']}, "
          f"{f['days_behind']} days behind, {f['backlog_days']} days waiting)")
    print(f"model: {res['status']}")
    pr = res["processing"]
    print(f"processing: {pr['status']} ({len(pr['failing'])} failing, "
          f"{len(pr['given_up'])} given up)")
    for r in res["reasons"]:
        print(f"  - {r}")
    for n in res["notes"]:
        print(f"  (note) {n}")
    return 0


def register(sub) -> None:
    pp = sub.add_parser("publish", help="write the public files and website for stored days")
    product_args(pp)
    pp.add_argument("--dates", help="comma-separated days that are already ingested")
    grid_args(pp)
    pp.set_defaults(func=cmd_publish)

    pk = sub.add_parser("health", help="retrain alarm: check the published days for drift")
    pk.add_argument("--out", default="public", help="website folder")
    pk.set_defaults(func=cmd_health)

    pd_ = sub.add_parser("daily", help="ingest, process and publish the days not yet published")
    product_args(pd_)
    pd_.add_argument("--dates", help="comma-separated days to (re)do instead of the look-back")
    pd_.add_argument("--lookback", type=int, default=3, help="days back from today to check")
    pd_.add_argument("--cleanup", action="store_true", help="delete processed data afterwards")
    grid_args(pd_)
    pd_.set_defaults(func=cmd_daily)
