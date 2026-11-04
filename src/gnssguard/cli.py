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
    print(f"When done, press 'Download {Path(args.gold).name}' and replace {args.gold} with it.")
    return 0


def _split_days(args) -> tuple[list[str], list[str], list[str]]:
    """Train / validation / test days. Default: test = the gold-set days, validation =
    the last stored day that is not a test day, train = every other stored day (also days
    after the test day, so retraining uses new data; the test days never enter)."""
    import pandas as pd

    from gnssguard.analysis.coverage import partitions

    stored = sorted({d for _, d, _ in partitions(args.cells)})
    if args.test_dates:
        test = _split(args.test_dates)
    elif Path(args.gold).exists():
        test = sorted(pd.read_csv(args.gold, dtype=str)["date"].unique())
    else:
        raise SystemExit("give --test-dates, or a gold set (--gold) to take them from")
    before = [d for d in stored if d not in test]
    val = _split(args.val_dates) or before[-1:]
    train = _split(args.train_dates) or [d for d in before if d not in val]
    if not train or not val:
        raise SystemExit(f"need stored days before the test day {test}: have {stored}")
    return train, val, test


def cmd_train(args) -> int:
    from gnssguard.model.train import build_table, importance, train

    tr, va, te = _split_days(args)
    print(f"train: {', '.join(tr)}\nvalidation: {', '.join(va)}\ntest (sealed): {', '.join(te)}")
    table, lm = build_table(args.data, args.cells, tr, va,
                            synth_region=_split(args.synth_train_regions),
                            synth_seeds=tuple(int(s) for s in _split(args.synth_seeds)),
                            spec=_grid(args), unlabel_borderline=args.unlabel_borderline)
    Path(args.table).parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.table, index=False)
    det = train(table, label_model=lm)
    det.meta["test_dates"] = te
    det.meta["training_options"] = {"synth_train_regions": _split(args.synth_train_regions),
                                    "unlabel_borderline": args.unlabel_borderline}
    det.save(args.model)
    rep = Path(args.report)
    rep.mkdir(parents=True, exist_ok=True)
    importance(det).to_csv(rep / "importance.csv", index=False)
    for t, info in det.meta["targets"].items():
        print(f"{t}: {info['train_rows']:,} rows, {info['train_positives']:,} positives, "
              f"{info['best_iteration']} trees, validation logloss {info['val_logloss']}")
    print(f"model: {args.model}   table: {args.table}")
    return 0


def cmd_evaluate(args) -> int:
    import pandas as pd

    from gnssguard.model.evaluate import run
    from gnssguard.model.train import Detectors

    det = Detectors.load(args.model)
    table = pd.read_parquet(args.table)
    test = _split(args.test_dates) or det.meta.get("test_dates")
    res = run(args.data, args.cells, table, det, test, args.gold, args.advisories, args.report,
              synth_region=args.synth_region, ablations=not args.no_ablations, spec=_grid(args))
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        for name in ("gold", "synthetic_mean", "edge", "synthetic_hard_mean", "edge_hard",
                     "synthetic_stress_mean", "edge_stress", "real_flag_rates", "localization",
                     "calibration", "advisories"):
            df = res.get(name)
            if df is not None and not df.empty:
                print(f"\n== {name}")
                print(df.to_string(index=False))
    print(f"\nreports: {args.report}/  (no per-cell predictions for the test day are written)")
    return 0


def cmd_rolling(args) -> int:
    import pandas as pd

    from gnssguard.analysis.coverage import partitions
    from gnssguard.model.rolling import run

    table = pd.read_parquet(args.table)
    stored = sorted({d for _, d, _ in partitions(args.cells)})
    days = _split(args.test_dates) or stored[-args.days:]
    print(f"rolling test days: {', '.join(days)}")
    res = run(args.data, args.cells, table, days, args.report,
              synth_region=_split(args.test_regions), n_seeds=args.seeds, spec=_grid(args),
              unlabel_borderline=args.unlabel_borderline)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        for name in ("rolling_synthetic_overall", "rolling_real_flag_rates"):
            df = res.get(name)
            if df is not None and not df.empty:
                if name == "rolling_real_flag_rates":
                    df = df.pivot_table(index=["region", "predictor"], columns="test_day",
                                        values=["flagged_share_dense", "flagged_share_sparse"])
                print(f"\n== {name}")
                print(df.to_string())
    print(f"\nreports: {args.report}/rolling_*.csv")
    return 0


def cmd_holdout_sample(args) -> int:
    import json

    from gnssguard.features.cells import load_cells
    from gnssguard.labels.holdout import holdout_sample
    from gnssguard.model.dataset import with_context
    from gnssguard.model.evaluate import rule_frame
    from gnssguard.model.train import Detectors

    out, strata_out = Path(args.out), Path(args.strata)
    for f in (out, strata_out):
        if f.exists() and not args.force:
            print(f"{f} exists; it may hold labels. Use --force to overwrite.", file=sys.stderr)
            return 1
    det = Detectors.load(args.model or _default_model())
    cells = with_context(load_cells(args.cells, [args.date], _split(args.regions)))
    rules = rule_frame(cells, det)["label"].astype(str)
    model = det.predict(cells)["label"].astype(str)
    gold, strata = holdout_sample(cells, rules, model, n=args.n, seed=args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(out, index=False)
    strata.to_csv(strata_out, index=False)
    counts = strata.groupby("stratum")["pool"].agg(["size", "first"]).rename(
        columns={"size": "sampled", "first": "pool"})
    print(f"{len(gold)} cell-windows -> {out}  (labeling file)")
    print(f"strata and frozen predictions -> {strata_out}  (do not open before labeling)")
    print(json.dumps({"model": str(args.model or _default_model()),
                      "sampled": {k: int(v) for k, v in counts["sampled"].items()}}))
    return 0


def cmd_holdout_eval(args) -> int:
    import json

    import pandas as pd

    from gnssguard.labels.holdout import holdout_eval

    gold = pd.read_csv(args.gold, dtype=str, keep_default_na=False)
    strata = pd.read_csv(args.strata, keep_default_na=False)
    strata["weight"] = strata["weight"].astype(float)
    res = holdout_eval(gold, strata)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))
    return 0


PRODUCT_REGIONS = "baltic,east_med,black_sea,punjab,central_europe"


def cmd_review_sample(args) -> int:
    from gnssguard.features.cells import load_cells
    from gnssguard.labels.review import review_sample
    from gnssguard.model.dataset import with_context
    from gnssguard.model.evaluate import rule_frame
    from gnssguard.model.train import Detectors

    out, groups_out = Path(args.out), Path(args.groups)
    for f in (out, groups_out):
        if f.exists() and not args.force:
            print(f"{f} exists; it may hold labels. Use --force to overwrite.", file=sys.stderr)
            return 1
    det = Detectors.load(args.model or _default_model())
    cells = with_context(load_cells(args.cells, _split(args.dates), _split(args.regions)))
    cells = cells.reset_index(drop=True)
    rules = rule_frame(cells, det)["label"].astype(str)
    model = det.predict(cells)["label"].astype(str)
    gold, groups = review_sample(cells, rules, model, seed=args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(out, index=False)
    groups.to_csv(groups_out, index=False)
    calm = (cells["region"] == "central_europe") & (cells["n_ac_usable"].fillna(0) >= 3)
    share = float(model[calm].isin(["jamming", "spoofing", "both"]).mean())
    print(groups.groupby("group").agg(cells=("gold_id", "size"), pool=("pool", "first"))
          .to_string())
    print(f"calm flagged share on these days (model): {share:.4f}")
    print(f"{len(gold)} cell-windows -> {out}  (labeling file)")
    print(f"groups and frozen predictions -> {groups_out}  (do not open before labeling)")
    return 0


def cmd_trust_report(args) -> int:
    from gnssguard.features.cells import load_cells
    from gnssguard.model import trust
    from gnssguard.model.dataset import with_context
    from gnssguard.model.train import Detectors

    rep = Path(args.report)
    rep.mkdir(parents=True, exist_ok=True)
    det = Detectors.load(args.model or _default_model())
    with pd_options():
        if Path(args.rolling, "rolling_synthetic.csv").exists():
            ri = trust.rolling_intervals(args.rolling)
            ri.to_csv(rep / "rolling_intervals.csv", index=False)
            print("== rolling results, 95% intervals over test days\n"
                  + ri[ri["predictor"] == "model"].to_string(index=False))
        else:
            print(f"no rolling results in {args.rolling}; run `gnssguard rolling` first")
        golds = _split(args.gold)
        import pandas as pd
        days = sorted({str(d) for f in golds for d in pd.read_csv(f)["date"].unique()})
        cells = with_context(load_cells(args.cells, days, None)).reset_index(drop=True)
        m = trust.human_scored(golds, cells, det)
        for by, name in ((None, "overall"), ("region", "region"), ("aircraft", "aircraft")):
            t = trust.reliability_by(m, by)
            t.to_csv(rep / f"calibration_human_{name}.csv", index=False)
            print(f"\n== score vs hand labels, {name} ({len(m)} decided cells)\n"
                  + t.to_string(index=False))
        train = det.meta.get("train_dates", [])
        ov = trust.aircraft_overlap(args.data, train, days, _split(args.regions))
        ov.to_csv(rep / "aircraft_overlap.csv", index=False)
        print("\n== aircraft also seen on training days\n" + ov.to_string(index=False))
    print(f"\nreports: {rep}/")
    return 0


def cmd_review_eval(args) -> int:
    import json

    import pandas as pd

    from gnssguard.labels.review import review_eval

    gold = pd.read_csv(args.gold, dtype=str, keep_default_na=False)
    groups = pd.read_csv(args.groups, keep_default_na=False)
    groups["weight"] = groups["weight"].astype(float)
    res = review_eval(gold, groups, args.calm_share)
    print(json.dumps(res, indent=1, default=str))
    return 0


def cmd_event_check(args) -> int:
    from gnssguard.model.eventcheck import run

    res = run(args.cells, args.model or _default_model(), args.events)
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


def pd_options():
    import pandas as pd

    return pd.option_context("display.width", 160, "display.max_rows", 200)


def _default_model() -> str:
    # the model committed with the repository (models/) is what the nightly job uses
    committed = Path("models/detector")
    return str(committed) if (committed / "meta.json").exists() else "data/models/detector"


def cmd_publish(args) -> int:
    from gnssguard.product.publish import publish_day
    from gnssguard.product.site import build_site

    regions = _split(args.regions)
    for day in _split(args.dates) or []:
        s = publish_day(day, regions, args.data, args.cells, args.model or _default_model(),
                        args.out, spec=_grid(args))
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

    dates = [dt.date.fromisoformat(d) for d in _split(args.dates)] if args.dates else None
    done = run(args.out, processed=args.data, cells_dir=args.cells,
               model_dir=args.model or _default_model(), config=args.config,
               regions=_split(args.regions), dates=dates, lookback=args.lookback,
               cleanup=args.cleanup, spec=_grid(args))
    for s in done:
        print(f"{s['date']}: " + ", ".join(f"{r} {v['status']}" for r, v in s["regions"].items()))
    if not done:
        print("nothing new to publish")
    return 0


def cmd_health(args) -> int:
    from gnssguard.product.health import write_health

    res = write_health(args.out)
    print(f"status: {res['status']}")
    for r in res["reasons"]:
        print(f"  - {r}")
    for n in res["notes"]:
        print(f"  (note) {n}")
    return 0


def _product_args(p) -> None:
    p.add_argument("--out", default="public", help="website folder")
    p.add_argument("--data", default="data/processed")
    p.add_argument("--cells", default="data/cells")
    p.add_argument("--model", help="model folder (default: models/detector if present)")
    p.add_argument("--config", default="configs/regions.yaml")
    p.add_argument("--regions", default=PRODUCT_REGIONS)


def _model_args(p) -> None:
    p.add_argument("--data", default="data/processed")
    p.add_argument("--cells", default="data/cells")
    p.add_argument("--gold", default="labels/gold.csv")
    p.add_argument("--model", default="data/models/detector")
    p.add_argument("--table", default="data/model/table.parquet")
    p.add_argument("--report", default="reports/model")
    p.add_argument("--synth-region", default="central_europe")
    p.add_argument("--test-dates")


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

    pr = sub.add_parser("train", help="train the jamming and spoofing detectors")
    _model_args(pr)
    pr.add_argument("--train-dates")
    pr.add_argument("--val-dates")
    pr.add_argument("--synth-seeds", default="0,1")
    pr.add_argument("--synth-train-regions", default="central_europe,east_med",
                    help="regions to inject training attacks into (comma-separated)")
    pr.add_argument("--unlabel-borderline", action="store_true",
                    help="leave weak negatives with some jamming evidence unlabeled")
    _grid_args(pr)
    pr.set_defaults(func=cmd_train)

    pe = sub.add_parser("evaluate", help="score the detectors on gold, synthetic, advisories")
    _model_args(pe)
    pe.add_argument("--advisories", default="configs/advisories.csv")
    pe.add_argument("--no-ablations", action="store_true", help="skip retraining variants")
    _grid_args(pe)
    pe.set_defaults(func=cmd_evaluate)

    pw = sub.add_parser("rolling", help="retrain and test day by day over the last N days")
    _model_args(pw)
    pw.add_argument("--days", type=int, default=7, help="test the last N stored days")
    pw.add_argument("--seeds", type=int, default=3, help="synthetic runs per profile per day")
    pw.add_argument("--test-regions", default="central_europe,black_sea",
                    help="regions to inject test attacks into (comma-separated)")
    pw.add_argument("--unlabel-borderline", action="store_true",
                    help="as in train; must match the table")
    _grid_args(pw)
    pw.set_defaults(func=cmd_rolling)

    pe = sub.add_parser("event-check", help="score the detectors on the public event catalog")
    pe.add_argument("--cells", default="data/cells")
    pe.add_argument("--model", help="model folder (default: models/detector if present)")
    pe.add_argument("--events", default="configs/events.csv")
    pe.add_argument("--report", default="reports/events")
    pe.set_defaults(func=cmd_event_check)

    ph = sub.add_parser("holdout-sample", help="blind stratified sample of a fresh day")
    ph.add_argument("--date", required=True)
    ph.add_argument("--cells", default="data/cells")
    ph.add_argument("--model", help="frozen model (default: models/detector if present)")
    ph.add_argument("--regions", default=PRODUCT_REGIONS)
    ph.add_argument("--n", type=int, default=150)
    ph.add_argument("--seed", type=int, default=20261101)
    ph.add_argument("--out", required=True, help="labeling file, e.g. labels/gold_2026-11-01.csv")
    ph.add_argument("--strata", required=True, help="strata file kept apart from the labeler")
    ph.add_argument("--force", action="store_true")
    ph.set_defaults(func=cmd_holdout_sample)

    pq = sub.add_parser("review-sample", help="blind sample for the false-alarm review set")
    pq.add_argument("--dates", required=True)
    pq.add_argument("--cells", default="data/cells")
    pq.add_argument("--model", help="frozen model (default: models/detector if present)")
    pq.add_argument("--regions", default=PRODUCT_REGIONS)
    pq.add_argument("--seed", type=int, default=20261104)
    pq.add_argument("--out", required=True)
    pq.add_argument("--groups", required=True, help="groups file kept apart from the labeler")
    pq.add_argument("--force", action="store_true")
    pq.set_defaults(func=cmd_review_sample)

    pz = sub.add_parser("trust-report", help="day intervals, calibration vs hand labels, "
                        "aircraft overlap")
    pz.add_argument("--rolling", default="reports/model", help="folder with rolling_*.csv")
    pz.add_argument("--gold", default="labels/gold.csv,labels/gold_2026-11-01.csv,"
                    "labels/review_1.csv")
    pz.add_argument("--cells", default="data/cells")
    pz.add_argument("--data", default="data/processed")
    pz.add_argument("--model", help="model folder (default: models/detector if present)")
    pz.add_argument("--regions", default=PRODUCT_REGIONS)
    pz.add_argument("--report", default="reports/trust")
    pz.set_defaults(func=cmd_trust_report)

    pu = sub.add_parser("review-eval", help="the pre-declared analysis of the review set")
    pu.add_argument("--gold", required=True)
    pu.add_argument("--groups", required=True)
    pu.add_argument("--calm-share", type=float, help="calm flagged share on the review days")
    pu.set_defaults(func=cmd_review_eval)

    pv = sub.add_parser("holdout-eval", help="the pre-declared analysis of a labeled holdout")
    pv.add_argument("--gold", required=True)
    pv.add_argument("--strata", required=True)
    pv.add_argument("--report", default="reports/holdout/result.json")
    pv.set_defaults(func=cmd_holdout_eval)

    pp = sub.add_parser("publish", help="write the public files and website for stored days")
    _product_args(pp)
    pp.add_argument("--dates", help="comma-separated days that are already ingested")
    _grid_args(pp)
    pp.set_defaults(func=cmd_publish)

    pk = sub.add_parser("health", help="retrain alarm: check the published days for drift")
    pk.add_argument("--out", default="public", help="website folder")
    pk.set_defaults(func=cmd_health)

    pd_ = sub.add_parser("daily", help="ingest, process and publish the days not yet published")
    _product_args(pd_)
    pd_.add_argument("--dates", help="comma-separated days to (re)do instead of the look-back")
    pd_.add_argument("--lookback", type=int, default=3, help="days back from today to check")
    pd_.add_argument("--cleanup", action="store_true", help="delete processed data afterwards")
    _grid_args(pd_)
    pd_.set_defaults(func=cmd_daily)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
