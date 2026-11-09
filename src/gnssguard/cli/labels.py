"""Hand labels: blind samples, the labeling tool and the pre-declared analyses."""

from __future__ import annotations

import sys
from pathlib import Path

from gnssguard.cli._common import (
    PRODUCT_REGIONS,
    default_model,
    pd_options,
    pd_series,
    sealed_refused,
    split,
)


def cmd_gold(args) -> int:
    import pandas as pd

    from gnssguard.labels.gold import sample_gold

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} exists; it may hold your hand labels. Use --force to overwrite.",
              file=sys.stderr)
        return 1
    if sealed_refused(split(args.dates), "a gold sample"):
        return 1
    labels = pd.read_parquet(args.labels)
    if args.dates is None and sealed_refused(labels["date"].astype(str).unique(),
                                             "a gold sample over all labeled days"):
        return 1
    gold = sample_gold(labels, n=args.n, seed=args.seed, dates=split(args.dates))
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
    if sealed_refused([args.date], "a holdout sample"):
        return 1
    det = Detectors.load(args.model or default_model())
    cells = with_context(load_cells(args.cells, [args.date], split(args.regions)))
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
    print(json.dumps({"model": str(args.model or default_model()),
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
    if sealed_refused(split(args.dates), "a review sample"):
        return 1
    det = Detectors.load(args.model or default_model())
    cells = with_context(load_cells(args.cells, split(args.dates), split(args.regions)))
    cells = cells.reset_index(drop=True)
    rules = rule_frame(cells, det)["label"].astype(str)
    model = det.predict(cells)["label"].astype(str)
    gold, groups = review_sample(cells, rules, model, seed=args.seed, key_order=args.key_order)
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


def cmd_relabel(args) -> int:
    import json

    import pandas as pd

    from gnssguard.labels.relabel import relabel_eval, relabel_sample

    if args.eval:
        res = relabel_eval(pd.read_csv(args.out, dtype=str, keep_default_na=False),
                           pd.read_csv(args.key, dtype=str, keep_default_na=False))
        print(json.dumps(res, indent=1))
        return 0
    for f in (Path(args.out), Path(args.key)):
        if f.exists() and not args.force:
            print(f"{f} exists; it may hold labels. Use --force to overwrite.", file=sys.stderr)
            return 1
    out, key = relabel_sample(args.seed)
    out.to_csv(args.out, index=False)
    key.to_csv(args.key, index=False)
    print(key["source_set"].value_counts().to_string())
    print(f"{len(out)} cells -> {args.out}  (labeling file, earlier labels hidden)")
    print(f"key -> {args.key}  (do not open before labeling)")
    return 0


def cmd_score_bands(args) -> int:
    import json

    import pandas as pd

    from gnssguard.labels.sealed import score_bands

    gold = pd.read_csv(args.gold, dtype={"label": str})
    groups = pd.read_csv(args.groups)
    pools = None
    if args.dates:  # weight to all judged cells of the sample's days (recomputed scores)
        import numpy as np

        from gnssguard.features.cells import load_cells
        from gnssguard.labels.sealed import band_pools
        from gnssguard.model.dataset import with_context
        from gnssguard.model.train import Detectors

        det = Detectors.load(args.model or default_model())
        cells = with_context(load_cells(args.cells, split(args.dates))).reset_index(drop=True)
        p = det.predict(cells, calibrated=True)
        score = pd.Series(np.maximum(p["p_jamming"], p["p_spoofing"]), index=cells.index)
        g = groups.set_index("gold_id")["score"]
        check = gold.merge(cells.assign(now=score)[["region", "date", "cell_lat", "cell_lon",
                                                     "window", "now"]]
                           .astype({"region": str, "date": str, "window": str}),
                           on=["region", "date", "cell_lat", "cell_lon", "window"], how="left")
        drift = (check.set_index("gold_id")["now"] - g).abs().max()
        if not drift <= 1e-3:
            print(f"recomputed scores differ from the drawn ones (max {drift}); not weighting",
                  file=sys.stderr)
        else:
            pools = band_pools(cells, score)
    res = score_bands(gold, groups, source=args.source or Path(args.gold).name, pools=pools)
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    for b in res["bands"]:
        line = (f"score {b['band']:>12}: {b['interference_share']:.0%} of reviewed "
                f"({b['ci95'][0]:.0%}-{b['ci95'][1]:.0%}), {b['cells']} decided cells")
        if b.get("population_share") is not None:
            lo, hi = b["population_ci95"]
            line += f"; weighted to all cells {b['population_share']:.0%} ({lo:.0%}-{hi:.0%})"
        print(line)
    print(f"-> {args.out}")
    return 0


def cmd_sealed_sample(args) -> int:
    import numpy as np
    import pandas as pd

    from gnssguard.features.cells import load_cells
    from gnssguard.labels import sealed
    from gnssguard.model.dataset import with_context
    from gnssguard.model.evaluate import rule_frame
    from gnssguard.model.train import Detectors

    out, groups_out = Path(args.out), Path(args.groups)
    frames_out = Path(args.frames) if args.frames else groups_out.with_name(
        groups_out.name.replace("_groups.csv", ".csv").replace(".csv", "_frames.csv"))
    files = [out, groups_out] + ([frames_out] if args.kind == "test" else [])
    for f in files:
        if f.exists() and not args.force:
            print(f"{f} exists; it may hold labels. Use --force to overwrite.", file=sys.stderr)
            return 1
    if args.kind != "calibration" and args.key_order != "shuffled":
        print("refused: the sealed test sample is always drawn with shuffled region-days",
              file=sys.stderr)
        return 1
    det = Detectors.load(args.model or default_model())
    cells = with_context(load_cells(args.cells, split(args.dates), split(args.regions)))
    cells = cells.reset_index(drop=True)
    missing = set(split(args.dates)) - set(cells["date"].astype(str))
    if missing:
        print(f"no cells for {sorted(missing)}: ingest and run `cells` first", file=sys.stderr)
        return 1
    rules = rule_frame(cells, det)["label"].astype(str)
    p = det.predict(cells, calibrated=True)
    score = pd_series(np.maximum(p["p_jamming"], p["p_spoofing"]), cells.index)
    model = p["label"].astype(str)
    kw: dict = {"seed": args.seed} if args.seed is not None else {}
    if args.kind == "calibration":
        gold, groups = sealed.calibration_sample(cells, score, rules, model,
                                                 key_order=args.key_order, **kw)
    else:
        from gnssguard.model.baselines import baseline_values

        extra = baseline_values(cells).join(pd.DataFrame(
            {"n_ac_usable": cells["n_ac_usable"].to_numpy(),
             "p_jamming": np.asarray(p["p_jamming"]), "p_spoofing": np.asarray(p["p_spoofing"])},
            index=cells.index)).round(4)
        gold, groups, frames = sealed.sealed_test_sample(cells, score, rules, model,
                                                         extra=extra, **kw)
    out.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(out, index=False)
    groups.to_csv(groups_out, index=False)
    print(groups.groupby("group").agg(cells=("gold_id", "size"), pool=("pool", "first"))
          .to_string())
    if args.kind == "test":
        frames.to_csv(frames_out, index=False)
        drawn = groups["group"].value_counts()
        short = {g: (int(drawn.get(g, 0)), n) for g, n in sealed.TEST_SIZES.items()
                 if drawn.get(g, 0) < n}
        if short:
            print(f"short groups (drawn, wanted): {short}; reported short, not refilled")
        print(f"pools per group and region-day -> {frames_out}")
    print(f"{len(gold)} cell-windows -> {out}  (labeling file)")
    print(f"groups and frozen predictions -> {groups_out}  (do not open before labeling)")
    return 0


def cmd_sealed_score(args) -> int:
    """Score the sealed test once (questions 1-8, primary endpoint)."""
    import hashlib
    import json

    import pandas as pd

    from gnssguard.labels.sealed_score import score
    from gnssguard.product.publish import git_commit

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} exists: the sealed test is scored once. --force only to redo a run "
              "that failed, never to rescore after changing anything.", file=sys.stderr)
        return 1
    files = {"gold": Path(args.gold), "groups": Path(args.groups), "frames": Path(args.frames)}
    gold = pd.read_csv(files["gold"], dtype={"label": str})
    blank = gold["label"].fillna("").str.strip().eq("")
    if blank.any():
        print(f"{int(blank.sum())} cells have no label (use 'unsure' where you cannot "
              "decide); the sealed test is scored only when every cell is labeled",
              file=sys.stderr)
        return 1
    res = score(gold, pd.read_csv(files["groups"]), pd.read_csv(files["frames"]))
    res["provenance"] = {
        "scored_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        **git_commit(),
        "inputs_sha256": {k: hashlib.sha256(f.read_bytes()).hexdigest()[:16]
                          for k, f in files.items()}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    pe = res["primary_endpoint"]
    print(f"primary endpoint (question 6): {'PASSES' if pe['passes'] else 'does not pass'}")
    print(f"  recall difference, model - NIC 0.25: {pe['recall_difference']}")
    print(f"  false-alarm difference: {pe['false_alarm_difference']}")
    print(f"all questions -> {out}")
    return 0


def cmd_trust_report(args) -> int:
    from gnssguard.features.cells import load_cells
    from gnssguard.model import trust
    from gnssguard.model.dataset import with_context
    from gnssguard.model.train import Detectors

    rep = Path(args.report)
    rep.mkdir(parents=True, exist_ok=True)
    det = Detectors.load(args.model or default_model())
    with pd_options():
        if Path(args.rolling, "rolling_synthetic.csv").exists():
            ri = trust.rolling_intervals(args.rolling)
            ri.to_csv(rep / "rolling_intervals.csv", index=False)
            print("== rolling results, 95% intervals over test days\n"
                  + ri[ri["predictor"] == "model"].to_string(index=False))
        else:
            print(f"no rolling results in {args.rolling}; run `gnssguard rolling` first")
        golds = split(args.gold)
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
        ov = trust.aircraft_overlap(args.data, train, days, split(args.regions))
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


def cmd_disagreement_sample(args) -> int:
    import numpy as np

    from gnssguard.features.cells import load_cells
    from gnssguard.labels.disagreement import disagreement_sample
    from gnssguard.model.dataset import with_context
    from gnssguard.model.evaluate import rule_frame
    from gnssguard.model.train import Detectors

    out, groups_out = Path(args.out), Path(args.groups)
    for f in (out, groups_out):
        if f.exists() and not args.force:
            print(f"{f} exists; it may hold labels. Use --force to overwrite.", file=sys.stderr)
            return 1
    sizes = None
    if args.only:
        from gnssguard.labels.disagreement import SIZES

        keep = split(args.only)
        unknown = set(keep) - set(SIZES)
        if unknown:
            print(f"unknown group(s) {sorted(unknown)}; groups: {list(SIZES)}", file=sys.stderr)
            return 1
        sizes = {g: (n if g in keep else 0) for g, n in SIZES.items()}
    sealed = {f"2026-11-{d:02d}" for d in range(5, 19)}
    dates = split(args.dates)
    if sealed & set(dates):
        print("refused: 5-18 Nov 2026 are the sealed test days", file=sys.stderr)
        return 1
    det = Detectors.load(args.model or default_model())
    cells = with_context(load_cells(args.cells, dates, split(args.regions)))
    cells = cells.reset_index(drop=True)
    rules = rule_frame(cells, det)["label"].astype(str)
    model = det.predict(cells)["label"].astype(str)
    p = det.predict(cells, calibrated=True)
    score = pd_series(np.maximum(p["p_jamming"], p["p_spoofing"]), cells.index)
    gold, groups = disagreement_sample(cells, score, rules, model, seed=args.seed,
                                       sizes=sizes, key_order=args.key_order)
    out.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(out, index=False)
    groups.to_csv(groups_out, index=False)
    print(groups.groupby("group").agg(cells=("gold_id", "size"), pool=("pool", "first"))
          .to_string())
    print(f"{len(gold)} cell-windows -> {out}  (labeling file)")
    print(f"groups and frozen predictions -> {groups_out}  (do not open before labeling)")
    return 0


def cmd_disagreement_eval(args) -> int:
    import pandas as pd

    from gnssguard.labels.disagreement import disagreement_eval

    res = disagreement_eval(pd.read_csv(args.gold, dtype={"label": str}),
                            pd.read_csv(args.groups))
    with pd_options():
        print(res.to_string(index=False))
    return 0


def register(sub) -> None:
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

    pd2 = sub.add_parser("disagreement-sample",
                         help="blind sample of cells where the model and the rules disagree")
    pd2.add_argument("--dates", required=True, help="never the sealed test days 5-18 Nov")
    pd2.add_argument("--regions")
    pd2.add_argument("--cells", default="data/cells")
    pd2.add_argument("--model", help="model folder (default: models/detector if present)")
    pd2.add_argument("--out", default="labels/disagreement_1.csv")
    pd2.add_argument("--groups", default="labels/disagreement_1_groups.csv")
    pd2.add_argument("--seed", type=int, default=20261108)
    pd2.add_argument("--key-order", choices=("shuffled", "alphabetical"), default="shuffled",
                     help="region-day order; alphabetical only to redraw sets from before 8 Nov")
    pd2.add_argument("--only", help="draw only these groups, e.g. D4_rules_unsure")
    pd2.add_argument("--force", action="store_true")
    pd2.set_defaults(func=cmd_disagreement_sample)

    pe2 = sub.add_parser("disagreement-eval", help="who the labeler sided with, per group")
    pe2.add_argument("--gold", default="labels/disagreement_1.csv")
    pe2.add_argument("--groups", default="labels/disagreement_1_groups.csv")
    pe2.set_defaults(func=cmd_disagreement_eval)

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
    pq.add_argument("--key-order", choices=("shuffled", "alphabetical"), default="shuffled",
                     help="region-day order; alphabetical only to redraw sets from before 8 Nov")
    pq.add_argument("--out", required=True)
    pq.add_argument("--groups", required=True, help="groups file kept apart from the labeler")
    pq.add_argument("--force", action="store_true")
    pq.set_defaults(func=cmd_review_sample)

    pr = sub.add_parser("relabel", help="blind relabel of earlier hand-labeled cells")
    pr.add_argument("--out", default="labels/relabel_1.csv")
    pr.add_argument("--key", default="labels/relabel_1_key.csv")
    pr.add_argument("--seed", type=int, default=20261105)
    pr.add_argument("--eval", action="store_true", help="score the second labels")
    pr.add_argument("--force", action="store_true")
    pr.set_defaults(func=cmd_relabel)

    pb = sub.add_parser("score-bands", help="reviewer share of interference per score band")
    pb.add_argument("--gold", default="labels/calibration_1.csv")
    pb.add_argument("--groups", default="labels/calibration_1_groups.csv")
    pb.add_argument("--source", help="text recorded as the source")
    pb.add_argument("--out", default="models/score_bands.json")
    pb.add_argument("--dates", help="the sample's days: also weight to all their judged cells")
    pb.add_argument("--cells", default="data/cells")
    pb.add_argument("--model", help="model folder (default: models/detector if present)")
    pb.set_defaults(func=cmd_score_bands)

    px = sub.add_parser("sealed-sample", help="blind sample for the calibration or sealed test set")
    px.add_argument("--kind", choices=["calibration", "test"], required=True)
    px.add_argument("--dates", required=True)
    px.add_argument("--cells", default="data/cells")
    px.add_argument("--model", help="frozen model (default: models/detector if present)")
    px.add_argument("--regions", default=PRODUCT_REGIONS)
    px.add_argument("--seed", type=int, help="default: 20261104 (calibration), 20261119 (test)")
    px.add_argument("--frames", help="test only: pools per group and region-day "
                    "(default: next to --groups, as *_frames.csv)")
    px.add_argument("--key-order", choices=("shuffled", "alphabetical"), default="shuffled",
                     help="region-day order; alphabetical only to redraw sets from before 8 Nov")
    px.add_argument("--out", required=True)
    px.add_argument("--groups", required=True)
    px.add_argument("--force", action="store_true")
    px.set_defaults(func=cmd_sealed_sample)

    pz = sub.add_parser("sealed-score", help="score the sealed test once (questions 1-8)")
    pz.add_argument("--gold", default="labels/sealed_test.csv")
    pz.add_argument("--groups", default="labels/sealed_test_groups.csv")
    pz.add_argument("--frames", default="labels/sealed_test_frames.csv")
    pz.add_argument("--out", default="results/sealed_test.json")
    pz.add_argument("--force", action="store_true",
                    help="only to redo a run that failed; never to rescore")
    pz.set_defaults(func=cmd_sealed_score)

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
