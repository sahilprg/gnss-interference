import numpy as np
import pandas as pd

from gnssguard.labels.disagreement import disagreement_eval, disagreement_sample


def cells_and_labels():
    rows, model, rules = [], [], []
    plan = ([("spoofing", "normal")] * 50 + [("normal", "jamming")] * 3
            + [("both", "jamming")] * 2 + [("normal", "unknown")] * 100
            + [("normal", "normal")] * 300 + [("jamming", "jamming")] * 50)
    for i, (m, r) in enumerate(plan):
        rows.append({"region": ["baltic", "east_med"][i % 2], "date": f"2026-10-{20 + i % 5}",
                     "cell_lat": 50.0 + i % 7, "cell_lon": 10.0 + i % 11,
                     "window": pd.Timestamp("2026-10-20T00:00Z") + pd.Timedelta(minutes=15 * i),
                     "n_ac_usable": 5})
        model.append(m)
        rules.append(r)
    c = pd.DataFrame(rows)
    return c, pd.Series(0.5, index=c.index), pd.Series(rules), pd.Series(model)


def test_sample_takes_only_disagreements_and_small_pools_whole():
    c, score, rules, model = cells_and_labels()
    gold, groups = disagreement_sample(c, score, rules, model)
    n = groups.groupby("group").size().to_dict()
    assert n == {"D1_model_flags": 30, "D2_rules_flag": 3, "D3_type": 2, "D4_rules_unsure": 20}
    assert (groups["model_label"] != groups["rule_label"]).all()
    assert "model_label" not in gold and gold["label"].eq("").all()  # blind file
    assert groups.loc[groups["group"] == "D1_model_flags", "pool"].iloc[0] == 50


def test_eval_says_who_was_right():
    gold = pd.DataFrame({"gold_id": ["D1", "D2", "D3", "D4"],
                         "label": ["spoofing", "normal", "unsure", "jamming"]})
    groups = pd.DataFrame({"gold_id": ["D1", "D2", "D3", "D4"],
                           "group": ["D1_model_flags", "D1_model_flags", "D1_model_flags",
                                     "D3_type"],
                           "pool": [50, 50, 50, 2],
                           "model_label": ["spoofing", "spoofing", "spoofing", "both"],
                           "rule_label": ["normal", "normal", "normal", "jamming"]})
    r = disagreement_eval(gold, groups).set_index("group")
    assert r.loc["D1_model_flags", "decided"] == 2 and r.loc["D1_model_flags", "unsure"] == 1
    d1 = r.loc["D1_model_flags"]
    assert d1["model_right"] == 1 and d1["rules_right"] == 1
    assert r.loc["D3_type", "rules_right"] == 1 and r.loc["D3_type", "model_right"] == 0


def _region_days(n_regions=5, n_days=14, per_key=30):
    regions = ["baltic", "black_sea", "central_europe", "east_med", "punjab"][:n_regions]
    rows = [{"region": r, "date": f"2026-11-{5 + d:02d}", "i": i}
            for r in regions for d in range(n_days) for i in range(per_key)]
    return pd.DataFrame(rows)


def test_spread_reaches_every_region_when_keys_outnumber_k():
    from gnssguard.labels.review import _spread

    df = _region_days()
    key = df["region"] + "|" + df["date"]  # 70 region-days
    old = _spread(df.index, key, 20, np.random.default_rng(1), key_order="alphabetical")
    assert set(df.loc[old, "region"]) == {"baltic", "black_sea"}  # the bug: first 20 keys
    for seed in range(20):
        new = _spread(df.index, key, 20, np.random.default_rng(seed))
        assert len(new) == 20 and len(set(key[new])) == 20  # still one per region-day
        # four per region, on random days (not the earliest)
        assert df.loc[new, "region"].value_counts().to_dict() == {
            r: 4 for r in ("baltic", "black_sea", "central_europe", "east_med", "punjab")}
        assert new == _spread(df.index, key, 20, np.random.default_rng(seed))  # seeded
    days = {df.loc[i, "date"] for s in range(20)
            for i in _spread(df.index, key, 20, np.random.default_rng(s))}
    assert len(days) == 14  # over seeds, every day gets drawn


def test_sealed_sample_spreads_every_group_over_regions():
    from gnssguard.labels.sealed import sealed_test_sample

    df = _region_days(per_key=12)
    n = len(df)
    rng = np.random.default_rng(0)
    i = df["i"].to_numpy()
    # per region-day: usable / sparse cells, some with one degraded aircraft (borderline)
    df["n_ac_usable"] = np.where(i < 8, 5, 1)
    df["n_ac_degraded"] = np.where(i % 4 == 1, 1, 0)
    df["nb_hot"] = 0
    df["cell_lat"], df["cell_lon"] = 50.0 + i, 20.0
    df["window"] = pd.Timestamp("2026-11-05", tz="UTC")
    model = pd.Series(np.where(i % 4 == 0, "jamming", "normal"), index=df.index)
    rules = pd.Series(np.where(i % 4 == 0, "jamming", "unknown"), index=df.index)
    score = pd.Series(rng.random(n), index=df.index)
    gold, groups = sealed_test_sample(df, score, rules, model)
    m = gold.merge(groups, on="gold_id")
    for name, g in m.groupby("group"):
        assert g["region"].nunique() == 5, (name, g["region"].value_counts().to_dict())


def test_alphabetical_order_reproduces_the_sets_drawn_before_8_nov():
    from gnssguard.labels.review import _spread

    def before(idx, by, k, rng):  # the function as it was until 8 Nov 2026
        groups = {v: list(rng.permutation(np.asarray(g)))
                  for v, g in idx.groupby(by.loc[idx]).items()}
        out = []
        while len(out) < k and any(groups.values()):
            for v in sorted(groups):
                if groups[v] and len(out) < k:
                    out.append(groups[v].pop())
        return out

    df = _region_days(per_key=7)
    key = df["region"] + "|" + df["date"]
    for k in (5, 20, 70, 150):
        assert (_spread(df.index, key, k, np.random.default_rng(k), key_order="alphabetical")
                == before(df.index, key, k, np.random.default_rng(k)))


def test_only_one_group_can_be_drawn(tmp_path, capsys):
    c, score, rules, model = cells_and_labels()
    sizes = {"D1_model_flags": 0, "D2_rules_flag": 0, "D3_type": 0, "D4_rules_unsure": 20}
    gold, groups = disagreement_sample(c, score, rules, model, sizes=sizes)
    assert groups["group"].unique().tolist() == ["D4_rules_unsure"] and len(gold) == 20

    from gnssguard import cli
    argv = ["disagreement-sample", "--dates", "2026-10-20", "--only", "D9_nonsense",
            "--out", str(tmp_path / "d.csv"), "--groups", str(tmp_path / "g.csv")]
    assert cli.main(argv) == 1 and "unknown group" in capsys.readouterr().err
