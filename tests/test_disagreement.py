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
