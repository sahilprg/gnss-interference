import numpy as np
import pandas as pd

from gnssguard.labels.review import review_eval, review_sample

W = pd.Timestamp("2026-10-27T10:00Z")


def day(n=600, seed=1):
    rng = np.random.default_rng(seed)
    cells = pd.DataFrame({
        "region": rng.choice(["central_europe", "baltic", "punjab"], n),
        "date": rng.choice(["2026-10-27", "2026-10-28"], n),
        "cell_lat": rng.integers(48, 58, n).astype(float),
        "cell_lon": rng.integers(8, 26, n).astype(float),
        "window": [W + pd.Timedelta(minutes=15 * i) for i in range(n)],
        "n_ac_usable": rng.integers(0, 9, n)})
    model = pd.Series(rng.choice(["normal", "jamming", "spoofing"], n, p=[0.85, 0.1, 0.05]))
    return cells, model.copy(), model


def test_sample_groups_sizes_blind_and_reproducible():
    cells, rules, model = day()
    gold, groups = review_sample(cells, rules, model)
    assert len(gold) == 100 and gold["gold_id"].is_unique
    assert "group" not in gold.columns and (gold["label"] == "").all()
    dense = cells.set_index(["cell_lat", "cell_lon", "window"])
    assert (dense.loc[list(zip(gold.cell_lat, gold.cell_lon, gold.window, strict=True)),
                      "n_ac_usable"] >= 3).all()
    calm_flag = groups[groups["group"] == "R1_calm_flag"]
    assert calm_flag["model_label"].isin(["jamming", "spoofing", "both"]).all()
    g2, _ = review_sample(cells, rules, model)
    pd.testing.assert_frame_equal(gold, g2)


def test_short_groups_move_to_calm_normal():
    cells, rules, model = day(n=200)
    gold, groups = review_sample(cells, rules, model)
    n = groups["group"].value_counts()
    assert n.get("R1_calm_flag", 0) < 50 and len(gold) == n.sum()
    want = 20 + (50 - n.get("R1_calm_flag", 0)) + (30 - n.get("R2_hotspot_flag", 0))
    pool = groups.loc[groups["group"] == "R3_calm_normal", "pool"].iloc[0]
    assert n["R3_calm_normal"] == min(want, pool)  # takes the shortfall, up to its pool


def test_eval_false_alarm_share():
    ids = [f"F{i:04d}" for i in range(30)]
    groups = pd.DataFrame({"gold_id": ids, "group": ["R1_calm_flag"] * 20 + ["R3_calm_normal"] * 10,
                           "weight": 1.0, "pool": 20, "rule_label": "jamming",
                           "model_label": ["jamming"] * 20 + ["normal"] * 10})
    gold = pd.DataFrame({"gold_id": ids, "label": ["normal"] * 15 + ["jamming"] * 4 + ["unsure"]
                         + ["normal"] * 10})
    r = review_eval(gold, groups, calm_flagged_share=0.006)
    p = r["primary_calm_flags"]
    assert p["decided"] == 19 and p["human_normal"] == 15 and p["unsure"] == 1
    assert p["estimated_real_false_alarm_rate"][1] == round(0.006 * 15 / 19, 5)
    assert r["R3_calm_normal"]["human_normal"] == 10
