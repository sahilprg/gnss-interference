import numpy as np
import pandas as pd
import pytest

from gnssguard.labels.holdout import (
    binom_two_sided,
    classify,
    holdout_eval,
    holdout_sample,
)

W = pd.Timestamp("2026-11-01T10:00Z")


def day(n=400, seed=0):
    rng = np.random.default_rng(seed)
    cells = pd.DataFrame({
        "region": rng.choice(["baltic", "punjab"], n), "date": "2026-11-01",
        "cell_lat": rng.integers(50, 60, n).astype(float),
        "cell_lon": rng.integers(10, 20, n).astype(float),
        "window": [W + pd.Timedelta(minutes=15 * i) for i in range(n)],
        "n_ac_usable": rng.integers(0, 8, n), "n_ac_degraded": rng.integers(0, 2, n),
        "nb_hot": 0})
    rules = pd.Series(np.where(cells["n_ac_usable"] < 3, "unknown",
                               rng.choice(["normal", "jamming"], n, p=[0.8, 0.2])))
    model = rules.where(rng.random(n) > 0.1, "jamming").where(
        cells["n_ac_usable"] >= 1, "unknown")
    model[(cells["n_ac_usable"] >= 3) & (rng.random(n) < 0.05)] = "normal"
    return cells, rules, model


def test_strata_are_the_declared_ones():
    cells, rules, model = day()
    s = classify(cells, rules, model)
    dense = cells["n_ac_usable"] >= 3
    flag = lambda x: x.isin(["jamming", "spoofing", "both"])  # noqa: E731
    assert (s[dense & (flag(rules) != flag(model))] == "S1_disagree").all()
    assert (s[(cells["n_ac_usable"].between(1, 2)) & flag(model)] == "S2_sparse").all()
    assert (s[cells["n_ac_usable"] == 0] == "").all()


def test_sample_is_blind_reproducible_and_capped():
    cells, rules, model = day()
    gold, strata = holdout_sample(cells, rules, model, n=150)
    again, _ = holdout_sample(cells, rules, model, n=150)
    assert gold.equals(again)
    assert list(gold.columns) == ["gold_id", "region", "date", "cell_lat", "cell_lon",
                                  "window", "label", "labeler", "notes"]  # nothing revealing
    counts = strata["stratum"].value_counts()
    assert counts.get("S1_disagree", 0) <= 40 and counts.get("S2_sparse", 0) <= 50
    assert len(gold) <= 150 and gold["gold_id"].is_unique
    s4 = strata[strata["stratum"] == "S4_random"]
    assert (s4["weight"] >= 1).all()


def test_binomial_and_verdict():
    assert binom_two_sided(5, 10) == pytest.approx(1.0)
    assert binom_two_sided(10, 10) == pytest.approx(2 / 1024)
    ids = [f"H{i:04d}" for i in range(12)]
    strata = pd.DataFrame({"gold_id": ids, "stratum": "S1_disagree", "weight": 1.0, "pool": 12,
                           "rule_label": "normal", "model_label": "jamming"})
    gold = pd.DataFrame({"gold_id": ids, "label": ["jamming"] * 11 + ["normal"]})
    res = holdout_eval(gold, strata)
    assert res["primary"]["human_sides_with_model"] == 11
    assert res["primary"]["verdict"] == "model better"
    few = holdout_eval(gold.head(5), strata.head(5))
    assert few["primary"]["verdict"].startswith("inconclusive")


def test_sparse_precision_verdict():
    ids = [f"H{i:04d}" for i in range(40)]
    strata = pd.DataFrame({"gold_id": ids, "stratum": "S2_sparse", "weight": 1.0, "pool": 40,
                           "rule_label": "unknown", "model_label": "jamming"})
    good = holdout_eval(pd.DataFrame({"gold_id": ids, "label": ["jamming"] * 34 + ["normal"] * 6}),
                        strata)
    assert good["primary_2_sparse_precision"]["verdict"] == "supported"
    bad = holdout_eval(pd.DataFrame({"gold_id": ids, "label": ["jamming"] * 8 + ["normal"] * 32}),
                       strata)
    assert bad["primary_2_sparse_precision"]["verdict"] == "not supported"


def test_day_weighting_keeps_each_group_at_its_size():
    # S3: 2 cells drawn from 1,000, one unsure; S4: 2 cells drawn from 1,000, both decided.
    strata = pd.DataFrame({
        "gold_id": ["A", "B", "C", "D"],
        "stratum": ["S3_borderline", "S3_borderline", "S4_random", "S4_random"],
        "weight": 500.0, "pool": 1000, "rule_label": "normal", "model_label": "normal"})
    gold = pd.DataFrame({"gold_id": ["A", "B", "C", "D"], "region": "baltic",
                         "label": ["jamming", "unsure", "normal", "normal"]})
    res = holdout_eval(gold, strata)
    # S3 agreement 0 (its one decided cell was missed), S4 agreement 1, equal sizes -> 0.5
    assert res["model_agreement"]["weighted_to_day"] == pytest.approx(0.5)
    assert res["rules_agreement"]["sample"] == pytest.approx(2 / 3, abs=1e-3)
