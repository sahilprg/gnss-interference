import numpy as np
import pandas as pd

from gnssguard.labels.sealed import calibration_sample, sealed_test_sample

W = pd.Timestamp("2026-09-15T00:00Z")


def day(n=2000, seed=3):
    rng = np.random.default_rng(seed)
    cells = pd.DataFrame({
        "region": rng.choice(["baltic", "black_sea", "central_europe"], n),
        "date": rng.choice(["2026-09-15", "2026-09-16"], n),
        "cell_lat": rng.integers(41, 58, n).astype(float),
        "cell_lon": rng.integers(8, 40, n).astype(float),
        "window": [W + pd.Timedelta(minutes=15 * i) for i in range(n)],
        "n_ac_usable": rng.integers(0, 12, n),
        "n_ac_degraded": rng.integers(0, 3, n), "n_ac_jump": 0, "n_ac_altdev": 0,
        "n_ac_speed_mismatch": 0, "n_ac_mlat_fallback": 0, "n_ac_low_nacp": 0,
        "n_ac_nic0": 0, "nb_hot": rng.integers(0, 2, n)})
    score = pd.Series(rng.choice([0.001, 0.3, 0.7, 0.999], n, p=[0.8, 0.01, 0.01, 0.18]))
    model = pd.Series(np.where(score >= 0.5, "jamming", "normal"))
    return cells, score, model.copy(), model


def test_calibration_fills_bands_and_moves_short_mid_band():
    cells, score, rules, model = day()
    gold, g = calibration_sample(cells, score, rules, model)
    n = g["group"].value_counts()
    assert len(gold) == 150 and n.sum() == 150
    dense_mid = ((cells["n_ac_usable"] >= 3) & score.between(0.1, 0.9, inclusive="left")).sum()
    assert n["C_mid"] == min(70, dense_mid)
    assert (g.loc[g["group"] == "C_high", "score"] >= 0.9).all()
    assert gold["gold_id"].str.startswith("K").all() and (gold["label"] == "").all()


def test_test_sample_groups_are_disjoint_and_blind():
    cells, score, rules, model = day()
    gold, g = sealed_test_sample(cells, score, rules, model)
    assert gold["gold_id"].is_unique and "group" not in gold
    assert set(g["group"]) <= {"T_random", "T_flag", "T_border", "T_sparse"}
    keyed = gold.merge(g, on="gold_id")
    assert keyed[["cell_lat", "cell_lon", "window"]].drop_duplicates().shape[0] == len(gold)
    sp = keyed[keyed["group"] == "T_sparse"]
    assert (sp["model_label"] == "jamming").all()
    g2 = sealed_test_sample(cells, score, rules, model)[0]
    pd.testing.assert_frame_equal(gold, g2)
