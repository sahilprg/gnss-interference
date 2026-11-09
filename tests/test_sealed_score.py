"""The sealed test's sampler and scoring, checked on fake labels before any real label
exists: the estimates recover known population values, and nothing breaks on empty or
small pools."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnssguard.labels import sealed_score as SS
from gnssguard.labels.review import _spread, inclusion
from gnssguard.labels.sealed import sealed_test_sample

REGIONS = ["baltic", "black_sea", "central_europe", "east_med", "punjab"]


def population(seed: int = 0):
    """Five regions x 14 days with very unequal region-day sizes (5 to 400 cells), a
    hidden truth, and predictors that track it imperfectly."""
    rng = np.random.default_rng(seed)
    rows = []
    for r_i, reg in enumerate(REGIONS):
        for d in range(14):
            n = int(rng.choice([5, 20, 80, 400]))
            rate = [0.6, 0.3, 0.02, 0.1, 0.25][r_i] * (0.5 + rng.random())
            for i in range(n):
                rows.append({"region": reg, "date": f"2026-11-{5 + d:02d}",
                             "cell_lat": 40.0 + i % 30, "cell_lon": 10.0 + i // 30,
                             "true": rng.random() < min(rate, 0.95)})
    c = pd.DataFrame(rows)
    c["window"] = pd.Timestamp("2026-11-05", tz="UTC") + pd.to_timedelta(
        rng.integers(0, 96, len(c)) * 15, unit="min")
    k = len(c)
    c["n_ac_usable"] = np.where(rng.random(k) < 0.08, rng.integers(1, 3, k), rng.integers(3, 12, k))
    use = c["n_ac_usable"]
    deg = np.where(c["true"], rng.binomial(use, 0.6), rng.binomial(use, 0.03))
    c["n_ac_degraded"] = deg
    c["n_ac_nacp_low"] = np.where(c["true"], rng.binomial(c["n_ac_usable"], 0.3), 0)
    c["nb_hot"] = (rng.random(k) < 0.1).astype(int)
    flag = np.where(c["true"], rng.random(k) < 0.85, rng.random(k) < 0.03)
    model = pd.Series(np.where(flag, "jamming", "normal"), index=c.index)
    rules = pd.Series(np.where(flag & (rng.random(k) < 0.9), "jamming",
                               np.where(deg == 1, "unknown", "normal")), index=c.index)
    score = pd.Series(np.where(flag, 0.9 + 0.1 * rng.random(k), 0.2 * rng.random(k)), index=c.index)
    p = pd.DataFrame({"p_jamming": score, "p_spoofing": 0.0}, index=c.index)
    return c, score, rules, model, p


def draw_and_label(c, score, rules, model, p, seed):
    from gnssguard.model.baselines import baseline_values

    extra = baseline_values(c).join(pd.DataFrame(
        {"n_ac_usable": c["n_ac_usable"], "p_jamming": p["p_jamming"],
         "p_spoofing": p["p_spoofing"]}, index=c.index))
    gold, groups, frames = sealed_test_sample(c, score, rules, model, seed=seed,
                                              extra=extra, n_sim=600)
    key = gold.merge(c.reset_index(), on=["region", "date", "cell_lat", "cell_lon", "window"],
                     how="left")
    gold["label"] = np.where(key["true"].to_numpy(bool), "jamming", "normal")
    return gold, groups, frames


def truth(c, model):
    """The population values the scoring is meant to estimate."""
    judged = c["n_ac_usable"] >= 3
    n_day = c[judged].groupby("date").size()
    w = c["date"].map(1 / n_day)
    flag = model == "jamming"
    t = c["true"]

    def dw(num, den):
        return float((w * (num & den)).sum() / (w * den).sum())
    return {"recall": dw(flag & t, t & judged), "fa": dw(flag & ~t, ~t & judged),
            "agree": dw(flag == t, judged),
            "false_alarm_share": float((~t[judged & flag]).mean())}


@pytest.mark.parametrize("pop_seed", [0, 1])
def test_scoring_recovers_population_values(pop_seed, monkeypatch):
    monkeypatch.setattr(SS, "BOOT", 30)
    c, score, rules, model, p = population(pop_seed)
    true = truth(c, model)
    est = {"recall": [], "fa": [], "agree": [], "false_alarm_share": []}
    for s in range(40):
        gold, groups, frames = draw_and_label(c, score, rules, model, p, seed=s)
        r = SS.score(gold, groups, frames)
        est["recall"].append(r["q6_q7_predictors"]["model"]["recall"]["estimate"])
        est["fa"].append(r["q6_q7_predictors"]["model"]["false_alarm_rate"]["estimate"])
        est["agree"].append(r["q1_agreement"]["model"]["estimate"])
        est["false_alarm_share"].append(
            r["q2"]["false_alarm_share_of_model_flags"]["estimate"])
    for k, v in est.items():
        # unbiased: the mean over 40 draws is close to the truth
        assert abs(np.mean(v) - true[k]) < 0.03, (k, np.mean(v), true[k])


def test_each_day_counts_the_same(monkeypatch):
    """Day weighting: a day with 10x the cells does not count 10x."""
    monkeypatch.setattr(SS, "BOOT", 30)
    c, score, rules, model, p = population(3)
    gold, groups, frames = draw_and_label(c, score, rules, model, p, seed=1)
    t = SS.load(gold, groups, frames)
    per_day = t.groupby("date")["w_day"].sum()
    # each day's weights add up to about 1 (its judged cells), whatever its size
    assert per_day.between(0.3, 3).all(), per_day.to_dict()


def test_inclusion_matches_the_sampler():
    keys = pd.Series([f"{r}|d{d}" for r in "abc" for d in range(6) for _ in range(3 + d)])
    idx = keys.index
    counts = keys.value_counts().to_dict()
    pi = inclusion(counts, 10, n_sim=4000, seed=1)
    hits = pd.Series(0.0, index=idx)
    for s in range(3000):
        hits.loc[_spread(idx, keys, 10, np.random.default_rng(1000 + s))] += 1
    emp = (hits / 3000).groupby(keys).mean()
    for k_, v in pi.items():
        assert abs(emp[k_] - v) < 0.03, (k_, emp[k_], v)


def test_empty_and_small_pools():
    assert _spread(pd.Index([]), pd.Series(dtype=str), 5, np.random.default_rng(0)) == []
    assert inclusion({}, 5) == {}
    c, score, rules, model, p = population(4)
    model = pd.Series("normal", index=c.index)  # no model flags: T_flag, T_sparse empty
    gold, groups, frames = sealed_test_sample(c, score, rules, model, n_sim=200)
    assert set(groups["group"]) <= {"T_random", "T_border"}
    small = c.groupby(["region", "date"]).head(1).reset_index(drop=True)  # 70 cells
    sm = pd.Series("jamming", index=small.index)
    g2 = sealed_test_sample(small, score.loc[small.index].reset_index(drop=True)
                            if False else pd.Series(0.95, index=small.index),
                            sm, sm, n_sim=200)[1]
    flags = g2[g2["group"] == "T_flag"]
    assert (flags["pi"] == 1.0).all()  # a pool smaller than its group: all taken, pi = 1
    nothing = c.iloc[:0]
    g3 = sealed_test_sample(nothing, score.iloc[:0], rules.iloc[:0], model.iloc[:0], n_sim=50)
    assert len(g3[0]) == 0


def test_scoring_rules(monkeypatch):
    monkeypatch.setattr(SS, "BOOT", 30)
    c, score, rules, model, p = population(5)
    gold, groups, frames = draw_and_label(c, score, rules, model, p, seed=2)
    gold.loc[gold.index[:5], "label"] = "unsure"
    r = SS.score(gold, groups, frames)
    assert r["unsure"] == 5 and r["decided"] == len(gold) - 5
    pe = r["primary_endpoint"]
    assert set(pe["unsure_bounds"]) == {"unsure_as_interference", "unsure_as_normal"}
    assert isinstance(pe["passes"], bool)
    gold.loc[gold.index[0], "label"] = "maybe"
    with pytest.raises(ValueError, match="labels must be"):
        SS.score(gold, groups, frames)


def test_sealed_score_command_scores_once(tmp_path, monkeypatch, capsys):
    import json

    from gnssguard import cli

    monkeypatch.setattr(SS, "BOOT", 30)
    c, score, rules, model, p = population(6)
    gold, groups, frames = draw_and_label(c, score, rules, model, p, seed=3)
    f = {n: tmp_path / f"{n}.csv" for n in ("gold", "groups", "frames")}
    blank = gold.copy()
    blank.loc[blank.index[0], "label"] = ""
    blank.to_csv(f["gold"], index=False)
    groups.to_csv(f["groups"], index=False)
    frames.to_csv(f["frames"], index=False)
    out = tmp_path / "sealed.json"
    argv = ["sealed-score", "--gold", str(f["gold"]), "--groups", str(f["groups"]),
            "--frames", str(f["frames"]), "--out", str(out)]
    assert cli.main(argv) == 1 and "no label" in capsys.readouterr().err  # every cell first
    gold.to_csv(f["gold"], index=False)
    assert cli.main(argv) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert set(res["provenance"]["inputs_sha256"]) == {"gold", "groups", "frames"}
    assert "passes" in res["primary_endpoint"]
    assert cli.main(argv) == 1 and "scored once" in capsys.readouterr().err
