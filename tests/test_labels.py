import numpy as np
import pandas as pd
import pytest

from gnssguard.labels.events import load_events, match_events
from gnssguard.labels.gold import sample_gold
from gnssguard.labels.labelmodel import DawidSkene, majority_vote
from gnssguard.labels.pipeline import combine
from gnssguard.labels.rules import ABSTAIN, apply_rules


def simulate(n=4000, prior=0.2, seed=0):
    """Truth + votes: 'good' is accurate, 'noisy' near random, 'one' one-sided."""
    rng = np.random.default_rng(seed)
    y = rng.random(n) < prior

    def two_sided(acc, cover):
        v = np.where(rng.random(n) < acc, y, ~y).astype(int)
        return np.where(rng.random(n) < cover, v, ABSTAIN)

    one = np.where(y & (rng.random(n) < 0.6), 1, ABSTAIN)
    votes = pd.DataFrame({"good": two_sided(0.95, 0.8), "good2": two_sided(0.9, 0.8),
                          "noisy": two_sided(0.55, 0.8), "one": one})
    return y, votes


def test_dawid_skene_learns_rule_quality():
    y, votes = simulate()
    m = DawidSkene().fit(votes)
    t = m.rule_table().set_index("rule")
    assert t.loc["good", "sensitivity"] > 0.9 and t.loc["good", "specificity"] > 0.9
    assert t.loc["noisy", "sensitivity"] < 0.7
    assert m.one_sided_ == ["one"]
    p = m.predict_proba(votes)
    acc = ((p >= 0.5) == y)[p.notna()].mean()
    mv = majority_vote(votes)
    acc_mv = ((mv >= 0.5) == y)[mv.notna()].mean()
    assert acc > 0.93 and acc >= acc_mv
    assert abs(m.prior_ - y.mean()) < 0.05


def test_all_abstain_gives_nan():
    votes = pd.DataFrame({"a": [1, 0, ABSTAIN], "b": [1, 0, ABSTAIN]})
    p = DawidSkene().fit(votes).predict_proba(votes)
    assert np.isnan(p.iloc[2]) and p.iloc[0] > p.iloc[1]


def cells_row(**kw):
    base = dict(n_ac_usable=5, n_ac_degraded=0, share_ac_degraded=0.0, n_ac_nic0=0,
                n_ac_nacp_low=0, share_ac_nacp_low=0.0, n_ac_mlat_fallback=0, n_ac_jump=0,
                n_ac_speed_mismatch=0, n_ac_altpair=5, n_ac_altdev=0, share_ac_altdev=0.0)
    base.update(kw)
    return base


def test_rules_vote_and_abstain():
    c = pd.DataFrame([
        cells_row(),  # healthy, busy
        cells_row(n_ac_degraded=4, share_ac_degraded=0.8, n_ac_nic0=3),
        cells_row(n_ac_usable=2, n_ac_altpair=2),  # too sparse to judge
        cells_row(n_ac_speed_mismatch=2),
    ])
    v = apply_rules(c)
    assert v.loc[0, "lf_nic_cluster"] == 0 and v.loc[1, "lf_nic_cluster"] == 1
    assert v.loc[1, "lf_nic_zero"] == 1
    assert v.loc[2, "lf_nic_cluster"] == ABSTAIN and v.loc[2, "lf_alt_mismatch"] == ABSTAIN
    assert v.loc[3, "lf_speed_mismatch"] == 1 and v.loc[0, "lf_speed_mismatch"] == ABSTAIN
    assert v.loc[0, "lf_mlat_fallback"] == ABSTAIN  # one-sided: never votes normal


def test_combine_labels():
    lab = combine(pd.Series([0.9, 0.9, 0.1, 0.1, np.nan]), pd.Series([0, 1, 1, 0, 0]),
                  pd.Series([True, True, True, True, False]))
    assert list(lab) == ["jamming", "both", "spoofing", "normal", "unknown"]


def test_events_match_and_validation(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text(
        "event_id,type,lat_min,lat_max,lon_min,lon_max,start_utc,end_utc,source_url,notes\n"
        "E1,Jamming,54.2,55.0,20.0,21.5,2026-10-26T10:00:00Z,2026-10-26T11:00:00Z,"
        "https://example.org/notam,\n")
    ev = load_events(path)
    cells = pd.DataFrame({
        "cell_lat": [54.0, 54.0, 56.0], "cell_lon": [20.0, 20.0, 20.0],
        "window": pd.to_datetime(["2026-10-26T10:45Z", "2026-10-26T11:00Z",
                                  "2026-10-26T10:15Z"], utc=True),
        "cell_deg": 1.0, "window_min": 15})
    m = match_events(cells, ev)
    assert m["event_id"].tolist()[0] == "E1" and m["event_type"].iloc[0] == "jamming"
    assert m["event_id"].isna().tolist()[1:] == [True, True]  # ends at 11:00; wrong lat

    path.write_text(path.read_text().replace("https://example.org/notam", ""))
    with pytest.raises(ValueError, match="source_url"):
        load_events(path)


def test_gold_sample_is_stratified_and_blind():
    rng = np.random.default_rng(0)
    n = 600
    labels = pd.DataFrame({
        "region": rng.choice(["a", "b"], n), "date": "2026-10-26",
        "cell_lat": rng.integers(50, 55, n).astype(float),
        "cell_lon": rng.integers(10, 15, n).astype(float),
        "window": pd.Timestamp("2026-10-26T10:00Z")
        + pd.to_timedelta(rng.integers(0, 96, n) * 15, "min"),
        "n_ac_usable": rng.integers(0, 10, n), "p_jamming": rng.random(n),
        "spoof_votes": (rng.random(n) < 0.05).astype(int)})
    g = sample_gold(labels, n=60, seed=1)
    assert len(g) == 60 and g["gold_id"].is_unique
    assert "p_jamming" not in g.columns and "spoof_votes" not in g.columns
    keyed = g.merge(labels, on=["region", "date", "cell_lat", "cell_lon", "window"])
    assert (keyed["n_ac_usable"] >= 3).all()
    assert keyed["p_jamming"].lt(0.2).any() and keyed["p_jamming"].ge(0.8).any()
