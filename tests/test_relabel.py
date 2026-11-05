import pandas as pd

from gnssguard.labels.relabel import cohen_kappa, relabel_eval, relabel_sample


def test_sample_hides_labels_and_keeps_key(tmp_path):
    rows = pd.DataFrame({"gold_id": [f"G{i}" for i in range(20)], "region": "baltic",
                         "date": "2026-10-26", "cell_lat": 55.0, "cell_lon": 20.0,
                         "window": "2026-10-26T00:00:00Z",
                         "label": ["jamming", "normal"] * 10, "labeler": "x", "notes": ""})
    f = tmp_path / "g.csv"
    rows.to_csv(f, index=False)
    out, key = relabel_sample(1, {"a": (str(f), 5)})
    assert len(out) == 5 and (out["label"] == "").all() and (out["labeler"] == "").all()
    assert out["gold_id"].str.startswith("B").all()
    assert set(key["source_id"]) <= set(rows["gold_id"]) and key["first_label"].ne("").all()


def test_eval_and_kappa():
    key = pd.DataFrame({"gold_id": ["B1", "B2", "B3", "B4"], "source_set": "a",
                        "source_id": ["G1", "G2", "G3", "G4"],
                        "first_label": ["jamming", "normal", "spoofing", "unsure"]})
    second = pd.DataFrame({"gold_id": ["B1", "B2", "B3", "B4"],
                           "label": ["jamming", "normal", "jamming", "normal"]})
    r = relabel_eval(second, key)
    assert r["decided_both"] == 3 and r["unsure_in_either"] == 1
    assert r["binary_agreement"] == 1.0 and r["exact_agreement"] == round(2 / 3, 3)
    assert cohen_kappa(pd.Series([1, 0]), pd.Series([1, 0])) == 1.0
