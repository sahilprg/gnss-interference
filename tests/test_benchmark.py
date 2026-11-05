from gnssguard.model.benchmark import summary_md


def test_summary_lists_missing_inputs_instead_of_hiding_them():
    doc = {"provenance": {"generated_at": "t", "code_version": "0", "model_hash": "abc",
                          "model_train_dates": ["2026-09-29", "2026-10-24"], "inputs": {}},
           "results": {
               "calibration_1": {"status": "ok", "bands": [
                   {"band": "below 0.1", "interference_share": 0.1}]},
               "rolling": {"status": "not run", "reason": "no rolling results given"}}}
    md = summary_md(doc)
    assert "below 0.1: 10%" in md and "Model hash abc" in md
    assert "`rolling`: no rolling results given" in md
