"""Apply the acceptance rules of docs/evaluation_protocol.md to the rolling results.

    python scripts/protocol_decide.py reports/protocol base A B C

Reads <folder>/<variant>/rolling_synthetic.csv and rolling_real_flag_rates.csv, written by
`gnssguard rolling`, and compares each variant with the first one (the baseline).
"""

import sys
from pathlib import Path

import pandas as pd

RECALL_DROP = 0.01      # rule 1
FA_RISE = 0.002         # rule 2 (0.2 percentage points)
CALM_RISE = 0.002       # rule 3, mean over days
CALM_OVER_RULES = 0.005  # rule 3, any single day
GAIN = 0.02             # rule 4
CALM = "central_europe"


def load(folder: Path, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    syn = pd.read_csv(folder / variant / "rolling_synthetic.csv")
    real = pd.read_csv(folder / variant / "rolling_real_flag_rates.csv")
    return syn, real


def summary(syn: pd.DataFrame) -> pd.DataFrame:
    m = syn[syn["predictor"] == "model"]
    return (m.groupby(["synth_region", "profile"])
            [["recall_jamming", "recall_spoofing", "false_positive_rate_away_from_events"]]
            .mean())


def calm(real: pd.DataFrame) -> pd.DataFrame:
    r = real[real["region"] == CALM].pivot_table(index="test_day", columns="predictor",
                                                 values="flagged_share_dense")
    return r


def decide(base: str, variant: str, folder: Path) -> dict:
    sb, rb = load(folder, base)
    sv, rv = load(folder, variant)
    a, b = summary(sb), summary(sv)
    d = (b - a).round(4)
    cb, cv = calm(rb), calm(rv)
    rule1 = bool((d[["recall_jamming", "recall_spoofing"]] >= -RECALL_DROP).all().all())
    rule2 = bool((d["false_positive_rate_away_from_events"] <= FA_RISE).all())
    calm_rise = float(cv["model"].mean() - cb["model"].mean())
    calm_over = float((cv["model"] - cv["rules"]).max())
    rule3 = calm_rise <= CALM_RISE and calm_over <= CALM_OVER_RULES
    best_gain = float(d[["recall_jamming", "recall_spoofing"]].max().max())
    rule4 = best_gain >= GAIN
    return {"variant": variant, "rule1_no_lost_recall": rule1,
            "worst_recall_change": float(d[["recall_jamming", "recall_spoofing"]].min().min()),
            "rule2_no_synthetic_false_alarms": rule2,
            "worst_false_alarm_change": float(d["false_positive_rate_away_from_events"].max()),
            "rule3_no_real_false_alarms": rule3, "calm_mean_change": round(calm_rise, 4),
            "calm_max_over_rules": round(calm_over, 4),
            "rule4_gain": rule4, "best_recall_gain": best_gain,
            "accepted": rule1 and rule2 and rule3 and rule4}, d


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    base, *variants = sys.argv[2:]
    rows = []
    pd.set_option("display.width", 200)
    print(f"== baseline ({base}), model, mean over the selection days")
    print(summary(load(folder, base)[0]).round(3).to_string())
    for v in variants:
        row, d = decide(base, v, folder)
        rows.append(row)
        print(f"\n== {v} minus {base}")
        print(d.to_string())
    out = pd.DataFrame(rows)
    print("\n== decisions")
    print(out.to_string(index=False))
    out.to_csv(folder / "decisions.csv", index=False)
