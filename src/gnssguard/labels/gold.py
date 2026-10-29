"""Sample cell-windows for the hand-checked gold test set.

The sample is stratified by region and by label-model confidence, so it contains
clear positives, clear negatives and the uncertain middle, not just easy cases.
Only cell-windows with enough aircraft to judge are eligible.

The CSV for labeling deliberately leaves out rule votes and model probabilities:
label from the map (`gnssguard map --gold ...`) without seeing what the model thinks.
Days used for the gold set must be excluded from training in Phase 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gnssguard.features.cells import CELL_KEYS
from gnssguard.labels.rules import MIN_USABLE

BINS = [(0.0, 0.2, "likely_normal"), (0.2, 0.8, "uncertain"), (0.8, 1.01, "likely_jamming")]
GOLD_COLUMNS = ["gold_id", "region", "date", *CELL_KEYS, "label", "labeler", "notes"]
GOLD_LABELS = ("jamming", "spoofing", "both", "normal", "unsure")


def sample_gold(labels: pd.DataFrame, n: int = 300, seed: int = 0,
                dates: list[str] | None = None) -> pd.DataFrame:
    df = labels[labels["n_ac_usable"] >= MIN_USABLE].copy()
    if dates:
        df = df[df["date"].isin(dates)]
    df = df[df["p_jamming"].notna()]
    if df.empty:
        raise ValueError("no eligible cell-windows (need labels with enough aircraft)")
    df["stratum"] = pd.cut(df["p_jamming"], [b[0] for b in BINS] + [BINS[-1][1]],
                           labels=[b[2] for b in BINS], right=False).astype(str)
    df.loc[df["spoof_votes"] >= 1, "stratum"] = "spoof_candidate"
    groups = df.groupby(["region", "stratum"], observed=True)
    per = max(1, int(np.ceil(n / max(groups.ngroups, 1))))
    rng = np.random.default_rng(seed)
    picks = pd.concat([g.iloc[rng.permutation(len(g))[:per]] for _, g in groups])
    # Small strata (few uncertain or spoofing cells) leave the sample short: top up at
    # random from everything not yet picked.
    if len(picks) < n:
        rest = df.drop(index=picks.index)
        picks = pd.concat([picks, rest.sample(n=min(n - len(picks), len(rest)),
                                              random_state=seed)])
    out = picks.sample(frac=1.0, random_state=seed).head(n).reset_index(drop=True)
    out["gold_id"] = [f"G{i:04d}" for i in range(1, len(out) + 1)]
    out["label"] = ""  # fill in one of GOLD_LABELS
    out["labeler"] = ""
    out["notes"] = ""
    return out[GOLD_COLUMNS]
