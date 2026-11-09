"""Train / validation / test days, checked before any model is trained.

Two purposes, kept apart:

* **benchmark** (the default): numbers that claim how well a model does on unseen
  days. Training and validation days are strictly earlier than every test day, and the
  three sets do not overlap.
* **deployment**: retraining the published model on all data. It may use days after the
  test days, but never the test days themselves, and its numbers are not benchmarks.

In both, the sealed test days (5-18 Nov 2026) are refused for training and validation
until the sealed test has been scored (results/sealed_test.json exists). Each trained
model records its split in a versioned manifest (`split.json`).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

SPLIT_SCHEMA = "split/1"
PURPOSES = ("benchmark", "deployment")
SEALED_DAYS = tuple((dt.date(2026, 11, 5) + dt.timedelta(days=i)).isoformat()
                    for i in range(14))
SEALED_RESULT = Path("results/sealed_test.json")


class SplitError(ValueError):
    pass


def check_split(train: list[str], val: list[str], test: list[str], purpose: str = "benchmark",
                sealed_scored: bool | None = None) -> None:
    """Raise SplitError unless the split is allowed for this purpose."""
    if purpose not in PURPOSES:
        raise SplitError(f"purpose must be one of {PURPOSES}")
    tr, va, te = set(train), set(val), set(test)
    for (a, b), both in {("train", "validation"): tr & va, ("train", "test"): tr & te,
                         ("validation", "test"): va & te}.items():
        if both:
            raise SplitError(f"{a} and {b} days overlap: {sorted(both)}")
    if purpose == "benchmark" and te:
        first = min(te)
        late = sorted(d for d in tr | va if d >= first)
        if late:
            raise SplitError(f"benchmark training and validation must be before the first "
                             f"test day {first}; these are not: {late}. Use --purpose "
                             "deployment to retrain on later days (not a benchmark).")
    if sealed_scored is None:
        sealed_scored = SEALED_RESULT.exists()
    sealed = sorted((tr | va) & set(SEALED_DAYS))
    if sealed and not sealed_scored:
        raise SplitError(f"sealed test days cannot be used for training or validation "
                         f"before the sealed test is scored: {sealed}")


def manifest(train: list[str], val: list[str], test: list[str], purpose: str) -> dict:
    body = {"schema": SPLIT_SCHEMA, "purpose": purpose, "train": sorted(train),
            "validation": sorted(val), "test": sorted(test), "sealed_days": list(SEALED_DAYS)}
    body["hash"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]
    return body


def refuse_sealed(days, what: str, sealed_scored: bool | None = None) -> None:
    """Raise SplitError if any of days is a sealed day and the sealed test is not scored:
    evaluating on, sampling from or training on them would spend the sealed test."""
    if sealed_scored is None:
        sealed_scored = SEALED_RESULT.exists()
    hit = sorted(set(map(str, days or [])) & set(SEALED_DAYS))
    if hit and not sealed_scored:
        raise SplitError(f"{what} refused on sealed test days before the sealed test is "
                         f"scored: {hit}")
