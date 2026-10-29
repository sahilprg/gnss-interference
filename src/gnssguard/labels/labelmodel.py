"""Combine rule votes into one probabilistic label per cell-window.

Two combiners:

* `majority_vote`: share of non-abstaining rules voting interference (baseline).
* `DawidSkene`: a generative label model fitted by EM. Each rule has a learned
  sensitivity (how often it votes interference when there is interference) and
  specificity (how often it votes normal when there is none), measured only over the
  cases where it votes. Rules that agree with the consensus earn weight; noisy rules
  lose it. No ground truth is needed.

Abstentions of two-sided rules carry no evidence. Those rules abstain mostly because a
cell is sparse, and sparseness is not a symptom of interference; letting the model
learn from their abstentions made it call sparse cells "interference" in early tests.

One-sided rules (they only ever vote interference, e.g. MLAT fallback) are different:
for them "did not fire" is the other outcome, so they are scored as fired / not fired.
Conditioning them on voting would make them uninformative, since they always say yes.

This is the classic Dawid & Skene (1979) model, the idea behind data-programming tools
such as Snorkel, implemented here in plain numpy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ABSTAIN = -1


def majority_vote(votes: pd.DataFrame) -> pd.Series:
    v = votes.where(votes != ABSTAIN)
    return v.mean(axis=1)  # NaN where every rule abstained


def _onehot(x: np.ndarray) -> np.ndarray:
    """(n, m) votes -> (n, m, 2) one-hot over (votes normal, votes interference)."""
    return np.stack([(x == 0), (x == 1)], axis=-1).astype(np.float64)


class DawidSkene:
    def __init__(self, max_iter: int = 200, tol: float = 1e-7, smoothing: float = 1.0,
                 prior: float | None = None, one_sided: list[str] | str = "auto"):
        self.max_iter = max_iter
        self.tol = tol
        self.smoothing = smoothing
        self.fixed_prior = prior
        self.one_sided = one_sided
        self.rules_: list[str] = []
        self.one_sided_: list[str] = []
        self.prior_: float = 0.5
        self.conf_: np.ndarray | None = None  # (m rules, 2 true classes, 2 votes)
        self.n_iter_ = 0

    def _encode(self, votes: pd.DataFrame) -> np.ndarray:
        """Votes as used by the model: one-sided rules' abstains become 'did not fire'."""
        x = votes[self.rules_].to_numpy().copy()
        for j, r in enumerate(self.rules_):
            if r in self.one_sided_:
                x[:, j] = np.where(x[:, j] == 1, 1, 0)
        return x

    def fit(self, votes: pd.DataFrame) -> DawidSkene:
        self.rules_ = list(votes.columns)
        if self.one_sided == "auto":
            self.one_sided_ = [c for c in self.rules_ if not (votes[c] == 0).any()
                               and (votes[c] == 1).any()]
        else:
            self.one_sided_ = list(self.one_sided)
        raw = votes[self.rules_].to_numpy()
        keep = (raw != ABSTAIN).any(axis=1)
        if not keep.any():
            raise ValueError("every rule abstained on every row")
        x = self._encode(votes)[keep]
        oh = _onehot(x)
        # initialise from the majority vote of the rules that actually voted
        q = np.clip(np.nanmean(np.where(raw[keep] == ABSTAIN, np.nan, raw[keep]), axis=1),
                    0.05, 0.95)
        prev = -np.inf
        for it in range(1, self.max_iter + 1):
            self.n_iter_ = it
            self._m_step(oh, q)
            q, ll = self._e_step(oh)
            if abs(ll - prev) < self.tol * max(1.0, abs(ll)):
                break
            prev = ll
        return self

    def _m_step(self, oh: np.ndarray, q: np.ndarray) -> None:
        w = np.stack([1 - q, q], axis=1)
        counts = np.einsum("nc,nmk->mck", w, oh) + self.smoothing
        self.conf_ = counts / counts.sum(axis=2, keepdims=True)
        self.prior_ = self.fixed_prior if self.fixed_prior is not None else float(
            np.clip(q.mean(), 1e-4, 1 - 1e-4))

    def _e_step(self, oh: np.ndarray) -> tuple[np.ndarray, float]:
        lj = np.einsum("nmk,mck->nc", oh, np.log(self.conf_))
        lj = lj + np.log([1 - self.prior_, self.prior_])
        mx = lj.max(axis=1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(lj - mx).sum(axis=1))
        return np.exp(lj[:, 1] - lse), float(lse.sum())

    def predict_proba(self, votes: pd.DataFrame) -> pd.Series:
        """P(interference). NaN where every rule abstained."""
        active = (votes[self.rules_].to_numpy() != ABSTAIN).any(axis=1)
        out = np.full(len(votes), np.nan)
        if active.any():
            out[active], _ = self._e_step(_onehot(self._encode(votes)[active]))
        return pd.Series(out, index=votes.index, name="p_interference")

    def rule_table(self) -> pd.DataFrame:
        """Learned sensitivity and specificity of each rule.

        Two-sided rules: measured over the cells where the rule votes.
        One-sided rules: fired vs did not fire, over all cells some rule voted on.
        """
        c = self.conf_
        return pd.DataFrame({
            "rule": [r.removeprefix("lf_") for r in self.rules_],
            "kind": ["one-sided" if r in self.one_sided_ else "two-sided" for r in self.rules_],
            "sensitivity": np.round(c[:, 1, 1], 3),
            "specificity": np.round(c[:, 0, 0], 3),
        })
