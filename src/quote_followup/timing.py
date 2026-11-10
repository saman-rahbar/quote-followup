"""What days without a reply say about the outcome.

A quote ends in a yes, a no, or silence until it expires. From settled quotes this learns, per stratum
(type of work by default), the share of yeses and of nos still to come after a days, and turns
silence into a log-odds shift:

    logit P(yes | quiet at a) = logit p + log S_yes(a) - log[ q S_no(a) + (1 - q) ]

q is the chance a lost quote is declined rather than left to expire. A mixture model for competing
outcomes (Larson and Dinse 1985). Assumes reply speed depends only on the stratum and the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

POOLED: tuple = ()


@dataclass(frozen=True)
class _Curves:
    win_days: np.ndarray  # sorted days from sending to acceptance
    decline_days: np.ndarray  # sorted days from sending to decline

    @staticmethod
    def _still_quiet(days: np.ndarray, age: np.ndarray) -> np.ndarray:
        # Share of replies still to come after `age` days. We count one extra pretend quote that never
        # replies, so a brand-new quote starts at exactly 1 and the share never drops all the way to 0.
        later = days.size - np.searchsorted(days, age, side="right")
        return (later + 1) / (days.size + 1)

    def log_odds_shift(self, age: np.ndarray, decline_share: np.ndarray) -> np.ndarray:
        quiet_loss = decline_share * self._still_quiet(self.decline_days, age) + (1 - decline_share)
        return np.log(self._still_quiet(self.win_days, age)) - np.log(quiet_loss)


@dataclass(frozen=True)
class DecisionTiming:
    """Reply-speed curves per stratum. A stratum with fewer than `min_quotes` settled quotes falls
    back to its parent stratum, ending at the pooled curve."""

    segments: tuple[str, ...]
    curves: dict[tuple, _Curves]

    @classmethod
    def fit(
        cls, settled: pd.DataFrame, segments: tuple[str, ...] = ("workflow_type",), min_quotes: int = 200
    ) -> DecisionTiming:
        days = settled["days_to_decision"].to_numpy(dtype=float)
        won = settled["won"].to_numpy() == 1
        declined = settled["status"].to_numpy() == "declined"

        def curves(rows: np.ndarray) -> _Curves:
            return _Curves(np.sort(days[rows][won[rows]]), np.sort(days[rows][declined[rows]]))

        fitted = {POOLED: curves(np.arange(len(settled)))}
        for depth in range(1, len(segments) + 1):
            for key, rows in settled.groupby(list(segments[:depth]), observed=True).indices.items():
                if len(rows) >= min_quotes:
                    fitted[key if isinstance(key, tuple) else (key,)] = curves(rows)
        return cls(tuple(segments), fitted)

    def log_odds_shift(self, quotes: pd.DataFrame, decline_share: np.ndarray) -> np.ndarray:
        """How much `age_days` without a reply lowers each quote's chance (on the log-odds scale)."""
        age = quotes["age_days"].to_numpy(dtype=float)
        decline_share = np.broadcast_to(np.asarray(decline_share, dtype=float), age.shape)
        if not self.segments:
            return self.curves[POOLED].log_odds_shift(age, decline_share)
        shift = np.empty(len(quotes))
        for key, rows in quotes.groupby(list(self.segments), observed=True).indices.items():
            curve = self._curve_for(key if isinstance(key, tuple) else (key,))
            shift[rows] = curve.log_odds_shift(age[rows], decline_share[rows])
        return shift

    def _curve_for(self, key: tuple) -> _Curves:
        for depth in range(len(key), -1, -1):
            if key[:depth] in self.curves:
                return self.curves[key[:depth]]
        raise AssertionError("the pooled curve is always present")
