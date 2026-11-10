"""Ways of ordering a morning's waiting quotes, behind one interface: `fit` on what was known at a
date, then `priority` per quote (higher means call sooner). The backtest compares them all through it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from .config import ModelSettings
from .propensity import LogisticScore
from .timeline import settled_quotes
from .timing import DecisionTiming

SILENCE = "days_quiet"

# The names people see in tables and charts. Defined once so every output uses the same words.
OUR_LIST = "Our list: chance x amount"
OUR_LIST_BY_CHANCE = "Our list, by chance only"
OUR_LIST_NO_SILENCE = "Our list, ignoring days without a reply"
DAYS_QUIET_X_AMOUNT = "Days without a reply x amount"
BIGGEST_FIRST = "Biggest quote first"
NEWEST_FIRST = "Newest quote first"
RANDOM_ORDER = "Random order"


class Scorer(Protocol):
    name: str

    def fit(self, quotes: pd.DataFrame, as_of: pd.Timestamp) -> Scorer: ...

    def priority(self, candidates: pd.DataFrame) -> np.ndarray:
        """Higher means call it sooner."""


RankBy = Literal["expected_value", "p_now", "expected_value_at_send"]


@dataclass
class FollowUpModel:
    """Our model: the chance of a yes today, times the quote amount.

    It learns three things from past quotes whose outcome is known: how likely a quote is to be
    accepted when it goes out, how often a customer who says no says it rather than going quiet,
    and how quickly yeses and nos usually arrive. timing.py explains how they fit together.
    """

    settings: ModelSettings
    rank_by: RankBy = "expected_value"
    name: str = OUR_LIST
    propensity: LogisticScore = field(init=False, repr=False)
    decline_share: LogisticScore = field(init=False, repr=False)
    timing: DecisionTiming = field(init=False, repr=False)
    trained_on: int = field(init=False, default=0)

    def fit(self, quotes: pd.DataFrame, as_of: pd.Timestamp) -> FollowUpModel:
        s = self.settings
        settled = settled_quotes(quotes, as_of, s.label_horizon_days)
        lost = settled[settled["won"] == 0]
        self.propensity = LogisticScore(s.categorical, s.numeric, s.regularization_c).fit(settled, settled["won"])
        self.decline_share = LogisticScore(s.categorical, s.numeric, s.regularization_c).fit(
            lost, lost["status"].eq("declined")
        )
        self.timing = DecisionTiming.fit(settled, s.timing_segments, s.timing_min_quotes)
        self.trained_on = len(settled)
        return self

    def assess(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """For each open quote: chance of a yes when sent and today, what it is worth, and why."""
        at_send = self.propensity.log_odds(candidates)
        q = self.decline_share.probability(candidates)
        silence = self.timing.log_odds_shift(candidates, q)
        p_now = expit(at_send + silence)
        reasons = self.propensity.contributions(candidates).assign(**{SILENCE: silence})
        amount = candidates["quoted_amount_usd"].to_numpy()
        return reasons.add_prefix("why_").assign(
            p_at_send=expit(at_send),
            p_now=p_now,
            decline_share=q,
            expected_value=p_now * amount,
            expected_value_at_send=expit(at_send) * amount,
            days_until_half=self._days_until_half(candidates, at_send, q, p_now),
        )

    def priority(self, candidates: pd.DataFrame) -> np.ndarray:
        return self.assess(candidates)[self.rank_by].to_numpy()

    def _days_until_half(
        self, candidates: pd.DataFrame, at_send: np.ndarray, q: np.ndarray, p_now: np.ndarray
    ) -> np.ndarray:
        """How many more quiet days until today's chance is cut in half (inf if not before it expires)."""
        days_left = (
            (candidates["expires_at"] - candidates["sent_at"]) / pd.Timedelta(days=1) - candidates["age_days"]
        ).to_numpy()
        result = np.full(len(candidates), np.inf)
        for step in np.arange(0.5, 30.5, 0.5):
            later = candidates.assign(age_days=candidates["age_days"] + step)
            halved = expit(at_send + self.timing.log_odds_shift(later, q)) <= p_now / 2
            result[np.isinf(result) & halved & (step <= days_left)] = step
        return result


@dataclass(frozen=True)
class RuleOfThumb:
    """A fixed order an owner could use with no model at all."""

    name: str
    rule: Callable[[pd.DataFrame], np.ndarray]

    def fit(self, quotes: pd.DataFrame, as_of: pd.Timestamp) -> RuleOfThumb:
        return self

    def priority(self, candidates: pd.DataFrame) -> np.ndarray:
        return self.rule(candidates)


@dataclass
class AgeCurveValue:
    """The toughest simple rival: how often quotes this quiet end in a yes, times the amount.

    It knows that silence matters but nothing else about the quote. Comparing it with our model
    shows what knowing the quote itself adds.
    """

    horizon_days: int
    name: str = DAYS_QUIET_X_AMOUNT
    base_log_odds: float = field(init=False, default=0.0)
    decline_share: float = field(init=False, default=0.0)
    timing: DecisionTiming = field(init=False, repr=False)

    def fit(self, quotes: pd.DataFrame, as_of: pd.Timestamp) -> AgeCurveValue:
        settled = settled_quotes(quotes, as_of, self.horizon_days)
        self.base_log_odds = float(logit(settled["won"].mean()))
        self.decline_share = float(settled.loc[settled["won"] == 0, "status"].eq("declined").mean())
        self.timing = DecisionTiming.fit(settled, segments=())
        return self

    def probability(self, candidates: pd.DataFrame) -> np.ndarray:
        return expit(self.base_log_odds + self.timing.log_odds_shift(candidates, self.decline_share))

    def priority(self, candidates: pd.DataFrame) -> np.ndarray:
        return self.probability(candidates) * candidates["quoted_amount_usd"].to_numpy()


def _shuffled(candidates: pd.DataFrame) -> np.ndarray:
    # Shuffle by a hash of the quote id, so the "random" order comes out the same on every run.
    return pd.util.hash_array(candidates["quote_id"].to_numpy()).astype(float)


RULES_OF_THUMB: tuple[RuleOfThumb, ...] = (
    RuleOfThumb(BIGGEST_FIRST, lambda c: c["quoted_amount_usd"].to_numpy()),
    RuleOfThumb(NEWEST_FIRST, lambda c: -c["age_days"].to_numpy()),
    RuleOfThumb(RANDOM_ORDER, _shuffled),
)


def all_scorers(settings: ModelSettings) -> list[Scorer]:
    """Our list, two cut-down versions of it, the toughest simple rival and the rules of thumb."""
    return [
        FollowUpModel(settings),
        FollowUpModel(settings, rank_by="p_now", name=OUR_LIST_BY_CHANCE),
        FollowUpModel(settings, rank_by="expected_value_at_send", name=OUR_LIST_NO_SILENCE),
        AgeCurveValue(settings.label_horizon_days),
        *RULES_OF_THUMB,
    ]
