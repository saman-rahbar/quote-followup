"""The model must never see anything recorded after the moment it is asked about."""

import numpy as np
import pandas as pd

from quote_followup.data import OUTCOME_COLUMNS
from quote_followup.scorers import FollowUpModel
from quote_followup.timeline import open_quotes, settled_quotes

AS_OF = pd.Timestamp("2024-10-01 08:00")


def test_open_quotes_are_exactly_the_unanswered_unexpired_ones(quotes):
    view = open_quotes(quotes, AS_OF)
    full = quotes.loc[view.index]
    assert (full["sent_at"] <= AS_OF).all()
    assert (full["expires_at"] > AS_OF).all()
    assert (full["decided_at"].isna() | (full["decided_at"] > AS_OF)).all()

    answered_before = quotes["decided_at"] <= AS_OF
    assert not view.index.isin(quotes.index[answered_before]).any()


def test_open_quotes_carry_no_outcome_columns(quotes):
    assert not set(OUTCOME_COLUMNS) & set(open_quotes(quotes, AS_OF).columns)


def test_training_uses_only_settled_quotes_from_before_the_horizon(quotes):
    settled = settled_quotes(quotes, AS_OF, horizon_days=45)
    assert (settled["sent_at"] < AS_OF - pd.Timedelta(days=45)).all()
    assert (settled["resolved_at"] <= AS_OF).all()
    assert settled["won"].notna().all()


def test_rewriting_the_future_does_not_change_todays_scores(quotes, model_settings):
    """Flip every outcome that lands after AS_OF. Fitting and scoring at AS_OF must not notice."""
    future = quotes["resolved_at"].isna() | (quotes["resolved_at"] > AS_OF)
    rewritten = quotes.copy()
    rewritten.loc[future, "won"] = 1 - rewritten.loc[future, "won"].fillna(0)
    rewritten.loc[future, "decided_at"] = rewritten.loc[future, "decided_at"] + pd.Timedelta(days=3)
    rewritten.loc[future, "days_to_decision"] = rewritten.loc[future, "days_to_decision"] + 3

    def scores(book):
        model = FollowUpModel(model_settings).fit(book, AS_OF)
        return model.assess(open_quotes(book, AS_OF))

    pd.testing.assert_frame_equal(scores(quotes), scores(rewritten))
    assert np.isfinite(scores(quotes)["p_now"]).all()
