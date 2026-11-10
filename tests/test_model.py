import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from quote_followup.propensity import LogisticScore
from quote_followup.scorers import SILENCE, FollowUpModel
from quote_followup.timeline import open_quotes, settled_quotes
from quote_followup.timing import POOLED, DecisionTiming

AS_OF = pd.Timestamp("2024-11-01 08:00")


def _toy_settled() -> pd.DataFrame:
    return pd.DataFrame({
        "workflow_type": ["recurring"] * 6,
        "won": [1, 1, 1, 0, 0, 0],
        "status": ["accepted"] * 3 + ["declined", "expired", "expired"],
        "days_to_decision": [0.5, 1.0, 1.5, 6.0, np.nan, np.nan],
    })


def _ages(*ages: float) -> pd.DataFrame:
    return pd.DataFrame({"workflow_type": "recurring", "age_days": list(ages)})


def test_silence_pushes_the_odds_down_while_wins_are_still_arriving():
    shift = DecisionTiming.fit(_toy_settled(), min_quotes=1).log_odds_shift(_ages(0.0, 0.7, 1.2, 3.0), 0.5)
    assert shift[0] == pytest.approx(0.0)
    assert np.all(np.diff(shift) < 0)


def test_silence_counts_for_less_when_losses_are_usually_declined_out_loud():
    # If this customer's losses would mostly be explicit declines, a quiet week is less likely to
    # be a loss in progress, so the shift is less negative.
    timing = DecisionTiming.fit(_toy_settled(), min_quotes=1)
    quiet_week = _ages(7.0)  # after the toy decline on day 6
    assert timing.log_odds_shift(quiet_week, 0.9)[0] > timing.log_odds_shift(quiet_week, 0.1)[0]


def test_with_no_declines_the_shift_is_just_the_win_curve():
    timing = DecisionTiming.fit(_toy_settled(), min_quotes=1)
    expected = np.log((0 + 1) / (3 + 1))  # no win answered after day 2, plus the never-answering pseudo-quote
    assert timing.log_odds_shift(_ages(2.0), 0.0)[0] == pytest.approx(expected)


def test_thin_strata_back_off_to_the_coarser_curve():
    settled = _toy_settled().assign(customer_channel=["referral"] * 5 + ["marketplace"])
    timing = DecisionTiming.fit(settled, ("workflow_type", "customer_channel"), min_quotes=5)
    assert ("recurring", "referral") in timing.curves
    assert ("recurring", "marketplace") not in timing.curves
    thin = _ages(2.0).assign(customer_channel="marketplace")
    parent = timing.curves[("recurring",)].log_odds_shift(np.array([2.0]), np.array([0.5]))
    assert timing.log_odds_shift(thin, 0.5) == pytest.approx(parent)


def test_unseen_stratum_falls_back_to_the_pooled_curve():
    timing = DecisionTiming.fit(_toy_settled(), min_quotes=1)
    unseen = pd.DataFrame({"workflow_type": ["snow"], "age_days": [2.0]})
    pooled = timing.curves[POOLED].log_odds_shift(np.array([2.0]), np.array([0.5]))
    assert timing.log_odds_shift(unseen, 0.5) == pytest.approx(pooled)


def test_reasons_add_up_to_the_score(quotes, model_settings):
    settled = settled_quotes(quotes, AS_OF, model_settings.label_horizon_days)
    s = model_settings
    score = LogisticScore(s.categorical, s.numeric, s.regularization_c).fit(settled, settled["won"])
    sample = quotes.head(50)
    total = score.contributions(sample).sum(axis=1) + score.baseline_log_odds()
    assert total.to_numpy() == pytest.approx(score.log_odds(sample))


def test_win_chance_now_is_the_send_time_chance_updated_by_silence(quotes, model_settings):
    model = FollowUpModel(model_settings).fit(quotes, AS_OF)
    candidates = open_quotes(quotes, AS_OF)
    assessed = model.assess(candidates)
    expected = logit(assessed["p_at_send"]) + assessed[f"why_{SILENCE}"]
    assert logit(assessed["p_now"]).to_numpy() == pytest.approx(expected.to_numpy())
    assert (assessed["expected_value"] <= candidates["quoted_amount_usd"]).all()
    assert assessed["decline_share"].between(0, 1).all()


def test_a_saved_model_scores_exactly_like_the_one_in_memory(quotes, model_settings, tmp_path):
    from quote_followup import artifacts

    model = FollowUpModel(model_settings).fit(quotes, AS_OF)
    model_path, summary_path = artifacts.save(model, AS_OF, tmp_path)
    loaded, fitted_at = artifacts.load(model_path)
    candidates = open_quotes(quotes, AS_OF)
    pd.testing.assert_frame_equal(model.assess(candidates), loaded.assess(candidates))
    assert fitted_at == AS_OF and summary_path.exists()
