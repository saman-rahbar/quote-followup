import numpy as np
import pandas as pd
import pytest

from quote_followup import backtest
from quote_followup.data import DataContractError, load_quotes

from .conftest import write_tables


def test_a_customer_filed_under_another_business_stops_the_load(tmp_path):
    folder = write_tables(tmp_path, n_quotes=50)
    quotes = pd.read_csv(folder / "quotes.csv")
    customers = pd.read_csv(folder / "customers.csv")
    stolen = customers.loc[customers["sp_id"] != quotes.loc[0, "sp_id"], "customer_id"].iloc[0]
    quotes.loc[0, "customer_id"] = stolen
    quotes.to_csv(folder / "quotes.csv", index=False)

    with pytest.raises(DataContractError, match="different SP"):
        load_quotes(folder)


def _morning() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sp_id": ["A"] * 4 + ["B"] * 2,
            "quote_id": list("abcdef"),
            "won": [1, 0, 1, 0, 0, 1],
            "quoted_amount_usd": [100.0, 900.0, 300.0, 50.0, 10.0, 20.0],
        }
    )


def test_top_k_scores_the_picks_against_the_best_possible_picks():
    morning = _morning()
    largest_first = morning["quoted_amount_usd"].to_numpy()
    out = backtest._top_k(morning, largest_first, k=1)
    # A: picks the $900 loss while the best pick was the $300 win. B: picks the $20 win.
    assert out.loc["A", ["hit_rate", "won_value", "best_value"]].tolist() == [0.0, 0.0, 300.0]
    assert out.loc["B", ["hit_rate", "won_value", "best_value"]].tolist() == [1.0, 20.0, 20.0]


def test_a_business_with_no_more_quotes_than_capacity_is_skipped():
    out = backtest._top_k(_morning(), np.zeros(6), k=2)
    assert list(out.index) == ["A"]


def test_summary_gaps_are_zero_for_the_reference_and_paired_for_the_rest():
    morning = _morning()
    lists = pd.concat(
        [
            backtest._top_k(morning, morning["won"] * morning["quoted_amount_usd"], 1).assign(scorer="oracle", k=1),
            backtest._top_k(morning, morning["quoted_amount_usd"], 1).assign(scorer="largest", k=1),
        ]
    ).reset_index()
    summary = backtest.summarise(lists, reference="oracle", draws=200, seed=0).set_index("scorer")
    assert summary.loc["oracle", "dollar_capture"] == pytest.approx(1.0)
    assert summary.loc["oracle", ["dollar_capture_gap_lo", "dollar_capture_gap_hi"]].tolist() == [0.0, 0.0]
    assert summary.loc["largest", "dollar_capture"] == pytest.approx(20 / 320)


def test_a_malformed_timestamp_stops_the_load_instead_of_becoming_text(tmp_path):
    folder = write_tables(tmp_path, n_quotes=50)
    quotes = pd.read_csv(folder / "quotes.csv", dtype=str)
    quotes.loc[3, "sent_at"] = "2024-13-45 25:00"
    quotes.to_csv(folder / "quotes.csv", index=False)
    with pytest.raises(ValueError):
        load_quotes(folder)


def test_the_audit_parser_flags_values_that_do_not_match_their_documented_type():
    from quote_followup.audit import _parses

    assert _parses(pd.Series(["2024-01-02T03:04:05", "2024-01-02"]), "timestamp").tolist() == [True, False]
    assert _parses(pd.Series(["True", "yes"]), "boolean").tolist() == [True, False]
    assert _parses(pd.Series(["3", "3.5"]), "integer").tolist() == [True, False]


def test_the_audit_catches_a_broken_quote_rule(quotes):
    from quote_followup.audit import quote_rules

    broken = quotes.copy()
    broken.loc[broken.index[0], "decided_at"] = broken.loc[broken.index[0], "sent_at"] - pd.Timedelta(days=1)
    failed = {detail.split(":")[0] for _, _, detail, ok in quote_rules(broken) if not ok}
    assert "decided after sent" in failed


def test_identical_orderings_have_a_zero_gap():
    morning = _morning()
    one = backtest._top_k(morning, morning["quoted_amount_usd"], 1)
    lists = pd.concat([one.assign(scorer="a", k=1), one.assign(scorer="b", k=1)]).reset_index()
    gap = backtest.paired_gap(lists, "a", "b", "dollar_capture", 1, draws=100, seed=0)
    assert (gap["gap"], gap["lo"], gap["hi"]) == (0.0, 0.0, 0.0)


def test_subgroup_check_reports_no_gap_when_predictions_equal_the_observed_rate():
    rng = np.random.default_rng(0)
    n = 4000
    frame = pd.DataFrame(
        {
            "quote_id": [f"Q{i}" for i in range(n)],
            "workflow_type": "recurring",
            "customer_channel": rng.choice(["referral", "organic"], n),
            "send_speed": "within a day",
            "age_days": rng.uniform(0, 30, n),
            "won": rng.integers(0, 2, n).astype(float),
        }
    )
    bands = pd.cut(frame["age_days"], backtest.AGE_BANDS, include_lowest=True)
    frame["p_now"] = frame.groupby(["customer_channel", bands], observed=True)["won"].transform("mean")
    cells = backtest.subgroup_calibration(frame)
    channel_cells = cells[cells["split"] == "customer_channel"]
    assert channel_cells["gap"].abs().max() == pytest.approx(0.0, abs=1e-12)
