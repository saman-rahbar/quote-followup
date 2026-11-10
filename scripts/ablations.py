"""The experiments that decided how the model is built (notebook 04 walks through them).

All of this uses data from before 1 Sep 2024, the day the final test starts, so the test stays a
fair one. It works like the final test does: on the first of April, May, June and July the model
learns from what was known then, and is scored on the quotes sent that month. Quotes sent after
17 Jul are left out, because some of their answers weren't in yet by 1 Sep.

    python scripts/ablations.py
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from quote_followup.backtest import subgroup_calibration, subgroup_summary
from quote_followup.config import load_settings
from quote_followup.data import load_quotes
from quote_followup.timeline import settled_quotes
from quote_followup.timing import DecisionTiming

ROOT = Path(__file__).resolve().parents[1]
BACKTEST_START = pd.Timestamp("2024-09-01")
ORIGINS = pd.to_datetime(["2024-04-01", "2024-05-01", "2024-06-01", "2024-07-01"])
LANDMARKS = (0, 1, 2, 3, 5, 7, 10, 14, 21)
CAT, NUM = ["workflow_type", "customer_channel"], ["log_hours_to_send"]


def with_history(quotes: pd.DataFrame, key: str, strength: float) -> pd.Series:
    """How often this business (or customer) won before, using only answers already in, pulled
    towards the overall rate when there is little history to go on."""
    done = quotes[quotes["won"].notna()].sort_values("resolved_at")
    done = done.assign(wins=done.groupby(key)["won"].cumsum(), n=done.groupby(key).cumcount() + 1)
    asof = pd.merge_asof(
        quotes[["quote_id", key, "sent_at"]].sort_values("sent_at"),
        done[[key, "resolved_at", "wins", "n"]],
        left_on="sent_at",
        right_on="resolved_at",
        by=key,
        allow_exact_matches=False,
    ).set_index("quote_id")
    prior = quotes.loc[quotes["sent_at"] < ORIGINS[0], "won"].mean()
    rate = (asof["wins"].fillna(0) + strength * prior) / (asof["n"].fillna(0) + strength)
    return quotes["quote_id"].map(rate)


def linear(cat: list[str], num: list[str]) -> Pipeline:
    return Pipeline(
        [
            (
                "enc",
                ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat), ("n", StandardScaler(), num)]),
            ),
            ("lr", LogisticRegression(max_iter=2000)),
        ]
    )


def boosted(cat: list[str], num: list[str]) -> Pipeline:
    return Pipeline(
        [
            (
                "enc",
                ColumnTransformer(
                    [("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat), ("n", "passthrough", num)]
                ),
            ),
            ("gb", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)),
        ]
    )


def landmark(frame: pd.DataFrame) -> pd.DataFrame:
    days = frame["days_to_decision"].fillna(np.inf)
    return pd.concat([frame[days > a].assign(age_days=float(a)) for a in LANDMARKS], ignore_index=True)


Predict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]  # (train, rows to score) -> probabilities


def rolling(quotes: pd.DataFrame, settings, predict: Predict, at_landmarks: bool) -> pd.DataFrame:
    """Predictions for every test month, each made only with data from before that month."""
    cutoff = BACKTEST_START - pd.Timedelta(days=settings.model.label_horizon_days)
    folds = []
    for start, end in zip(ORIGINS, [*ORIGINS[1:], cutoff], strict=True):
        train = settled_quotes(quotes, start, settings.model.label_horizon_days)
        fold = quotes[(quotes["sent_at"] >= start) & (quotes["sent_at"] < min(end, cutoff)) & quotes["won"].notna()]
        rows = landmark(fold) if at_landmarks else fold
        folds.append(rows.assign(p=predict(train, rows)))
    return pd.concat(folds, ignore_index=True)


def summary(name: str, scored: pd.DataFrame, subgroups: bool = False) -> dict:
    y, p = scored["won"].astype(int), scored["p"]
    row = {"option": name, "rows": len(scored), "auc": roc_auc_score(y, p), "log_loss": log_loss(y, p)}
    if subgroups:
        cells = subgroup_calibration(scored.rename(columns={"p": "p_now"}))
        found = subgroup_summary(cells)
        row |= {
            "groups_missed_beyond_luck": found["subgroup_cells_beyond"],
            "worst_miss_z": found["subgroup_max_abs_z"],
            "luck_threshold_z": found["subgroup_z_threshold"],
        }
    return row


def with_candidate_inputs(quotes: pd.DataFrame) -> pd.DataFrame:
    """Extra facts to try in the model, all of them known when the quote goes out."""
    return quotes.assign(
        log_amount=np.log(quotes["quoted_amount_usd"]),
        sp_history=with_history(quotes, "sp_id", 10),
        customer_history=with_history(quotes, "customer_id", 3),
        weekend=(quotes["sent_at"].dt.day_of_week >= 5).astype(int),
    )


def send_time_ablations(quotes: pd.DataFrame, settings) -> pd.DataFrame:
    """Which facts about a quote earn their place. `quotes` needs `with_candidate_inputs` first."""
    rows = []
    every_cat, every_num = CAT + ["industry", "customer_type"], NUM + ["log_amount", "line_item_count"]
    send_time = {
        "type of work, customer source, sending speed (chosen)": (linear, CAT, NUM),
        "+ industry, customer type, quote size, number of items": (linear, every_cat, every_num),
        "+ the business's past win rate": (linear, CAT, NUM + ["sp_history"]),
        "+ the customer's past win rate": (linear, CAT, NUM + ["customer_history"]),
        "+ sent on a weekend": (linear, CAT, NUM + ["weekend"]),
        "without type of work": (linear, ["customer_channel"], NUM),
        "without customer source": (linear, ["workflow_type"], NUM),
        "without sending speed (quote size instead)": (linear, CAT, ["log_amount"]),
        "gradient boosting with every fact": (boosted, every_cat, every_num),
    }
    for name, (make, cat, num) in send_time.items():

        def predict(train, score, make=make, cols=cat + num, cat=cat, num=num):
            return make(cat, num).fit(train[cols], train["won"].astype(int)).predict_proba(score[cols])[:, 1]

        rows.append(summary(name, rolling(quotes, settings, predict, at_landmarks=False)))
    return pd.DataFrame(rows)


def silence_ablations(quotes: pd.DataFrame, settings) -> pd.DataFrame:
    """Different ways of using "days without a reply", each scored at every age a quote was waiting."""

    def bayes(make, segments, competing=True):
        """Chance at sending, adjusted for silence. `competing` lets "says no vs goes quiet" vary by quote."""

        def predict(train, score):
            at_send = make(CAT, NUM).fit(train[CAT + NUM], train["won"].astype(int))
            lost = train[train["won"] == 0]
            if competing:
                share = make(CAT, NUM).fit(lost[CAT + NUM], lost["status"].eq("declined").astype(int))
                q = share.predict_proba(score[CAT + NUM])[:, 1]
            else:
                q = lost["status"].eq("declined").mean()
            timing = DecisionTiming.fit(train, segments, settings.model.timing_min_quotes)
            return expit(logit(at_send.predict_proba(score[CAT + NUM])[:, 1]) + timing.log_odds_shift(score, q))

        return predict

    def direct(make):
        def predict(train, score):
            cols = CAT + NUM + ["age_days"]
            lm = landmark(train)
            return make(CAT, NUM + ["age_days"]).fit(lm[cols], lm["won"].astype(int)).predict_proba(score[cols])[:, 1]

        return predict

    def age_only(train, score):
        lm = landmark(train)
        return score["age_days"].map(lm.groupby("age_days")["won"].mean()).to_numpy()

    silence = {
        "one says-no-or-goes-quiet figure per type of work": bayes(linear, ("workflow_type",), competing=False),
        "says-no-or-goes-quiet chance per customer (chosen)": bayes(linear, ("workflow_type",)),
        "  same, reply speed also learned by customer source": bayes(linear, ("workflow_type", "customer_channel")),
        "  same, reply speed also learned by sending speed": bayes(linear, ("workflow_type", "send_speed")),
        "  same, with gradient boosting instead": bayes(boosted, ("workflow_type",)),
        "days waiting as one more fact": direct(linear),
        "days waiting as one more fact, gradient boosting": direct(boosted),
        "days waiting only": age_only,
    }
    return pd.DataFrame(
        [
            summary(name, rolling(quotes, settings, predict, at_landmarks=True), subgroups=True)
            for name, predict in silence.items()
        ]
    )


def main() -> None:
    settings = load_settings(ROOT / "config.toml")
    quotes = with_candidate_inputs(load_quotes(settings.data_dir))
    table = pd.concat(
        [
            send_time_ablations(quotes, settings).assign(question="which facts"),
            silence_ablations(quotes, settings).assign(question="using days without a reply"),
        ]
    )
    table.to_csv(ROOT / "outputs" / "ablations.csv", index=False)
    print(table.to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
