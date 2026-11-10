"""What could be known on a given date, and nothing more.

Everything in the project, learning and scoring alike, gets its quotes through one of these two
functions with a date attached. That is what stops the model from peeking at the future, which
would make the test results look better than real life.
"""

from __future__ import annotations

import pandas as pd

from .data import OUTCOME_COLUMNS


def open_quotes(quotes: pd.DataFrame, as_of: pd.Timestamp, sp_id: str | None = None) -> pd.DataFrame:
    """Quotes that, at `as_of`, had gone out, had no reply yet and hadn't expired.

    Anything that records how a quote ended is removed, so the model can't see it even by accident.
    """
    alive = (
        (quotes["sent_at"] <= as_of)
        & (quotes["expires_at"] > as_of)
        & (quotes["decided_at"].isna() | (quotes["decided_at"] > as_of))
    )
    if sp_id is not None:
        alive &= quotes["sp_id"].eq(sp_id)
    view = quotes.loc[alive].drop(columns=list(OUTCOME_COLUMNS))
    return view.assign(age_days=(as_of - view["sent_at"]) / pd.Timedelta(days=1))


def settled_quotes(quotes: pd.DataFrame, as_of: pd.Timestamp, horizon_days: int) -> pd.DataFrame:
    """Past quotes whose ending was known at `as_of`, and so are safe to learn from.

    A quote counts if it went out at least `horizon_days` earlier and its outcome was on record by
    `as_of`. Why the extra wait: among recent quotes, the quick yeses are already in but the slow
    nos aren't, so learning from them would make yeses look more common than they are.
    """
    cutoff = as_of - pd.Timedelta(days=horizon_days)
    settled = quotes["won"].notna() & (quotes["sent_at"] < cutoff) & (quotes["resolved_at"] <= as_of)
    return quotes.loc[settled]
