"""The morning follow-up board: the list an owner actually reads."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit

from .scorers import SILENCE, FollowUpModel
from .timeline import open_quotes

# A push smaller than this barely moves the chance (two or three points), so it is not worth a mention.
_MIN_REASON = 0.10

WORK = {"recurring": "recurring service", "project": "one-off project", "emergency": "emergency job"}
CHANNEL = {
    "referral": "referred by someone",
    "repeat_offline": "repeat customer",
    "organic": "found you on their own",
    "paid_search": "came from a paid ad",
    "marketplace": "came from a marketplace",
}


def _phrase(feature: str, row: pd.Series) -> str:
    match feature:
        case "workflow_type":
            return WORK.get(row["workflow_type"], row["workflow_type"])
        case "customer_channel":
            return CHANNEL.get(row["customer_channel"], row["customer_channel"].replace("_", " "))
        case "log_hours_to_send":
            hours = row["hours_to_send"]
            return f"sent within {np.ceil(hours):.0f} hours" if hours < 24 else f"took {hours / 24:.1f} days to send"
        case _ if feature == SILENCE:
            return f"no reply for {row['age_days']:.1f} days"
        case _:
            return feature.replace("_", " ")


def _reasons(row: pd.Series, features: list[str]) -> str:
    """The biggest things pushing this quote's chance up or down, in words."""
    pushes = sorted(((row[f"why_{f}"], f) for f in features), key=lambda x: -abs(x[0]))
    pushes = [(v, f) for v, f in pushes[:3] if abs(v) >= _MIN_REASON]
    helps = [_phrase(f, row) for v, f in pushes if v > 0]
    hurts = [_phrase(f, row) for v, f in pushes if v < 0]
    parts = ([f"helps: {', '.join(helps)}"] if helps else []) + ([f"hurts: {', '.join(hurts)}"] if hurts else [])
    return "; ".join(parts) or "a typical quote"


def followup_board(
    model: FollowUpModel, quotes: pd.DataFrame, sp_id: str, as_of: pd.Timestamp, capacity: int
) -> pd.DataFrame:
    """Every open quote for one business at `as_of`, best bet first, with the top `capacity` flagged."""
    candidates = open_quotes(quotes, as_of, sp_id)
    board = candidates.join(model.assess(candidates)).sort_values(
        ["expected_value", "quote_id"], ascending=[False, True]
    )
    features = [*model.propensity.features, SILENCE]
    return board.assign(
        chase_today=np.arange(len(board)) < capacity,
        why=board.apply(_reasons, axis=1, features=features),
        days_to_expiry=(board["expires_at"] - as_of) / pd.Timedelta(days=1),
    )


BOARD_COLUMNS = {
    "quote_id": ("quote", str),
    "workflow_type": ("work", lambda w: WORK.get(w, w)),
    "quoted_amount_usd": ("amount", "${:,.0f}".format),
    "age_days": ("days waiting", "{:.1f}".format),
    "p_now": ("chance of yes", "{:.0%}".format),
    "expected_value": ("worth today", "${:,.0f}".format),
    "days_until_half": ("chance halves in", lambda d: "after expiry" if np.isinf(d) else f"{d:.1f} days"),
    "why": ("why", str),
}


def _table(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "  (none)"
    cells = [[header for header, _ in BOARD_COLUMNS.values()]]
    cells += [[fmt(row[col]) for col, (_, fmt) in BOARD_COLUMNS.items()] for _, row in rows.iterrows()]
    widths = [max(len(line[i]) for line in cells) for i in range(len(BOARD_COLUMNS))]
    return "\n".join(
        "  " + "  ".join(text.ljust(w) for text, w in zip(line, widths, strict=True)).rstrip() for line in cells
    )


def render(board: pd.DataFrame, capacity: int) -> str:
    """A plain-text table, split into today's calls and the rest."""
    chase, rest = board[board["chase_today"]], board[~board["chase_today"]]
    return f"Call today (top {capacity}):\n{_table(chase)}\n\nCan wait:\n{_table(rest)}"


def explain(model: FollowUpModel, row: pd.Series) -> pd.Series:
    """Where one quote's number comes from: a typical quote's score plus one push per fact.

    The pushes are on the log-odds scale, which is how the evidence adds up. Their total turns back
    into the chance on the board, exactly.
    """
    labels = {
        "workflow_type": f"type of work: {WORK.get(row['workflow_type'], row['workflow_type'])}",
        "customer_channel": f"customer source: {CHANNEL.get(row['customer_channel'], row['customer_channel'])}",
        "log_hours_to_send": f"quote sent after {row['hours_to_send']:.1f} hours",
    }
    pieces = {"a typical quote starts at": model.propensity.baseline_log_odds()}
    pieces |= {labels.get(f, f): row[f"why_{f}"] for f in model.propensity.features}
    pieces[f"no reply for {row['age_days']:.1f} days"] = row[f"why_{SILENCE}"]
    pieces["total"] = sum(pieces.values())
    pieces["chance of a yes (from the total)"] = float(1 / (1 + np.exp(-pieces["total"])))
    assert abs(pieces["total"] - logit(row["p_now"])) < 1e-9, "the pieces must add up to the board's number"
    return pd.Series(pieces, name="adds to the score").round(4)


def with_outcomes(board: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """The board next to what really happened to each quote, which the model could not see."""
    ended = quotes.set_index("quote_id").loc[board["quote_id"], ["status", "decided_at"]]
    shown = board.set_index("quote_id")[["quoted_amount_usd", "age_days", "p_now", "expected_value"]].round(3)
    return shown.join(ended).rename(
        columns={
            "quoted_amount_usd": "amount",
            "age_days": "days waiting",
            "p_now": "chance of yes",
            "expected_value": "worth today",
            "status": "how it ended",
            "decided_at": "reply came",
        }
    )
