"""Reads the quotes, along with who the customer is and which business sent them.

This is the only file that opens the CSVs. Everything else works on the one tidy table it returns,
so moving to a different source (a database, an API) means changing this file and nothing else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Columns that only get filled in once the customer has replied (or the quote expired). Listing them
# here lets timeline.open_quotes strip them out before the model sees a quote.
OUTCOME_COLUMNS = ("status", "decided_at", "decline_reason", "won", "resolved_at", "days_to_decision")

_WON = {"accepted": 1.0, "declined": 0.0, "expired": 0.0, "open": np.nan}


class DataContractError(ValueError):
    """Raised when the data breaks a rule the model depends on."""


# Every date in the files is written like 2024-05-15T10:20:00. We insist on exactly that, so a
# badly written date stops the run instead of turning into a blank that nobody notices.
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"


def read_table(
    path: Path, columns: list[str], timestamps: tuple[str, ...] = (), numbers: dict[str, type] | None = None
) -> pd.DataFrame:
    """Read a file as plain text, then convert the named date and number columns, strictly."""
    frame = pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False, na_values=[""])
    for column in timestamps:
        frame[column] = pd.to_datetime(frame[column], format=TIMESTAMP_FORMAT, errors="raise")
    return frame.astype(numbers or {})


def load_quotes(data_dir: Path | str) -> pd.DataFrame:
    data_dir = Path(data_dir)
    quotes = read_table(
        data_dir / "quotes.csv",
        [
            "quote_id",
            "sp_id",
            "customer_id",
            "property_id",
            "workflow_type",
            "created_at",
            "sent_at",
            "expires_at",
            "decided_at",
            "status",
            "quoted_amount_usd",
            "line_item_count",
            "decline_reason",
        ],
        timestamps=("created_at", "sent_at", "expires_at", "decided_at"),
        numbers={"quoted_amount_usd": float, "line_item_count": int},
    )
    customers = read_table(
        data_dir / "customers.csv",
        ["customer_id", "sp_id", "customer_type", "acquisition_channel"],
    ).rename(columns={"acquisition_channel": "customer_channel"})
    providers = read_table(
        data_dir / "service_providers.csv",
        ["sp_id", "synthetic_business_name", "industry"],
    ).rename(columns={"synthetic_business_name": "business_name"})

    # Matching on the business as well as the customer keeps every record with the business that
    # owns it. A customer filed under the wrong business won't match, and the check below stops the run.
    quotes = quotes.merge(
        customers, on=["sp_id", "customer_id"], how="left", validate="many_to_one", indicator="customer_match"
    )
    quotes = quotes.merge(providers, on="sp_id", how="left", validate="many_to_one", indicator="sp_match")
    _check_contract(quotes)
    return _derive(quotes.drop(columns=["customer_match", "sp_match"]))


def _check_contract(quotes: pd.DataFrame) -> None:
    orphans = quotes["customer_match"].eq("left_only").sum()
    if orphans:
        raise DataContractError(f"{orphans} quotes point at a customer owned by a different SP")
    if quotes["sp_match"].eq("left_only").any():
        raise DataContractError("some quotes belong to an SP missing from service_providers")
    unknown = set(quotes["status"]) - set(_WON)
    if unknown:
        raise DataContractError(f"unexpected quote status values: {sorted(unknown)}")
    if (quotes["sent_at"] < quotes["created_at"]).any():
        raise DataContractError("some quotes were sent before they were created")


def _derive(quotes: pd.DataFrame) -> pd.DataFrame:
    day = pd.Timedelta(days=1)
    hours_to_send = (quotes["sent_at"] - quotes["created_at"]) / pd.Timedelta(hours=1)
    expired = quotes["status"].eq("expired")
    return quotes.assign(
        won=quotes["status"].map(_WON),
        hours_to_send=hours_to_send,
        log_hours_to_send=np.log1p(hours_to_send),
        send_speed=np.where(hours_to_send <= 24, "within a day", "after a day"),
        # When the ending became known: the reply date, or the expiry date if nobody replied.
        resolved_at=quotes["decided_at"].fillna(quotes["expires_at"].where(expired)),
        days_to_decision=(quotes["decided_at"] - quotes["sent_at"]) / day,
    ).sort_values("sent_at", ignore_index=True)
