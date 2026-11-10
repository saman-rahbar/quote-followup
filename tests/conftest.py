"""A small synthetic quote book, written as CSVs so tests run the real loader without the real data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quote_followup.config import ModelSettings, load_settings
from quote_followup.data import load_quotes

ROOT = Path(__file__).resolve().parents[1]


def write_tables(folder: Path, n_quotes: int = 1500, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    sps = [f"SP{i:02d}" for i in range(6)]
    providers = pd.DataFrame({"sp_id": sps, "synthetic_business_name": [f"Biz {s}" for s in sps],
                              "industry": "cleaning"})
    customers = pd.DataFrame({
        "customer_id": [f"CU{i:04d}" for i in range(300)],
        "sp_id": rng.choice(sps, 300),
        "customer_type": "residential",
        "acquisition_channel": rng.choice(["referral", "marketplace", "organic"], 300),
    })

    owner = customers.sample(n_quotes, replace=True, random_state=seed).reset_index(drop=True)
    created = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.uniform(0, 330, n_quotes), unit="D")
    sent = created + pd.to_timedelta(rng.exponential(24, n_quotes), unit="h")
    workflow = rng.choice(["recurring", "project", "emergency"], n_quotes)
    p_win = np.where(workflow == "project", 0.5, 0.75) * np.where(owner["acquisition_channel"] == "referral", 1.1, 0.9)
    won = rng.uniform(size=n_quotes) < p_win
    declined = ~won & (rng.uniform(size=n_quotes) < 0.5)
    # Wins are answered fast and declines slower, which is what the timing model is meant to pick up.
    delay = np.where(won, rng.exponential(1.5, n_quotes), rng.exponential(4, n_quotes))
    decided = pd.Series(sent + pd.to_timedelta(delay, unit="D")).where(won | declined)
    status = np.select([won, declined], ["accepted", "declined"], "expired")

    quotes = pd.DataFrame({
        "quote_id": [f"QU{i:05d}" for i in range(n_quotes)],
        "sp_id": owner["sp_id"], "customer_id": owner["customer_id"], "property_id": "PR0",
        "workflow_type": workflow, "created_at": created, "sent_at": sent,
        "expires_at": sent + pd.Timedelta(days=30), "decided_at": decided, "status": status,
        "quoted_amount_usd": rng.lognormal(6, 1, n_quotes).round(2), "line_item_count": 2,
        "decline_reason": None,
    })
    folder.mkdir(parents=True, exist_ok=True)
    # The supplied files write timestamps as 2024-01-31T09:06:00, whole seconds, and the loader
    # accepts nothing else, so the fixture does the same.
    for column in ("created_at", "sent_at", "expires_at", "decided_at"):
        quotes[column] = pd.to_datetime(quotes[column]).dt.floor("s")
    quotes.to_csv(folder / "quotes.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S")
    customers.to_csv(folder / "customers.csv", index=False)
    providers.to_csv(folder / "service_providers.csv", index=False)
    return folder


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    return write_tables(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session")
def quotes(data_dir) -> pd.DataFrame:
    return load_quotes(data_dir)


@pytest.fixture(scope="session")
def model_settings() -> ModelSettings:
    return load_settings(ROOT / "config.toml").model
