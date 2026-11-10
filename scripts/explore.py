"""Why quotes: the scan that picked the problem.

An outcome is worth predicting only if it moves with something known in advance. For each
candidate outcome this reports its rate across the attributes a model would use.

    python scripts/explore.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quote_followup.data import read_table

ROOT = Path(__file__).resolve().parents[1]


def _spread(name: str, frame: pd.DataFrame, outcome: str, drivers: list[str]) -> list[dict]:
    rows = []
    for driver in drivers:
        rates = frame.groupby(driver, observed=True)[outcome].mean()
        rows.append({"outcome": name, "overall": frame[outcome].mean(), "driver": driver,
                     "lowest": rates.min(), "highest": rates.max(), "groups": len(rates)})
    return rows


def outcome_scan(data_dir: Path) -> pd.DataFrame:
    providers = read_table(data_dir / "service_providers.csv", ["sp_id", "industry", "plan_tier", "subscription_end_at"])
    customers = read_table(data_dir / "customers.csv", ["customer_id", "customer_type", "acquisition_channel"])
    jobs = read_table(data_dir / "jobs.csv", ["job_id", "workflow_type"])

    invoices = read_table(data_dir / "invoices.csv", ["invoice_id", "job_id", "sp_id", "customer_id", "sent_at",
                                                      "due_at", "status", "invoice_amount_usd"],
                          timestamps=("sent_at", "due_at"), numbers={"invoice_amount_usd": float})
    invoices = (invoices[invoices["sent_at"].notna() & invoices["status"].ne("void")]
                .merge(customers, on="customer_id").merge(jobs, on="job_id").merge(providers, on="sp_id"))
    invoices = invoices.assign(
        never_paid=invoices["status"].isin(["overdue", "sent"]),
        amount_quintile=pd.qcut(invoices["invoice_amount_usd"], 5),
        terms_days=(invoices["due_at"] - invoices["sent_at"]).dt.days,
    )

    visits = read_table(data_dir / "visits.csv", ["visit_id", "job_id", "sp_id", "status", "reschedule_count"])
    visits = visits[visits["status"] != "scheduled"].merge(jobs, on="job_id").merge(providers, on="sp_id")
    visits = visits.assign(cancelled=visits["status"].eq("cancelled"))

    quotes = read_table(data_dir / "quotes.csv", ["quote_id", "customer_id", "workflow_type", "created_at",
                                                  "sent_at", "status"], timestamps=("created_at", "sent_at"))
    quotes = quotes[quotes["status"] != "open"].merge(customers, on="customer_id")
    hours = (quotes["sent_at"] - quotes["created_at"]).dt.total_seconds() / 3600
    quotes = quotes.assign(accepted=quotes["status"].eq("accepted"),
                           hours_to_send=pd.cut(hours, [-1, 4, 24, 48, 1e4], labels=["<4h", "4-24h", "1-2d", ">2d"]))

    rows = (_spread("invoice never paid", invoices, "never_paid",
                    ["customer_type", "acquisition_channel", "workflow_type", "industry", "plan_tier",
                     "amount_quintile", "terms_days"])
            + _spread("visit cancelled", visits, "cancelled", ["workflow_type", "industry", "reschedule_count"])
            + _spread("quote accepted", quotes, "accepted", ["workflow_type", "acquisition_channel", "hours_to_send"]))
    churned = providers["subscription_end_at"].notna()
    rows.append({"outcome": "SP churned", "overall": churned.mean(), "driver": f"{churned.sum()} of {len(providers)} SPs",
                 "lowest": None, "highest": None, "groups": None})
    return pd.DataFrame(rows)


def main() -> None:
    table = outcome_scan(ROOT / "data")
    print(table.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
