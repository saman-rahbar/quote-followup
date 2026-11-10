"""The exploration behind the project: which problem to pick, and what the quotes look like.

Every table in notebooks 01 to 03 comes from a function here, so the notebooks only call and show.

    python scripts/explore.py        # the scan that picked the problem
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency
from sklearn.linear_model import LogisticRegression

from quote_followup import audit
from quote_followup.board import CHANNEL, WORK
from quote_followup.data import read_table
from quote_followup.timeline import open_quotes

ROOT = Path(__file__).resolve().parents[1]


def _plain(frame: pd.DataFrame) -> pd.DataFrame:
    """Swap data codes like `repeat_offline` for the words people use."""
    names = WORK | CHANNEL
    return frame.rename(index=lambda v: names.get(v, v), columns=lambda v: names.get(v, v))


def _spread(name: str, frame: pd.DataFrame, outcome: str, drivers: list[str]) -> list[dict]:
    rows = []
    for driver in drivers:
        rates = frame.groupby(driver, observed=True)[outcome].mean()
        rows.append(
            {
                "outcome": name,
                "overall": frame[outcome].mean(),
                "split by": driver,
                "lowest group": rates.min(),
                "highest group": rates.max(),
            }
        )
    return rows


def outcome_scan(data_dir: Path) -> pd.DataFrame:
    """Four candidate problems: does each outcome change with things known in advance?"""
    providers = read_table(
        data_dir / "service_providers.csv", ["sp_id", "industry", "plan_tier", "subscription_end_at"]
    )
    customers = read_table(data_dir / "customers.csv", ["customer_id", "customer_type", "acquisition_channel"])
    jobs = read_table(data_dir / "jobs.csv", ["job_id", "workflow_type"])

    invoices = read_table(
        data_dir / "invoices.csv",
        ["invoice_id", "job_id", "sp_id", "customer_id", "sent_at", "due_at", "status", "invoice_amount_usd"],
        timestamps=("sent_at", "due_at"),
        numbers={"invoice_amount_usd": float},
    )
    invoices = (
        invoices[invoices["sent_at"].notna() & invoices["status"].ne("void")]
        .merge(customers, on="customer_id")
        .merge(jobs, on="job_id")
        .merge(providers, on="sp_id")
    )
    invoices = invoices.assign(
        never_paid=invoices["status"].isin(["overdue", "sent"]),
        amount=pd.qcut(invoices["invoice_amount_usd"], 5),
        payment_terms=(invoices["due_at"] - invoices["sent_at"]).dt.days,
    )

    visits = read_table(data_dir / "visits.csv", ["visit_id", "job_id", "sp_id", "status", "reschedule_count"])
    visits = visits[visits["status"] != "scheduled"].merge(jobs, on="job_id").merge(providers, on="sp_id")
    visits = visits.assign(cancelled=visits["status"].eq("cancelled"))

    quotes = read_table(
        data_dir / "quotes.csv",
        ["quote_id", "customer_id", "workflow_type", "created_at", "sent_at", "status"],
        timestamps=("created_at", "sent_at"),
    )
    quotes = quotes[quotes["status"] != "open"].merge(customers, on="customer_id")
    hours = (quotes["sent_at"] - quotes["created_at"]).dt.total_seconds() / 3600
    quotes = quotes.assign(
        accepted=quotes["status"].eq("accepted"),
        sending_speed=pd.cut(hours, [-1, 4, 24, 48, 1e4], labels=["<4h", "4-24h", "1-2d", ">2d"]),
    )

    rows = (
        _spread(
            "invoice never paid",
            invoices,
            "never_paid",
            [
                "customer_type",
                "acquisition_channel",
                "workflow_type",
                "industry",
                "plan_tier",
                "amount",
                "payment_terms",
            ],
        )
        + _spread("visit cancelled", visits, "cancelled", ["workflow_type", "industry", "reschedule_count"])
        + _spread("quote accepted", quotes, "accepted", ["workflow_type", "acquisition_channel", "sending_speed"])
    )
    left = providers["subscription_end_at"].notna()
    rows.append(
        {"outcome": "business left Jobber", "overall": left.mean(), "split by": f"{left.sum()} of {len(providers)}"}
    )
    return pd.DataFrame(rows)


def data_facts(data_dir: Path, quotes: pd.DataFrame) -> pd.Series:
    """Facts about the extract that shaped the design."""
    raw = audit.read_raw(data_dir)
    latest = max(pd.to_datetime(raw[t][c]).max() for t, c in [("payments", "paid_at"), ("product_events", "event_at")])
    stale = quotes[quotes["status"] == "open"]
    return pd.Series(
        {
            "latest payment or product event": f"{latest:%d %b %Y}",
            "latest expiry date of any quote": f"{quotes['expires_at'].max():%d %b %Y}",
            "quotes still marked open": len(stale),
            "  of those, already past their expiry": f"{(stale['expires_at'] < latest).mean():.0%}",
            "replies more than 45 days after sending": int((quotes["days_to_decision"] > 45).sum()),
            "replies in total": int(quotes["decided_at"].notna().sum()),
        }
    )


def quote_value_check(data_dir: Path, quotes: pd.DataFrame) -> pd.DataFrame:
    """Is the quote amount a fair measure of a win? Compare it with the job's planned and invoiced value."""
    jobs = read_table(
        data_dir / "jobs.csv", ["job_id", "quote_id", "planned_revenue_usd"], numbers={"planned_revenue_usd": float}
    )
    invoices = read_table(
        data_dir / "invoices.csv", ["job_id", "status", "invoice_amount_usd"], numbers={"invoice_amount_usd": float}
    )
    invoiced = invoices[invoices["status"] != "void"].groupby("job_id")["invoice_amount_usd"].sum()
    won = quotes.merge(jobs, on="quote_id").join(invoiced.rename("invoiced"), on="job_id")
    ratios = pd.DataFrame(
        {
            "planned value / quote": won["planned_revenue_usd"] / won["quoted_amount_usd"],
            "invoiced value / quote": won["invoiced"] / won["quoted_amount_usd"],
            "work": won["workflow_type"],
        }
    )
    return _plain(ratios.groupby("work").median().round(3))


def quote_endings(quotes: pd.DataFrame) -> pd.DataFrame:
    ends = quotes.groupby("status").agg(
        quotes=("quote_id", "size"),
        median_amount=("quoted_amount_usd", "median"),
        total_amount=("quoted_amount_usd", "sum"),
    )
    return ends.assign(share=ends["quotes"] / ends["quotes"].sum()).round(3)


def reply_times(quotes: pd.DataFrame) -> pd.DataFrame:
    """Days from sending to the reply, for quotes that got one."""
    answered = quotes[quotes["decided_at"].notna()]
    return answered.groupby("status")["days_to_decision"].describe(percentiles=[0.25, 0.5, 0.75, 0.9]).round(2)


def waiting_per_morning(quotes: pd.DataFrame, start: str, end: str, hour: int = 8) -> pd.Series:
    """How many quotes each business had waiting for a reply, every morning between two dates."""
    mornings = pd.date_range(start, end, freq="D") + pd.Timedelta(hours=hour)
    return pd.concat([open_quotes(quotes, m).groupby("sp_id").size() for m in mornings])


def business_spread(quotes: pd.DataFrame) -> dict[str, float]:
    """Do businesses win more or less than their mix of work and customers explains?

    A ratio near 1 means they scatter about as much as chance alone would make them.
    """
    known = quotes[quotes["won"].notna()]
    expected = known.groupby(known["workflow_type"] + "|" + known["customer_channel"])["won"].transform("mean")
    per = (
        pd.DataFrame(
            {"won": known["won"], "expected": expected, "var": expected * (1 - expected), "sp": known["sp_id"]}
        )
        .groupby("sp")
        .sum()
    )
    statistic = ((per["won"] - per["expected"]) ** 2 / per["var"]).sum()
    dof = len(per) - 1
    return {
        "scatter compared with chance": round(float(statistic / dof), 2),
        "p-value": round(float(chi2.sf(statistic, dof)), 3),
    }


def speed_checks(quotes: pd.DataFrame) -> pd.Series:
    """Is "sent sooner, accepted more" just bigger quotes in disguise? Two checks."""
    known = quotes[quotes["won"].notna()]
    log_amount = np.log(known["quoted_amount_usd"])

    def speed_effect(inputs: pd.DataFrame) -> float:
        fit = LogisticRegression(C=1e6, max_iter=5000).fit(inputs, known["won"])
        return round(fit.coef_[0][list(inputs.columns).index("log_hours_to_send")], 3)

    base = pd.get_dummies(known[["workflow_type", "customer_channel"]], drop_first=True).astype(float)
    base = base.assign(log_hours_to_send=known["log_hours_to_send"])
    link = {
        f"link between sending time and quote size, {WORK[w]}": round(
            np.corrcoef(g["log_hours_to_send"], log_amount[g.index])[0, 1], 3
        )
        for w, g in known.groupby("workflow_type")
    }
    return pd.Series(
        link
        | {
            "effect of sending time, allowing for work and customer source": speed_effect(base),
            "  ... and also for quote size and number of items": speed_effect(
                base.assign(log_amount=log_amount, line_items=known["line_item_count"])
            ),
        }
    )


def speed_by_size(quotes: pd.DataFrame) -> pd.DataFrame:
    """Share accepted, by sending speed, within each type of work and quote size."""
    known = quotes[quotes["won"].notna()]
    size = pd.qcut(np.log(known["quoted_amount_usd"]), 4, labels=["smallest", "small", "large", "largest"])
    table = (
        known.assign(size=size)
        .pivot_table(index=["workflow_type", "size"], columns="send_speed", values="won", observed=True)
        .dropna()
    )
    return _plain(table.round(3))


def still_quiet(quotes: pd.DataFrame, days: tuple[int, ...] = (0, 3, 7, 14)) -> pd.Series:
    """Of the quotes still without a reply after N days, the share that end up a yes."""
    known = quotes[quotes["won"].notna()]
    waited = known["days_to_decision"].fillna(np.inf)
    return pd.Series({f"no reply after {d} days": round(known.loc[waited > d, "won"].mean(), 3) for d in days})


def loss_mix(quotes: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Of lost quotes, the share that ended in silence, by customer source; and how likely that is luck."""
    lost = quotes[quotes["won"] == 0]
    share = lost.assign(silent=lost["status"].eq("expired")).pivot_table(
        index="customer_channel", columns="workflow_type", values="silent"
    )
    luck = pd.Series(
        {
            WORK[w]: chi2_contingency(pd.crosstab(g["customer_channel"], g["status"]))[1]
            for w, g in lost.groupby("workflow_type")
        },
        name="chance this pattern is luck (p)",
    )
    return _plain(share.round(3)), luck


def days_to_yes(quotes: pd.DataFrame) -> pd.DataFrame:
    """Median days until a yes, by customer source and type of work."""
    accepted = quotes[quotes["won"] == 1]
    return _plain(
        accepted.pivot_table(
            index="customer_channel", columns="workflow_type", values="days_to_decision", aggfunc="median"
        ).round(2)
    )


def main() -> None:
    print(outcome_scan(ROOT / "data").to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()


def other_patterns(data_dir: Path, quotes: pd.DataFrame) -> pd.DataFrame:
    """Leads for future work, noticed while exploring but not followed up."""
    jobs = read_table(
        data_dir / "jobs.csv",
        ["job_id", "quote_id", "status", "planned_revenue_usd"],
        numbers={"planned_revenue_usd": float},
    )
    invoices = read_table(
        data_dir / "invoices.csv",
        ["invoice_id", "job_id", "status", "balance_usd"],
        numbers={"balance_usd": float},
    )
    payments = read_table(
        data_dir / "payments.csv", ["invoice_id", "payment_status", "payment_method", "is_jobber_payment"]
    )
    visits = read_table(data_dir / "visits.csv", ["status", "reschedule_count"], numbers={"reschedule_count": int})
    providers = read_table(
        data_dir / "service_providers.csv", ["sp_id", "subscription_end_at"], timestamps=("subscription_end_at",)
    )
    events = read_table(data_dir / "product_events.csv", ["sp_id", "event_at"], timestamps=("event_at",))

    accepted = quotes[quotes["status"] == "accepted"]
    no_job = accepted[~accepted["quote_id"].isin(jobs["quote_id"])]
    done = jobs[jobs["status"] == "completed"]
    unbilled = done[~done["job_id"].isin(invoices["job_id"])]
    overdue = invoices[invoices["status"] == "overdue"]
    jpay = payments[payments["is_jobber_payment"] == "True"]
    fail_rate = jpay.groupby("payment_method")["payment_status"].apply(lambda s: s.eq("failed").mean())
    failed = payments[payments["payment_status"] == "failed"]
    settled = failed["invoice_id"].isin(payments.loc[payments["payment_status"] == "succeeded", "invoice_id"])
    finished = visits[visits["status"] != "scheduled"]
    cancelled = finished.groupby(finished["reschedule_count"].clip(upper=3))["status"].apply(
        lambda s: s.eq("cancelled").mean()
    )
    sent = quotes.groupby(quotes["sent_at"].dt.to_period("M")).size()
    left = events.merge(providers.dropna(), on="sp_id")
    after = left["event_at"] > left["subscription_end_at"]
    declined = quotes[quotes["status"] == "declined"]
    reasons = declined["decline_reason"].value_counts(normalize=True)
    unpaid = (~overdue["invoice_id"].isin(payments["invoice_id"])).sum()

    rows = [
        (
            "accepted quotes that never became a job",
            f"{len(no_job):,} (${no_job['quoted_amount_usd'].sum() / 1e6:.1f}M)",
        ),
        (
            "completed jobs never invoiced",
            f"{len(unbilled):,} (${unbilled['planned_revenue_usd'].sum() / 1e6:.1f}M planned)",
        ),
        (
            "overdue invoices with no payment at all",
            f"{unpaid:,} of {len(overdue):,} (${overdue['balance_usd'].sum() / 1e6:.1f}M)",
        ),
        ("failed payments that went through Jobber Payments", f"{(failed['is_jobber_payment'] == 'True').mean():.0%}"),
        (
            "Jobber Payments failure rate, bank transfer vs card",
            f"{fail_rate['bank_transfer']:.0%} vs {fail_rate['card']:.0%}",
        ),
        ("failed payments whose invoice was paid later", f"{settled.mean():.0%}"),
        (
            "visits cancelled: overall, after 3+ reschedules",
            f"{finished['status'].eq('cancelled').mean():.0%}, {cancelled[3]:.0%}",
        ),
        ("quotes sent: January vs November 2024 (December is partial)", f"{sent.iloc[0]:,} vs {sent.iloc[-2]:,}"),
        (
            "cancelled businesses still active afterwards",
            f"{left.loc[after, 'sp_id'].nunique()} of {providers['subscription_end_at'].notna().sum()}",
        ),
        (
            "declines by reason (each), and with no reason",
            f"{reasons.min():.0%} to {reasons.max():.0%}, {declined['decline_reason'].isna().mean():.0%}",
        ),
    ]
    return pd.DataFrame(rows, columns=["pattern", "number"]).set_index("pattern")


def preprocessing_steps(quotes: pd.DataFrame, as_of: str = "2024-09-01", horizon_days: int = 45) -> pd.DataFrame:
    """From raw rows to what the model learns from and scores, with the row count after each step."""
    from quote_followup.timeline import settled_quotes

    known = quotes[quotes["won"].notna()]
    settled = settled_quotes(quotes, pd.Timestamp(as_of), horizon_days)
    steps = [
        ("read quotes.csv as text, parse dates and amounts strictly", len(quotes)),
        ("join customer and business on (sp_id, customer_id); all match", len(quotes)),
        ("derive won, hours_to_send, log(1 + hours), send_speed, days_to_decision", len(quotes)),
        ("drop the 1,444 stale 'open' quotes from learning and scoring", len(known)),
        (f"learning set on {as_of}: sent {horizon_days}+ days earlier, ending on record", len(settled)),
        ("in the model: one-hot encode work and source, standardise log hours", len(settled)),
    ]
    return pd.DataFrame(steps, columns=["step", "rows"]).set_index("step")
