# %% [markdown]
# # 01 · Can we trust the data?
#
# Before analysing anything, I want to be sure of four things, because every result later
# depends on them:
#
# 1. The files I'm using are exactly the ones in the `data.zip` we were given.
# 2. Every value reads correctly: dates as dates, amounts as numbers, nothing lost on the way in.
# 3. Every record points to something real. A quote belongs to a real customer of the same
#    business, a job to a real property, and so on.
# 4. My counts of quotes match the counts the platform keeps separately.
#
# The checks live in one place (`quote_followup.audit`), so this notebook, the command line
# (`quote-followup audit --zip ../data.zip`) and the automated tests all run the same code.

# %%
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "scripts"))

from quote_followup import audit
from quote_followup.config import load_settings
from quote_followup.data import load_quotes, read_table

pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 100)
settings = load_settings(ROOT / "config.toml")
DATA, ZIP, DICTIONARY = settings.data_dir, ROOT.parent / "data.zip", ROOT / "reference" / "DATA_DICTIONARY.csv"

# %% [markdown]
# ## How the tables connect
#
# This is the diagram that came with the data. Each business (SP) has its own customers, and
# everything below them (properties, quotes, jobs, visits, invoices, payments) belongs to that
# same business. When I check a link, I check it within the business, so a quote attached to
# another business's customer counts as broken.

# %%
display(Image(ROOT / "reference" / "ERD.png", width=640))

# %% [markdown]
# ## Running every check
#
# One row per check. `ok` is True when it passes.

# %%
quotes = load_quotes(DATA)
report = audit.run(DATA, DICTIONARY, quotes, ZIP if ZIP.exists() else None)
summary = report.groupby("check", sort=False)["ok"].agg(checks="size", passed="sum")
display(summary)
print(f"{report['ok'].sum()} of {len(report)} checks pass")

# %% [markdown]
# Every check passes. The two groups worth reading in full are the links between tables and the
# cross-checks against the platform's own counts.

# %%
display(report[report["check"].isin(["links", "reconciles"])].reset_index(drop=True))

# %% [markdown]
# The cross-checks are the strongest evidence that I'm reading the quotes correctly.
# `daily_sp_metrics` is a separate summary the platform keeps: how many quotes each business
# created, sent and had accepted, day by day. Counting the same things myself from the quotes
# table gives exactly the same numbers on every single business-day. It also confirms what the
# `decided_at` column means, because the platform counts an acceptance on that date. On top of
# that, every quote has its own "quote created" event in the product log, with the same timestamp.

# %% [markdown]
# ## How the model reads quotes
#
# The loader reads every column as plain text first, then converts dates using the exact format
# in the files (like `2024-05-15T10:20:00`), and stops if anything doesn't fit. The usual
# shortcut in pandas would silently leave a bad date as text, and nothing would complain.

# %%
print(quotes.dtypes.to_string())
display(quotes.head(3).T)

# %% [markdown]
# A few columns are worked out from the quote itself:
#
# | column | what it means |
# |---|---|
# | `won` | 1 if accepted, 0 if declined or expired, blank for the 1,444 still marked "open" |
# | `hours_to_send` | hours between writing the quote and sending it |
# | `send_speed` | "within a day" or "after a day" |
# | `resolved_at` | when the ending became known: the reply date, or the expiry date if nobody replied |
# | `days_to_decision` | days between sending and the reply; blank if the quote expired |

# %% [markdown]
# ## Four facts that shape the rest of the work

# %%
raw = audit.read_raw(DATA)
latest = {f"{t}.{c}": pd.to_datetime(raw[t][c]).max()
          for t in audit.TABLES for c in raw[t].columns if c.endswith("_at") or c == "reporting_date"}
latest = pd.Series(latest).sort_values(ascending=False)
display(latest.head(5).to_frame("latest date in the column"))

open_quotes = quotes[quotes["status"] == "open"]
print(f"quotes still marked open: {len(open_quotes):,}")
print(f"latest expiry date of any quote: {quotes['expires_at'].max():%d %b %Y}")
print(f"open quotes already past their expiry by 19 Mar 2025: {(open_quotes['expires_at'] < '2025-03-19').mean():.0%}")

# %% [markdown]
# **1. The "open" quotes aren't really open.** The data was pulled on or after 19 Mar 2025
# (payments and product events run up to that day). Every quote had expired by 23 Jan 2025. So
# the 1,444 quotes still marked "open" aren't waiting for an answer; they look like a glitch in
# the export. I leave them out, and notebook 05 shows that counting them as lost changes nothing.

# %%
late = quotes["days_to_decision"] > 45
print(f"replies that came more than 45 days after sending: {late.sum()} of {quotes['decided_at'].notna().sum():,}")

jobs = read_table(DATA / "jobs.csv", ["job_id", "quote_id", "planned_revenue_usd"], numbers={"planned_revenue_usd": float})
invoices = read_table(DATA / "invoices.csv", ["job_id", "status", "invoice_amount_usd"], numbers={"invoice_amount_usd": float})
invoiced = invoices[invoices["status"] != "void"].groupby("job_id")["invoice_amount_usd"].sum()
won = quotes.merge(jobs, on="quote_id").join(invoiced.rename("invoiced"), on="job_id")
value = won.assign(planned_vs_quote=won["planned_revenue_usd"] / won["quoted_amount_usd"],
                   invoiced_vs_quote=won["invoiced"] / won["quoted_amount_usd"])
display(value.groupby("workflow_type")[["planned_vs_quote", "invoiced_vs_quote"]].median().round(3))

# %% [markdown]
# **2. Waiting 45 days before learning from a quote loses almost nothing.** Only a handful of
# replies ever came later than that.
#
# **3. The quoted amount is a fair measure of what a win is worth.** For every type of work,
# recurring included, the job that follows a won quote is planned and invoiced at within a few
# percent of the quote. So "worth today" can use the quote amount.
#
# **4. Customers who say no don't all behave alike.** Notebook 03 covers this one.
