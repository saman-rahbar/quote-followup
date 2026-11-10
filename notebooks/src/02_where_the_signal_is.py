# %% [markdown]
# # 02 · Which problem is worth solving?
#
# The brief left the problem open, so I started by asking a simple question of each candidate:
# does the outcome change depending on things we know in advance? If it doesn't, no model can
# predict it, and a list that looks clever would just be noise.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "scripts"))

import explore
from quote_followup.config import load_settings
from quote_followup.data import load_quotes
from quote_followup.timeline import open_quotes

pd.set_option("display.width", 160)
settings = load_settings(ROOT / "config.toml")
quotes = load_quotes(settings.data_dir)

# %% [markdown]
# ## Four candidates
#
# For each one: how often it happens overall, and how much that rate moves between the lowest and
# highest group of each thing we know in advance.

# %%
scan = explore.outcome_scan(settings.data_dir)
display(scan.round(3))

# %% [markdown]
# - **Invoices that never get paid.** This was my first idea, since getting paid is a real worry
#   for owners. But the rate sits between 21% and 26% no matter how you slice it. There is nothing
#   for a model to learn.
# - **Cancelled visits.** These only move with rescheduling, and that happens after the booking,
#   so it can't warn anyone in time.
# - **Businesses leaving Jobber.** Only 30 out of 500 left, far too few to learn from or test on.
# - **Quotes being accepted.** This moves a lot: from 55% to 79% depending on the type of work,
#   58% to 76% depending on how the customer found the business, and 61% to 73% depending on how
#   quickly the quote went out.
#
# So, quotes. The rest of this notebook looks at them more closely.

# %% [markdown]
# ## How quotes end

# %%
ends = quotes.groupby("status").agg(quotes=("quote_id", "size"), median_amount=("quoted_amount_usd", "median"),
                                     total_amount=("quoted_amount_usd", "sum"))
display(ends.assign(share=ends["quotes"] / ends["quotes"].sum()).round(3))

labelled = quotes[quotes["won"].notna()]
lost = labelled[labelled["won"] == 0]
print(f"lost, out of quotes with a known ending: {len(lost) / len(labelled):.1%}, "
      f"worth ${lost['quoted_amount_usd'].sum() / 1e6:.1f}M")

# %%
answered = quotes[quotes["decided_at"].notna()]
display(answered.groupby("status")["days_to_decision"].describe(percentiles=[0.25, 0.5, 0.75, 0.9]).round(2))

# %% [markdown]
# Replies come fast. Half of all yeses arrive within about a day and a half, and nos a little
# later. Expired quotes never get a reply at all. So how long a quote has gone without a reply
# tells us something about how it will end. That idea is at the heart of the model.

# %% [markdown]
# ## How many quotes is an owner waiting on?
#
# Every morning at 8am from September to December, I counted each business's quotes that had gone
# out, had no reply yet, and hadn't expired.

# %%
mornings = pd.date_range("2024-09-01", "2024-12-20", freq="D") + pd.Timedelta(hours=8)
counts = pd.concat([open_quotes(quotes, m).groupby("sp_id").size() for m in mornings])
print(counts.describe().round(2).to_string())
print(f"business-mornings with 2 or more quotes waiting: {(counts >= 2).sum():,}")

# %% [markdown]
# On a morning when a business has anything waiting, it's usually two quotes, and sometimes as many
# as 21. "Which one do I call first?" comes up on tens of thousands of business-mornings.

# %% [markdown]
# ## Does each business need its own model?
#
# If some businesses simply win more than others, beyond what their mix of work and customers
# explains, one shared model would be wrong for them. To test it, I predicted each business's
# number of wins from its mix alone, and checked whether the real numbers scatter around those
# predictions more than chance would.

# %%
from scipy.stats import chi2

cell = labelled["workflow_type"] + "|" + labelled["customer_channel"]
expected = labelled.groupby(cell)["won"].transform("mean")
per_sp = pd.DataFrame({"won": labelled["won"], "expected": expected, "variance": expected * (1 - expected),
                       "sp_id": labelled["sp_id"]}).groupby("sp_id").sum()
statistic = ((per_sp["won"] - per_sp["expected"]) ** 2 / per_sp["variance"]).sum()
dof = len(per_sp) - 1
print(f"scatter vs chance: {statistic / dof:.2f} (1.0 means exactly what chance predicts), p = {chi2.sf(statistic, dof):.3f}")

# %% [markdown]
# Businesses scatter about as much as luck alone would produce (a ratio close to 1, and a p-value
# above the usual 0.05 bar). So one shared model can serve every business. That's lucky, because a
# typical business sends only about 50 quotes a year, nowhere near enough for a model of its own.
