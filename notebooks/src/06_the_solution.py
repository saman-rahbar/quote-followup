# %% [markdown]
# # 06 · What the owner sees
#
# It's 8am on Thursday 12 December 2024 at SP0267. The model has learned from past quotes whose
# endings were known at least 45 days earlier. It cannot see anything that happened after 8am today.
#
# I chose this morning because the different ways of ordering disagree on it, which makes it a good
# illustration. The fair comparison across every morning is in notebook 05.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image, display
from scipy.special import logit

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "scripts"))

from quote_followup.board import CHANNEL, WORK, followup_board, render
from quote_followup.config import load_settings
from quote_followup.data import load_quotes
from quote_followup.scorers import SILENCE, FollowUpModel

pd.set_option("display.width", 170)
settings = load_settings(ROOT / "config.toml")
quotes = load_quotes(settings.data_dir)
as_of = pd.Timestamp("2024-12-12 08:00")
model = FollowUpModel(settings.model).fit(quotes, as_of)
board = followup_board(model, quotes, "SP0267", as_of, settings.followup.daily_capacity)
print(f"learned from {model.trained_on:,} past quotes\n")
print(render(board, settings.followup.daily_capacity))

# %% [markdown]
# How to read it: **chance of yes** is the chance this quote ends in a yes, given everything known
# this morning. **Worth today** is that chance times the amount, and the list is sorted by it.
# **Chance halves in** is how many more days without a reply before that chance drops by half, a
# rough sense of how urgent the call is. **Why** lists the biggest things pushing the chance up or down.

# %% [markdown]
# ## Where the top quote's number comes from
#
# The model starts from a typical quote and adds or subtracts a fixed amount for each fact. The
# amounts are on the log-odds scale, a standard way of adding up evidence for a yes or a no. The
# total converts back to a chance. Nothing is hidden: the pieces add up exactly.

# %%
top = board.iloc[0]
labels = {"workflow_type": f"type of work: {WORK[top['workflow_type']]}",
          "customer_channel": f"customer source: {CHANNEL[top['customer_channel']]}",
          "log_hours_to_send": f"quote sent after {top['hours_to_send']:.1f} hours"}
pieces = {"a typical quote starts at": model.propensity.baseline_log_odds()}
pieces |= {labels[feature]: top[f"why_{feature}"] for feature in model.propensity.features}
pieces[f"no reply for {top['age_days']:.1f} days"] = top[f"why_{SILENCE}"]
pieces = pd.Series(pieces, name="adds to the score")
display(pieces.round(3).to_frame())
print(f"total {pieces.sum():.4f}, which is a chance of {top['p_now']:.1%}  (check: {logit(top['p_now']):.4f})")
print(f"{top['p_now']:.1%} x ${top['quoted_amount_usd']:,.0f} = ${top['expected_value']:,.0f} worth today")

# %% [markdown]
# Notice the last line. This customer came by referral, and referred customers who turn a quote
# down usually say so (notebook 03). So nearly two days without a reply only nudges the chance down a little.

# %% [markdown]
# ## What actually happened
#
# The real endings, which the model could not see.

# %%
outcome = quotes.set_index("quote_id").loc[board["quote_id"], ["status", "decided_at"]]
shown = board.set_index("quote_id")[["quoted_amount_usd", "age_days", "p_now", "expected_value", "chase_today"]]
display(shown.round(3).join(outcome).rename(columns={
    "quoted_amount_usd": "amount", "age_days": "days waiting", "p_now": "chance of yes",
    "expected_value": "worth today", "chase_today": "call today", "status": "how it ended", "decided_at": "reply came"}))

# %% [markdown]
# The top two were accepted that afternoon and the next evening. Calling the biggest quote first
# would have meant a \$13.8k project that had been quiet for nearly three weeks; it expired. Calling
# the newest first would have meant a \$168 quote; it expired too.
#
# The third pick is that same \$13.8k quote at a 3% chance. It's on the list because 3% of \$13.8k
# is still worth more than most small quotes. The list goes after dollars, so a big long shot can
# outrank a small likely yes. An owner who would rather count yeses can order by chance instead.

# %% [markdown]
# ## A morning where it gets it wrong

# %%
as_of_wrong = pd.Timestamp("2024-12-17 08:00")
model_wrong = FollowUpModel(settings.model).fit(quotes, as_of_wrong)
wrong = followup_board(model_wrong, quotes, "SP0236", as_of_wrong, settings.followup.daily_capacity)
outcome = quotes.set_index("quote_id").loc[wrong["quote_id"], ["status"]]
display(wrong.set_index("quote_id")[["quoted_amount_usd", "age_days", "p_now", "expected_value"]].round(3)
        .join(outcome).rename(columns={"quoted_amount_usd": "amount", "age_days": "days waiting",
                                       "p_now": "chance of yes", "expected_value": "worth today", "status": "how it ended"}))

# %% [markdown]
# Second on the list was a \$3.4k project at 48%; it expired. A \$2.6k project at 44%, fourth on the
# list, was accepted. When two quotes have nearly the same chance, which one comes through is close
# to a coin toss, and no model can know which customer will reply.

# %% [markdown]
# ## Three weeks of the board, replayed
#
# ![The board replayed every four hours for three weeks](../assets/board.gif)

# %% [markdown]
# ## What this does and doesn't show
#
# - **It finds the quotes most likely to say yes. It doesn't prove a call changes anything.** The data
#   has no record of follow-up calls, so some of these yeses would have come anyway. To measure the
#   real effect of calling, a business would skip follow-ups on a random share of quotes for a while
#   and compare (an approach called uplift modelling, Radcliffe and Surry 2011).
# - **For putting quotes in order, the simple "days waiting × amount" list does as well** (notebook 05,
#   H2). The full model is there so that the chance shown for each quote is right for that kind of
#   quote, and so the reasons are real.
# - **The data is synthetic.** The patterns found here are the ones the data's creators built in.
