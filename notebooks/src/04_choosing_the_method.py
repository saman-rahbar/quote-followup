# %% [markdown]
# # 04 · Choosing how to build the model
#
# Every decision about the model is made in this notebook, using only data from before 1 Sep 2024,
# the day the final test starts. If I tuned the model on the same months I test it on, it would
# look better than it really is.
#
# The practice runs work like the final test. On the first of April, May, June and July, the model
# learns from what was known by then and is scored on the quotes sent that month. Quotes sent after
# 17 July are left out, because some of their replies hadn't arrived by 1 September.
#
# Two questions, in order: which facts about a quote belong in the model, and how best to use
# "days without a reply". The main yardstick is **log loss**, a standard score for predicted chances
# that punishes confident mistakes (lower is better). When two options tie on it, I pick the one
# whose chances hold up better group by group (H3).

# %%
import sys
from pathlib import Path

import pandas as pd
from IPython.display import display

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "scripts"))

import ablations
from quote_followup.config import load_settings
from quote_followup.data import load_quotes

pd.set_option("display.width", 160)
settings = load_settings(ROOT / "config.toml")
quotes = ablations.with_candidate_inputs(load_quotes(settings.data_dir))
print("practice runs start on:", ", ".join(f"{d:%d %b}" for d in ablations.ORIGINS),
      "| last quote used was sent on:", f"{ablations.BACKTEST_START - pd.Timedelta(days=settings.model.label_horizon_days):%d %b %Y}")

# %% [markdown]
# ## Which facts about a quote matter?
#
# Every fact tried here is known the moment the quote goes out. The two "history" facts are a
# business's (or customer's) past win rate, using only replies already in, pulled towards the
# overall rate when there's little history. Besides log loss, the table shows **AUC**: the chance
# that a quote which said yes was rated higher than one that said no (0.5 is a coin flip, 1 is perfect).

# %%
send_time = ablations.send_time_ablations(quotes, settings)
display(send_time.round(4))

# %% [markdown]
# Three facts do the work: the type of work, how the customer found the business, and how fast the
# quote went out. Adding industry, customer type, quote size, number of items, weekend sending, or
# either history rate adds nothing, which matches the finding in notebook 02 that businesses don't
# differ beyond their mix. Taking any one of the three out makes things worse, the type of work
# most of all.
#
# A more complex model (gradient boosting) does clearly worse. With only a few real patterns to
# find, its extra flexibility mostly picks up noise, and it couldn't explain its scores the way
# the simple model can.

# %% [markdown]
# ## How best to use "days without a reply"?
#
# Each option is scored at every point a quote was still waiting (after 0, 1, 2, 3, 5, 7, 10, 14
# and 21 days). The last columns are the H3 check: how many groups of similar quotes the chances
# missed by more than luck explains, against the threshold shown.

# %%
silence = ablations.silence_ablations(quotes, settings)
display(silence.round(4))

# %% [markdown]
# What this shows:
#
# - **Letting "says no vs goes quiet" depend on the customer is the fix that matters.** My first
#   version used one figure per type of work and missed in 3 groups. Letting it vary by customer,
#   as notebook 03 suggests, scores better (0.5705 against 0.5717) and misses in just 1.
# - **Learning reply speed separately for smaller groups doesn't help.** Splitting by sending speed
#   ties on log loss (0.5704), and splitting by customer source is slightly worse (0.5711). Both miss
#   in more groups (3 and 2), so reply speed stays learned per type of work.
# - **The approach beats the usual alternative**, which simply adds "days waiting" as one more fact
#   in the model (van Houwelingen 2007). It also keeps "days without a reply" as its own reason on
#   the board.
# - **Gradient boosting does worse here too.**
# - **Days waiting on its own is badly off for many groups** (27 misses). That matters for H2 and H3.
#
# So the model going forward is the one in `config.toml`: the three facts, a "says no vs goes quiet"
# chance for each customer, and reply speeds learned per type of work.

# %%
print((ROOT / "config.toml").read_text().split("[followup]")[0])
