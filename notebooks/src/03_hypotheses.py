# %% [markdown]
# # 03 · What I set out to test
#
# The user is a business owner looking at their waiting quotes in the morning and deciding which
# one to call first. Here are the four claims I tested, written so that the data could prove each
# one wrong.
#
# | | The claim | How I test it | It's wrong if |
# |---|---|---|---|
# | **H1** | Calling quotes in order of *chance of a yes today × amount* brings in more money than the simple orders an owner might use | Replay every morning from 1 Sep to 20 Dec 2024. For the first call, compare dollars won with the most any single pick could have won | it doesn't clearly beat *newest quote first* and *biggest quote first* |
# | **H2** | Knowing the quote itself (type of work, customer source, sending speed) makes that order better than knowing only how long it has waited | The same replay, against a list that uses only days waiting × amount | the difference is too small to rule out luck |
# | **H3** | The chance on the board is right for each kind of quote, not just on average | Group similar quotes and compare the chance given with what happened | more groups miss than luck would explain |
# | **H4** | When the answer is no, whether a customer says so or just goes quiet depends on how they found the business | Compare how lost quotes ended, by customer source | the difference could be chance in any type of work |
#
# **"Clearly beat" and "luck"** come from the same method throughout. I re-ran each comparison a
# thousand times on random re-draws of the businesses and looked at the middle 95% of results. If
# that whole range sits above zero, the difference is real, not a fluke of which businesses
# happened to be in the data.
#
# **To be upfront about how these came about.** H1 was the original claim. The first time I tested
# it, I measured each morning's share of the dollars and averaged those shares. I switched to total
# dollars won over total available, because that is what an owner cares about. Notebook 05 shows
# both. H2, H3 and H4 came out of a second review, after the first results, so I re-made every
# modelling choice using only data from before September (notebook 04) to keep the test fair.
#
# The rest of this notebook shows the patterns that led to the model.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "scripts"))

import style
from quote_followup.board import CHANNEL
from quote_followup.config import load_settings
from quote_followup.data import load_quotes

style.apply()
pd.set_option("display.width", 160)
settings = load_settings(ROOT / "config.toml")
quotes = load_quotes(settings.data_dir)
labelled = quotes[quotes["won"].notna()]

# %% [markdown]
# ## Quotes sent sooner are accepted more often
#
# This is a pattern, not proof that speed causes the yes. Maybe slow quotes are for harder jobs.
# Two checks make the obvious explanations unlikely: sending speed has nothing to do with the
# quote's size, and the effect of speed stays the same when I account for size and number of items.

# %%
print("link between hours-to-send and quote size, within each type of work (0 means none):")
print(labelled.groupby("workflow_type")[["log_hours_to_send", "quoted_amount_usd"]]
      .apply(lambda g: np.corrcoef(g["log_hours_to_send"], np.log(g["quoted_amount_usd"]))[0, 1]).round(3).to_string())


def speed_effect(inputs: pd.DataFrame) -> float:
    fit = LogisticRegression(C=1e6, max_iter=5000).fit(inputs, labelled["won"])
    return fit.coef_[0][list(inputs.columns).index("log_hours_to_send")]


base = pd.get_dummies(labelled[["workflow_type", "customer_channel"]], drop_first=True).astype(float)
base = base.assign(log_hours_to_send=labelled["log_hours_to_send"])
fuller = base.assign(log_amount=np.log(labelled["quoted_amount_usd"]), line_items=labelled["line_item_count"])
print(f"\neffect of sending speed, allowing for type of work and customer source: {speed_effect(base):.3f}")
print(f"  ... and also for quote size and number of items:                    {speed_effect(fuller):.3f}")

# %%
amount_quartile = pd.qcut(np.log(labelled["quoted_amount_usd"]), 4, labels=["smallest", "small", "large", "largest"])
within = labelled.assign(size=amount_quartile).pivot_table(
    index=["workflow_type", "size"], columns="send_speed", values="won", observed=True).dropna()
display(within.round(3).rename(index=style.WORKFLOW_NAMES))

# %% [markdown]
# For every type of work and every size of quote, the ones sent within a day were accepted more
# often.

# %% [markdown]
# ## Every quiet day makes a yes less likely
#
# Of the quotes still without a reply after a given number of days, how many end up a yes?

# %%
fig, ax = plt.subplots(figsize=(7, 3.8))
for workflow, colour in style.WORKFLOW_COLOURS.items():
    sub = labelled[labelled["workflow_type"] == workflow]
    days = sub["days_to_decision"].fillna(np.inf).to_numpy()
    ages = np.arange(0, 22)
    ax.plot(ages, [sub["won"].to_numpy()[days > a].mean() for a in ages], "-o", ms=3.5, color=colour,
            label=style.WORKFLOW_NAMES[workflow])
ax.set(xlabel="days since the quote was sent, still no reply", ylabel="share that still end up a yes", ylim=(0, 0.85))
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.legend()
plt.show()

days = labelled["days_to_decision"].fillna(np.inf)
for a in (0, 3, 7, 14):
    print(f"no reply after {a:>2} days: {labelled.loc[days > a, 'won'].mean():.1%} still end up a yes")

# %% [markdown]
# ## H4: some customers say no, others just go quiet
#
# A quote can be lost in two ways: the customer says no, or never replies and it expires. If
# everyone behaved the same way, a quiet week would mean the same thing for every quote. They don't.

# %%
lost = labelled[labelled["won"] == 0]
mix = lost.assign(expired=lost["status"].eq("expired")).pivot_table(
    index="customer_channel", columns="workflow_type", values="expired")
display(mix.round(3).rename(index=CHANNEL, columns=style.WORKFLOW_NAMES)
        .rename_axis(index="how the customer found the business", columns="lost quotes that ended in silence"))
for workflow in style.WORKFLOW_COLOURS:
    table = pd.crosstab(lost.loc[lost["workflow_type"] == workflow, "customer_channel"],
                        lost.loc[lost["workflow_type"] == workflow, "status"])
    chi, p, dof, _ = chi2_contingency(table)
    print(f"{style.WORKFLOW_NAMES[workflow]:<18} chance this pattern is luck: p = {p:.1e}")

# %% [markdown]
# H4 holds for every type of work, and the odds of it being luck are vanishingly small. Repeat
# and referred customers who turn a quote down usually say so. Marketplace leads usually just stop
# replying. So a quiet week from a repeat customer is much less worrying than the same week from a
# marketplace lead.
#
# My first model missed this. It treated silence the same for everyone within a type of work, and
# so it underrated good customers whose quotes had been waiting a few days. Notebook 04 shows the
# fix working.
#
# ## How the model works, in plain terms
#
# A quote ends in a yes, a no, or silence. The model learns three things from past quotes:
#
# 1. **How likely a yes is**, when the quote goes out, from the type of work, how the customer found
#    the business, and how fast the quote was sent.
# 2. **If it's a no, how likely the customer is to say so** rather than go quiet, from the same facts.
# 3. **How quickly yeses and nos usually arrive**, for each type of work.
#
# Each morning it combines them: the longer a quote waits without a reply, the more it looks like
# one that will end in silence, and the chance of a yes comes down, faster for customers who usually
# just go quiet. For the statistically minded, with *p* the chance of a yes at sending, *q* the
# chance a no is said out loud, and *S* the share of yeses or nos still to come after *a* days:
#
# $$\operatorname{logit} P(\text{yes} \mid \text{no reply after } a \text{ days}) = \operatorname{logit} p + \log S_\text{yes}(a) - \log\big[q\,S_\text{no}(a) + (1-q)\big]$$
#
# This is known as a mixture model for competing outcomes (Larson and Dinse 1985). It assumes reply
# speed depends on the type of work and on the answer, but not on anything else. Here's how well that
# holds:

# %%
accepted = labelled[labelled["won"] == 1]
display(accepted.pivot_table(index="customer_channel", columns="workflow_type", values="days_to_decision",
                             aggfunc="median").round(2).rename(index=CHANNEL, columns=style.WORKFLOW_NAMES)
        .rename_axis(index="how the customer found the business", columns="median days until the yes"))

# %% [markdown]
# For recurring and emergency work, yeses arrive at the same speed whoever the customer is. For
# one-off projects they vary by about a day, so that's where the assumption is weakest. Notebook 05
# shows where this still causes small misses.
