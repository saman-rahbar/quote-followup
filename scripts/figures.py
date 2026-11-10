"""Static figures for the README. Run after `quote-followup backtest`.

python scripts/figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402

from quote_followup.backtest import subgroup_calibration  # noqa: E402
from quote_followup.config import load_settings  # noqa: E402
from quote_followup.data import load_quotes  # noqa: E402
from quote_followup.scorers import DAYS_QUIET_X_AMOUNT as AGE_CURVE  # noqa: E402
from quote_followup.scorers import OUR_LIST as MODEL

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def wilson(wins: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    p = wins / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def signals(quotes: pd.DataFrame) -> None:
    """Three patterns behind the model: speed of sending, days without a reply, and who goes quiet."""
    labelled = quotes[quotes["won"].notna()]
    fig, (left, right, third) = plt.subplots(1, 3, figsize=(16, 4.4))

    bins, names = [-1, 4, 24, 48, 1e4], ["under\n4 hours", "4 to 24\nhours", "1 to 2\ndays", "over\n2 days"]
    speed = labelled.assign(band=pd.cut(labelled["hours_to_send"], bins, labels=names))
    for workflow, colour in style.WORKFLOW_COLOURS.items():
        g = speed[speed["workflow_type"] == workflow].groupby("band", observed=True)["won"].agg(["sum", "count"])
        lo, hi = wilson(g["sum"].to_numpy(), g["count"].to_numpy())
        x = np.arange(len(g))
        left.fill_between(x, lo, hi, color=colour, alpha=0.12, linewidth=0)
        left.plot(x, g["sum"] / g["count"], "-o", color=colour, ms=6)
        left.annotate(
            style.WORKFLOW_NAMES[workflow],
            (x[-1], (g["sum"] / g["count"]).iloc[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=style.INK_2,
        )
    left.set_xticks(range(len(names)), names)
    left.set_xlim(-0.3, len(names) + 0.3)
    left.set_ylim(0.4, 0.9)
    left.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    left.set_xlabel("time between writing a quote and sending it")
    left.set_ylabel("share of quotes accepted")
    left.set_title("Quotes sent sooner are accepted more often")

    ages = np.arange(0, 22)
    for workflow, colour in style.WORKFLOW_COLOURS.items():
        sub = labelled[labelled["workflow_type"] == workflow]
        days = sub["days_to_decision"].fillna(np.inf).to_numpy()
        rate = [sub["won"].to_numpy()[days > a].mean() for a in ages]
        right.plot(ages, rate, "-o", color=colour, ms=4, label=style.WORKFLOW_NAMES[workflow])
    right.legend(loc="upper right")
    right.set_xlim(-0.5, 21.5)
    right.set_xticks([0, 3, 7, 14, 21])
    right.set_ylim(0, 0.85)
    right.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    right.set_xlabel("days since the quote was sent, still no reply")
    right.set_ylabel("share that still end up a yes")
    right.set_title("Every quiet day makes a yes less likely")

    lost = labelled[labelled["won"] == 0]
    share = lost.assign(expired=lost["status"].eq("expired")).pivot_table(
        index="customer_channel", columns="workflow_type", values="expired"
    )
    channels = share.mean(axis=1).sort_values().index
    width = 0.26
    for i, (workflow, colour) in enumerate(style.WORKFLOW_COLOURS.items()):
        x = np.arange(len(channels)) + (i - 1) * width
        third.bar(
            x, share.loc[channels, workflow], width=width - 0.03, color=colour, label=style.WORKFLOW_NAMES[workflow]
        )
    third.set_xticks(range(len(channels)), [style.CHANNEL_NAMES.get(c, c) for c in channels], fontsize=9)
    third.set_ylim(0, 0.85)
    third.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    third.set_ylabel("lost quotes that ended in silence")
    third.set_xlabel("how the customer found the business")
    third.set_title("Some customers say no, others just go quiet")
    third.legend(loc="upper left")
    third.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=3)
    fig.savefig(ASSETS / "signals.png")
    plt.close(fig)


def evidence(summary: pd.DataFrame, cells: pd.DataFrame, age_cells: pd.DataFrame) -> None:
    """Are the chances right for each kind of quote, and does the list win more than simpler orders?"""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), gridspec_kw={"width_ratios": [1, 1.3, 1.3]})

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=style.AXIS, lw=1)
    ax.scatter(
        age_cells["predicted"],
        age_cells["observed"],
        s=np.sqrt(age_cells["quotes"]) * 2.5,
        color=style.MUTED,
        alpha=0.55,
        label="days without a reply only",
        edgecolor="white",
        lw=0.5,
    )
    ax.scatter(
        cells["predicted"],
        cells["observed"],
        s=np.sqrt(cells["quotes"]) * 2.5,
        color=style.HIGHLIGHT,
        alpha=0.8,
        label="our list",
        edgecolor="white",
        lw=0.5,
    )
    ax.set(
        xlim=(0, 0.95),
        ylim=(0, 0.95),
        ylabel="share that actually said yes",
        xlabel=(
            "chance we gave\n(each dot is a group of similar quotes: same work,\n"
            "customer source or sending speed, and days waiting)"
        ),
    )
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Our chances hold up for each kind of quote")

    one = summary[summary["k"] == 1].set_index("scorer")
    order = one.sort_values("dollar_capture").index
    for ax, metric, title, label in [
        (
            axes[1],
            "dollar_capture",
            "Dollars won from the first call of the day",
            "share of what the best possible pick would have won",
        ),
        (axes[2], "hit_rate", "How often the first call ended in a yes", "share of mornings"),
    ]:
        y = np.arange(len(order))
        rows = one.loc[order]
        colours = [style.HIGHLIGHT if s == MODEL else style.INK_2 if s == AGE_CURVE else style.MUTED for s in order]
        ax.hlines(y, rows[f"{metric}_lo"], rows[f"{metric}_hi"], colors=colours, lw=2)
        ax.scatter(rows[metric], y, color=colours, s=40, zorder=3)
        for yi, value in zip(y, rows[metric], strict=True):
            ax.annotate(
                f"{value:.0%}",
                (value, yi),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=style.INK_2,
            )
        ax.set_yticks(y, order if metric == "dollar_capture" else [""] * len(y))
        ax.set_xlim(0.15, 0.95)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_xlabel(label)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[1].get_yticklabels()[list(order).index(MODEL)].set_color(style.HIGHLIGHT)
    fig.tight_layout(w_pad=2)
    fig.savefig(ASSETS / "evidence.png")
    plt.close(fig)


def architecture() -> None:
    """How the code is laid out, one box per file. Written for engineers, so it uses file names."""
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)

    def box(x, y, w, h, title, body, colour=style.INK_2):
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=colour, lw=1.4)
        )
        ax.text(x + 0.15, y + h - 0.2, title, weight="bold", va="top", color=style.INK, fontsize=10.5)
        ax.text(x + 0.15, y + h - 0.55, body, va="top", color=style.INK_2, fontsize=9, linespacing=1.4)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=style.MUTED, lw=1.2))

    box(0.1, 1.6, 2.0, 1.5, "data.py", "quotes, customers,\nservice providers\njoined on (sp_id,\ncustomer_id)")
    box(
        2.6,
        1.6,
        2.1,
        1.5,
        "timeline.py",
        "open_quotes(as_of)\nsettled_quotes(as_of)\nnothing after as_of\ngets through",
    )
    box(
        5.2,
        2.65,
        2.4,
        1.55,
        "propensity.py",
        "two logistic regressions\non workflow, channel,\nhours to send: P(win)\nand P(decline | lost)",
        style.HIGHLIGHT,
    )
    box(
        5.2,
        0.45,
        2.4,
        1.55,
        "timing.py",
        "how fast accepts and\ndeclines arrive, per\nworkflow; expiries\nnever answer",
        style.HIGHLIGHT,
    )
    box(8.1, 1.6, 1.8, 1.5, "scorers.py", "logit p_now =\nlogit p_send + shift\nexpected $ = p × $\n+ rules of thumb")
    box(10.3, 2.65, 1.6, 1.25, "board.py", "this morning's\nlist and reasons")
    box(10.3, 0.75, 1.6, 1.25, "backtest.py", "walk forward,\nresample SPs")
    arrow((2.1, 2.35), (2.6, 2.35))
    arrow((4.7, 2.6), (5.2, 3.3))
    arrow((4.7, 2.1), (5.2, 1.3))
    arrow((7.6, 3.3), (8.1, 2.6))
    arrow((7.6, 1.3), (8.1, 2.1))
    arrow((9.9, 2.6), (10.3, 3.2))
    arrow((9.9, 2.1), (10.3, 1.4))
    ax.text(
        0.1,
        4.45,
        "Every step only sees what was known on the day it is run for",
        weight="bold",
        fontsize=12,
        color=style.INK,
    )
    fig.savefig(ASSETS / "architecture.png")
    plt.close(fig)


def main() -> None:
    style.apply()
    ASSETS.mkdir(exist_ok=True)
    settings = load_settings(ROOT / "config.toml")
    quotes = load_quotes(settings.data_dir)
    summary = pd.read_csv(ROOT / "outputs" / "backtest_summary.csv")
    predictions = pd.read_csv(ROOT / "outputs" / "backtest_predictions.csv")
    cells = subgroup_calibration(predictions)
    age_cells = subgroup_calibration(predictions.assign(p_now=predictions["p_age_only"]))
    signals(quotes)
    evidence(summary, cells, age_cells)
    architecture()
    print(f"figures written to {ASSETS}/")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------
# Exploratory charts for the notebooks. Each returns a closed figure, so a notebook shows it once
# as the cell's result whatever matplotlib backend is active.

STATUS_COLOURS = {"accepted": style.WON, "declined": "#ec835a", "expired": "#fab219", "open": style.MUTED}


def _done(fig):
    fig.tight_layout()
    plt.close(fig)
    return fig


def _percent(ax, axis: str = "y") -> None:
    getattr(ax, f"{axis}axis").set_major_formatter(lambda v, _: f"{v:.0%}")


def eda_missing(data_dir: Path):
    """Share of missing values in every column that has any, across all ten tables."""
    from quote_followup import audit

    raw = audit.read_raw(data_dir)
    gaps = pd.Series(
        {f"{t}.{c}": raw[t][c].isna().mean() for t in audit.TABLES for c in raw[t].columns if raw[t][c].isna().any()}
    ).sort_values()
    fig, ax = plt.subplots(figsize=(9, 0.3 * len(gaps) + 1.2))
    ax.barh(gaps.index, gaps.values, color=style.HIGHLIGHT, height=0.6)
    for y, value in enumerate(gaps.values):
        ax.annotate(f"{value:.0%}", (value, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5)
    _percent(ax, "x")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="y", visible=False)
    ax.set_title("Missing values by column (columns with any)")
    return _done(fig)


def eda_transform(quotes: pd.DataFrame):
    """Hours from writing to sending, before and after the log transform the model uses."""
    fig, (raw, logged) = plt.subplots(1, 2, figsize=(11, 3.6))
    raw.hist(quotes["hours_to_send"], bins=60, color=style.MUTED)
    raw.set(title="Hours to send, as recorded", xlabel="hours", ylabel="quotes")
    logged.hist(quotes["log_hours_to_send"], bins=60, color=style.HIGHLIGHT)
    logged.set(title="After log(1 + hours)", xlabel="log(1 + hours)")
    raw.text(0.95, 0.9, f"skew {quotes['hours_to_send'].skew():.1f}", transform=raw.transAxes, ha="right")
    logged.text(0.95, 0.9, f"skew {quotes['log_hours_to_send'].skew():.1f}", transform=logged.transAxes, ha="right")
    return _done(fig)


def eda_quotes(quotes: pd.DataFrame):
    """Quotes at a glance: volume over time, how they end, how big they are, how fast replies come."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    volume, mix, amount, reply = axes.ravel()

    monthly = quotes.groupby([quotes["sent_at"].dt.to_period("M"), "workflow_type"]).size().unstack()
    for workflow, colour in style.WORKFLOW_COLOURS.items():
        volume.plot(
            monthly.index.to_timestamp(),
            monthly[workflow],
            "-o",
            ms=4,
            color=colour,
            label=style.WORKFLOW_NAMES[workflow],
        )
    volume.set(title="Quotes sent per month", ylabel="quotes")
    volume.legend()

    shares = pd.crosstab(quotes["workflow_type"], quotes["status"], normalize="index")[list(STATUS_COLOURS)]
    left = np.zeros(len(shares))
    labels = [style.WORKFLOW_NAMES[w] for w in shares.index]
    for status, colour in STATUS_COLOURS.items():
        mix.barh(labels, shares[status], left=left, color=colour, label=status, height=0.6)
        left += shares[status].to_numpy()
    _percent(mix, "x")
    mix.set(title="How quotes end", xlim=(0, 1))
    mix.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    mix.grid(axis="y", visible=False)

    bins = np.logspace(np.log10(quotes["quoted_amount_usd"].min()), np.log10(quotes["quoted_amount_usd"].max()), 50)
    for workflow, colour in style.WORKFLOW_COLOURS.items():
        values = quotes.loc[quotes["workflow_type"] == workflow, "quoted_amount_usd"]
        amount.hist(values, bins=bins, histtype="step", lw=2, color=colour, label=style.WORKFLOW_NAMES[workflow])
    amount.set(xscale="log", title="Quote amount (log scale)", ylabel="quotes")
    amount.xaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    amount.legend()

    answered = quotes[quotes["decided_at"].notna()]
    days = np.arange(0, 21, 1)
    for status in ("accepted", "declined"):
        reply.hist(
            answered.loc[answered["status"] == status, "days_to_decision"].clip(upper=20),
            bins=days,
            histtype="step",
            lw=2,
            density=True,
            color=STATUS_COLOURS[status],
            label=status,
        )
    reply.set(title="Days from sending to reply (expired quotes never reply)", xlabel="days", ylabel="share")
    reply.legend()
    return _done(fig)


def eda_acceptance(quotes: pd.DataFrame):
    """Acceptance rate for each customer source and type of work."""
    from quote_followup.board import CHANNEL

    known = quotes[quotes["won"].notna()]
    rates = known.pivot_table(index="customer_channel", columns="workflow_type", values="won")
    rates = rates.loc[rates.mean(axis=1).sort_values().index, list(style.WORKFLOW_COLOURS)]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    image = ax.imshow(rates.to_numpy(), cmap="Blues", vmin=0.3, vmax=0.95, aspect="auto")
    for (i, j), value in np.ndenumerate(rates.to_numpy()):
        ax.text(j, i, f"{value:.0%}", ha="center", va="center", color="white" if value > 0.7 else style.INK)
    ax.set_xticks(range(rates.shape[1]), [style.WORKFLOW_NAMES[w] for w in rates.columns])
    ax.set_yticks(range(rates.shape[0]), [CHANNEL[c] for c in rates.index])
    ax.grid(False)
    ax.set_title("Share of quotes accepted")
    fig.colorbar(image, ax=ax, format=lambda v, _: f"{v:.0%}", shrink=0.8)
    return _done(fig)


def eda_waiting(waiting: pd.Series):
    """How many quotes a business had waiting, across every business-morning with at least one."""
    counts = waiting.clip(upper=10).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.bar(counts.index, counts.values, color=style.HIGHLIGHT, width=0.7)
    ax.set_xticks(counts.index, [str(i) if i < 10 else "10+" for i in counts.index])
    ax.set(
        title="Quotes waiting per business, each morning (Sep to Dec 2024)", xlabel="quotes waiting", ylabel="mornings"
    )
    ax.grid(axis="x", visible=False)
    return _done(fig)


def chance_check(predictions: pd.DataFrame):
    """After scoring: how the chances are spread, and whether each band of chances came true."""
    fig, (spread, check) = plt.subplots(1, 2, figsize=(12, 4))
    spread.hist(predictions["p_now"], bins=40, color=style.HIGHLIGHT)
    spread.set(title="Chances given, every quote-morning", xlabel="chance of a yes", ylabel="quote-mornings")
    _percent(spread, "x")

    bands = predictions.assign(band=pd.qcut(predictions["p_now"], 12, duplicates="drop"))
    bands = bands.groupby("band", observed=True).agg(given=("p_now", "mean"), came_true=("won", "mean"))
    check.plot([0, 1], [0, 1], color=style.AXIS, lw=1)
    check.plot(bands["given"], bands["came_true"], "-o", color=style.HIGHLIGHT)
    check.set(xlim=(0, 0.85), ylim=(0, 0.85), title="Chance given vs share that said yes", xlabel="chance given")
    check.set_ylabel("share that said yes")
    _percent(check, "x")
    _percent(check, "y")
    return _done(fig)


def silence_by_customer(model, workflow: str = "recurring", hours: float = 4.0):
    """How the chance falls with each quiet day, for one type of work, by how the customer found the business."""
    from scipy.special import expit

    from quote_followup.board import CHANNEL

    ages = np.linspace(0, 29, 117)
    fig, ax = plt.subplots(figsize=(8, 4))
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#898781"]
    for channel, colour in zip(CHANNEL, colours, strict=True):
        frame = pd.DataFrame(
            {
                "workflow_type": workflow,
                "customer_channel": channel,
                "log_hours_to_send": np.log1p(hours),
                "age_days": ages,
            }
        )
        q = model.decline_share.probability(frame)
        chance = expit(model.propensity.log_odds(frame) + model.timing.log_odds_shift(frame, q))
        ax.plot(ages, chance, color=colour, label=CHANNEL[channel])
    _percent(ax)
    ax.set(
        title=f"{style.WORKFLOW_NAMES[workflow]}, quote sent within {hours:.0f} hours",
        xlabel="days with no reply",
        ylabel="chance of a yes",
        ylim=(0, 0.9),
    )
    ax.legend(title="customer", alignment="left")
    return _done(fig)
