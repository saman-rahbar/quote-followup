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
from quote_followup.scorers import DAYS_QUIET_X_AMOUNT as AGE_CURVE, OUR_LIST as MODEL  # noqa: E402

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
        left.annotate(style.WORKFLOW_NAMES[workflow], (x[-1], (g["sum"] / g["count"]).iloc[-1]), xytext=(8, 0),
                      textcoords="offset points", va="center", color=style.INK_2)
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
        index="customer_channel", columns="workflow_type", values="expired")
    channels = share.mean(axis=1).sort_values().index
    width = 0.26
    for i, (workflow, colour) in enumerate(style.WORKFLOW_COLOURS.items()):
        x = np.arange(len(channels)) + (i - 1) * width
        third.bar(x, share.loc[channels, workflow], width=width - 0.03, color=colour, label=style.WORKFLOW_NAMES[workflow])
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
    ax.scatter(age_cells["predicted"], age_cells["observed"], s=np.sqrt(age_cells["quotes"]) * 2.5,
               color=style.MUTED, alpha=0.55, label="days without a reply only", edgecolor="white", lw=0.5)
    ax.scatter(cells["predicted"], cells["observed"], s=np.sqrt(cells["quotes"]) * 2.5,
               color=style.HIGHLIGHT, alpha=0.8, label="our list", edgecolor="white", lw=0.5)
    ax.set(xlim=(0, 0.95), ylim=(0, 0.95), ylabel="share that actually said yes",
           xlabel="chance we gave\n(each dot is a group of similar quotes: same work,\ncustomer source or sending speed, and days waiting)")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Our chances hold up for each kind of quote")

    one = summary[summary["k"] == 1].set_index("scorer")
    order = one.sort_values("dollar_capture").index
    for ax, metric, title, label in [
        (axes[1], "dollar_capture", "Dollars won from the first call of the day", "share of what the best possible pick would have won"),
        (axes[2], "hit_rate", "How often the first call ended in a yes", "share of mornings"),
    ]:
        y = np.arange(len(order))
        rows = one.loc[order]
        colours = [style.HIGHLIGHT if s == MODEL else style.INK_2 if s == AGE_CURVE else style.MUTED for s in order]
        ax.hlines(y, rows[f"{metric}_lo"], rows[f"{metric}_hi"], colors=colours, lw=2)
        ax.scatter(rows[metric], y, color=colours, s=40, zorder=3)
        for yi, value in zip(y, rows[metric]):
            ax.annotate(f"{value:.0%}", (value, yi), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=9, color=style.INK_2)
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
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="white", ec=colour, lw=1.4))
        ax.text(x + 0.15, y + h - 0.2, title, weight="bold", va="top", color=style.INK, fontsize=10.5)
        ax.text(x + 0.15, y + h - 0.55, body, va="top", color=style.INK_2, fontsize=9, linespacing=1.4)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=style.MUTED, lw=1.2))

    box(0.1, 1.6, 2.0, 1.5, "data.py", "quotes, customers,\nservice providers\njoined on (sp_id,\ncustomer_id)")
    box(2.6, 1.6, 2.1, 1.5, "timeline.py", "open_quotes(as_of)\nsettled_quotes(as_of)\nnothing after as_of\ngets through")
    box(5.2, 2.65, 2.4, 1.55, "propensity.py", "two logistic regressions\non workflow, channel,\nhours to send: P(win)\nand P(decline | lost)", style.HIGHLIGHT)
    box(5.2, 0.45, 2.4, 1.55, "timing.py", "how fast accepts and\ndeclines arrive, per\nworkflow; expiries\nnever answer", style.HIGHLIGHT)
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
    ax.text(0.1, 4.45, "Every step only sees what was known on the day it is run for",
            weight="bold", fontsize=12, color=style.INK)
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
