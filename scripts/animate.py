"""An animated follow-up board for one business, over three weeks of December 2024.

Each frame moves the clock on four hours. Each dot is a quote waiting for a reply: further right
means it has waited longer, higher means a better chance of a yes, bigger means more money. The
faint line ahead of a dot shows where its chance goes if the customer stays quiet. The three ringed
dots are the ones to call today.

    python scripts/animate.py [--sp SP0267] [--start 2024-12-01] [--days 19]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402

from quote_followup.config import load_settings  # noqa: E402
from quote_followup.data import load_quotes  # noqa: E402
from quote_followup.scorers import FollowUpModel  # noqa: E402
from quote_followup.timeline import open_quotes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEP = pd.Timedelta(hours=4)
FADE_FRAMES = 9
WORK_SHORT = {"recurring": "recurring", "project": "project", "emergency": "emergency"}


def size(amount: np.ndarray) -> np.ndarray:
    return 25 + 3.2 * np.sqrt(amount)


def money(amount: float) -> str:
    return f"${amount / 1000:.1f}k" if amount >= 1000 else f"${amount:,.0f}"


def trajectories(model: FollowUpModel, frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each quote's win chance over the rest of its life if no answer comes."""
    paths = []
    for _, row in frame.iterrows():
        ages = np.linspace(row["age_days"], 30, 40)
        ahead = pd.DataFrame({"workflow_type": row["workflow_type"], "age_days": ages})
        shift = model.timing.log_odds_shift(ahead, row["decline_share"])
        paths.append((ages, expit(logit(row["p_at_send"]) + shift)))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp", default="SP0267")
    parser.add_argument("--start", default="2024-12-01")
    parser.add_argument("--days", type=int, default=19)
    parser.add_argument("--out", default=str(ROOT / "assets" / "board.gif"))
    args = parser.parse_args()

    style.apply()
    settings = load_settings(ROOT / "config.toml")
    quotes = load_quotes(settings.data_dir)
    start = pd.Timestamp(args.start) + pd.Timedelta(hours=settings.followup.morning_hour)
    # Fitted once, as the monthly refit in the backtest would be, on what was known at the start.
    model = FollowUpModel(settings.model).fit(quotes, pd.Timestamp(args.start))
    capacity = settings.followup.daily_capacity
    moments = [start + i * STEP for i in range(args.days * 6 + 1)]
    mine = quotes[quotes["sp_id"] == args.sp]
    name = mine["business_name"].iloc[0]

    last_seen: dict[str, tuple[float, float, float]] = {}  # quote -> (age, p_now, amount)
    fading: list[dict] = []  # answers and expiries still on screen
    log: list[str] = []

    fig = plt.figure(figsize=(12.5, 6.2))
    ax = fig.add_axes([0.06, 0.11, 0.56, 0.76])
    panel = fig.add_axes([0.66, 0.05, 0.33, 0.85])

    def draw(i: int) -> None:
        now = moments[i]
        ax.clear()
        panel.clear()
        panel.set_axis_off()

        # What happened since the last frame: answers and expiries, shown where the dot last was.
        window = (now - STEP, now)
        for _, q in mine.iterrows():
            if q["quote_id"] not in last_seen:
                continue
            answered = pd.notna(q["decided_at"]) and window[0] < q["decided_at"] <= now
            lapsed = q["status"] in ("expired", "open") and window[0] < q["expires_at"] <= now
            if answered or lapsed:
                age, p, amount = last_seen.pop(q["quote_id"])
                won = q["status"] == "accepted"
                label = f"yes, {money(amount)}" if won else ("said no" if answered else "expired, no reply")
                fading.append({"age": age, "p": p, "amount": amount, "won": won, "label": label, "left": FADE_FRAMES})
                when = (q["decided_at"] if answered else q["expires_at"]).strftime("%a %d %b")
                mark = "✓" if won else "✗"
                outcome = "yes" if won else ("no" if answered else "expired")
                log.insert(0, f"{mark}  {when}  {money(amount):>7}  {WORK_SHORT[q['workflow_type']]:<10} {outcome}")

        frame = open_quotes(quotes, now, args.sp)
        assessed = model.assess(frame)
        frame = frame.join(assessed).sort_values(["expected_value", "quote_id"], ascending=[False, True])

        for (ages, path), (_, row) in zip(trajectories(model, frame), frame.iterrows()):
            ax.plot(ages, path, color=style.WORKFLOW_COLOURS[row["workflow_type"]], alpha=0.22, lw=1.2)
        colours = [style.WORKFLOW_COLOURS[w] for w in frame["workflow_type"]]
        ax.scatter(frame["age_days"], frame["p_now"], s=size(frame["quoted_amount_usd"]), c=colours,
                   edgecolor="white", linewidth=1.5, zorder=3)
        top = frame.head(capacity)
        ax.scatter(top["age_days"], top["p_now"], s=size(top["quoted_amount_usd"]) + 160,
                   facecolor="none", edgecolor=style.INK, linewidth=1.6, zorder=4)
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            ax.annotate(f"#{rank}  {money(row['quoted_amount_usd'])}", (row["age_days"], row["p_now"]),
                        xytext=(12, 10), textcoords="offset points", fontsize=10, weight="bold", color=style.INK)

        for event in fading:
            alpha = event["left"] / FADE_FRAMES
            colour = style.WON if event["won"] else style.LOST
            ax.scatter([event["age"]], [event["p"]], s=size(np.array([event["amount"]])), c=colour, alpha=alpha * 0.8, zorder=2)
            ax.annotate(event["label"], (event["age"], event["p"]), xytext=(10, -14), textcoords="offset points",
                        color=colour, alpha=alpha, fontsize=10, weight="bold")
            event["left"] -= 1
        fading[:] = [e for e in fading if e["left"] > 0]

        for _, row in frame.iterrows():
            last_seen[row["quote_id"]] = (row["age_days"], row["p_now"], row["quoted_amount_usd"])

        ax.set_xlim(-0.8, 30)
        ax.set_ylim(0, 0.95)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_xlabel("days since the quote was sent, still no reply")
        ax.set_ylabel("chance of a yes today")
        ax.set_title(f"{name} ({args.sp})", loc="left")
        ax.text(0, 1.06, now.strftime("%A %d %b %Y, %H:%M"), transform=ax.transAxes, color=style.INK_2, fontsize=11)
        handles = [Line2D([], [], marker="o", ls="", color=c, label=style.WORKFLOW_NAMES[w], ms=8)
                   for w, c in style.WORKFLOW_COLOURS.items()]
        handles.append(Line2D([], [], marker="o", ls="", mfc="none", mec=style.INK, ms=11, label=f"call today (top {capacity})"))
        ax.legend(handles=handles, loc="upper right", fontsize=9.5)

        panel.text(0, 1.0, "Call today", weight="bold", fontsize=12.5, va="top")
        y = 0.93
        for rank, (_, row) in enumerate(frame.iterrows(), start=1):
            if rank == capacity + 1:
                y -= 0.02
                panel.text(0, y, "Can wait", weight="bold", fontsize=11, va="top", color=style.INK_2)
                y -= 0.06
            colour = style.INK if rank <= capacity else style.MUTED
            panel.text(0, y, f"{rank}. {money(row['quoted_amount_usd']):>7}  {WORK_SHORT[row['workflow_type']]:<10}"
                             f"{row['p_now']:>4.0%} chance",
                       family="monospace", fontsize=10, va="top", color=colour)
            y -= 0.055
            if rank == 8:
                break
        panel.text(0, 0.36, "Replies so far", weight="bold", fontsize=11, va="top", color=style.INK_2)
        for j, line in enumerate(log[:6]):
            colour = style.WON if line.startswith("✓") else style.MUTED
            panel.text(0, 0.30 - j * 0.05, line, family="monospace", fontsize=9.5, va="top", color=colour)

    frames = list(range(len(moments))) + [len(moments) - 1] * 6
    animation = FuncAnimation(fig, lambda i: draw(i), frames=frames, interval=140)
    animation.save(args.out, writer=PillowWriter(fps=7), dpi=88)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
