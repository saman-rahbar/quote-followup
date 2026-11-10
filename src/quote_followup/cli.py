"""Command line entry points.

quote-followup board --sp SP0027 --as-of 2024-11-14       the worked example
quote-followup board --sp SP0027 --as-of 2024-11-14 --hindsight
quote-followup backtest                                     replay Sep to Dec 2024
quote-followup audit --zip ../data.zip                      check the data is read as supplied
quote-followup train --as-of 2024-12-12                     save a fitted model to models/
quote-followup board --sp SP0267 --as-of 2024-12-12 --model models/quote_followup_2024-12-12.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import artifacts, audit, backtest
from .board import followup_board, render, with_outcomes
from .config import Settings, load_settings
from .data import load_quotes
from .scorers import FollowUpModel


def _morning(settings: Settings, day: str) -> pd.Timestamp:
    return pd.Timestamp(day) + pd.Timedelta(hours=settings.followup.morning_hour)


def _board(settings: Settings, args: argparse.Namespace) -> None:
    quotes = load_quotes(settings.data_dir)
    if not quotes["sp_id"].eq(args.sp).any():
        raise SystemExit(f"No business called {args.sp} in the data.")
    as_of = _morning(settings, args.as_of)
    if args.model:
        model, fitted_at = artifacts.load(Path(args.model))
        if fitted_at > as_of:
            raise SystemExit(
                f"That model was fitted on {fitted_at:%d %b %Y %H:%M}, after this morning. "
                "Using it would let the list see the future."
            )
    else:
        model, fitted_at = FollowUpModel(settings.model).fit(quotes, as_of), as_of
    capacity = args.top or settings.followup.daily_capacity
    board = followup_board(model, quotes, args.sp, as_of, capacity)

    name = quotes.loc[quotes["sp_id"].eq(args.sp), "business_name"].iloc[0]
    print(f"{name} ({args.sp}), morning of {as_of:%a %d %b %Y}")
    print(
        f"Learned from {model.trained_on:,} past quotes with a known outcome, all sent before "
        f"{fitted_at - pd.Timedelta(days=settings.model.label_horizon_days):%d %b %Y}.\n"
    )
    print(render(board, capacity))

    if args.hindsight:
        print("\nWhat actually happened (the model could not see this):")
        print(with_outcomes(board, quotes)[["how it ended", "reply came"]].to_string())

    if args.out:
        board.to_csv(args.out, index=False)
        print(f"\nFull board written to {args.out}")


def _backtest(settings: Settings, args: argparse.Namespace) -> None:
    ev = backtest.evaluate(load_quotes(settings.data_dir), settings)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ev.summary.to_csv(out / "backtest_summary.csv", index=False)
    ev.result.predictions.to_csv(out / "backtest_predictions.csv", index=False)
    ev.monthly.to_csv(out / "backtest_by_month.csv", index=False)
    ev.subgroups.to_csv(out / "subgroup_calibration.csv", index=False)
    ev.comparisons.to_csv(out / "paired_comparisons.csv", index=False)
    (out / "prediction_quality.json").write_text(ev.quality["our model"].to_json(indent=2))

    shown = ev.summary[["scorer", "k", "dollar_capture", "dollar_capture_lo", "dollar_capture_hi", "hit_rate"]]
    print(backtest.for_people(shown).to_string(index=False, float_format="{:.3f}".format))
    print(f"\nWritten to {out}/")


def _train(settings: Settings, args: argparse.Namespace) -> None:
    as_of = _morning(settings, args.as_of)
    model = FollowUpModel(settings.model).fit(load_quotes(settings.data_dir), as_of)
    model_path, summary_path = artifacts.save(model, as_of, Path(args.out_dir))
    print(f"Fitted on {model.trained_on:,} past quotes, as of {as_of:%d %b %Y %H:%M}.")
    print(f"Model saved to {model_path}\nWhat it learned, in plain JSON: {summary_path}")


def _audit(settings: Settings, args: argparse.Namespace) -> None:
    quotes = load_quotes(settings.data_dir)
    report = audit.run(settings.data_dir, Path(args.dictionary), quotes, Path(args.zip) if args.zip else None)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    report.to_csv(Path(args.out_dir) / "data_audit.csv", index=False)
    print(report.to_string(index=False))
    failed = (~report["ok"]).sum()
    print(f"\n{len(report) - failed} of {len(report)} checks pass.")
    if failed:
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="quote-followup")
    parser.add_argument("--config", default="config.toml")
    commands = parser.add_subparsers(dest="command", required=True)

    board = commands.add_parser("board", help="today's follow-up list for one business")
    board.add_argument("--sp", required=True)
    board.add_argument("--as-of", required=True, help="date, e.g. 2024-11-14; scored at the configured morning hour")
    board.add_argument("--top", type=int, help="how many quotes to chase (defaults to config)")
    board.add_argument("--hindsight", action="store_true", help="also print what happened to each quote")
    board.add_argument("--out", help="write the full board to this CSV")
    board.add_argument("--model", help="use a model saved by `train` instead of fitting a fresh one")
    board.set_defaults(run=_board)

    replay = commands.add_parser("backtest", help="replay past mornings against simple rules of thumb")
    replay.add_argument("--out-dir", default="outputs")
    replay.set_defaults(run=_backtest)

    train = commands.add_parser("train", help="fit the model as of a date and save it, with a readable summary")
    train.add_argument("--as-of", required=True, help="date, e.g. 2024-12-12; fitted at the configured morning hour")
    train.add_argument("--out-dir", default="models")
    train.set_defaults(run=_train)

    check = commands.add_parser("audit", help="check the CSVs against the zip, the data dictionary and each other")
    check.add_argument("--zip", help="the supplied data.zip, to compare file bytes")
    check.add_argument("--dictionary", default="reference/DATA_DICTIONARY.csv")
    check.add_argument("--out-dir", default="outputs")
    check.set_defaults(run=_audit)

    args = parser.parse_args(argv)
    args.run(load_settings(args.config), args)
