"""Would the main result change if I'd made a different judgement call? Rerun the test to see.

    python scripts/sensitivity.py

Each variant changes one thing: treating the 1,444 quotes still marked "open" as lost, waiting
longer before learning from a quote, or letting the reply-speed curves use smaller groups.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from quote_followup import backtest
from quote_followup.config import load_settings
from quote_followup.data import load_quotes
from quote_followup.scorers import RANDOM_ORDER, RULES_OF_THUMB, AgeCurveValue, FollowUpModel

ROOT = Path(__file__).resolve().parents[1]


def open_as_lost(quotes: pd.DataFrame) -> pd.DataFrame:
    """Count quotes still marked "open" as lost: no reply, expired on their expiry date."""
    stale = quotes["status"].eq("open")
    out = quotes.copy()
    out.loc[stale, "won"] = 0.0
    out.loc[stale, "status"] = "expired"
    out.loc[stale, "resolved_at"] = out.loc[stale, "expires_at"]
    return out


def run(quotes: pd.DataFrame, settings, draws: int = 300) -> pd.DataFrame:
    m = settings.model
    variants = {
        "nothing (as built)": (quotes, m),
        "stale open quotes counted as lost": (open_as_lost(quotes), m),
        "wait 60 days, not 45, before learning from a quote": (quotes, replace(m, label_horizon_days=60)),
        "reply speeds may use groups of 50 quotes, not 200": (quotes, replace(m, timing_min_quotes=50)),
        "reply speeds also learned by customer source": (quotes, replace(m, timing_segments=("workflow_type", "customer_channel"))),
    }
    rows = []
    for name, (book, model_settings) in variants.items():
        scorers = [FollowUpModel(model_settings), AgeCurveValue(model_settings.label_horizon_days),
                   *[r for r in RULES_OF_THUMB if r.name != RANDOM_ORDER]]
        result = backtest.run(book, scorers, settings, ks=(1,))
        summary = backtest.summarise(result.lists, scorers[0].name, draws, settings.backtest.seed).set_index("scorer")
        quality = backtest.prediction_quality(result.predictions)
        model, curve = scorers[0].name, scorers[1].name
        rules = summary.loc[[r.name for r in scorers[2:]], "dollar_capture"]
        best_rule = rules.idxmax()
        rows.append({
            "what changed": name,
            "our list": summary.loc[model, "dollar_capture"],
            "days waiting x amount": summary.loc[curve, "dollar_capture"],
            "best rule of thumb": best_rule,
            "its dollars won": rules.max(),
            "our lead over it": -summary.loc[best_rule, "dollar_capture_gap"],
            "lead low": -summary.loc[best_rule, "dollar_capture_gap_hi"],
            "lead high": -summary.loc[best_rule, "dollar_capture_gap_lo"],
            "lead over days waiting x amount": -summary.loc[curve, "dollar_capture_gap"],
            "average chance gap": quality["expected_calibration_error"],
            "groups missed beyond luck": quality["subgroup_cells_beyond"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    settings = load_settings(ROOT / "config.toml")
    table = run(load_quotes(settings.data_dir), settings)
    table.to_csv(ROOT / "outputs" / "sensitivity.csv", index=False)
    print(table.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
