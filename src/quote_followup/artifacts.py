"""Saving a fitted model, and a plain summary of what it learned, so it can be reused and reviewed.

The model file is a Python pickle. Only load model files this project wrote itself: a pickle from
somewhere else can run any code when it is opened.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .scorers import FollowUpModel


def _weights(score) -> dict[str, float]:
    """What each fact adds to the score, by name (e.g. "workflow_type=emergency": 0.59)."""
    encode, fitted = score._pipeline[0], score._pipeline[-1]
    names = [name.split("__", 1)[1] for name in encode.get_feature_names_out()]
    return {name: round(float(w), 4) for name, w in zip(names, fitted.coef_[0], strict=True)}


def summary(model: FollowUpModel, as_of: pd.Timestamp) -> dict:
    """A readable record of the fitted model: when, on what, and what it learned."""
    curves = {}
    for key, curve in model.timing.curves.items():
        curves[" / ".join(key) or "all quotes"] = {
            "yeses learned from": int(curve.win_days.size),
            "median days until a yes": round(float(np.median(curve.win_days)), 2) if curve.win_days.size else None,
            "nos learned from": int(curve.decline_days.size),
            "median days until a no": round(float(np.median(curve.decline_days)), 2)
            if curve.decline_days.size
            else None,
        }
    return {
        "fitted as of": str(as_of),
        "learned from quotes": model.trained_on,
        "facts used": list(model.propensity.features),
        "chance of a yes: typical quote (log odds)": round(model.propensity.baseline_log_odds(), 4),
        "chance of a yes: what each fact adds (log odds, standardised)": _weights(model.propensity),
        "says no out loud: what each fact adds (log odds, standardised)": _weights(model.decline_share),
        "reply speeds": curves,
    }


def save(model: FollowUpModel, as_of: pd.Timestamp, folder: Path) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"quote_followup_{as_of:%Y-%m-%d}"
    model_path, summary_path = folder / f"{stem}.pkl", folder / f"{stem}.json"
    with open(model_path, "wb") as handle:
        pickle.dump({"as_of": as_of, "model": model}, handle)
    summary_path.write_text(json.dumps(summary(model, as_of), indent=2))
    return model_path, summary_path


def load(path: Path) -> tuple[FollowUpModel, pd.Timestamp]:
    """Load a model this project saved. Never point this at a file from anywhere else."""
    with open(path, "rb") as handle:
        stored = pickle.load(handle)
    return stored["model"], stored["as_of"]


def scorecard(model: FollowUpModel) -> pd.DataFrame:
    """The points each fact adds, in plain words: towards a yes, and towards saying no out loud."""
    from .board import CHANNEL, WORK

    names = {f"workflow_type_{k}": v for k, v in WORK.items()}
    names |= {f"customer_channel_{k}": v for k, v in CHANNEL.items()}
    names["log_hours_to_send"] = "each step slower in sending the quote"
    table = pd.DataFrame(
        {
            "points towards a yes": _weights(model.propensity),
            "points towards saying no out loud": _weights(model.decline_share),
        }
    )
    return table.rename(index=names).rename_axis("fact")
