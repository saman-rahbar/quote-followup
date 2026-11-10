"""The settings for a run, read once from config.toml. Change the file, not the code."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ModelSettings:
    categorical: tuple[str, ...]
    numeric: tuple[str, ...]
    regularization_c: float
    label_horizon_days: int
    timing_segments: tuple[str, ...]
    timing_min_quotes: int


@dataclass(frozen=True)
class FollowUpSettings:
    daily_capacity: int
    morning_hour: int


@dataclass(frozen=True)
class BacktestSettings:
    start: pd.Timestamp
    end: pd.Timestamp
    bootstrap_draws: int
    seed: int


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    model: ModelSettings
    followup: FollowUpSettings
    backtest: BacktestSettings


def load_settings(path: Path | str = "config.toml") -> Settings:
    path = Path(path)
    raw = tomllib.loads(path.read_text())
    model, followup, backtest = raw["model"], raw["followup"], raw["backtest"]
    return Settings(
        data_dir=(path.parent / raw["data"]["dir"]).resolve(),
        model=ModelSettings(
            categorical=tuple(model["categorical"]),
            numeric=tuple(model["numeric"]),
            regularization_c=float(model["regularization_c"]),
            label_horizon_days=int(model["label_horizon_days"]),
            timing_segments=tuple(model["timing_segments"]),
            timing_min_quotes=int(model["timing_min_quotes"]),
        ),
        followup=FollowUpSettings(**followup),
        backtest=BacktestSettings(
            start=pd.Timestamp(backtest["start"]),
            end=pd.Timestamp(backtest["end"]),
            bootstrap_draws=int(backtest["bootstrap_draws"]),
            seed=int(backtest["seed"]),
        ),
    )
