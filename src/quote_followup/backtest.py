"""Replay of past mornings, and every score that comes out of it.

Each month every method refits on settled quotes, then ranks each business's waiting quotes at 8am
on every day of that month; outcomes are joined only afterwards (Tashman 2000). This measures who the
list finds, not what a call changes: the data has no record of follow-ups.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .config import Settings
from .scorers import (
    BIGGEST_FIRST,
    DAYS_QUIET_X_AMOUNT,
    NEWEST_FIRST,
    OUR_LIST,
    OUR_LIST_BY_CHANCE,
    AgeCurveValue,
    FollowUpModel,
    Scorer,
    all_scorers,
)
from .timeline import open_quotes


@dataclass(frozen=True)
class BacktestResult:
    lists: pd.DataFrame  # one row per scorer, business, morning and list size k
    predictions: pd.DataFrame  # the main model's win chance for every candidate morning, with outcome


def _mornings(settings: Settings) -> list[pd.Timestamp]:
    bt = settings.backtest
    days = pd.date_range(bt.start, bt.end, freq="D")
    return list(days + pd.Timedelta(hours=settings.followup.morning_hour))


def _top_k(candidates: pd.DataFrame, priority: np.ndarray, k: int) -> pd.DataFrame:
    """For each business that morning: did the quotes at the top say yes, and how many dollars came in
    compared with the best possible pick."""
    c = candidates.assign(priority=priority, won_value=candidates["won"] * candidates["quoted_amount_usd"])
    c = c[c.groupby("sp_id")["quote_id"].transform("size") > k]
    ranked = c.sort_values(["sp_id", "priority", "quote_id"], ascending=[True, False, True])
    picked = ranked.groupby("sp_id").head(k).groupby("sp_id")
    best = c.sort_values("won_value", ascending=False).groupby("sp_id").head(k).groupby("sp_id")
    out = pd.DataFrame(
        {
            "hit_rate": picked["won"].mean(),
            "won_value": picked["won_value"].sum(),
            "best_value": best["won_value"].sum(),
            "n_open": c.groupby("sp_id").size(),
        }
    )
    return out.assign(mornings=1)


def run(
    quotes: pd.DataFrame, scorers: list[Scorer], settings: Settings, ks: tuple[int, ...] = (1, 3)
) -> BacktestResult:
    lists, predictions = [], []
    # Keep the chances our model gave (and the simple rival's), to check later whether they came true.
    logged = next(s for s in scorers if isinstance(s, FollowUpModel))
    age_curve = next((s for s in scorers if isinstance(s, AgeCurveValue)), None)
    fitted_for: pd.Timestamp | None = None
    for morning in _mornings(settings):
        month = morning.normalize().replace(day=1)
        if month != fitted_for:
            refit_at = max(month, settings.backtest.start)
            for scorer in scorers:
                scorer.fit(quotes, refit_at)
            fitted_for = month

        candidates = open_quotes(quotes, morning)
        # Only now do we attach what actually happened. Quotes still marked "open" in the data have
        # no known outcome, so they can't be scored and are left out.
        candidates = candidates.assign(won=quotes.loc[candidates.index, "won"]).dropna(subset=["won"])
        if candidates.empty:
            continue
        for scorer in scorers:
            priority = scorer.priority(candidates)
            for k in ks:
                lists.append(_top_k(candidates, priority, k).assign(scorer=scorer.name, morning=morning, k=k))
        assessed = logged.assess(candidates)
        logged_columns = ["quote_id", "sp_id", "workflow_type", "customer_channel", "send_speed", "age_days", "won"]
        logged_rows = candidates[logged_columns].join(assessed[["p_at_send", "p_now"]]).assign(morning=morning)
        if age_curve is not None:
            logged_rows["p_age_only"] = age_curve.probability(candidates)
        predictions.append(logged_rows)

    return BacktestResult(
        lists=pd.concat(lists).reset_index(),
        predictions=pd.concat(predictions, ignore_index=True),
    )


# How each score is worked out: add up the top line and the bottom line over all mornings, then
# divide. For dollars that means total dollars won over total dollars available, which is what an
# owner cares about. Averaging each morning's share instead would count a $60 morning the same as
# a $6,000 one.
METRICS = {
    "hit_rate": ("hit_rate", "mornings"),
    "dollar_capture": ("won_value", "best_value"),
}


def summarise(lists: pd.DataFrame, reference: str, draws: int, seed: int) -> pd.DataFrame:
    """Each metric per scorer and k, the gap to `reference`, and 95% intervals.

    Intervals resample whole businesses (Field and Welsh 2007), paired across scorers. `_adj` columns
    widen the gap intervals by Bonferroni across the rivals."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, at_k in lists.groupby("k"):
        sp_ids = np.sort(at_k["sp_id"].unique())
        picks = rng.integers(0, sp_ids.size, size=(draws, sp_ids.size))
        weights = np.stack([np.bincount(p, minlength=sp_ids.size) for p in picks])
        family = at_k["scorer"].nunique() - 1
        tail = 2.5 / max(family, 1)

        def boot(group: pd.DataFrame, metric: str, sp_ids=sp_ids, weights=weights) -> tuple[float, np.ndarray]:
            num, den = METRICS[metric]
            per_sp = group.groupby("sp_id")[[num, den]].sum().reindex(sp_ids, fill_value=0)
            top, bottom = per_sp[num].to_numpy(), per_sp[den].to_numpy()
            return top.sum() / bottom.sum(), (weights @ top) / (weights @ bottom)

        ref = {m: boot(at_k[at_k["scorer"] == reference], m) for m in METRICS}
        for scorer, group in at_k.groupby("scorer", sort=False):
            row = {"scorer": scorer, "k": k, "mornings": len(group), "businesses": group["sp_id"].nunique()}
            for m in METRICS:
                point, resampled = boot(group, m)
                gaps = resampled - ref[m][1]
                row |= {
                    m: point,
                    f"{m}_lo": np.percentile(resampled, 2.5),
                    f"{m}_hi": np.percentile(resampled, 97.5),
                    f"{m}_gap": point - ref[m][0],
                    f"{m}_gap_lo": np.percentile(gaps, 2.5),
                    f"{m}_gap_hi": np.percentile(gaps, 97.5),
                    f"{m}_gap_lo_adj": np.percentile(gaps, tail),
                    f"{m}_gap_hi_adj": np.percentile(gaps, 100 - tail),
                }
            rows.append(row)
    return pd.DataFrame(rows)


def paired_gap(
    lists: pd.DataFrame, first: str, second: str, metric: str, k: int, draws: int, seed: int
) -> dict[str, float]:
    """How much better `first` scores than `second`, with a 95% range from re-drawing businesses."""
    at_k = lists[lists["k"] == k]
    sp_ids = np.sort(at_k["sp_id"].unique())
    picks = np.random.default_rng(seed).integers(0, sp_ids.size, size=(draws, sp_ids.size))
    weights = np.stack([np.bincount(p, minlength=sp_ids.size) for p in picks])
    num, den = METRICS[metric]

    def resample(name: str) -> tuple[float, np.ndarray]:
        per_sp = at_k[at_k["scorer"] == name].groupby("sp_id")[[num, den]].sum().reindex(sp_ids, fill_value=0)
        top, bottom = per_sp[num].to_numpy(), per_sp[den].to_numpy()
        return top.sum() / bottom.sum(), (weights @ top) / (weights @ bottom)

    (a, a_draws), (b, b_draws) = resample(first), resample(second)
    lo, hi = np.percentile(a_draws - b_draws, [2.5, 97.5])
    return {"first": first, "second": second, "metric": metric, "k": k, "gap": a - b, "lo": lo, "hi": hi}


def by_month(lists: pd.DataFrame) -> pd.DataFrame:
    """Scores month by month, to check the result isn't driven by one unusual stretch."""
    month = pd.to_datetime(lists["morning"]).dt.to_period("M").astype(str)
    sums = (
        lists.assign(month=month)
        .groupby(["k", "scorer", "month"], sort=False)[["hit_rate", "mornings", "won_value", "best_value"]]
        .sum()
    )
    return pd.DataFrame(
        {
            "hit_rate": sums["hit_rate"] / sums["mornings"],
            "dollar_capture": sums["won_value"] / sums["best_value"],
            "mornings": sums["mornings"],
        }
    ).reset_index()


AGE_BANDS = (0, 1, 3, 7, 14, 30)


def subgroup_calibration(predictions: pd.DataFrame, min_rows: int = 100) -> pd.DataFrame:
    """Predicted vs observed win rate in work x (source or send speed) x age cells.

    z uses a standard error clustered by quote, since one quote spans many mornings. Cells with no
    wins or no losses are skipped: the clustered error is zero there."""
    bands = pd.cut(predictions["age_days"], AGE_BANDS, include_lowest=True).astype(str)
    cells = []
    for second in ("customer_channel", "send_speed"):
        for key, g in predictions.assign(age_band=bands).groupby(["workflow_type", second, "age_band"], observed=True):
            wins = g["won"].sum()
            if len(g) < min_rows or wins in (0, len(g)):
                continue
            residual = g["won"] - g["p_now"]
            se = np.sqrt((residual.groupby(g["quote_id"]).sum() ** 2).sum()) / len(g)
            cells.append(
                {
                    "workflow_type": key[0],
                    "split": second,
                    "level": key[1],
                    "age_band": key[2],
                    "rows": len(g),
                    "quotes": g["quote_id"].nunique(),
                    "predicted": g["p_now"].mean(),
                    "observed": g["won"].mean(),
                    "gap": residual.mean(),
                    "z": residual.mean() / se,
                }
            )
    return pd.DataFrame(cells)


def subgroup_summary(cells: pd.DataFrame, alpha: float = 0.05) -> dict[str, float]:
    """How many groups miss by more than luck would explain, allowing for how many groups we checked."""
    threshold = float(norm.ppf(1 - alpha / (2 * len(cells))))
    return {
        "subgroup_cells": len(cells),
        "subgroup_z_threshold": threshold,
        "subgroup_cells_beyond": int((cells["z"].abs() > threshold).sum()),
        "subgroup_max_abs_z": float(cells["z"].abs().max()),
        "subgroup_max_abs_gap": float(cells["gap"].abs().max()),
    }


def prediction_quality(predictions: pd.DataFrame, bins: int = 10) -> dict[str, float]:
    """Discrimination and calibration of the win chance.

    Pooled AUC counts a quote once per waiting morning, which favours long-ignored losses; the
    first-morning and same-age AUCs do not."""
    y, p = predictions["won"].to_numpy(), predictions["p_now"].to_numpy()
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    bucket = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    gaps = [abs(p[bucket == b].mean() - y[bucket == b].mean()) * (bucket == b).mean() for b in range(bins)]
    first = predictions.sort_values("age_days").groupby("quote_id").head(1)
    bands = pd.cut(predictions["age_days"], AGE_BANDS, include_lowest=True)
    within_age = predictions.groupby(bands, observed=True).apply(
        lambda g: roc_auc_score(g["won"], g["p_now"]), include_groups=False
    )
    return {
        "quote_mornings": int(len(y)),
        "quotes": int(predictions["quote_id"].nunique()),
        "auc_pooled": roc_auc_score(y, p),
        "auc_pooled_age_only": roc_auc_score(y, -predictions["age_days"]),
        "auc_first_morning": roc_auc_score(first["won"], first["p_now"]),
        "auc_within_age_band_min": float(within_age.min()),
        "auc_within_age_band_max": float(within_age.max()),
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "expected_calibration_error": float(np.sum(gaps)),
    } | subgroup_summary(subgroup_calibration(predictions))


# The comparisons every report shows: (first, second, measure, number of calls).
COMPARISONS = [
    (OUR_LIST, NEWEST_FIRST, "dollar_capture", 1),
    (OUR_LIST, BIGGEST_FIRST, "dollar_capture", 1),
    (OUR_LIST, DAYS_QUIET_X_AMOUNT, "dollar_capture", 1),
    (OUR_LIST, DAYS_QUIET_X_AMOUNT, "dollar_capture", 3),
    (OUR_LIST, NEWEST_FIRST, "hit_rate", 1),
    (OUR_LIST_BY_CHANCE, NEWEST_FIRST, "hit_rate", 1),
    (OUR_LIST_BY_CHANCE, NEWEST_FIRST, "hit_rate", 3),
]


@dataclass(frozen=True)
class Evaluation:
    """Everything the replay produces, in one place, for the command line and the notebooks alike."""

    result: BacktestResult
    summary: pd.DataFrame
    comparisons: pd.DataFrame
    first_measure: pd.Series
    quality: pd.DataFrame
    subgroups: pd.DataFrame
    simple_subgroups: pd.DataFrame
    monthly: pd.DataFrame


def evaluate(quotes: pd.DataFrame, settings: Settings) -> Evaluation:
    """Replay every morning with every way of ordering, then score them all."""
    result = run(quotes, all_scorers(settings.model), settings)
    draws, seed = settings.backtest.bootstrap_draws, settings.backtest.seed
    one = result.lists[(result.lists["k"] == 1) & (result.lists["best_value"] > 0)]
    predictions = result.predictions
    simple = predictions.assign(p_now=predictions["p_age_only"])
    return Evaluation(
        result=result,
        summary=summarise(result.lists, OUR_LIST, draws, seed),
        comparisons=pd.DataFrame([paired_gap(result.lists, a, b, m, k, draws, seed) for a, b, m, k in COMPARISONS]),
        # The average of each morning's share: shown for contrast, since it weighs a $60 morning
        # the same as a $6,000 one.
        first_measure=(one["won_value"] / one["best_value"]).groupby(one["scorer"], sort=False).mean(),
        quality=pd.DataFrame(
            {"our model": prediction_quality(predictions), "days waiting only": prediction_quality(simple)}
        ),
        subgroups=subgroup_calibration(predictions),
        simple_subgroups=subgroup_calibration(simple),
        monthly=by_month(result.lists),
    )


PLAIN = {
    "scorer": "way of ordering",
    "k": "calls",
    "first": "this order",
    "second": "compared with",
    "metric": "measure",
    "gap": "difference",
    "lo": "low",
    "hi": "high",
    "dollar_capture": "dollars won",
    "dollar_capture_lo": "dollars low",
    "dollar_capture_hi": "dollars high",
    "hit_rate": "yes rate",
    "hit_rate_lo": "yes low",
    "hit_rate_hi": "yes high",
    "quote_mornings": "quote-mornings scored",
    "quotes": "different quotes",
    "auc_pooled": "AUC, every waiting morning counted",
    "auc_pooled_age_only": "AUC of days waiting alone",
    "auc_first_morning": "AUC, each quote counted once",
    "auc_within_age_band_min": "AUC among quotes the same age, lowest",
    "auc_within_age_band_max": "AUC among quotes the same age, highest",
    "log_loss": "log loss (lower is better)",
    "brier": "Brier score (lower is better)",
    "expected_calibration_error": "average gap between chance and reality",
    "subgroup_cells": "groups checked",
    "subgroup_z_threshold": "luck threshold (z)",
    "subgroup_cells_beyond": "groups off by more than luck",
    "subgroup_max_abs_z": "worst miss (z)",
    "subgroup_max_abs_gap": "worst miss (share of quotes)",
}


def for_people(table: pd.DataFrame) -> pd.DataFrame:
    """The same table with plain-language labels, for reading rather than for files."""
    return table.rename(columns=PLAIN, index=PLAIN).replace({"metric": PLAIN, "measure": PLAIN})


def strict_lead(summary: pd.DataFrame, rivals: list[str], k: int = 1) -> pd.DataFrame:
    """Our lead in dollars won over each rival, with the range widened for comparing many at once."""
    rows = summary[summary["k"] == k].set_index("scorer").loc[rivals]
    return pd.DataFrame(
        {
            "our lead": -rows["dollar_capture_gap"],
            "low (strict)": -rows["dollar_capture_gap_hi_adj"],
            "high (strict)": -rows["dollar_capture_gap_lo_adj"],
        }
    ).rename_axis("compared with")
