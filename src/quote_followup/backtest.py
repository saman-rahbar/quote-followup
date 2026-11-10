"""Replays past mornings to see which way of ordering quotes would have done best.

We go month by month from September to December 2024. On the first of each month every method
relearns from what was known by then, and then, every morning of that month, puts each business's
waiting quotes in order. Only after that do we look up which quotes were actually accepted. This
is the standard way to test a forecast on data it has never seen (Tashman 2000).

It tells us whether the list puts the quotes that went on to say yes at the top. It can't tell
us whether calling them changed anything, because the data has no record of follow-up calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .config import Settings
from .scorers import AgeCurveValue, FollowUpModel, Scorer
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
    out = pd.DataFrame({
        "hit_rate": picked["won"].mean(),
        "won_value": picked["won_value"].sum(),
        "best_value": best["won_value"].sum(),
        "n_open": c.groupby("sp_id").size(),
    })
    return out.assign(mornings=1)


def run(quotes: pd.DataFrame, scorers: list[Scorer], settings: Settings, ks: tuple[int, ...] = (1, 3)) -> BacktestResult:
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
    """Each method's scores, how far each is from `reference`, and how sure we can be.

    The ranges come from re-running the numbers many times on random re-draws of the businesses
    (a bootstrap). We re-draw whole businesses because one business's mornings are alike, and
    treating them as independent would make us look more certain than we are (Field and Welsh 2007).
    Every method sees the same re-draws, so the differences are like-for-like. The `_adj` ranges
    are widened because several methods are compared at once (a Bonferroni correction).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for k, at_k in lists.groupby("k"):
        sp_ids = np.sort(at_k["sp_id"].unique())
        picks = rng.integers(0, sp_ids.size, size=(draws, sp_ids.size))
        weights = np.stack([np.bincount(p, minlength=sp_ids.size) for p in picks])
        family = at_k["scorer"].nunique() - 1
        tail = 2.5 / max(family, 1)

        def boot(group: pd.DataFrame, metric: str) -> tuple[float, np.ndarray]:
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


def paired_gap(lists: pd.DataFrame, first: str, second: str, metric: str, k: int,
               draws: int, seed: int) -> dict[str, float]:
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
    sums = lists.assign(month=month).groupby(["k", "scorer", "month"], sort=False)[
        ["hit_rate", "mornings", "won_value", "best_value"]].sum()
    return pd.DataFrame({
        "hit_rate": sums["hit_rate"] / sums["mornings"],
        "dollar_capture": sums["won_value"] / sums["best_value"],
        "mornings": sums["mornings"],
    }).reset_index()


AGE_BANDS = (0, 1, 3, 7, 14, 30)


def subgroup_calibration(predictions: pd.DataFrame, min_rows: int = 100) -> pd.DataFrame:
    """Did the chances we gave come true, group by group?

    Being right on average can hide being too high for one kind of quote and too low for another.
    So we group quotes by type of work, by how the customer found the business (or how fast the
    quote went out), and by days waiting, then compare the chance we gave with what happened. The
    z column says how surprising each gap is. A quote shows up on many mornings, so its mornings
    are treated as one unit when judging that. Groups where every quote said yes, or every quote
    said no, are skipped, because the surprise measure breaks down there.
    """
    bands = pd.cut(predictions["age_days"], AGE_BANDS, include_lowest=True).astype(str)
    cells = []
    for second in ("customer_channel", "send_speed"):
        for key, g in predictions.assign(age_band=bands).groupby(["workflow_type", second, "age_band"], observed=True):
            wins = g["won"].sum()
            if len(g) < min_rows or wins in (0, len(g)):
                continue
            residual = g["won"] - g["p_now"]
            se = np.sqrt((residual.groupby(g["quote_id"]).sum() ** 2).sum()) / len(g)
            cells.append({"workflow_type": key[0], "split": second, "level": key[1], "age_band": key[2],
                          "rows": len(g), "quotes": g["quote_id"].nunique(), "predicted": g["p_now"].mean(),
                          "observed": g["won"].mean(), "gap": residual.mean(), "z": residual.mean() / se})
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
    """How good the chances are: do they separate yeses from nos, and do they come true?

    AUC is the chance that a random quote that said yes was given a higher chance than a random one
    that said no (0.5 is a coin flip, 1 is perfect). The pooled figures count a quote once per
    morning it waited, so a quote ignored for 20 days counts 20 times, mostly as an easy "no". The
    first-morning and same-age figures don't have that problem, so they are the fairer ones.
    """
    y, p = predictions["won"].to_numpy(), predictions["p_now"].to_numpy()
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    bucket = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    gaps = [abs(p[bucket == b].mean() - y[bucket == b].mean()) * (bucket == b).mean() for b in range(bins)]
    first = predictions.sort_values("age_days").groupby("quote_id").head(1)
    bands = pd.cut(predictions["age_days"], AGE_BANDS, include_lowest=True)
    within_age = predictions.groupby(bands, observed=True).apply(
        lambda g: roc_auc_score(g["won"], g["p_now"]), include_groups=False)
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
