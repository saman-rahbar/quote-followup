"""A simple, explainable scoring model, used twice.

It's a logistic regression: each fact about a quote (type of work, how the customer found the
business, how fast the quote went out) adds or subtracts a fixed amount, and the total becomes a
chance between 0 and 1. We use it once for "will they say yes?" and once for "if they say no, will
they say it or just go quiet?" (timing.py explains why the second one matters).

A more complex model (gradient boosting) did worse when we tested it (notebook 04). This one also
lets us show exactly how much each fact moved a quote's chance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class LogisticScore:
    def __init__(self, categorical: tuple[str, ...], numeric: tuple[str, ...], c: float = 1.0):
        self.categorical = categorical
        self.numeric = numeric
        self._pipeline = Pipeline([
            ("encode", ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(categorical)),
                ("num", StandardScaler(), list(numeric)),
            ])),
            ("fit", LogisticRegression(C=c, max_iter=1000)),
        ])
        self._typical: np.ndarray | None = None

    @property
    def features(self) -> tuple[str, ...]:
        return self.categorical + self.numeric

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> LogisticScore:
        self._pipeline.fit(frame[list(self.features)], target.astype(int))
        # Reasons are measured against a typical quote, so a positive number means "better than
        # usual" and a negative one "worse than usual".
        self._typical = self._per_feature(frame).mean(axis=0)
        return self

    def log_odds(self, frame: pd.DataFrame) -> np.ndarray:
        return self._pipeline.decision_function(frame[list(self.features)])

    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        return expit(self.log_odds(frame))

    def baseline_log_odds(self) -> float:
        """The score of a typical quote, the starting point that `contributions` add to."""
        return float(self._pipeline[-1].intercept_[0] + self._typical.sum())

    def contributions(self, frame: pd.DataFrame) -> pd.DataFrame:
        """How much each fact moved this quote's score up or down. They add up to the full score."""
        return pd.DataFrame(self._per_feature(frame) - self._typical, columns=self.features, index=frame.index)

    def _per_feature(self, frame: pd.DataFrame) -> np.ndarray:
        encode, model = self._pipeline[0], self._pipeline[-1]
        terms = encode.transform(frame[list(self.features)]) * model.coef_[0]
        widths = [len(c) for c in encode.named_transformers_["cat"].categories_] + [1] * len(self.numeric)
        edges = np.cumsum([0, *widths])
        return np.column_stack([terms[:, a:b].sum(axis=1) for a, b in zip(edges[:-1], edges[1:])])
