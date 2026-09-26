"""Fit finalized URL phishing classifiers from engineered training data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from url_phishing_classification.config import FINAL_MODEL_SPEC, RANDOM_STATE, TARGET_COLUMN


def build_feature_matrix(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    """Return a validated numeric model matrix for one finalized feature set."""
    missing_features = sorted(set(features) - set(frame.columns))
    if missing_features:
        raise KeyError(f"Missing engineered features: {missing_features}")

    return frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def make_logistic_pipeline(C: float) -> Pipeline:
    """Create the finalized preprocessing and logistic-regression pipeline."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    max_iter=2_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def fit_final_model(frame: pd.DataFrame) -> Pipeline:
    """Fit the finalized all-engineered model from a labelled feature-engineered frame."""
    if TARGET_COLUMN not in frame:
        raise ValueError(f"Missing required target column: {TARGET_COLUMN!r}")

    model = make_logistic_pipeline(FINAL_MODEL_SPEC.C)
    model.fit(build_feature_matrix(frame, FINAL_MODEL_SPEC.features), frame[TARGET_COLUMN].astype(int))
    return model
