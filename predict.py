"""Predict phishing risk for one raw URL using the saved model artifact."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from url_phishing_classification.config import FINAL_MODEL_SPEC, OPERATING_THRESHOLD
from url_phishing_classification.preprocessing import engineer_features
from url_phishing_classification.training import build_feature_matrix

MODEL_PATH = Path(__file__).resolve().parent / "models" / FINAL_MODEL_SPEC.artifact_name


@lru_cache(maxsize=1)
def _load_model():
    """Load the trusted, locally saved model once, on the first prediction."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}. Run train.py first.")
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "feature_names_in_") or not hasattr(model, "classes_"):
        raise TypeError(f"Model artifact has no expected feature schema or classes: {MODEL_PATH}")
    if 1 not in model.classes_:
        raise ValueError(f"Model artifact has no phishing class (1): {MODEL_PATH}")
    return model


def predict_probabilities(urls: Sequence[str]) -> list[float]:
    """Score raw URLs in order using the same path as single-URL prediction."""
    values = list(urls)
    if any(not isinstance(url, str) for url in values):
        raise TypeError("Every URL must be a string.")
    if not values:
        return []

    model = _load_model()
    engineered = engineer_features(pd.DataFrame({"url": values}))
    features = build_feature_matrix(engineered, tuple(model.feature_names_in_))
    phishing_column = list(model.classes_).index(1)
    probabilities = model.predict_proba(features)[:, phishing_column]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("The saved model produced an invalid phishing probability.")
    return [float(probability) for probability in probabilities]


def predict(url: str) -> dict[str, str | float]:
    """Return a phishing/clean label and phishing probability for one raw URL.

    Empty or malformed strings are scored through the normal feature pipeline.
    Non-string inputs are rejected with TypeError.
    """
    probability = predict_probabilities([url])[0]
    return {
        "label": "phishing" if probability >= OPERATING_THRESHOLD else "clean",
        "probability": probability,
    }
