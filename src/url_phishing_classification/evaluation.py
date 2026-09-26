"""Preprocessing, overlap checks, and metrics for held-out URL evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score

from url_phishing_classification.config import OPERATING_THRESHOLD, TARGET_COLUMN
from url_phishing_classification.preprocessing import deduplicate_urls, engineer_features, normalize_url


def preprocess_for_evaluation(
    frame: pd.DataFrame, url_column: str = "url", label_column: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, int]:
    """Apply training-time URL preprocessing to one held-out frame."""
    if url_column not in frame:
        raise ValueError(f"Missing required URL column: {url_column!r}")

    result = frame.copy()
    result["normalized_url"] = result[url_column].fillna("").astype(str).map(normalize_url)
    rows_before_deduplication = len(result)
    result = deduplicate_urls(result, label_column=label_column)
    return engineer_features(result, url_column=url_column), rows_before_deduplication


def _nonempty_strings(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column].fillna("").astype(str).str.strip()
    return values[values.ne("")]


def _add_overlap_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a host-path key for near-duplicate URL detection."""
    result = frame.copy()
    host = result["hostname"].fillna("").astype(str).str.strip()
    path = result["path"].fillna("").astype(str).str.strip()
    result["hostname_path"] = np.where(host.ne(""), host + path, "")
    return result


def overlap_summary(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize shared URL components between training and held-out data."""
    train_keys = _add_overlap_keys(train_frame)
    test_keys = _add_overlap_keys(test_frame)
    fields = [
        "normalized_url",
        "hostname_path",
        "hostname",
        "registered_domain",
        "domain",
        "subdomain",
        "suffix",
        "path",
        "query",
    ]
    records = []
    for field in fields:
        train_values = _nonempty_strings(train_keys, field)
        test_values = _nonempty_strings(test_keys, field)
        train_set, test_set = set(train_values), set(test_values)
        shared = train_set & test_set
        matching_rows = test_values.isin(shared).sum()
        records.append(
            {
                "overlap_key": field,
                "train_unique_values": len(train_set),
                "test_unique_values": len(test_set),
                "shared_unique_values": len(shared),
                "test_unique_overlap_rate": len(shared) / len(test_set) if test_set else np.nan,
                "test_rows_with_train_match": matching_rows,
                "test_row_overlap_rate": matching_rows / len(test_frame) if len(test_frame) else np.nan,
            }
        )
    return pd.DataFrame(records)


def evaluate_probabilities(
    y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float = OPERATING_THRESHOLD
) -> dict[str, float | int]:
    """Calculate ranking and classification metrics at a given threshold."""
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    false_positive_rate = fp / (fp + tn) if fp + tn else np.nan
    false_negative_rate = fn / (fn + tp) if fn + tp else np.nan
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "false_positive_rate": float(false_positive_rate) if not np.isnan(false_positive_rate) else np.nan,
        "false_negative_rate": float(false_negative_rate) if not np.isnan(false_negative_rate) else np.nan,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
