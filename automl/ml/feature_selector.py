"""
feature_selector.py
-------------------
Feature selection logic for Classification, Regression, and Clustering.
No Flask imports — pure ML logic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
    VarianceThreshold,
)


# ---------------------------------------------------------------------------
# Correlation-based selection
# ---------------------------------------------------------------------------

def correlation_selection(
    X: pd.DataFrame, threshold: float = 0.95
) -> Tuple[List[str], List[str]]:
    """
    Remove features that are highly correlated with each other.
    Keeps one from each correlated pair.
    Returns (kept_features, dropped_features).
    """
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    kept = [c for c in X.columns if c not in to_drop]
    return kept, to_drop


# ---------------------------------------------------------------------------
# Mutual Information
# ---------------------------------------------------------------------------

def mutual_info_selection(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    top_k: int = None,
    threshold: float = 0.01,
) -> Tuple[List[str], List[float]]:
    """
    Select features by mutual information score.
    Returns (selected_features, scores_dict).
    """
    X_arr = X.values
    if problem_type == "classification":
        scores = mutual_info_classif(X_arr, y, random_state=42)
    else:
        scores = mutual_info_regression(X_arr, y, random_state=42)

    score_series = pd.Series(scores, index=X.columns).sort_values(ascending=False)

    if top_k:
        selected = score_series.head(top_k).index.tolist()
    else:
        selected = score_series[score_series >= threshold].index.tolist()

    return selected, score_series.to_dict()


# ---------------------------------------------------------------------------
# Feature Importance (quick RandomForest fit)
# ---------------------------------------------------------------------------

def importance_selection(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    top_k: int = None,
    threshold: float = 0.01,
) -> Tuple[List[str], List[float]]:
    """
    Select features by RandomForest feature importance.
    Returns (selected_features, importance_dict).
    """
    if problem_type == "classification":
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)

    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    if top_k:
        selected = importance.head(top_k).index.tolist()
    else:
        selected = importance[importance >= threshold].index.tolist()

    return selected, importance.to_dict()


# ---------------------------------------------------------------------------
# Variance Threshold (for Clustering)
# ---------------------------------------------------------------------------

def variance_selection(
    X: pd.DataFrame, threshold: float = 0.01
) -> Tuple[List[str], List[str]]:
    """
    Remove low-variance features. Good for clustering.
    Returns (kept_features, dropped_features).
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    kept = X.columns[selector.get_support()].tolist()
    dropped = [c for c in X.columns if c not in kept]
    return kept, dropped


# ---------------------------------------------------------------------------
# Summary builder for UI display
# ---------------------------------------------------------------------------

def build_feature_summary(
    X: pd.DataFrame, y: pd.Series = None, problem_type: str = "classification"
) -> List[dict]:
    """
    Build a per-feature summary for the UI:
    name, dtype, missing_count, unique_count, variance, importance_score (if supervised).
    """
    summary = []
    for col in X.columns:
        entry = {
            "name": col,
            "dtype": str(X[col].dtype),
            "missing": int(X[col].isnull().sum()),
            "unique": int(X[col].nunique()),
            "variance": round(float(X[col].var()) if X[col].dtype != "object" else 0, 4),
        }
        summary.append(entry)
    return summary
