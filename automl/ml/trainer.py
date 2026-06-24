"""
trainer.py
----------
Model registry + training logic for Classification, Regression, Clustering.
No Flask imports — pure ML logic.
"""
from __future__ import annotations

import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score,
)
from sklearn.preprocessing import LabelEncoder

# XGBoost with graceful fallback
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBClassifier = GradientBoostingClassifier
    XGBRegressor = GradientBoostingRegressor


# ---------------------------------------------------------------------------
# Model Registries
# ---------------------------------------------------------------------------

def get_classification_models() -> Dict[str, Any]:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True, random_state=42, max_iter=2000, cache_size=200),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=50, max_depth=5, random_state=42,
            eval_metric="logloss", verbosity=0, tree_method="hist"
        )
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)
    return models


def get_regression_models() -> Dict[str, Any]:
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0, max_iter=2000),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost Regressor"] = XGBRegressor(
            n_estimators=50, max_depth=5, random_state=42,
            verbosity=0, tree_method="hist"
        )
    else:
        models["Gradient Boosting Regressor"] = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
    return models


def get_clustering_models(n_clusters: int = 3) -> Dict[str, Any]:
    return {
        "KMeans": KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
        "Agglomerative Clustering": AgglomerativeClustering(n_clusters=n_clusters),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
        "Gaussian Mixture": GaussianMixture(n_components=n_clusters, random_state=42),
    }


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_classification(
    X_train, X_test, y_train, y_test,
    model_names: List[str],
    cv: int = 0,  # 0 = no CV, else number of folds
) -> List[Dict]:
    """
    Train selected classification models and compute metrics.
    Returns a list of result dicts sorted by F1 Score descending.
    """
    all_models = get_classification_models()
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    n_classes = len(le.classes_)
    multi = "macro" if n_classes > 2 else "binary"

    results = []
    for name in model_names:
        if name not in all_models:
            continue
        model = all_models[name]
        start = time.time()

        if cv > 0:
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            X_all = np.vstack([X_train, X_test])
            y_all = np.hstack([y_train_enc, y_test_enc])
            cv_scores = cross_val_score(model, X_all, y_all, cv=skf, scoring="f1_weighted", n_jobs=-1)
            model.fit(X_train, y_train_enc)
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
        else:
            model.fit(X_train, y_train_enc)
            cv_mean = None
            cv_std = None

        elapsed = round(time.time() - start, 2)
        preds = model.predict(X_test)

        try:
            proba = model.predict_proba(X_test)
            if n_classes == 2:
                roc = float(roc_auc_score(y_test_enc, proba[:, 1]))
            else:
                roc = float(roc_auc_score(y_test_enc, proba, multi_class="ovr", average="macro"))
        except Exception:
            roc = None

        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test_enc, preds).tolist()

        result = {
            "model": name,
            "accuracy": round(float(accuracy_score(y_test_enc, preds)), 4),
            "precision": round(float(precision_score(y_test_enc, preds, average=multi, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test_enc, preds, average=multi, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test_enc, preds, average=multi, zero_division=0)), 4),
            "roc_auc": round(roc, 4) if roc is not None else "N/A",
            "confusion_matrix": cm,
            "cv_mean": round(cv_mean, 4) if cv_mean is not None else None,
            "cv_std": round(cv_std, 4) if cv_std is not None else None,
            "train_time": elapsed,
            "trained_model": model,
            "label_encoder": le,
            "classes": le.classes_.tolist(),
        }
        results.append(result)

    results.sort(key=lambda r: r["f1"], reverse=True)
    return results


def train_regression(
    X_train, X_test, y_train, y_test,
    model_names: List[str],
    cv: int = 0,
) -> List[Dict]:
    """
    Train selected regression models and compute metrics.
    Returns a list of result dicts sorted by R² descending.
    """
    all_models = get_regression_models()
    results = []

    for name in model_names:
        if name not in all_models:
            continue
        model = all_models[name]
        start = time.time()

        if cv > 0:
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            X_all = np.vstack([X_train, X_test])
            y_all = np.hstack([y_train, y_test])
            cv_scores = cross_val_score(model, X_all, y_all, cv=kf, scoring="r2", n_jobs=-1)
            model.fit(X_train, y_train)
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
        else:
            model.fit(X_train, y_train)
            cv_mean = None
            cv_std = None

        elapsed = round(time.time() - start, 2)
        preds = model.predict(X_test)

        result = {
            "model": name,
            "mae": round(float(mean_absolute_error(y_test, preds)), 4),
            "mse": round(float(mean_squared_error(y_test, preds)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
            "r2": round(float(r2_score(y_test, preds)), 4),
            "cv_mean": round(cv_mean, 4) if cv_mean is not None else None,
            "cv_std": round(cv_std, 4) if cv_std is not None else None,
            "train_time": elapsed,
            "trained_model": model,
        }
        results.append(result)

    results.sort(key=lambda r: r["r2"], reverse=True)
    return results


def train_clustering(
    X: np.ndarray,
    model_names: List[str],
    n_clusters: int = 3,
) -> List[Dict]:
    """
    Train selected clustering models and compute metrics.
    Returns a list of result dicts.
    """
    all_models = get_clustering_models(n_clusters)
    results = []

    for name in model_names:
        if name not in all_models:
            continue
        model = all_models[name]
        start = time.time()

        try:
            if hasattr(model, "fit_predict"):
                labels = model.fit_predict(X)
            else:
                model.fit(X)
                labels = model.predict(X)
        except Exception as e:
            results.append({"model": name, "error": str(e)})
            continue

        elapsed = round(time.time() - start, 2)

        unique_labels = set(labels)
        if len(unique_labels) < 2 or (len(unique_labels) == 1 and -1 in unique_labels):
            sil = None
            db = None
        else:
            mask = labels != -1
            try:
                sil = round(float(silhouette_score(X[mask], labels[mask])), 4)
                db = round(float(davies_bouldin_score(X[mask], labels[mask])), 4)
            except Exception:
                sil = None
                db = None

        # Cluster distribution
        from collections import Counter
        dist = dict(Counter(labels.tolist()))

        result = {
            "model": name,
            "silhouette": sil,
            "davies_bouldin": db,
            "n_clusters_found": len(unique_labels),
            "cluster_distribution": {str(k): v for k, v in dist.items()},
            "train_time": elapsed,
            "trained_model": model,
            "labels": labels.tolist(),
        }
        results.append(result)

    if results:
        results.sort(
            key=lambda r: r.get("silhouette") or -999,
            reverse=True
        )
    return results


# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------

def get_feature_importance(model, feature_names: List[str]) -> List[Dict]:
    """Extract feature importances from a trained model."""
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)

    if importances is None or len(importances) != len(feature_names):
        return []

    total = importances.sum()
    if total == 0:
        total = 1

    result = []
    for fname, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        result.append({
            "feature": fname,
            "importance": round(float(imp), 6),
            "contribution": round(float(imp / total) * 100, 2),
        })
    return result


# ---------------------------------------------------------------------------
# Train/test split helper
# ---------------------------------------------------------------------------

def make_split(
    X: np.ndarray, y,
    test_size: float = 0.2,
    stratify: bool = False,
):
    strat = y if stratify else None
    try:
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=strat)
    except Exception:
        return train_test_split(X, y, test_size=test_size, random_state=42)
