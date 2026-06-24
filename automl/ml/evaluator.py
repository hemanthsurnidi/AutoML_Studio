"""
evaluator.py
------------
Prepares chart data and metric summaries for the results dashboard.
Outputs Chart.js-compatible JSON dicts — no Matplotlib, no server-side images.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Correlation heatmap data
# ---------------------------------------------------------------------------

def correlation_heatmap_data(df: pd.DataFrame, max_cols: int = 20) -> Dict:
    """Compute correlation matrix and return Chart.js-compatible data."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] > max_cols:
        num_df = num_df.iloc[:, :max_cols]

    corr = num_df.corr().round(3)
    cols = corr.columns.tolist()
    values = []
    for i, row_label in enumerate(cols):
        for j, col_label in enumerate(cols):
            values.append({
                "x": col_label,
                "y": row_label,
                "v": float(corr.iloc[i, j]),
            })
    return {"labels": cols, "values": values}


# ---------------------------------------------------------------------------
# Missing value heatmap data
# ---------------------------------------------------------------------------

def missing_heatmap_data(df: pd.DataFrame) -> List[Dict]:
    """Per-column missing value stats for heatmap display."""
    total = len(df)
    result = []
    for col in df.columns:
        missing = int(df[col].isnull().sum())
        result.append({
            "column": col,
            "missing": missing,
            "present": total - missing,
            "missing_pct": round(missing / total * 100, 2) if total > 0 else 0,
        })
    return result


# ---------------------------------------------------------------------------
# Distribution data (histogram per column)
# ---------------------------------------------------------------------------

def distribution_data(df: pd.DataFrame, max_cols: int = 10) -> List[Dict]:
    """Generate histogram data for numeric columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    result = []
    for col in num_cols:
        clean = df[col].dropna()
        if len(clean) == 0:
            continue
        counts, bin_edges = np.histogram(clean, bins=20)
        labels = [f"{round(bin_edges[i], 2)}" for i in range(len(bin_edges) - 1)]
        result.append({
            "column": col,
            "labels": labels,
            "counts": counts.tolist(),
        })
    return result


# ---------------------------------------------------------------------------
# Confusion matrix formatter
# ---------------------------------------------------------------------------

def format_confusion_matrix(cm: List[List[int]], classes: List[str]) -> Dict:
    """Format confusion matrix for heatmap rendering."""
    return {
        "matrix": cm,
        "classes": [str(c) for c in classes],
    }


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------

def classification_comparison_table(results: List[Dict]) -> List[Dict]:
    """Return a sorted, serializable list of classification results."""
    rows = []
    for r in results:
        rows.append({
            "model": r["model"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "roc_auc": r["roc_auc"],
            "train_time": r["train_time"],
            "cv_mean": r.get("cv_mean"),
            "cv_std": r.get("cv_std"),
        })
    return rows


def regression_comparison_table(results: List[Dict]) -> List[Dict]:
    rows = []
    for r in results:
        rows.append({
            "model": r["model"],
            "mae": r["mae"],
            "mse": r["mse"],
            "rmse": r["rmse"],
            "r2": r["r2"],
            "train_time": r["train_time"],
            "cv_mean": r.get("cv_mean"),
            "cv_std": r.get("cv_std"),
        })
    return rows


def clustering_comparison_table(results: List[Dict]) -> List[Dict]:
    rows = []
    for r in results:
        if "error" in r:
            rows.append({"model": r["model"], "error": r["error"]})
            continue
        rows.append({
            "model": r["model"],
            "silhouette": r["silhouette"],
            "davies_bouldin": r["davies_bouldin"],
            "n_clusters_found": r["n_clusters_found"],
            "cluster_distribution": r["cluster_distribution"],
            "train_time": r["train_time"],
        })
    return rows


# ---------------------------------------------------------------------------
# Feature importance chart data
# ---------------------------------------------------------------------------

def feature_importance_chart(importance_list: List[Dict], top_n: int = 15) -> Dict:
    """Return Chart.js horizontal bar chart data for feature importance."""
    top = importance_list[:top_n]
    return {
        "labels": [r["feature"] for r in top],
        "values": [r["contribution"] for r in top],
    }
