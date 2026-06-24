"""
blueprints/configure.py
-----------------------
Steps 2–5: Problem type, target selection, preprocessing config,
feature selection, train/test configuration.
"""
import json
import pandas as pd
import numpy as np

from flask import (
    Blueprint, render_template, request, redirect, url_for, session, flash
)
from automl.utils.session_store import load_state, update_state
from automl.ml.preprocessor import infer_column_types, ColumnPreprocessConfig, PreprocessConfig
from automl.ml.trainer import get_classification_models, get_regression_models, get_clustering_models

configure_bp = Blueprint("configure", __name__)


@configure_bp.route("/configure", methods=["GET"])
def configure():
    sid = session.get("sid")
    if not sid:
        flash("Session expired. Please upload a dataset first.", "error")
        return redirect(url_for("upload.index"))

    state = load_state(sid)
    if not state:
        flash("Session data not found. Please upload a dataset again.", "error")
        return redirect(url_for("upload.index"))

    # Load dataset
    try:
        df = pd.read_csv(state["csv_path"])
    except Exception:
        flash("Could not reload dataset. Please upload again.", "error")
        return redirect(url_for("upload.index"))

    col_types = state.get("col_types", infer_column_types(df))
    columns = df.columns.tolist()

    # Build per-column stats for UI
    col_stats = []
    for col in columns:
        missing = int(df[col].isnull().sum())
        unique = int(df[col].nunique())
        col_stats.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "inferred_type": col_types.get(col, "numerical"),
            "missing": missing,
            "unique": unique,
        })

    classification_models = list(get_classification_models().keys())
    regression_models = list(get_regression_models().keys())
    clustering_models = list(get_clustering_models().keys())

    return render_template(
        "configure.html",
        columns=columns,
        col_stats=col_stats,
        col_types=col_types,
        classification_models=classification_models,
        regression_models=regression_models,
        clustering_models=clustering_models,
        rows=state.get("rows", 0),
        cols=state.get("cols", 0),
        filename=state.get("original_filename", "dataset.csv"),
    )


@configure_bp.route("/configure", methods=["POST"])
def save_configure():
    sid = session.get("sid")
    if not sid:
        flash("Session expired.", "error")
        return redirect(url_for("upload.index"))

    state = load_state(sid)
    form = request.form

    problem_type = form.get("problem_type", "classification")
    target = form.get("target_column", "") if problem_type != "clustering" else ""

    # Collect column preprocessing configs
    col_configs = {}
    try:
        df = pd.read_csv(state["csv_path"])
    except Exception:
        flash("Could not reload dataset.", "error")
        return redirect(url_for("upload.index"))

    for col in df.columns:
        if col == target:
            continue
        col_type = form.get(f"col_type_{col}", "ignore")
        cfg = {
            "col_type": col_type,
            "missing_num": form.get(f"missing_num_{col}", "median"),
            "scaling": form.get(f"scaling_{col}", "standard"),
            "transform": form.get(f"transform_{col}", "none"),
            "outlier": form.get(f"outlier_{col}", "none"),
            "missing_cat": form.get(f"missing_cat_{col}", "mode"),
            "encoding": form.get(f"encoding_{col}", "onehot"),
        }
        col_configs[col] = cfg

    # Global preprocessing options
    global_opts = {
        "remove_duplicates": "remove_duplicates" in form,
        "remove_constant": "remove_constant" in form,
        "remove_correlated": "remove_correlated" in form,
        "correlation_threshold": float(form.get("correlation_threshold", 0.95)),
        "remove_low_variance": "remove_low_variance" in form,
        "variance_threshold": float(form.get("variance_threshold", 0.01)),
        "extract_datetime": "extract_datetime" in form,
    }

    # Feature selection
    feature_selection_mode = form.get("feature_selection_mode", "all")
    selected_features_manual = form.getlist("selected_features")
    feature_selection_method = form.get("feature_selection_method", "none")
    top_k_features = form.get("top_k_features", "")
    top_k = int(top_k_features) if top_k_features.isdigit() else None

    # Train/test split config
    test_size = float(form.get("test_size", 0.2))
    cv_folds = int(form.get("cv_folds", 0))
    stratify = "stratify" in form

    # Model selection
    selected_models = form.getlist("selected_models")

    # Clustering config
    n_clusters = int(form.get("n_clusters", 3))

    update_state(sid, {
        "problem_type": problem_type,
        "target": target,
        "col_configs": col_configs,
        "global_opts": global_opts,
        "feature_selection_mode": feature_selection_mode,
        "selected_features_manual": selected_features_manual,
        "feature_selection_method": feature_selection_method,
        "top_k_features": top_k,
        "test_size": test_size,
        "cv_folds": cv_folds,
        "stratify": stratify,
        "selected_models": selected_models,
        "n_clusters": n_clusters,
    })

    return redirect(url_for("training.train"))
