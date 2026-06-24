"""
blueprints/training.py
----------------------
Step 6: Preprocessing, feature selection, model training, results storage.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd

from flask import Blueprint, render_template, redirect, url_for, session, flash
from automl.utils.session_store import load_state, update_state
from automl.ml.preprocessor import (
    PreprocessConfig, ColumnPreprocessConfig,
    apply_global_preprocessing, build_column_transformer,
    get_output_feature_names,
)
from automl.ml.feature_selector import (
    correlation_selection, mutual_info_selection,
    importance_selection, variance_selection,
)
from automl.ml.trainer import (
    train_classification, train_regression, train_clustering,
    make_split, get_feature_importance,
    get_classification_models, get_regression_models, get_clustering_models,
)
from automl.ml.evaluator import (
    classification_comparison_table, regression_comparison_table,
    clustering_comparison_table, feature_importance_chart,
    correlation_heatmap_data, missing_heatmap_data, distribution_data,
)
from config import Config

training_bp = Blueprint("training", __name__)


@training_bp.route("/train")
def train():
    sid = session.get("sid")
    if not sid:
        flash("Session expired. Please start again.", "error")
        return redirect(url_for("upload.index"))

    state = load_state(sid)
    if not state or "problem_type" not in state:
        flash("Configuration not found. Please configure your training.", "error")
        return redirect(url_for("configure.configure"))

    try:
        df_raw = pd.read_csv(state["csv_path"])
    except Exception as e:
        flash(f"Could not load dataset: {e}", "error")
        return redirect(url_for("upload.index"))

    problem_type = state["problem_type"]
    target = state.get("target", "")
    col_configs = state.get("col_configs", {})
    global_opts = state.get("global_opts", {})
    selected_models = state.get("selected_models", [])
    n_clusters = state.get("n_clusters", 3)
    test_size = state.get("test_size", 0.2)
    cv_folds = state.get("cv_folds", 0)
    stratify = state.get("stratify", False)
    feature_selection_mode = state.get("feature_selection_mode", "all")
    feature_selection_method = state.get("feature_selection_method", "none")
    top_k = state.get("top_k_features")
    selected_features_manual = state.get("selected_features_manual", [])

    # --- 1. Separate target ---
    if problem_type != "clustering" and target:
        if target not in df_raw.columns:
            flash(f"Target column '{target}' not found in dataset.", "error")
            return redirect(url_for("configure.configure"))
        df_clean = df_raw.dropna(subset=[target])
        if len(df_clean) < len(df_raw):
            global_steps_log_init = [f"Dropped {len(df_raw) - len(df_clean)} rows with missing target values"]
        else:
            global_steps_log_init = []
        y = df_clean[target].copy()
        df = df_clean.drop(columns=[target])
    else:
        y = None
        df = df_raw.copy()
        global_steps_log_init = []

    # --- 2. Build PreprocessConfig ---
    preprocess_config = PreprocessConfig(
        remove_duplicates=global_opts.get("remove_duplicates", True),
        remove_constant=global_opts.get("remove_constant", True),
        remove_correlated=global_opts.get("remove_correlated", False),
        correlation_threshold=global_opts.get("correlation_threshold", 0.95),
        remove_low_variance=global_opts.get("remove_low_variance", False),
        variance_threshold=global_opts.get("variance_threshold", 0.01),
        extract_datetime=global_opts.get("extract_datetime", True),
    )

    for col, cfg in col_configs.items():
        if col in df.columns:
            preprocess_config.columns[col] = ColumnPreprocessConfig(**cfg)

    # Ensure any unconfirmed columns get a default config
    from automl.ml.preprocessor import infer_column_types
    auto_types = infer_column_types(df)
    for col in df.columns:
        if col not in preprocess_config.columns:
            ctype = auto_types.get(col, "numerical")
            preprocess_config.columns[col] = ColumnPreprocessConfig(col_type=ctype)

    # --- 3. Global preprocessing (dedup, constant removal, datetime) ---
    df, global_steps_log = apply_global_preprocessing(df, preprocess_config)
    global_steps_log = global_steps_log_init + global_steps_log

    # Align y with remaining rows in df
    if y is not None:
        y = y.loc[df.index]
        if len(y) == 0:
            flash("No samples remaining after preprocessing.", "error")
            return redirect(url_for("configure.configure"))

    # Store chart data BEFORE preprocessing for missing/distribution views
    missing_data = missing_heatmap_data(df_raw.drop(columns=[target]) if target else df_raw)
    dist_data = distribution_data(df_raw.drop(columns=[target]) if target else df_raw)

    # --- 4. Build ColumnTransformer and fit_transform ---
    # Update config with remaining columns
    remaining_cols = df.columns.tolist()
    for col in list(preprocess_config.columns.keys()):
        if col not in remaining_cols:
            del preprocess_config.columns[col]

    ct = build_column_transformer(df, preprocess_config)
    try:
        X_processed = ct.fit_transform(df)
    except Exception as e:
        flash(f"Preprocessing failed: {e}", "error")
        return redirect(url_for("configure.configure"))

    # Get output feature names
    try:
        feature_names = list(ct.get_feature_names_out())
        # Clean up prefixes added by ColumnTransformer
        feature_names = [
            n.split("__")[-1] if "__" in n else n for n in feature_names
        ]
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]

    # --- 5. Feature Selection ---
    X_df = pd.DataFrame(X_processed, columns=feature_names)

    if feature_selection_mode == "manual" and selected_features_manual:
        valid = [f for f in selected_features_manual if f in feature_names]
        if valid:
            X_df = X_df[valid]
            feature_names = valid

    elif feature_selection_mode == "auto" and y is not None:
        if feature_selection_method == "correlation":
            kept, dropped = correlation_selection(X_df, threshold=0.95)
            X_df = X_df[kept]
            feature_names = kept
        elif feature_selection_method == "mutual_info":
            kept, _ = mutual_info_selection(X_df, y, problem_type, top_k=top_k)
            X_df = X_df[kept]
            feature_names = kept
        elif feature_selection_method == "importance":
            kept, _ = importance_selection(X_df, y, problem_type, top_k=top_k)
            X_df = X_df[kept]
            feature_names = kept

    elif feature_selection_mode == "auto" and y is None:
        # Clustering
        if feature_selection_method == "variance":
            kept, _ = variance_selection(X_df, threshold=0.01)
            X_df = X_df[kept]
            feature_names = kept
        elif feature_selection_method == "correlation":
            kept, _ = correlation_selection(X_df, threshold=0.95)
            X_df = X_df[kept]
            feature_names = kept

    X_final = X_df.values

    # --- 6. Corr heatmap on processed features ---
    corr_data = correlation_heatmap_data(X_df)

    # --- 7. Default model selection ---
    if not selected_models:
        if problem_type == "classification":
            selected_models = list(get_classification_models().keys())
        elif problem_type == "regression":
            selected_models = list(get_regression_models().keys())
        else:
            selected_models = list(get_clustering_models().keys())

    # --- 8. Train ---
    trained_results = []
    feature_importance_data = []

    if problem_type == "classification":
        if y is None:
            flash("Target column required for classification.", "error")
            return redirect(url_for("configure.configure"))
        X_train, X_test, y_train, y_test = make_split(X_final, y, test_size, stratify)
        raw_results = train_classification(X_train, X_test, y_train, y_test, selected_models, cv=cv_folds)
        trained_results = classification_comparison_table(raw_results)

        # Feature importance from best model
        if raw_results:
            best = raw_results[0]
            fi = get_feature_importance(best["trained_model"], feature_names)
            feature_importance_data = fi
            fi_chart = feature_importance_chart(fi)
            # Save best model
            _save_bundle(sid, best["trained_model"], ct, feature_names, target, problem_type,
                         label_encoder=best.get("label_encoder"))
        else:
            fi_chart = {"labels": [], "values": []}

        # Confusion matrix for best model
        confusion_matrices = []
        for r in raw_results:
            confusion_matrices.append({
                "model": r["model"],
                "matrix": r.get("confusion_matrix", []),
                "classes": r.get("classes", []),
            })
        update_state(sid, {"confusion_matrices": confusion_matrices})

    elif problem_type == "regression":
        if y is None:
            flash("Target column required for regression.", "error")
            return redirect(url_for("configure.configure"))
        y_num = pd.to_numeric(y, errors="coerce").fillna(y.median() if hasattr(y, "median") else 0)
        X_train, X_test, y_train, y_test = make_split(X_final, y_num, test_size)
        raw_results = train_regression(X_train, X_test, y_train, y_test, selected_models, cv=cv_folds)
        trained_results = regression_comparison_table(raw_results)

        if raw_results:
            best = raw_results[0]
            fi = get_feature_importance(best["trained_model"], feature_names)
            feature_importance_data = fi
            fi_chart = feature_importance_chart(fi)
            _save_bundle(sid, best["trained_model"], ct, feature_names, target, problem_type)
        else:
            fi_chart = {"labels": [], "values": []}

    else:  # clustering
        raw_results = train_clustering(X_final, selected_models, n_clusters)
        trained_results = clustering_comparison_table(raw_results)
        fi_chart = {"labels": [], "values": []}

        if raw_results:
            best = raw_results[0]
            _save_bundle(sid, best.get("trained_model"), ct, feature_names, target, problem_type)

    # --- 9. Save results to session ---
    update_state(sid, {
        "trained_results": trained_results,
        "feature_importance": feature_importance_data,
        "fi_chart": fi_chart,
        "corr_data": corr_data,
        "missing_data": missing_data,
        "dist_data": dist_data,
        "feature_names": feature_names,
        "global_steps_log": global_steps_log,
        "training_done": True,
    })

    return redirect(url_for("results.show_results"))


def _save_bundle(sid, model, ct, feature_names, target, problem_type, label_encoder=None):
    """Save trained model + pipeline to disk."""
    bundle = {
        "model": model,
        "column_transformer": ct,
        "feature_names": feature_names,
        "target": target,
        "problem_type": problem_type,
        "label_encoder": label_encoder,
    }
    path = os.path.join(Config.MODEL_FOLDER, f"{sid}_model.pkl")
    joblib.dump(bundle, path)
    update_state(sid, {"model_path": path})
