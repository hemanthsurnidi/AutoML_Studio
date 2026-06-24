"""
blueprints/predict.py
---------------------
Step 8: Manual prediction form + batch CSV prediction.
"""
import os
import io
import csv
import joblib
import pandas as pd
import numpy as np

from flask import (
    Blueprint, render_template, request, redirect, url_for,
    session, flash, send_file, make_response
)
from automl.utils.session_store import load_state, update_state
from automl.utils.validators import validate_csv_upload

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict")
def predict():
    sid = session.get("sid")
    if not sid:
        flash("Session expired.", "error")
        return redirect(url_for("upload.index"))

    state = load_state(sid)
    if not state.get("model_path"):
        flash("No trained model found. Please train a model first.", "error")
        return redirect(url_for("results.show_results"))

    bundle = _load_bundle(state["model_path"])
    if not bundle:
        flash("Could not load model.", "error")
        return redirect(url_for("results.show_results"))

    return render_template(
        "predict.html",
        features=bundle["feature_names"],
        target=state.get("target", "prediction"),
        problem_type=state.get("problem_type", "classification"),
        manual_result=None,
        batch_results=None,
    )


@predict_bp.route("/predict/manual", methods=["POST"])
def manual_predict():
    sid = session.get("sid")
    state = load_state(sid)
    bundle = _load_bundle(state.get("model_path", ""))
    if not bundle:
        flash("Model not found.", "error")
        return redirect(url_for("predict.predict"))

    feature_names = bundle["feature_names"]
    model = bundle["model"]
    ct = bundle["column_transformer"]
    le = bundle.get("label_encoder")

    # Build input row
    try:
        raw_inputs = {}
        for f in feature_names:
            val = request.form.get(f, "0")
            try:
                raw_inputs[f] = float(val)
            except ValueError:
                raw_inputs[f] = val

        input_df = pd.DataFrame([raw_inputs])
        prediction_raw = model.predict(input_df.values)[0]

        if le is not None:
            try:
                prediction = le.inverse_transform([int(prediction_raw)])[0]
            except Exception:
                prediction = prediction_raw
        else:
            prediction = prediction_raw

        # Probability if available
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba_vals = model.predict_proba(input_df.values)[0]
                classes = le.classes_.tolist() if le else list(range(len(proba_vals)))
                proba = [
                    {"class": str(c), "prob": round(float(p) * 100, 2)}
                    for c, p in zip(classes, proba_vals)
                ]
            except Exception:
                proba = None

        return render_template(
            "predict.html",
            features=feature_names,
            target=state.get("target", "prediction"),
            problem_type=state.get("problem_type", "classification"),
            manual_result={
                "prediction": str(prediction),
                "inputs": raw_inputs,
                "proba": proba,
            },
            batch_results=None,
        )
    except Exception as e:
        flash(f"Prediction failed: {e}", "error")
        return redirect(url_for("predict.predict"))


@predict_bp.route("/predict/batch", methods=["POST"])
def batch_predict():
    sid = session.get("sid")
    state = load_state(sid)
    bundle = _load_bundle(state.get("model_path", ""))
    if not bundle:
        flash("Model not found.", "error")
        return redirect(url_for("predict.predict"))

    file = request.files.get("batch_file")
    valid, err = validate_csv_upload(file)
    if not valid:
        flash(err, "error")
        return redirect(url_for("predict.predict"))

    try:
        batch_df = pd.read_csv(file)
    except Exception as e:
        flash(f"Could not read batch CSV: {e}", "error")
        return redirect(url_for("predict.predict"))

    feature_names = bundle["feature_names"]
    model = bundle["model"]
    le = bundle.get("label_encoder")

    # Align columns
    missing_cols = [f for f in feature_names if f not in batch_df.columns]
    if missing_cols:
        flash(f"Batch CSV is missing columns: {missing_cols}", "error")
        return redirect(url_for("predict.predict"))

    X_batch = batch_df[feature_names].fillna(0).values
    raw_preds = model.predict(X_batch)

    if le is not None:
        try:
            preds = le.inverse_transform(raw_preds.astype(int))
        except Exception:
            preds = raw_preds
    else:
        preds = raw_preds

    result_df = batch_df.copy()
    target_col = state.get("target", "prediction")
    result_df[f"{target_col}_predicted"] = preds

    # Build CSV in memory for download
    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    csv_bytes = io.BytesIO(csv_buffer.getvalue().encode("utf-8"))

    # Store for display
    preview = result_df.head(20).fillna("").astype(str).to_dict(orient="records")
    preview_cols = result_df.columns.tolist()

    update_state(sid, {
        "batch_csv_data": csv_buffer.getvalue(),
        "batch_preview": preview,
        "batch_preview_cols": preview_cols,
    })

    return render_template(
        "predict.html",
        features=feature_names,
        target=target_col,
        problem_type=state.get("problem_type", "classification"),
        manual_result=None,
        batch_results={
            "preview": preview,
            "preview_cols": preview_cols,
            "total": len(result_df),
        },
    )


@predict_bp.route("/predict/download_batch")
def download_batch():
    sid = session.get("sid")
    state = load_state(sid)
    csv_data = state.get("batch_csv_data", "")
    if not csv_data:
        flash("No batch predictions to download.", "error")
        return redirect(url_for("predict.predict"))

    response = make_response(csv_data)
    response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
    response.headers["Content-Type"] = "text/csv"
    return response


def _load_bundle(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None
