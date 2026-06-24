"""
blueprints/upload.py
--------------------
Step 1: CSV upload, validation, dataset summary, first-20-rows preview.
"""
import os
import uuid
import pandas as pd
import numpy as np

from flask import (
    Blueprint, render_template, request, redirect, url_for, session, flash
)
from werkzeug.utils import secure_filename

from automl.utils.validators import validate_csv_upload
from automl.utils.session_store import new_session, save_state, load_state
from automl.ml.preprocessor import infer_column_types
from config import Config

upload_bp = Blueprint("upload", __name__)


@upload_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@upload_bp.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("dataset")
    valid, err = validate_csv_upload(file)
    if not valid:
        flash(err, "error")
        return redirect(url_for("upload.index"))

    # Save with unique name
    original_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    save_path = os.path.join(Config.UPLOAD_FOLDER, unique_name)
    file.save(save_path)

    # Read CSV
    try:
        df = pd.read_csv(save_path)
    except Exception as e:
        flash(f"Could not read CSV: {e}", "error")
        return redirect(url_for("upload.index"))

    if df.empty:
        flash("The uploaded CSV file is empty.", "error")
        return redirect(url_for("upload.index"))

    # Build column summary
    col_types = infer_column_types(df)
    summary = []
    for col in df.columns:
        missing = int(df[col].isnull().sum())
        missing_pct = round(missing / len(df) * 100, 1) if len(df) > 0 else 0
        unique = int(df[col].nunique())
        sample_vals = df[col].dropna().head(3).tolist()
        summary.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "inferred_type": col_types.get(col, "unknown"),
            "missing": missing,
            "missing_pct": missing_pct,
            "unique": unique,
            "sample": [str(v) for v in sample_vals],
        })

    # First 20 rows as records (JSON-safe)
    preview_rows = df.head(20).fillna("").astype(str).to_dict(orient="records")
    preview_cols = df.columns.tolist()

    # Skip expensive duplicate calculation on upload to prevent timeouts
    duplicate_count = 0

    # Create new session
    sid = new_session()
    session["sid"] = sid
    save_state(sid, {
        "csv_path": save_path,
        "original_filename": original_name,
        "rows": len(df),
        "cols": len(df.columns),
        "columns": df.columns.tolist(),
        "col_types": col_types,
        "duplicate_count": duplicate_count,
    })

    return redirect(url_for("upload.dataset_info"))


@upload_bp.route("/debug_upload", methods=["GET"])
def debug_upload():
    """Development-only: simulate an upload with sample.csv to enable browser testing."""
    import flask
    if not flask.current_app.debug:
        return "Not available in production", 403

    # Find sample.csv relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sample_path = os.path.join(project_root, "sample.csv")
    if not os.path.exists(sample_path):
        return f"sample.csv not found at {sample_path}", 404

    try:
        df = pd.read_csv(sample_path)
    except Exception as e:
        return f"Could not read sample.csv: {e}", 500

    col_types = infer_column_types(df)
    duplicate_count = 0

    sid = new_session()
    session["sid"] = sid
    save_state(sid, {
        "csv_path": sample_path,
        "original_filename": "sample.csv",
        "rows": len(df),
        "cols": len(df.columns),
        "columns": df.columns.tolist(),
        "col_types": col_types,
        "duplicate_count": duplicate_count,
    })
    return redirect(url_for("upload.dataset_info"))


@upload_bp.route("/dataset_info", methods=["GET"])
def dataset_info():
    sid = session.get("sid")
    if not sid:
        flash("Session expired. Please upload a dataset first.", "error")
        return redirect(url_for("upload.index"))

    state = load_state(sid)
    if not state or "csv_path" not in state:
        flash("Session data not found. Please upload a dataset again.", "error")
        return redirect(url_for("upload.index"))

    try:
        df = pd.read_csv(state["csv_path"])
    except Exception as e:
        flash(f"Could not load dataset: {e}", "error")
        return redirect(url_for("upload.index"))

    # Re-infer or load column types
    col_types = state.get("col_types", infer_column_types(df))
    summary = []
    for col in df.columns:
        missing = int(df[col].isnull().sum())
        missing_pct = round(missing / len(df) * 100, 1) if len(df) > 0 else 0
        unique = int(df[col].nunique())
        sample_vals = df[col].dropna().head(3).tolist()
        summary.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "inferred_type": col_types.get(col, "unknown"),
            "missing": missing,
            "missing_pct": missing_pct,
            "unique": unique,
            "sample": [str(v) for v in sample_vals],
        })

    preview_rows = df.head(20).fillna("").astype(str).to_dict(orient="records")
    preview_cols = df.columns.tolist()
    duplicate_count = state.get("duplicate_count", 0)

    return render_template(
        "dataset_info.html",
        rows=len(df),
        cols=len(df.columns),
        summary=summary,
        preview_cols=preview_cols,
        preview_rows=preview_rows,
        duplicate_count=duplicate_count,
        original_filename=state.get("original_filename", "dataset.csv"),
    )

