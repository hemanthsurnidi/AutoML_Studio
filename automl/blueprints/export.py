"""
blueprints/export.py
--------------------
Step 9: Export trained model, preprocessing pipeline, predictions CSV, PDF report.
"""
import os
import joblib
from flask import (
    Blueprint, render_template, redirect, url_for, session, flash, send_file, make_response
)
from automl.utils.session_store import load_state
from automl.ml.reporter import generate_pdf_report
from config import Config

export_bp = Blueprint("export", __name__)


@export_bp.route("/export")
def export():
    sid = session.get("sid")
    if not sid:
        flash("Session expired.", "error")
        return redirect(url_for("upload.index"))

    state = load_state(sid)
    has_model = bool(state.get("model_path") and os.path.exists(state.get("model_path", "")))
    has_batch = bool(state.get("batch_csv_data"))

    return render_template(
        "export.html",
        has_model=has_model,
        has_batch=has_batch,
        problem_type=state.get("problem_type", ""),
        target=state.get("target", ""),
        filename=state.get("original_filename", "dataset.csv"),
    )


@export_bp.route("/export/model")
def download_model():
    sid = session.get("sid")
    state = load_state(sid)
    path = state.get("model_path", "")

    if not path or not os.path.exists(path):
        flash("No trained model found.", "error")
        return redirect(url_for("export.export"))

    return send_file(path, as_attachment=True, download_name="automl_model.pkl")


@export_bp.route("/export/pipeline")
def download_pipeline():
    sid = session.get("sid")
    state = load_state(sid)
    model_path = state.get("model_path", "")

    if not model_path or not os.path.exists(model_path):
        flash("No pipeline found.", "error")
        return redirect(url_for("export.export"))

    try:
        bundle = joblib.load(model_path)
        pipeline_path = model_path.replace("_model.pkl", "_pipeline.pkl")
        joblib.dump({
            "column_transformer": bundle.get("column_transformer"),
            "feature_names": bundle.get("feature_names"),
        }, pipeline_path)
        return send_file(pipeline_path, as_attachment=True, download_name="automl_pipeline.pkl")
    except Exception as e:
        flash(f"Could not export pipeline: {e}", "error")
        return redirect(url_for("export.export"))


@export_bp.route("/export/predictions")
def download_predictions():
    sid = session.get("sid")
    state = load_state(sid)
    csv_data = state.get("batch_csv_data", "")

    if not csv_data:
        flash("No batch predictions available. Run batch prediction first.", "error")
        return redirect(url_for("predict.predict"))

    response = make_response(csv_data)
    response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
    response.headers["Content-Type"] = "text/csv"
    return response


@export_bp.route("/export/report")
def download_report():
    sid = session.get("sid")
    state = load_state(sid)

    if not state.get("training_done"):
        flash("No training results to report on.", "error")
        return redirect(url_for("export.export"))

    problem_type = state.get("problem_type", "classification")
    trained_results = state.get("trained_results", [])
    report_path = os.path.join(Config.MODEL_FOLDER, f"{sid}_report.pdf")

    success = generate_pdf_report(
        session_state=state,
        results=trained_results,
        problem_type=problem_type,
        output_path=report_path,
    )

    if success and os.path.exists(report_path):
        return send_file(report_path, as_attachment=True, download_name="automl_report.pdf")
    elif os.path.exists(report_path.replace(".pdf", ".txt")):
        return send_file(
            report_path.replace(".pdf", ".txt"),
            as_attachment=True,
            download_name="automl_report.txt"
        )
    else:
        flash("Report generation failed.", "error")
        return redirect(url_for("export.export"))
