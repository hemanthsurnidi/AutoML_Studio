"""
blueprints/results.py
---------------------
Step 7: Results dashboard — metrics, charts, model comparison.
"""
import json
from flask import Blueprint, render_template, redirect, url_for, session, flash
from automl.utils.session_store import load_state

results_bp = Blueprint("results", __name__)


@results_bp.route("/results")
def show_results():
    sid = session.get("sid")
    if not sid:
        flash("Session expired.", "error")
        return redirect(url_for("upload.index"))

    state = load_state(sid)
    if not state.get("training_done"):
        flash("Training not complete. Please configure and train first.", "error")
        return redirect(url_for("configure.configure"))

    problem_type = state.get("problem_type", "classification")
    trained_results = state.get("trained_results", [])
    feature_importance = state.get("feature_importance", [])
    fi_chart = state.get("fi_chart", {"labels": [], "values": []})
    corr_data = state.get("corr_data", {"labels": [], "values": []})
    missing_data = state.get("missing_data", [])
    dist_data = state.get("dist_data", [])
    confusion_matrices = state.get("confusion_matrices", [])
    global_steps_log = state.get("global_steps_log", [])
    feature_names = state.get("feature_names", [])
    target = state.get("target", "")

    best_model = trained_results[0] if trained_results else {}

    return render_template(
        "results.html",
        problem_type=problem_type,
        trained_results=trained_results,
        feature_importance=feature_importance[:15],
        fi_chart_json=json.dumps(fi_chart),
        corr_data_json=json.dumps(corr_data),
        missing_data_json=json.dumps(missing_data),
        dist_data_json=json.dumps(dist_data),
        confusion_matrices_json=json.dumps(confusion_matrices),
        global_steps_log=global_steps_log,
        feature_names=feature_names,
        target=target,
        best_model=best_model,
        total_models=len(trained_results),
    )
