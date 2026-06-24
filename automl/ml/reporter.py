"""
reporter.py
-----------
PDF report generation using ReportLab with graceful fallback
to a simple text-based report if ReportLab is not installed.
"""
from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_pdf_report(
    session_state: Dict,
    results: List[Dict],
    problem_type: str,
    output_path: str,
) -> bool:
    """
    Generate a PDF report at output_path.
    Returns True if PDF was generated, False if fallback text was used.
    """
    if REPORTLAB_AVAILABLE:
        _generate_reportlab_pdf(session_state, results, problem_type, output_path)
        return True
    else:
        _generate_text_report(session_state, results, problem_type, output_path.replace(".pdf", ".txt"))
        return False


def _generate_reportlab_pdf(session_state, results, problem_type, output_path):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    story = []

    accent = colors.HexColor("#6c63ff")
    dark = colors.HexColor("#1a1a2e")

    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        textColor=accent, fontSize=24, spaceAfter=6
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        textColor=dark, fontSize=14, spaceAfter=4
    )
    body_style = styles["Normal"]

    # --- Title ---
    story.append(Paragraph("AutoML Studio — Model Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(HRFlowable(width="100%", thickness=2, color=accent))
    story.append(Spacer(1, 0.4 * cm))

    # --- Dataset Summary ---
    story.append(Paragraph("1. Dataset Summary", h2_style))
    ds = [
        ["Metric", "Value"],
        ["Rows", str(session_state.get("rows", "—"))],
        ["Columns", str(session_state.get("cols", "—"))],
        ["Problem Type", problem_type.capitalize()],
        ["Target Column", str(session_state.get("target", "—"))],
    ]
    t = Table(ds, colWidths=[7 * cm, 9 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), accent),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f3ff")]),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4 * cm))

    # --- Results ---
    story.append(Paragraph("2. Model Results", h2_style))

    if problem_type == "classification" and results:
        headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Time(s)"]
        data = [headers]
        for r in results:
            data.append([
                r.get("model", ""),
                str(r.get("accuracy", "")),
                str(r.get("precision", "")),
                str(r.get("recall", "")),
                str(r.get("f1", "")),
                str(r.get("roc_auc", "")),
                str(r.get("train_time", "")),
            ])
    elif problem_type == "regression" and results:
        headers = ["Model", "MAE", "MSE", "RMSE", "R²", "Time(s)"]
        data = [headers]
        for r in results:
            data.append([
                r.get("model", ""),
                str(r.get("mae", "")),
                str(r.get("mse", "")),
                str(r.get("rmse", "")),
                str(r.get("r2", "")),
                str(r.get("train_time", "")),
            ])
    elif problem_type == "clustering" and results:
        headers = ["Model", "Silhouette", "Davies-Bouldin", "Clusters", "Time(s)"]
        data = [headers]
        for r in results:
            if "error" not in r:
                data.append([
                    r.get("model", ""),
                    str(r.get("silhouette", "N/A")),
                    str(r.get("davies_bouldin", "N/A")),
                    str(r.get("n_clusters_found", "")),
                    str(r.get("train_time", "")),
                ])
    else:
        data = [["No results available"]]

    col_w = [16 * cm / len(data[0])] * len(data[0])
    t2 = Table(data, colWidths=col_w)
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), accent),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f3ff")]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.4 * cm))

    # --- Footer ---
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Paragraph("AutoML Studio — Powered by scikit-learn & Flask", body_style))

    doc.build(story)


def _generate_text_report(session_state, results, problem_type, output_path):
    lines = [
        "AutoML Studio — Model Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        f"Problem Type: {problem_type}",
        f"Target: {session_state.get('target', 'N/A')}",
        f"Rows: {session_state.get('rows', 'N/A')}",
        f"Columns: {session_state.get('cols', 'N/A')}",
        "",
        "Model Results:",
        "-" * 40,
    ]
    for r in results:
        lines.append(str(r))
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
