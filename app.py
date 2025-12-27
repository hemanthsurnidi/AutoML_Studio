from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np
import joblib

from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "saved_models"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------- DATASET UPLOAD ----------------
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["dataset"]
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    # First attempt: normal read
    df = pd.read_csv(path)

    # If column names are numeric → no header in CSV
    if all(isinstance(col, (int, float)) or str(col).isdigit() for col in df.columns):
        df = pd.read_csv(path, header=None)
        df.columns = [f"feature_{i}" for i in range(df.shape[1])]

    rows, cols = df.shape

    column_info = [
        (c, str(df[c].dtype), int(df[c].isnull().sum()))
        for c in df.columns
    ]

    # Save cleaned dataset back
    df.to_csv(path, index=False)

    return render_template(
        "dataset_info.html",
        rows=rows,
        cols=cols,
        column_info=column_info,
        filename=filename
    )


# ---------------- TRAIN MODEL ----------------
@app.route("/select_target", methods=["POST"])
def select_target():
    steps = []

    target = request.form["target"]
    filename = request.form["filename"]

    df = pd.read_csv(os.path.join(UPLOAD_FOLDER, filename))
    steps.append(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

    y = df[target]
    X = df.drop(columns=[target])
    steps.append(f"Target column selected: {target}")

    X = X.select_dtypes(include=["int64", "float64"])
    steps.append("Only numerical features selected.")

    # Detect problem type
    if y.dtype == "object" or y.nunique() <= 15:
        problem_type = "Classification"
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        steps.append("Detected classification problem (few unique values).")
    else:
        problem_type = "Regression"
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        steps.append("Detected regression problem (continuous target).")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    Xp = pipeline.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        Xp, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    steps.append(f"{model.__class__.__name__} selected and trained.")

    # Evaluation
    if problem_type == "Regression":
        preds = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, preds))
        metric = "RMSE"
    else:
        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds)
        metric = "Accuracy"

    # Feature importance
    importances = model.feature_importances_
    total = np.sum(importances)

    feature_importance = []
    for f, s in sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True):
        pct = (s / total) * 100
        level = "High" if pct >= 20 else "Medium" if pct >= 10 else "Low"
        feature_importance.append({
            "feature": f,
            "importance": round(s, 4),
            "contribution": level
        })

    steps.append("Feature importance calculated for transparency.")

    joblib.dump(
        {
            "model": model,
            "preprocess": pipeline,
            "features": X.columns.tolist(),
            "target": target,
            "problem_type": problem_type
        },
        os.path.join(MODEL_FOLDER, "final_model.pkl")
    )

    steps.append("Final model saved successfully.")

    return render_template(
        "model_ready.html",
        metric=metric,
        score=round(score, 4),
        steps=steps,
        feature_importance=feature_importance
    )

# ---------------- MANUAL PREDICTION ----------------
@app.route("/manual_predict")
def manual_predict():
    bundle = joblib.load(os.path.join(MODEL_FOLDER, "final_model.pkl"))
    return render_template(
        "manual_predict.html",
        features=bundle["features"],
        target=bundle["target"]
    )

@app.route("/run_manual_prediction", methods=["POST"])
def run_manual_prediction():
    bundle = joblib.load(os.path.join(MODEL_FOLDER, "final_model.pkl"))

    data = {f: float(request.form[f]) for f in bundle["features"]}
    df = pd.DataFrame([data])

    X = bundle["preprocess"].transform(df)
    prediction = bundle["model"].predict(X)[0]

    return render_template(
        "manual_result.html",
        target=bundle["target"],
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)
