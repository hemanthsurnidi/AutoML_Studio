from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "saved_models"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

DATA_STORE = {}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("dataset")
    if not file:
        return "No file uploaded"

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    df = pd.read_csv(path)
    DATA_STORE["df"] = df

    summary = []
    for col in df.columns:
        summary.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "missing": int(df[col].isnull().sum())
        })

    return render_template(
        "dataset_info.html",
        rows=df.shape[0],
        cols=df.shape[1],
        summary=summary,
        columns=df.columns
    )

@app.route("/select_target", methods=["POST"])
def select_target():
    df = DATA_STORE.get("df")
    target = request.form.get("target")

    processing_steps = []
    processing_steps.append("Dataset loaded")
    processing_steps.append(f"Target selected: {target}")

    y = df[target]
    X = df.drop(columns=[target])
    X = X.select_dtypes(include=["int64", "float64"])

    processing_steps.append("Selected numeric features")

    problem_type = "classification" if y.dtype == "object" or y.nunique() <= 15 else "regression"
    processing_steps.append(f"Detected problem type: {problem_type}")

    preprocess = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    X_processed = preprocess.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.25, random_state=42
    )

    if problem_type == "classification":
        model = RandomForestClassifier(
            n_estimators=20,
            max_depth=6,
            random_state=42,
            n_jobs=1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds)
        metric = "Accuracy"
    else:
        model = RandomForestRegressor(
            n_estimators=20,
            max_depth=6,
            random_state=42,
            n_jobs=1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, preds))
        metric = "RMSE"

    importances = model.feature_importances_
    total = importances.sum()

    feature_importance = []
    for f, imp in zip(X.columns, importances):
        feature_importance.append({
            "feature": f,
            "importance": round(imp, 4),
            "contribution": round((imp / total) * 100, 2)
        })

    bundle = {
        "model": model,
        "preprocess": preprocess,
        "features": list(X.columns),
        "target": target,
        "problem_type": problem_type
    }

    joblib.dump(bundle, os.path.join(MODEL_FOLDER, "model.pkl"))

    return render_template(
        "model_ready.html",
        metric=metric,
        score=round(score, 4),
        problem_type=problem_type,
        processing_steps=processing_steps,
        feature_importance=feature_importance
    )

@app.route("/manual_predict")
def manual_predict():
    bundle = joblib.load(os.path.join(MODEL_FOLDER, "model.pkl"))
    return render_template("manual_predict.html", features=bundle["features"])

@app.route("/run_manual_prediction", methods=["POST"])
def run_manual_prediction():
    bundle = joblib.load(os.path.join(MODEL_FOLDER, "model.pkl"))

    data = {f: float(request.form.get(f)) for f in bundle["features"]}
    df = pd.DataFrame([data])
    X = bundle["preprocess"].transform(df)
    prediction = bundle["model"].predict(X)[0]

    return render_template("manual_result.html", prediction=prediction)

@app.route("/download_model")
def download_model():
    return send_file(os.path.join(MODEL_FOLDER, "model.pkl"), as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
