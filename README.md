---
title: AutoML Studio
emoji: 🚀
colorFrom: indigo
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# AutoML Studio

AutoML Studio is a comprehensive, web-based machine learning application that allows users to automatically build, configure, evaluate, and export machine learning models from tabular CSV datasets. The project is designed with a strong focus on customizability, beautiful UI/UX, and explainability.

## 🚀 Key Features

- **Interactive Step-by-Step Wizard:** A seamless interface to configure your machine learning pipeline.
- **Multiple Problem Types:** Automatically handles Classification, Regression, and Clustering.
- **Advanced Preprocessing:** Configure data cleaning per column (imputation, scaling, outlier handling, log transformations, categorical encoding).
- **Feature Selection:** Filter columns manually or use auto-selection (Correlation, Mutual Info, Feature Importance, Variance Threshold).
- **Hyper-Fast Parallel Training:** Trains multiple models concurrently using optimized parameters (e.g., `XGBoost` hist tree method, `n_jobs=-1`).
- **Rich Visualizations:** Interactive charts for feature importance, data distributions, model comparisons, and confusion matrices.
- **Interactive Predictions:** Make manual predictions using your trained models with real-time probability bars.
- **Model Export:** Download your trained pipeline (preprocessor + model) as a ready-to-deploy `.pkl` file.
- **Mobile Responsive:** A modern, glassmorphic UI that works beautifully on desktops, tablets, and phones.

## 🧠 Supported Models

| Problem Type | Supported Models |
|-------------|----------|
| **Classification** | Logistic Regression, Decision Tree, Random Forest, XGBoost, Gradient Boosting, KNN, Naive Bayes, SVM |
| **Regression** | Linear Regression, Ridge, Lasso, Decision Tree Regressor, Random Forest Regressor, XGBoost Regressor, Gradient Boosting Regressor |
| **Clustering** | KMeans, DBSCAN, Agglomerative Clustering, Gaussian Mixture |

## 🏗️ Tech Stack

- **Backend:** Python, Flask, Werkzeug
- **Machine Learning:** Scikit-learn, XGBoost, SciPy
- **Data Handling:** Pandas, NumPy
- **Frontend:** Vanilla HTML, CSS, JavaScript (Mobile-first, Glassmorphism design)
- **Visualizations:** Chart.js
- **Deployment Ready:** Render

## 🔄 Application Workflow

1. **Upload Dataset:** Upload any CSV file. The app automatically infers data types and stats.
2. **Problem Type:** Choose your target column and select Classification, Regression, or Clustering.
3. **Preprocessing:** Configure global and column-specific cleaning strategies.
4. **Feature Selection:** Choose specific features or apply automatic filtering methods.
5. **Train / Test Split:** Configure dataset split ratio and cross-validation strategy.
6. **Model Training:** Select multiple models to train and compare simultaneously.
7. **Results Dashboard:** Review metrics, compare models, and view feature importance charts.
8. **Predictions:** Input custom values to see what the best model predicts.
9. **Export:** Download the pipeline to use in your own applications.

## ⚙️ Local Setup Instructions

```bash
git clone https://github.com/hemanthsurnidi/AutoML_Studio.git
cd AutoML_Studio
python -m venv venv

# Activate Virtual Environment (Windows)
venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt

# Run the Application
python app.py
```

Open your browser and go to `http://127.0.0.1:5000`

## 👤 Author

**Hemanth Surnidi**  
B.Tech Computer Science  
Aspiring Data Scientist  

## 📜 License

This project is intended for educational and demonstration purposes.
