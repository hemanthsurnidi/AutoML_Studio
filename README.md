# AutoML Studio

AutoML Studio is a web-based machine learning application that automatically builds, evaluates, and explains machine learning models from CSV datasets. The project is designed with a strong focus on correctness, transparency, and explainability, rather than functioning as a black-box AutoML tool.

## 🚀 Features

- 📁 Upload any CSV dataset
- 🎯 Select target column
- 🤖 Automatically detects problem type:
  - Regression
  - Classification
- 🧠 Automatically selects the appropriate Random Forest model
- 📊 Evaluates model performance
  - RMSE for Regression
  - Accuracy for Classification
- 🔍 Displays detailed preprocessing steps
- 📈 Shows feature importance for model explainability
- ✍️ Manual prediction using the trained model
- 🎨 Clean, professional, and user-friendly web interface

## 🧠 Supported Problem Types

| Problem Type | Supported | Notes |
|-------------|----------|-------|
| Regression | ✅ Yes | Continuous numeric targets |
| Classification | ✅ Yes | Binary and multi-class problems |
| Clustering | ❌ No | Planned for future versions |

## 🏗️ Tech Stack

- Backend: Python, Flask
- Machine Learning: Scikit-learn
- Models: RandomForestRegressor, RandomForestClassifier
- Frontend: HTML, CSS
- Data Handling: Pandas, NumPy
- Deployment Ready: Render

## 🔄 Application Workflow

1. Upload CSV file
2. Review dataset overview (rows, columns, missing values)
3. Select the target column
4. Automatic detection of problem type
5. Model training and evaluation
6. Review preprocessing steps and feature importance
7. Perform manual predictions using the trained model

## 📊 Explainability Focus

AutoML Studio is intentionally built to avoid black-box behavior. Every preprocessing step is shown to the user, the reason behind model selection is clearly explained, feature importance is displayed to justify model decisions, and evaluation metrics are presented in a clear and understandable way. The goal is not only to achieve good accuracy, but also to ensure trust, clarity, and understanding.

## 📁 Project Structure

AutoML_Studio/
├── app.py
├── requirements.txt
├── static/
│   └── style.css
├── templates/
│   ├── index.html
│   ├── dataset_info.html
│   ├── model_ready.html
│   ├── manual_predict.html
│   └── manual_result.html
├── uploads/
├── saved_models/

## ⚙️ Local Setup Instructions

git clone https://github.com/hemanthsurnidi/AutoML_Studio.git
cd AutoML_Studio
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py

Open your browser and go to http://127.0.0.1:5000

## 🚫 Limitations

- Clustering and other unsupervised learning problems are not supported in the current version
- Only numerical features are handled
- Advanced hyperparameter tuning is not included

## 🔮 Future Enhancements

- Clustering support (KMeans, DBSCAN)
- Batch prediction through CSV upload
- Model comparison and selection
- Probability-based outputs for classification
- Production-scale deployment and monitoring

## 👤 Author

Hemanth Surnidi  
B.Tech Computer Science  
Aspiring Data Scientist  

## 📜 License

This project is intended for educational and demonstration purposes.
