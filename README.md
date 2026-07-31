# 📉 Telco Customer Churn Analysis & Prediction Dashboard

An interactive Data Science web application built with Streamlit to analyze customer churn patterns, train a Machine Learning model, tune decision thresholds, and interpret feature importances using SHAP.

This project was developed as part of the Applied Data Science Course (AI-Assisted Mini-Project).

---

## 📌 Features

- Exploratory Data Analysis (EDA): Interactive summary metrics, missing value inspection, and dynamically generated distribution charts for numeric and categorical features.
- Data Preprocessing & Feature Engineering: Custom binary, ordinal, and one-hot encoding for customer demographic and subscription features.
- Model Training & Evaluation: Trains a GradientBoostingClassifier with stratified train/test split, reporting Accuracy, ROC-AUC, Precision, Recall, Confusion Matrix, and ROC Curve.
- Interactive Threshold Tuning: Dynamic Streamlit sidebar slider allowing real-time adjustment of the classification threshold to observe trade-offs in Precision and Recall.
- Explainable AI (SHAP): TreeExplainer integration rendering a SHAP summary plot for top feature importances.

---

## 📁 Project Structure
```text
.
├── app.py                      # Main Streamlit application
├── Telco-Customer-Churn.csv    # Kaggle Telco Churn dataset
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```
---

## 🛠️ Installation & Setup

Follow these steps to run the application locally:

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create a Virtual Environment (Optional but recommended)
#### On Windows
python -m venv venv
venv\Scripts\activate

#### On macOS/Linux
python3 -m venv venv
source venv/bin/activate

### 3. Install Dependencies
pip install -r requirements.txt

---

## 🚀 How to Run

Launch the Streamlit dashboard using:

streamlit run app.py

The application will open automatically in your browser at http://localhost:8501.

---

## 📊 Dataset

The project uses the Telco Customer Churn dataset from Kaggle, containing information about a telecommunications company's customers and whether they churned within the last month.

---

## 🛠️ Tech Stack

- Python 3.10+
- Streamlit (UI / Web App Framework)
- Pandas & NumPy (Data Manipulation)
- Scikit-Learn (Machine Learning Pipeline)
- Matplotlib & Seaborn (Data Visualization)
- SHAP (Model Interpretability & Feature Importance)