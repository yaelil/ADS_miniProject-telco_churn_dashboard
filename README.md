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
## 🔄 Workflow

1. **Data Cleaning & Preprocessing:** 
   - Handled missing and whitespace values across numeric and categorical features.
   - Standardized categorical data and removed unnecessary identifier columns.
2. **Feature Engineering & Encoding:** 
   - Applied target-appropriate encoding strategies: Binary encoding for binary indicators, Ordinal encoding for ordered variables ("Contract"), and One-Hot Encoding for multi-class nominal features("PaymentMethod").
3. **Model Training & Evaluation:** 
   - Split data using a stratified train/test split to preserve class ratios.
   - Trained a `GradientBoostingClassifier` and evaluated performance using Accuracy, ROC-AUC, Precision-Recall metrics, and Confusion Matrices.

---

## 💡 Key Analytical Insights & Performance

- **Primary Churn Drivers:** Feature importance via SHAP revealed that Contract type (month-to-month), Tenure length, and Fiber Optic internet service (`IsFiber`) are the primary predictors of customer churn.
- **Deep-Dive: Fiber Optic Churn (`IsFiber`) & Pricing Dynamics:** 
  - **SHAP & Correlation Insights:** Exploratory analysis via feature correlation heatmap highlights a very strong positive correlation ($r = 0.79$) between `IsFiber` and `MonthlyCharges`.
  - **Propensity to Churn:** SHAP value distribution confirms that `IsFiber = 1` (red points) consistently increases churn probability (SHAP values > 0). 
  - **Business Context:** Fiber optic is a premium service with significantly higher recurring costs. High churn in this segment is driven by a combination of elevated monthly bills, higher service quality expectations, and competitive promotional target marketing.
- **Interactive Decision Thresholds:** Rather than relying solely on default probability cutoffs ($0.5$), the app provides a dynamic threshold slider—allowing stakeholders to prioritize either Recall (detecting more potential churners) or Precision (reducing false alerts).

---
## 🛠️ Tech Stack

- Python 3.10+
- Streamlit (UI / Web App Framework)
- Pandas & NumPy (Data Manipulation)
- Scikit-Learn (Machine Learning Pipeline)
- Matplotlib & Seaborn (Data Visualization)
- SHAP (Model Interpretability & Feature Importance)