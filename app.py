import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score,
    roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split


st.title("Telco Churn Analysis")
#################
# Loading data 
#################
@st.cache_data
def load_csv():
    return pd.read_csv('Telco-Customer-Churn.csv')
df = load_csv()

# Display the first 5 rows
st.subheader("First 5 rows of the data")
st.dataframe(df.head())

# Display basic metrics
st.subheader("Dataset Information")
col1, col2 = st.columns(2)
col1.metric("Total Rows", len(df))
col2.metric("Total Columns", len(df.columns))   

# Create summary DataFrame for column info
col_info = pd.DataFrame({
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isnull().sum(),
    "Percent Missing": (df.isnull().mean() * 100).round(2)
}).reset_index().rename(columns={"index": "Column"})
st.dataframe(col_info)

#######################
# Visualizations
#######################
# Drop the 'customerID' column
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)
# Convert specified columns to numeric, coercing errors to NaN
for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
# Fill missing values with 0 ONLY for TotalCharges where tenure is 0
df['TotalCharges'] = df['TotalCharges'].fillna(0)
st.subheader("Bar Plots for Features")

columns = df.columns
n_cols = 4
n_rows = 5
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 22))

axes = axes.flatten()

for i, col in enumerate(columns):
    ax = axes[i]
    if pd.api.types.is_numeric_dtype(df[col]):
        # Plot histogram with KDE for numeric columns
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color='skyblue', edgecolor='black')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    else:
        # Plot bar plot for categorical columns
        df[col].value_counts().plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
    ax.set_title(col)
    ax.tick_params(axis='x', rotation=45)

# Remove any unused subplots if columns < 20
for j in range(i+1, n_rows*n_cols):
    fig.delaxes(axes[j])

plt.tight_layout()
st.pyplot(fig)

########################################################
# Preprocessing Data
########################################################
st.header("Cleaning and Preprocessing Data")
df_processed = df.copy()
# 1. Churn (target): Yes -> 1, No -> 0
df_processed["Churn"] = (df_processed["Churn"] == "Yes").astype(int)

# 2. Gender -> IsMale: Male -> 1, Female -> 0
df_processed["IsMale"] = (df_processed["gender"] == "Male").astype(int)
df_processed = df_processed.drop("gender", axis=1)

# 3. SeniorCitizen -> IsSenior (already 0/1)
df_processed["IsSenior"] = df_processed["SeniorCitizen"]
df_processed = df_processed.drop("SeniorCitizen", axis=1)

# 4. Partner, Dependents, PhoneService, PaperlessBilling: Yes -> 1, No -> 0
for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
    df_processed[col] = (df_processed[col] == "Yes").astype(int)

# 5. IsFiber: 1 if Fiber optic, 0 otherwise (DSL or No)
df_processed["IsFiber"] = (df_processed["InternetService"] == "Fiber optic").astype(int)

# 6. Online add-ons and InternetService: 1 if Yes / has service, 0 otherwise
for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
    df_processed[col] = (df_processed[col] == "Yes").astype(int)
df_processed["InternetService"] = df_processed["InternetService"].isin(["DSL", "Fiber optic"]).astype(int)

# 7. MultipleLines: 1 if Yes, 0 if No or No phone service
df_processed["MultipleLines"] = (df_processed["MultipleLines"] == "Yes").astype(int)

# 8. Contract: ordinal encoding (Month-to-month=0, One year=1, Two year=2)
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
df_processed["Contract"] = df_processed["Contract"].map(contract_map)

# 9. PaymentMethod: one-hot encoding
df_processed = pd.get_dummies(df_processed, columns=["PaymentMethod"], dtype=int)

################################
# Visualize the cleaned data
################################
st.subheader("Clean Dataset Preview")
st.dataframe(df_processed.head())

st.subheader("Correlation Heatmap")
corr_matrix = df_processed.select_dtypes(include=[np.number]).corr()
fig_corr, ax_corr = plt.subplots(figsize=(16, 14))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax_corr, square=True)
ax_corr.set_title("Feature Correlation Heatmap")
plt.tight_layout()
st.pyplot(fig_corr)

st.subheader("Key Features vs Churn")
plot_features = ["InternetService", "IsFiber", "StreamingTV", "PhoneService", "Contract", "tenure"]
feature_labels = {
    "IsFiber": {0: "No Fiber", 1: "Fiber"},
    "Contract": {0: "Month-to-month", 1: "One year", 2: "Two year"},
    "InternetService": {0: "No Internet", 1: "Has Internet"},
    "PhoneService": {0: "No", 1: "Yes"},
    "StreamingTV": {0: "No Streaming", 1: "Has Streaming"},
}

fig_feats, axes_feats = plt.subplots(2, 3, figsize=(14, 10))
axes_feats = axes_feats.flatten()

for ax, feature in zip(axes_feats, plot_features):
    if feature == "tenure":
        sns.boxplot(data=df_processed, x="Churn", y="tenure", ax=ax, palette="coolwarm")
        ax.set_xticklabels(["No Churn", "Churn"])
        ax.set_xlabel("Churn")
        ax.set_ylabel("Tenure (months)")
    else:
        plot_df = df_processed[[feature, "Churn"]].copy()
        plot_df[feature] = plot_df[feature].map(feature_labels[feature])
        sns.countplot(data=plot_df, x=feature, hue="Churn", ax=ax, palette="coolwarm")
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, ["No Churn", "Churn"], title="Churn")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=15)
    ax.set_title(f"{feature} vs Churn")

plt.tight_layout()
st.pyplot(fig_feats)

########################################################
# Model Training & Evaluation
########################################################
st.write("---")
st.header("🤖 Model Training & Evaluation")

@st.cache_resource
def train_model(df_proc):
    # 1. Separate features and target variable
    X = df_proc.drop(columns=['Churn'])
    y = df_proc['Churn']
    # 2. Split dataset into stratified train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )
    # 3. Initialize and train Gradient Boosting Classifier
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train, y_train)
    return gb_model, X_test, y_test

gb_model, X_test, y_test = train_model(df_processed)

# Predict classes and probability estimates for the test set
y_pred = gb_model.predict(X_test)
y_proba = gb_model.predict_proba(X_test)[:, 1]

# 4. Calculate evaluation metrics
acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
recall = recall_score(y_test, y_pred, pos_label=1)
precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)

# Display key performance indicators in Streamlit metrics
col1, col2 = st.columns(2)
col1.metric("Gradient Boosting Accuracy", f"{acc:.2%}")
col2.metric("ROC-AUC Score", f"{roc_auc:.3f}")

# 5. Render Confusion Matrix and ROC Curve side by side
st.subheader("📊 Model Performance Plots")
plot_col1, plot_col2 = st.columns(2)

with plot_col1:
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax_cm,
        cbar=False,
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    plt.tight_layout()
    st.pyplot(fig_cm)

    # Display Recall and Precision right under the matrix
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Churn Recall", f"{recall:.2%}")
    m_col2.metric("Churn Precision", f"{precision:.2%}")

with plot_col2:
    st.markdown("**ROC Curve**")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
    ax_roc.plot(
        fpr, tpr, color="#3b82f6", label=f"GB (AUC = {roc_auc:.2f})"
    )
    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    plt.tight_layout()
    st.pyplot(fig_roc)

########################################################
# Threshold Adjustment Slider
########################################################
st.subheader("⚙️ Decision Threshold Tuning")
threshold = st.slider("Select Churn Threshold", 0.1, 0.9, 0.5, 0.05)

# Predict based on custom threshold
y_pred_custom = (y_proba >= threshold).astype(int)

# Recalculate Confusion Matrix with new threshold
cm_custom = confusion_matrix(y_test, y_pred_custom)
rec_custom = recall_score(y_test, y_pred_custom, pos_label=1)
prec_custom = precision_score(y_test, y_pred_custom, pos_label=1, zero_division=0)
acc_custom = accuracy_score(y_test, y_pred_custom)
# Create two columns to constrain the matrix size (matching top plot size)
col_cm, col_empty = st.columns(2)

with col_cm:
    st.markdown(f"**Confusion Matrix (Threshold = {threshold:.2f})**")
    fig_cm2, ax_cm2 = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm_custom, annot=True, fmt="d", cmap="Blues", ax=ax_cm2,
        cbar=False, xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    ax_cm2.set_xlabel("Predicted")
    ax_cm2.set_ylabel("Actual")
    plt.tight_layout()
    st.pyplot(fig_cm2)

    # Display Recall and Precision right under the matrix
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Churn Recall", f"{rec_custom:.2%}")
    m_col2.metric("Churn Precision", f"{prec_custom:.2%}")


# Display key performance indicators in Streamlit metrics
col1, col2 = st.columns(2)
col1.metric("Gradient Boosting Accuracy - custom threshold", f"{acc_custom:.2%}")

########################################################
# SHAP analysis for feature importance interpretation
########################################################
st.header("SHAP Analysis")
st.subheader("🧬 Top 10 Features Impact (SHAP Values)")
st.write(
    "SHAP values explain the impact of each feature on the model's churn prediction."
)

with st.spinner("Calculating SHAP values..."):
    # Initialize TreeExplainer optimized for tree-based ensemble models
    explainer = shap.TreeExplainer(gb_model)
    shap_values = explainer(X_test)

    # Plot summary for the top 10 features
    fig_shap, ax_shap = plt.subplots(figsize=(8, 5))
    shap.summary_plot(
        shap_values.values,
        X_test,
        max_display=10,
        show=False,
        plot_type="dot",
    )
    plt.tight_layout()
    st.pyplot(fig_shap)

