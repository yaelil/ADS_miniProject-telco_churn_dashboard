import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("Telco Churn Analysis")

# Load the CSV file
df = pd.read_csv('Telco-Customer-Churn.csv')

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

# Drop the 'customerID' column
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)
# Convert specified columns to numeric, coercing errors to NaN
for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

st.subheader("Bar Plots for Each Column (4x5 Grid)")

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

st.subheader("Cleaning and Preprocessing Data")

# 1. Churn (target): Yes -> 1, No -> 0
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# 2. Gender -> IsMale: Male -> 1, Female -> 0
df["IsMale"] = (df["gender"] == "Male").astype(int)
df = df.drop("gender", axis=1)

# 3. SeniorCitizen -> IsSenior (already 0/1)
df["IsSenior"] = df["SeniorCitizen"]
df = df.drop("SeniorCitizen", axis=1)

# 4. Partner, Dependents, PhoneService, PaperlessBilling: Yes -> 1, No -> 0
for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
    df[col] = (df[col] == "Yes").astype(int)

# 5. IsFiber: 1 if Fiber optic, 0 otherwise (DSL or No)
df["IsFiber"] = (df["InternetService"] == "Fiber optic").astype(int)

# 6. Online add-ons and InternetService: 1 if Yes / has service, 0 otherwise
for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
    df[col] = (df[col] == "Yes").astype(int)
df["InternetService"] = df["InternetService"].isin(["DSL", "Fiber optic"]).astype(int)

# 7. MultipleLines: 1 if Yes, 0 if No or No phone service
df["MultipleLines"] = (df["MultipleLines"] == "Yes").astype(int)

# 8. Contract: ordinal encoding (Month-to-month=0, One year=1, Two year=2)
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
df["Contract"] = df["Contract"].map(contract_map)

# 9. PaymentMethod: one-hot encoding
df = pd.get_dummies(df, columns=["PaymentMethod"], dtype=int)

st.subheader("Clean Dataset Preview")
st.dataframe(df.head())

st.subheader("Correlation Heatmap")
corr_matrix = df.select_dtypes(include=[np.number]).corr()
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
        sns.boxplot(data=df, x="Churn", y="tenure", ax=ax, palette="coolwarm")
        ax.set_xticklabels(["No Churn", "Churn"])
        ax.set_xlabel("Churn")
        ax.set_ylabel("Tenure (months)")
    else:
        plot_df = df[[feature, "Churn"]].copy()
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

