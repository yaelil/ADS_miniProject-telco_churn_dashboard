import streamlit as st
import pandas as pd

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
