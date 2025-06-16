import streamlit as st
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = "ml-loan-demo/data/loan_data_dev.csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df = pd.get_dummies(df, drop_first=True)
    df.dropna(inplace=True)
    return df


def train(df):
    X = df.drop("LoanAmount", axis=1)
    y = df["LoanAmount"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    metrics = { "rmse": rmse, "mae": mae, "r2": r2 }

    return model, metrics, X.columns.tolist()

st.title("Loan Amount App")

df = load_data(DATA_PATH)

st.write(f"Data Loaded: { df.shape[0]} rows")

if st.button("Train"):
    model, metrics, features = train(df)
    st.success("Model Trained")
    st.metric(label="RMSE", value = round(metrics["rmse"], 2))
    st.metric(label="MAE", value = round(metrics["mae"], 2))
    st.metric(label="r2", value = round(metrics["r2"], 2))

    with st.form(key="Predict Form"):
        input = {}
        for feature in features:
            val = st.number_input(feature, value = float(df[feature].mean()))
            input[feature] = val
        submitted = st.form_submit_button("Predict")

        X_new = pd.DataFrame([input])
        predict = model.predict(X_new)[0]
        st.write(f"Predicated Loan Amount: {predict:.2f}")
           
            




