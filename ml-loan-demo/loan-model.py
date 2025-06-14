import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
# from dotenv import load_dotenv
import mlflow
from mlflow.models import infer_signature

# load_dotenv(".env.prod")
# data_path = os.getenv("DATA_PATH")

data_path = "ml-loan-demo/data/loan_data_prod.csv"

df = pd.read_csv(data_path)
df = pd.get_dummies(df, drop_first=True)
df.dropna(inplace=True)

X = df.drop("LoanAmount", axis=1)
y = df["LoanAmount"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


with mlflow.start_run():

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("📊 Model Evaluation")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")

    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("r2_square", r2)


    mlflow.sklearn.log_model(
        model,
        name = "model",
        input_example=X_train.iloc[:1],
        signature=infer_signature(X_train, y_train)
    )

    # os.makedirs("model", exist_ok=True)
    # with open("/models/loan_model.pkl", "wb") as f:
    #     pickle.dump(model, f)
    # print("\n✅ Model saved to /models/loan_model.pkl")
