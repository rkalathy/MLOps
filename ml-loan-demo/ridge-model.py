import argparse
import mlflow
import pandas as pd
from sklearn.linear_model import Ridge
from mlflow.models import infer_signature


parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default = 1.0)
parser.add_argument("--data", type=str, default = "ml-loan-demo/raw_data.csv")

args = parser.parse_args()

# mlflow.set_tracking_uri("http://localhost:5000")

with mlflow.start_run():

    df = pd.read_csv(args.data)
    df_clean = df.dropna()
    mlflow.log_metric("rows_after_clean", len(df_clean))
    df_clean.to_csv("ml-loan-demo/cleaned_data.csv", index=False)

    X = df_clean[["x"]]
    Y = df_clean["y"]
    model = Ridge(alpha=args.alpha)
    model.fit(X, Y)

    mlflow.log_param("alpha", args.alpha)
    mlflow.log_metric("r2_score", model.score(X, Y))

    mlflow.sklearn.log_model(
        model,
        name="ridge-model",
        input_example=X.head(1),
        signature=infer_signature(X, Y)
    )