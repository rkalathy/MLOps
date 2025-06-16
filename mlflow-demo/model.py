import argparse
import mlflow
import pandas as pd
from sklearn.linear_model import Ridge
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--data",  type=str,   default="ml-loan-demo/raw_data.csv")
parser.add_argument("--registered_model_name", type=str, default="ridge-model-demo")
args = parser.parse_args()

with mlflow.start_run() as run:
    df = pd.read_csv(args.data)
    df_clean = df.dropna()
    df_clean.to_csv("ml-loan-demo/cleaned_data.csv", index=False)

    X = df_clean[["x"]]
    Y = df_clean["y"]
    model = Ridge(alpha=args.alpha)
    model.fit(X, Y)

    print(f"Trained Ridge model: {model}")
    print(f" • Coef:      {model.coef_}")
    print(f" • Intercept: {model.intercept_}")

    mlflow.sklearn.log_model(
        model,
        name="ridge-model",
        input_example=X.head(1),
        signature=infer_signature(X, Y),
        registered_model_name=args.registered_model_name
    )

    run_id = run.info.run_id
    print(f"MLflow run ID: {run_id}")

client = MlflowClient()
all_versions = client.search_model_versions(f"name='{args.registered_model_name}'")
mv = next(v for v in all_versions if v.run_id == run_id)

print(f"Registered as: models:/{args.registered_model_name}/{mv.version}")
