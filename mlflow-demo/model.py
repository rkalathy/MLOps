import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import dagshub



dagshub.init(repo_owner='frontenddevtrainer', repo_name='MLOps', mlflow=True)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and log a RandomForest on Iris"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the forest"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Max depth of each tree"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Frac of data to reserve for testing"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed"
    )
    return parser.parse_args()

def load_data(test_size, random_state):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target,
        test_size=test_size,
        random_state=random_state,
        stratify=iris.target
    )
    return X_train, X_test, y_train, y_test

def main():
    args = parse_args()

    # point MLflow at your tracking server if not default:
    # mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run(run_name="rf_iris"):
        # log hyper-params
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        X_train, X_test, y_train, y_test = load_data(
            test_size=args.test_size,
            random_state=args.random_state
        )

        # train
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state
        )
        model.fit(X_train, y_train)

        # predict & log metrics
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)

        # register model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="rf_model",
        )


        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
