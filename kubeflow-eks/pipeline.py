import kfp.dsl as dsl
from kfp.dsl import pipeline, component

@component(base_image="python:3.9")
def train_model():
    import joblib
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    import os

    iris = load_iris()
    X, y = iris.data, iris.target

    model = RandomForestClassifier()
    model.fit(X, y)

    os.makedirs("/tmp/model", exist_ok=True)
    joblib.dump(model, "/tmp/model/model.joblib")
    print("✅ Model saved.")

@pipeline(name="iris-training-pipeline")
def iris_pipeline():
    train_model()
