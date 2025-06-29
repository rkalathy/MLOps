from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib, os

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()
model.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.joblib")
print("Model saved.")
