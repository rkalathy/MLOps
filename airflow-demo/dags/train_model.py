import pickle
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_save():
    # Load into DataFrame
    data = load_iris(as_frame=True)
    df = pd.concat([data.data, data.target.rename("species")], axis=1)

    # 1. Data cleanup (drop missing)
    df = df.dropna()

    # 2. Feature creation
    df["petal_area"] = df["petal length (cm)"] * df["petal width (cm)"]

    # 3. Split
    X = df.drop("species", axis=1)
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Train
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)

    # 5. Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.2%}")

    # 6. Persist
    with open('/tmp/iris_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print("Model saved to /tmp/iris_model.pkl")
