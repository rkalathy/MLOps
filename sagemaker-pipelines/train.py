import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

train_data = pd.read_csv('/opt/ml/input/data/train/train.csv')

X = train_data.drop("label", axis=1)
y = train_data["label"]

model = LogisticRegression()
model.fit(X, y)

os.makedirs('/opt/ml/model', exist_ok=True)
joblib.dump(model, '/opt/ml/model/model.joblib')