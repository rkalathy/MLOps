from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import os

app = FastAPI()

MODEL_PATH = "/models/loan_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

class Features(BaseModel):
    age: float
    income: float
    credit_score: float
    employed: int  # 1 or 0

@app.post("/predict")
def predict(features: Features):
    try:
        X = np.array([[features.age, features.income, features.credit_score, features.employed]])
        prediction = model.predict(X)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
