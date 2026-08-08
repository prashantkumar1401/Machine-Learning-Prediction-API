from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = FastAPI(title="Machine Learning Prediction API", version="1.0.0")

# Small synthetic educational dataset.
X = np.array([
    [20, 25000, 1], [25, 32000, 2], [30, 45000, 3], [35, 52000, 5],
    [40, 65000, 7], [50, 80000, 10], [22, 28000, 1], [45, 70000, 8],
    [28, 40000, 2], [55, 90000, 12]
])
y = np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 1])

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)


class Customer(BaseModel):
    age: int = Field(ge=18, le=100)
    income: float = Field(ge=0)
    tenure: float = Field(ge=0, le=60)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    features = np.array([[customer.age, customer.income, customer.tenure]])
    prediction = int(model.predict(features)[0])
    probability = float(max(model.predict_proba(features)[0]))
    return {
        "prediction": "higher-risk" if prediction == 1 else "lower-risk",
        "confidence": round(probability, 3),
    }
