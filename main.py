# ============================================
# Imports
# ============================================

import os
import time
import joblib

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================
# Global model
# ============================================

model = None


# ============================================
# Pydantic input schema
# ============================================

class EmployeeInput(BaseModel):
    age: int
    salary: float


# ============================================
# Application startup
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global model

    model_path = os.getenv("MODEL_PATH", "model.pkl")

    model = joblib.load(model_path)

    print(f"Model loaded from: {model_path}")

    yield


# ============================================
# FastAPI application
# ============================================

app = FastAPI(
    title="Employee Prediction API",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================
# Latency middleware
# ============================================

@app.middleware("http")
async def latency_middleware(request, call_next):

    start_time = time.time()

    response = await call_next(request)

    latency = time.time() - start_time

    response.headers["X-Process-Time"] = str(round(latency, 4))

    return response


# ============================================
# Health endpoint
# ============================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================
# Readiness endpoint
# ============================================

@app.get("/ready")
def ready():

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    return {
        "status": "ready",
        "model_loaded": True
    }


# ============================================
# Prediction endpoint
# ============================================

@app.post("/predict")
def predict(data: EmployeeInput):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    prediction = model.predict([
        [data.age, data.salary]
    ])

    probability = model.predict_proba([
        [data.age, data.salary]
    ])[0][1]

    return {
        "prediction": int(prediction[0]),
        "probability": round(float(probability), 4)
    }


# ============================================
# Batch prediction endpoint
# ============================================

@app.post("/predict_batch")
def predict_batch(data: list[EmployeeInput]):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    input_data = [
        [employee.age, employee.salary]
        for employee in data
    ]

    predictions = model.predict(input_data)

    probabilities = model.predict_proba(input_data)[:, 1]

    results = []

    for prediction, probability in zip(
        predictions,
        probabilities
    ):

        results.append({
            "prediction": int(prediction),
            "probability": round(
                float(probability),
                4
            )
        })

    return {
        "results": results
    }

