# ============================================
# Imports
# ============================================

import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================
# Training data
# ============================================

X = np.array([
    [22, 25000],
    [25, 30000],
    [28, 35000],
    [30, 40000],
    [32, 45000],
    [35, 50000],
    [38, 55000],
    [40, 60000],
    [45, 70000],
    [50, 80000],
])

y = np.array([
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
])


# ============================================
# Preprocessing + ML model
# ============================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])


# ============================================
# Train model
# ============================================

pipeline.fit(X, y)


# ============================================
# Save model
# ============================================

joblib.dump(pipeline, "model.pkl")

print("Model trained successfully.")
print("model.pkl created successfully.")   
