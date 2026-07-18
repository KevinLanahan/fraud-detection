"""
Train and persist an Isolation Forest model on synthetic transaction data.

Run once before starting the app:
    python ml/train.py

The model file is saved to ml/isolation_forest.pkl and loaded at startup.

Synthetic data design:
  - 95% "normal" transactions: amounts $10-$2000, all hours, weekdays/weekends
  - 5% "anomalous" injections: very high amounts, extreme hours, round numbers
    (these calibrate the Isolation Forest's contamination parameter)
"""

import os
import sys
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "isolation_forest.pkl")
SEED = 42
N_NORMAL = 5000
N_ANOMALOUS = 250

rng = np.random.default_rng(SEED)


def generate_normal():
    amounts = rng.lognormal(mean=5.5, sigma=1.2, size=N_NORMAL)   # ~$100-$2000
    hours   = rng.integers(6, 22, size=N_NORMAL).astype(float)    # business hours
    weekend = rng.integers(0, 2, size=N_NORMAL).astype(float)
    return np.column_stack([amounts, np.log1p(amounts), hours, weekend])


def generate_anomalous():
    # High-amount transactions
    high_amounts = rng.uniform(8000, 50000, size=N_ANOMALOUS)
    # Off-hours (midnight to 4am)
    off_hours = rng.integers(0, 4, size=N_ANOMALOUS).astype(float)
    weekend = rng.integers(0, 2, size=N_ANOMALOUS).astype(float)
    return np.column_stack([high_amounts, np.log1p(high_amounts), off_hours, weekend])


def train():
    normal    = generate_normal()
    anomalous = generate_anomalous()
    X = np.vstack([normal, anomalous])

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,   # ~5% of data is fraudulent — matches our synthetic ratio
        max_samples="auto",
        random_state=SEED,
    )
    model.fit(X)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, OUTPUT_PATH)
    print(f"Model trained on {len(X)} samples and saved to {OUTPUT_PATH}")

    # Quick sanity check
    normal_scores    = model.decision_function(normal[:20])
    anomalous_scores = model.decision_function(anomalous[:20])
    print(f"Normal avg score:    {normal_scores.mean():.4f}  (expect > 0)")
    print(f"Anomalous avg score: {anomalous_scores.mean():.4f} (expect < 0)")


if __name__ == "__main__":
    train()
