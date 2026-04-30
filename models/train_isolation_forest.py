"""
Model A — Isolation Forest for real-time anomaly detection.

Trains on the normal (ATT_FLAG == 0) portion of BATADAL training data.
Saves the fitted model to models/isolation_forest.pkl.

Usage
-----
  python models/train_isolation_forest.py

Outputs
-------
  models/isolation_forest.pkl
  models/if_sensor_cols.pkl   (list of feature columns used)
  models/sensor_stats.pkl     (mean/std/min/max per sensor, for thresholds)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from utils.data_loader import load_batadal, train_test_split_batadal, compute_sensor_stats

MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR   = os.path.join(ROOT, "data")


def train(
    contamination: float = 0.05,
    n_estimators:  int   = 200,
    random_state:  int   = 42,
) -> IsolationForest:

    print("=" * 60)
    print("  Training Isolation Forest — Anomaly Detection")
    print("=" * 60)

    # 1. Load data
    df, sensor_cols = load_batadal(DATA_DIR)
    print(f"  Rows: {len(df)} | Sensors: {len(sensor_cols)}")
    print(f"  Attack rows: {df['ATT_FLAG'].sum()} ({df['ATT_FLAG'].mean()*100:.1f}%)")

    train_df, test_df = train_test_split_batadal(df, train_ratio=0.7)

    # 2. Feature matrix (sensor values only)
    X_train = train_df[sensor_cols].values
    X_test  = test_df[sensor_cols].values
    y_test  = test_df["ATT_FLAG"].values

    # 3. Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 4. Train Isolation Forest
    #    contamination = expected fraction of anomalies in training data
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_s)
    print("  Model trained.")

    # 5. Evaluate on test split
    #    IsolationForest returns -1 for anomaly, +1 for normal
    raw_preds = model.predict(X_test_s)
    y_pred    = (raw_preds == -1).astype(int)   # 1 = anomaly
    scores    = -model.score_samples(X_test_s)  # higher = more anomalous

    if y_test.sum() > 0:
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["normal", "attack"]))
        try:
            auc = roc_auc_score(y_test, scores)
            print(f"  ROC-AUC: {auc:.4f}")
        except Exception:
            pass
    else:
        print("  (No labeled anomalies in test split — skipping metrics)")

    # 6. Sensor statistics (used by DigitalTwin for thresholds)
    stats = compute_sensor_stats(train_df, sensor_cols)

    # 7. Save artefacts
    joblib.dump(model,       os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    joblib.dump(scaler,      os.path.join(MODELS_DIR, "if_scaler.pkl"))
    joblib.dump(sensor_cols, os.path.join(MODELS_DIR, "if_sensor_cols.pkl"))
    joblib.dump(stats,       os.path.join(MODELS_DIR, "sensor_stats.pkl"))
    print(f"\n  Saved to {MODELS_DIR}")

    return model, scaler, sensor_cols, stats


def load_model():
    """Load previously trained Isolation Forest and accessories."""
    model       = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    scaler      = joblib.load(os.path.join(MODELS_DIR, "if_scaler.pkl"))
    sensor_cols = joblib.load(os.path.join(MODELS_DIR, "if_sensor_cols.pkl"))
    stats       = joblib.load(os.path.join(MODELS_DIR, "sensor_stats.pkl"))
    return model, scaler, sensor_cols, stats


def predict(model, scaler, sensor_cols, reading_dict: dict):
    """
    Score a single reading dict  {sensor_id: value, ...}.
    Returns (is_anomaly: bool, anomaly_score: float 0-1).
    """
    values = np.array([[reading_dict.get(c, 0.0) for c in sensor_cols]])
    scaled = scaler.transform(values)
    raw    = model.predict(scaled)[0]
    score  = float(-model.score_samples(scaled)[0])
    # Normalise score to [0, 1] heuristically
    norm_score = min(max((score + 0.5) / 1.5, 0.0), 1.0)
    return (raw == -1), norm_score


if __name__ == "__main__":
    train()
