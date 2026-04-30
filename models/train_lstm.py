"""
Model B — LSTM for N-step-ahead failure probability prediction.

Architecture
------------
  Input  : sliding window of W timesteps × F features
  Output : sigmoid probability of anomaly in the next HORIZON steps

Outputs
-------
  models/lstm_model.keras   (saved Keras model)
  models/lstm_scaler.pkl    (StandardScaler)
  models/lstm_sensor_cols.pkl
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Fix Windows CP1252 terminal encoding (Keras progress bar uses Unicode chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"

import joblib
import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from utils.data_loader import load_batadal, train_test_split_batadal

MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR   = os.path.join(ROOT, "data")

WINDOW_SIZE = 24   # look back 24 timesteps (hours)
HORIZON     = 6    # predict if anomaly occurs within next 6 steps
BATCH_SIZE  = 64
EPOCHS      = 20


def build_sequences(X: np.ndarray, y: np.ndarray, window: int, horizon: int):
    """
    Create sliding-window sequences.
    Label = 1 if any anomaly occurs in the next `horizon` steps.
    """
    Xs, ys = [], []
    for i in range(len(X) - window - horizon):
        Xs.append(X[i : i + window])
        ys.append(int(y[i + window : i + window + horizon].max()))
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train():
    print("=" * 60)
    print("  Training LSTM — Failure Prediction")
    print("=" * 60)

    # Lazy import TF to avoid loading it at module level
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score

    # 1. Load
    df, sensor_cols = load_batadal(DATA_DIR)
    train_df, test_df = train_test_split_batadal(df, train_ratio=0.7)

    # 2. Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[sensor_cols].values)
    X_test  = scaler.transform(test_df[sensor_cols].values)
    y_train = train_df["ATT_FLAG"].values
    y_test  = test_df["ATT_FLAG"].values

    # 3. Build sequences
    X_tr_seq, y_tr_seq = build_sequences(X_train, y_train, WINDOW_SIZE, HORIZON)
    X_te_seq, y_te_seq = build_sequences(X_test,  y_test,  WINDOW_SIZE, HORIZON)
    print(f"  Train sequences: {X_tr_seq.shape} | Test sequences: {X_te_seq.shape}")
    print(f"  Train positive rate: {y_tr_seq.mean()*100:.1f}%")

    # 4. Class weights for imbalanced labels
    pos = y_tr_seq.sum()
    neg = len(y_tr_seq) - pos
    class_weight = {0: 1.0, 1: max(neg / (pos + 1e-6), 1.0)}
    print(f"  Class weights: {class_weight}")

    # 5. Build model
    n_features = X_tr_seq.shape[2]
    model = Sequential([
        LSTM(64, input_shape=(WINDOW_SIZE, n_features), return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1,  activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model.summary()

    # 6. Train
    callbacks = [
        EarlyStopping(monitor="val_auc", patience=5, restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]
    model.fit(
        X_tr_seq, y_tr_seq,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    # 7. Evaluate
    y_prob = model.predict(X_te_seq, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    print("\n  Test Classification Report:")
    print(classification_report(y_te_seq, y_pred, target_names=["normal", "failure"]))
    if y_te_seq.sum() > 0:
        auc = roc_auc_score(y_te_seq, y_prob)
        print(f"  ROC-AUC: {auc:.4f}")

    # 8. Save
    model.save(os.path.join(MODELS_DIR, "lstm_model.keras"))
    joblib.dump(scaler,      os.path.join(MODELS_DIR, "lstm_scaler.pkl"))
    joblib.dump(sensor_cols, os.path.join(MODELS_DIR, "lstm_sensor_cols.pkl"))
    print(f"\n  Saved to {MODELS_DIR}")

    return model, scaler, sensor_cols


def load_model():
    import tensorflow as tf
    model       = tf.keras.models.load_model(os.path.join(MODELS_DIR, "lstm_model.keras"))
    scaler      = joblib.load(os.path.join(MODELS_DIR, "lstm_scaler.pkl"))
    sensor_cols = joblib.load(os.path.join(MODELS_DIR, "lstm_sensor_cols.pkl"))
    return model, scaler, sensor_cols


def predict_proba(model, scaler, sensor_cols, window_buffer: list) -> float:
    """
    Predict failure probability from a buffer of recent readings.
    window_buffer : list of dicts [{sensor_id: value, ...}]
    Returns float in [0, 1].
    """
    if len(window_buffer) < WINDOW_SIZE:
        return 0.0
    mat = np.array([[r.get(c, 0.0) for c in sensor_cols]
                    for r in window_buffer[-WINDOW_SIZE:]], dtype=np.float32)
    mat_s = scaler.transform(mat)
    seq   = mat_s[np.newaxis, ...]           # (1, W, F)
    prob  = float(model.predict(seq, verbose=0)[0][0])
    return prob


if __name__ == "__main__":
    train()
