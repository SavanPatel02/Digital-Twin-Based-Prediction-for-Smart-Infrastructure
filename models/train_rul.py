"""
Model C — Remaining Useful Life (RUL) Regression.

Estimates how many timesteps remain before the next failure event.
Uses XGBoost regressor trained on hand-crafted rolling-window features.

Outputs
-------
  models/rul_model.pkl
  models/rul_scaler.pkl
  models/rul_sensor_cols.pkl
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from utils.data_loader import load_batadal, train_test_split_batadal

MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR   = os.path.join(ROOT, "data")

WINDOW = 12   # rolling window for feature extraction


def compute_rul_labels(att_flag: np.ndarray) -> np.ndarray:
    """
    For each timestep, compute how many steps until the NEXT anomaly event.
    Timesteps after the last anomaly get label = max_rul.
    Timesteps within an anomaly window get label = 0.
    """
    n = len(att_flag)
    rul = np.full(n, n, dtype=float)

    # Walk backwards
    next_attack = n
    for i in range(n - 1, -1, -1):
        if att_flag[i] == 1:
            next_attack = i
            rul[i] = 0
        else:
            rul[i] = next_attack - i

    # Clip to a reasonable max (e.g. 200 steps)
    return np.clip(rul, 0, 200).astype(float)


def extract_features(df: pd.DataFrame, sensor_cols: list, window: int) -> pd.DataFrame:
    """
    Rolling-window statistical features per sensor.
    Features: mean, std, min, max, trend (last - first) over `window` rows.
    """
    feats = {}
    for col in sensor_cols:
        s = df[col]
        feats[f"{col}_mean"]  = s.rolling(window, min_periods=1).mean()
        feats[f"{col}_std"]   = s.rolling(window, min_periods=1).std().fillna(0)
        feats[f"{col}_min"]   = s.rolling(window, min_periods=1).min()
        feats[f"{col}_max"]   = s.rolling(window, min_periods=1).max()
        feats[f"{col}_trend"] = s.diff(window).fillna(0)
    return pd.DataFrame(feats, index=df.index)


def train():
    from xgboost import XGBRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score

    print("=" * 60)
    print("  Training RUL Regressor (XGBoost)")
    print("=" * 60)

    df, sensor_cols = load_batadal(DATA_DIR)
    train_df, test_df = train_test_split_batadal(df, train_ratio=0.7)

    # Features
    X_train_feat = extract_features(train_df, sensor_cols, WINDOW)
    X_test_feat  = extract_features(test_df,  sensor_cols, WINDOW)

    # Labels
    y_train = compute_rul_labels(train_df["ATT_FLAG"].values)
    y_test  = compute_rul_labels(test_df["ATT_FLAG"].values)

    # Drop NaN rows created by rolling
    X_train_feat = X_train_feat.dropna()
    y_train = y_train[X_train_feat.index]
    X_test_feat  = X_test_feat.dropna()
    y_test  = y_test[X_test_feat.index]

    # Scale
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_feat)
    X_te_s = scaler.transform(X_test_feat)

    print(f"  Train: {X_tr_s.shape} | Test: {X_te_s.shape}")
    print(f"  Mean RUL (train): {y_train.mean():.1f} | (test): {y_test.mean():.1f}")

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_tr_s, y_train,
        eval_set=[(X_te_s, y_test)],
        verbose=50,
    )

    y_pred = model.predict(X_te_s)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"\n  MAE: {mae:.2f} steps | R²: {r2:.4f}")

    # Save
    feat_cols = list(X_train_feat.columns)
    joblib.dump(model,       os.path.join(MODELS_DIR, "rul_model.pkl"))
    joblib.dump(scaler,      os.path.join(MODELS_DIR, "rul_scaler.pkl"))
    joblib.dump(sensor_cols, os.path.join(MODELS_DIR, "rul_sensor_cols.pkl"))
    joblib.dump(feat_cols,   os.path.join(MODELS_DIR, "rul_feat_cols.pkl"))
    print(f"\n  Saved to {MODELS_DIR}")

    return model, scaler, sensor_cols, feat_cols


def load_model():
    model       = joblib.load(os.path.join(MODELS_DIR, "rul_model.pkl"))
    scaler      = joblib.load(os.path.join(MODELS_DIR, "rul_scaler.pkl"))
    sensor_cols = joblib.load(os.path.join(MODELS_DIR, "rul_sensor_cols.pkl"))
    feat_cols   = joblib.load(os.path.join(MODELS_DIR, "rul_feat_cols.pkl"))
    return model, scaler, sensor_cols, feat_cols


def predict_rul(model, scaler, sensor_cols, feat_cols, window_buffer: list) -> float:
    """
    Predict RUL from a buffer of recent readings.
    Returns estimated steps until next failure (clipped to [0, 200]).
    """
    if len(window_buffer) < WINDOW:
        return 200.0

    mat = pd.DataFrame(window_buffer[-WINDOW:])
    present = [c for c in sensor_cols if c in mat.columns]
    for c in sensor_cols:
        if c not in mat.columns:
            mat[c] = 0.0

    feat_df = pd.DataFrame(index=[0])
    for col in sensor_cols:
        s = mat[col].values
        feat_df[f"{col}_mean"]  = s.mean()
        feat_df[f"{col}_std"]   = s.std() if len(s) > 1 else 0.0
        feat_df[f"{col}_min"]   = s.min()
        feat_df[f"{col}_max"]   = s.max()
        feat_df[f"{col}_trend"] = float(s[-1]) - float(s[0])

    # Align to training features
    for fc in feat_cols:
        if fc not in feat_df.columns:
            feat_df[fc] = 0.0
    feat_df = feat_df[feat_cols]

    scaled = scaler.transform(feat_df.values)
    rul    = float(model.predict(scaled)[0])
    return float(np.clip(rul, 0, 200))


if __name__ == "__main__":
    train()
