"""
BATADAL dataset loader and real-time streaming simulator.

BATADAL (Battle of the Attack Detection ALgorithms) is a water distribution
network dataset with labeled anomalies injected into sensor readings.
Download from: https://www.batadal.net/data.html
  - training_dataset_1.csv  (normal operations)
  - training_dataset_2.csv  (contains attack periods, with ATT_FLAG column)
  - test_dataset.csv

Place CSVs inside digital_twin_project/data/
"""

import time
import os
import numpy as np
import pandas as pd
from typing import Generator, Dict, Any, Optional, Tuple


# Columns in BATADAL that represent sensor readings
SENSOR_COLS = [
    "L_T1", "L_T2", "L_T3", "L_T4", "L_T5", "L_T6", "L_T7",
    "F_PU1", "F_PU2", "F_PU3", "F_PU4", "F_PU5", "F_PU6",
    "F_PU7", "F_PU8", "F_PU9", "F_PU10", "F_PU11",
    "F_V2",
    "P_J280", "P_J269", "P_J300", "P_J256", "P_J289",
    "P_J415", "P_J302", "P_J306", "P_J307", "P_J317",
    "P_J14", "P_J422",
    "S_PU1", "S_PU2", "S_PU3", "S_PU4", "S_PU5",
    "S_PU6", "S_PU7", "S_PU8", "S_PU9", "S_PU10", "S_PU11",
    "S_V2",
]

# Sensor groups for UI display
SENSOR_GROUPS = {
    "Tank Levels":    [c for c in SENSOR_COLS if c.startswith("L_")],
    "Flow (Pumps)":   [c for c in SENSOR_COLS if c.startswith("F_")],
    "Pressure":       [c for c in SENSOR_COLS if c.startswith("P_")],
    "Pump Status":    [c for c in SENSOR_COLS if c.startswith("S_PU")],
    "Valve Status":   [c for c in SENSOR_COLS if c.startswith("S_V")],
}


def _find_csv(data_dir: str) -> Optional[str]:
    """Return path to first BATADAL CSV found in data_dir."""
    for name in [
        "training_dataset_2.csv",
        "training_dataset_1.csv",
        "test_dataset.csv",
        "batadal_dataset.csv",
        "BATADAL_dataset04.csv",
        "BATADAL_dataset03.csv",
    ]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return p
    # fallback: any CSV in the directory
    for f in os.listdir(data_dir):
        if f.endswith(".csv"):
            return os.path.join(data_dir, f)
    return None


def load_batadal(data_dir: str = "data") -> Tuple[pd.DataFrame, list]:
    """
    Load a BATADAL CSV and return (dataframe, sensor_columns_present).

    If no file is found, generate synthetic data so the system still runs.
    """
    csv_path = _find_csv(data_dir)

    if csv_path:
        df = pd.read_csv(csv_path, sep=",", skipinitialspace=True)
        df.columns = [c.strip().upper() for c in df.columns]

        # Parse timestamp
        if "DATETIME" in df.columns:
            df["DATETIME"] = pd.to_datetime(df["DATETIME"], format="%d/%m/%y %H", errors="coerce")
            if df["DATETIME"].isna().all():
                df["DATETIME"] = pd.to_datetime(df["DATETIME"], dayfirst=True, errors="coerce")
        elif "DATE" in df.columns:
            df["DATETIME"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")

        # Ensure ATT_FLAG column exists
        if "ATT_FLAG" not in df.columns:
            df["ATT_FLAG"] = 0
        else:
            # BATADAL test files use -999 for unlabelled rows — treat as normal (0)
            df["ATT_FLAG"] = pd.to_numeric(df["ATT_FLAG"], errors="coerce").fillna(0)
            df["ATT_FLAG"] = df["ATT_FLAG"].apply(lambda x: 0 if x < 0 else int(x))

        present = [c for c in SENSOR_COLS if c in df.columns]
        df[present] = df[present].apply(pd.to_numeric, errors="coerce")
        df[present] = df[present].ffill().fillna(0)
        return df, present

    # ── Synthetic fallback ──────────────────────────────────────────────────
    print("[DataLoader] No BATADAL CSV found — generating synthetic data.")
    return _generate_synthetic(), SENSOR_COLS[:10]


def _generate_synthetic(n_rows: int = 5000) -> pd.DataFrame:
    """Generate plausible synthetic water-network sensor data."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2018-01-01", periods=n_rows, freq="1h")
    cols = SENSOR_COLS[:10]

    data = {c: rng.normal(loc=50, scale=5, size=n_rows).clip(0) for c in cols}

    # Inject 3 attack windows
    for _ in range(3):
        start = rng.integers(500, n_rows - 200)
        end = start + rng.integers(50, 150)
        for c in cols[:3]:
            data[c][start:end] *= rng.uniform(0.3, 2.5)

    df = pd.DataFrame(data)
    df["DATETIME"] = timestamps
    df["ATT_FLAG"] = 0

    # Mark attack periods
    for start in [600, 1800, 3500]:
        df.loc[start : start + 100, "ATT_FLAG"] = 1

    return df


def stream_batadal(
    df: pd.DataFrame,
    sensor_cols: list,
    speed: float = 1.0,
    start_row: int = 0,
) -> Generator[Dict[str, Any], None, None]:
    """
    Yield one row at a time as a dict simulating real-time sensor feed.

    Parameters
    ----------
    df          : DataFrame returned by load_batadal()
    sensor_cols : list of sensor column names
    speed       : seconds to sleep between rows (0 = as fast as possible)
    start_row   : row index to start streaming from
    """
    for idx in range(start_row, len(df)):
        row = df.iloc[idx]
        reading: Dict[str, Any] = {
            "row_index":  idx,
            "timestamp":  row.get("DATETIME", pd.Timestamp.now()),
            "att_flag":   int(row.get("ATT_FLAG", 0)),
            "sensors":    {col: float(row[col]) for col in sensor_cols if col in row.index},
        }
        yield reading
        if speed > 0:
            time.sleep(speed)


def train_test_split_batadal(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically — no shuffling for time-series."""
    split = int(len(df) * train_ratio)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)


def compute_sensor_stats(df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    """Return per-sensor mean/std/min/max — used to set thresholds."""
    return df[sensor_cols].agg(["mean", "std", "min", "max"]).T


# ── CLI test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    df, cols = load_batadal(DATA_DIR)
    print(f"Loaded {len(df)} rows, {len(cols)} sensors")
    print(f"Attack rows: {df['ATT_FLAG'].sum()}")
    print("\nFirst 3 streamed readings:")
    for i, reading in enumerate(stream_batadal(df, cols, speed=0)):
        print(reading)
        if i >= 2:
            break
