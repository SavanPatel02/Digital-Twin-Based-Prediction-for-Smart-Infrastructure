"""
DigitalTwin — Layer 2 of the smart infrastructure digital twin system.

Maintains a live in-memory state model of all sensors, detects threshold
violations, stores history, and supports what-if fault injection.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ── Threshold / normal-range descriptor ─────────────────────────────────────

@dataclass
class SensorThreshold:
    mean: float = 0.0
    std:  float = 1.0
    lo:   Optional[float] = None   # hard lower bound (from dataset min)
    hi:   Optional[float] = None   # hard upper bound (from dataset max)
    sigma_factor: float = 3.0      # alert if |value - mean| > sigma_factor * std


@dataclass
class SensorReading:
    sensor_id: str
    value:     float
    timestamp: Any
    anomaly:   bool = False
    score:     float = 0.0         # ML anomaly score (filled by ML layer)


# ── Main DigitalTwin class ───────────────────────────────────────────────────

class DigitalTwin:
    """
    Digital representation of a water distribution network (BATADAL).

    Responsibilities
    ----------------
    - update(sensors, timestamp)  : ingest a new sensor snapshot
    - flag_anomaly(sensor_id)     : mark a sensor as anomalous
    - inject_fault(sensor_id, v)  : override a sensor value (what-if mode)
    - get_state()                 : return current snapshot of all sensors
    - get_history(sensor_id, n)   : last n readings for a sensor
    - get_alerts()                : list of currently active alerts
    - set_thresholds(stats_df)    : configure normal ranges from training stats
    - register_callback(fn)       : called with (sensor_id, reading) on anomaly
    """

    def __init__(
        self,
        sensor_ids: List[str],
        history_len: int = 500,
    ):
        self._sensors: List[str] = list(sensor_ids)
        self._history_len = history_len
        self._lock = threading.Lock()

        # current live state
        self._state:      Dict[str, SensorReading] = {}
        # per-sensor rolling history
        self._history:    Dict[str, deque] = {s: deque(maxlen=history_len) for s in sensor_ids}
        # active alerts  {sensor_id: SensorReading}
        self._alerts:     Dict[str, SensorReading] = {}
        # injected faults  {sensor_id: override_value}
        self._faults:     Dict[str, float] = {}
        # normal range thresholds
        self._thresholds: Dict[str, SensorThreshold] = {
            s: SensorThreshold() for s in sensor_ids
        }
        # external anomaly callbacks
        self._callbacks:  List[Callable] = []

        self._total_updates = 0
        self._anomaly_count = 0
        self._last_timestamp = None
        self._what_if_mode = False

    # ── Public API ───────────────────────────────────────────────────────────

    def set_thresholds(self, stats_df) -> None:
        """
        Configure per-sensor thresholds from a stats DataFrame.
        Expected index = sensor_id, columns = [mean, std, min, max].
        """
        for sensor_id, row in stats_df.iterrows():
            if sensor_id in self._thresholds:
                self._thresholds[sensor_id] = SensorThreshold(
                    mean=float(row.get("mean", 0)),
                    std=max(float(row.get("std", 1)), 1e-6),
                    lo=float(row.get("min", -1e9)),
                    hi=float(row.get("max",  1e9)),
                )

    def update(self, sensors: Dict[str, float], timestamp=None) -> None:
        """
        Ingest a new sensor snapshot.  Apply injected faults first, then
        check thresholds.
        """
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            self._total_updates += 1
            self._last_timestamp = timestamp

            for sensor_id, raw_value in sensors.items():
                # Apply injected fault override
                value = self._faults.get(sensor_id, raw_value)

                anomaly = self._check_threshold(sensor_id, value)
                score   = self._threshold_score(sensor_id, value) if anomaly else 0.0

                reading = SensorReading(
                    sensor_id=sensor_id,
                    value=value,
                    timestamp=timestamp,
                    anomaly=anomaly,
                    score=score,
                )
                self._state[sensor_id] = reading
                self._history[sensor_id].append(reading)

                if anomaly:
                    self._anomaly_count += 1
                    self._alerts[sensor_id] = reading
                    self._fire_callbacks(sensor_id, reading)
                else:
                    # Clear alert once sensor returns to normal
                    self._alerts.pop(sensor_id, None)

    def flag_anomaly(self, sensor_id: str, score: float = 1.0) -> None:
        """Externally flag a sensor as anomalous (called by ML layer)."""
        with self._lock:
            if sensor_id in self._state:
                self._state[sensor_id].anomaly = True
                self._state[sensor_id].score = score
                self._alerts[sensor_id] = self._state[sensor_id]
                self._anomaly_count += 1

    def inject_fault(self, sensor_id: str, value: float) -> None:
        """
        Override a sensor's value for what-if simulation.
        Pass value=None to remove the fault.
        """
        with self._lock:
            self._what_if_mode = True
            if value is None:
                self._faults.pop(sensor_id, None)
                if not self._faults:
                    self._what_if_mode = False
            else:
                self._faults[sensor_id] = value

    def clear_fault(self, sensor_id: str) -> None:
        self.inject_fault(sensor_id, None)

    def clear_all_faults(self) -> None:
        with self._lock:
            self._faults.clear()
            self._what_if_mode = False

    def get_state(self) -> Dict[str, Dict]:
        """Return a JSON-serialisable snapshot of all current sensor readings."""
        with self._lock:
            return {
                sid: {
                    "value":     r.value,
                    "timestamp": str(r.timestamp),
                    "anomaly":   r.anomaly,
                    "score":     r.score,
                    "injected":  sid in self._faults,
                }
                for sid, r in self._state.items()
            }

    def get_history(self, sensor_id: str, n: Optional[int] = None) -> List[SensorReading]:
        """Return last n readings for sensor_id (all if n is None)."""
        with self._lock:
            hist = list(self._history.get(sensor_id, []))
            return hist[-n:] if n else hist

    def get_history_df(self, sensor_id: str, n: Optional[int] = None):
        """Return history as a pandas DataFrame (value, timestamp, anomaly)."""
        import pandas as pd
        readings = self.get_history(sensor_id, n)
        if not readings:
            return pd.DataFrame(columns=["timestamp", "value", "anomaly"])
        return pd.DataFrame(
            [{"timestamp": r.timestamp, "value": r.value, "anomaly": r.anomaly}
             for r in readings]
        )

    def get_alerts(self) -> Dict[str, Dict]:
        """Return currently active anomaly alerts."""
        with self._lock:
            return {
                sid: {"value": r.value, "timestamp": str(r.timestamp), "score": r.score}
                for sid, r in self._alerts.items()
            }

    def get_summary(self) -> Dict[str, Any]:
        """High-level statistics about the twin's current run."""
        with self._lock:
            n_sensors = len(self._state)
            n_alerts  = len(self._alerts)
            return {
                "total_updates":  self._total_updates,
                "anomaly_count":  self._anomaly_count,
                "active_alerts":  n_alerts,
                "total_sensors":  n_sensors,
                "what_if_mode":   self._what_if_mode,
                "faults_active":  list(self._faults.keys()),
                "last_timestamp": str(self._last_timestamp),
                "health_pct":     round((1 - n_alerts / max(n_sensors, 1)) * 100, 1),
            }

    def register_callback(self, fn: Callable) -> None:
        """Register fn(sensor_id, SensorReading) — called on each anomaly."""
        self._callbacks.append(fn)

    def sensor_ids(self) -> List[str]:
        return list(self._sensors)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _check_threshold(self, sensor_id: str, value: float) -> bool:
        th = self._thresholds.get(sensor_id)
        if th is None:
            return False
        # Hard bounds check
        if th.lo is not None and value < th.lo:
            return True
        if th.hi is not None and value > th.hi:
            return True
        # Statistical z-score check
        if th.std > 0:
            z = abs(value - th.mean) / th.std
            if z > th.sigma_factor:
                return True
        return False

    def _threshold_score(self, sensor_id: str, value: float) -> float:
        """
        Return a [0, 1] severity score for a threshold violation.
        Uses how many sigma the value is from the mean, capped at 1.0.
        Hard-bound violations (outside min/max) always score >= 0.8.
        """
        th = self._thresholds.get(sensor_id)
        if th is None:
            return 1.0
        # Hard bounds violation
        if (th.lo is not None and value < th.lo) or (th.hi is not None and value > th.hi):
            if th.std > 0:
                z = abs(value - th.mean) / th.std
                return min(z / (th.sigma_factor * 3), 1.0)
            return 0.9
        # Statistical z-score violation
        if th.std > 0:
            z = abs(value - th.mean) / th.std
            # Map z from [sigma_factor, 3*sigma_factor] → [0.3, 1.0]
            normalised = (z - th.sigma_factor) / (th.sigma_factor * 2)
            return round(min(max(normalised, 0.3), 1.0), 3)
        return 0.5

    def _fire_callbacks(self, sensor_id: str, reading: SensorReading) -> None:
        for fn in self._callbacks:
            try:
                fn(sensor_id, reading)
            except Exception:
                pass

    def __repr__(self) -> str:
        s = self.get_summary()
        return (
            f"DigitalTwin(sensors={s['total_sensors']}, "
            f"alerts={s['active_alerts']}, "
            f"updates={s['total_updates']}, "
            f"health={s['health_pct']}%)"
        )
