"""
Live data fetcher — USGS National Water Information System (NWIS).

Fetches real-time water sensor readings from actual US monitoring stations.
No API key required. Data updates every 15 minutes.

API docs: https://waterservices.usgs.gov/rest/IV-Service.html

Sensors fetched
---------------
  00060 — Discharge / Flow rate  (ft³/s)
  00065 — Gage height / Water level (ft)
  00010 — Water temperature (°C)
  00095 — Specific conductance (µS/cm)
  00300 — Dissolved oxygen (mg/L)

Stations used (geographically spread across the US water network)
-----------------------------------------------------------------
  01646500 — Potomac River at Little Falls, MD
  02087500 — Neuse River at Kinston, NC
  07374000 — Mississippi River at Baton Rouge, LA
  09380000 — Colorado River at Lees Ferry, AZ
  11447650 — Sacramento River at Sacramento, CA
  03213700 — Tug Fork at Williamson, WV
  05420500 — Mississippi River at Clinton, IA
  06710000 — South Platte River at Denver, CO
"""

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# ── Station config ─────────────────────────────────────────────────────────────

STATIONS = {
    "Potomac_MD":       "01646500",
    "Neuse_NC":         "02087500",
    "Mississippi_LA":   "07374000",
    "Colorado_AZ":      "09380000",
    "Sacramento_CA":    "11447650",
    "TugFork_WV":       "03213700",
    "Mississippi_IA":   "05420500",
    "SouthPlatte_CO":   "06710000",
}

# USGS parameter codes → friendly names
PARAM_CODES = {
    "00060": "Flow_cfs",        # Discharge ft³/s
    "00065": "Level_ft",        # Gage height ft
    "00010": "Temp_C",          # Water temperature
    "00095": "Conductance",     # Specific conductance
    "00300": "DissolvedO2",     # Dissolved oxygen
}

USGS_BASE = "https://waterservices.usgs.gov/nwis/iv/"

# Sensor groups matching BATADAL-style categories for the dashboard
LIVE_SENSOR_GROUPS = {
    "Water Level":      [f"{name}_Level_ft"      for name in STATIONS],
    "Flow Rate":        [f"{name}_Flow_cfs"       for name in STATIONS],
    "Temperature":      [f"{name}_Temp_C"         for name in STATIONS],
    "Conductance":      [f"{name}_Conductance"    for name in STATIONS],
    "Dissolved O₂":     [f"{name}_DissolvedO2"    for name in STATIONS],
}


# ── Fetcher ────────────────────────────────────────────────────────────────────

class USGSLiveFetcher:
    """
    Polls the USGS Instantaneous Values API and returns a flat sensor dict
    suitable for feeding into DigitalTwin.update().

    Usage
    -----
        fetcher = USGSLiveFetcher()
        reading = fetcher.fetch()
        # reading["sensors"] = {"Potomac_MD_Flow_cfs": 4230.0, ...}
    """

    def __init__(self, cache_seconds: int = 900):
        self._cache_seconds = cache_seconds   # USGS updates every 15 min
        self._last_fetch:  Optional[float] = None
        self._cached:      Optional[Dict]  = None
        self._sensor_cols: Optional[List[str]] = None
        self._stats_cache: Optional[pd.DataFrame] = None
        self._history:     List[Dict] = []     # rolling history for stats

    # ── Public ────────────────────────────────────────────────────────────────

    def fetch(self) -> Dict[str, Any]:
        """
        Return a reading dict  {timestamp, sensors:{sensor_id: value}, att_flag}.
        Uses cached value if data is still fresh (< 15 min old).
        Between real fetches, adds small realistic noise so charts animate.
        """
        now = time.time()
        needs_refresh = (
            self._last_fetch is None
            or (now - self._last_fetch) >= self._cache_seconds
        )

        if needs_refresh:
            fresh = self._fetch_from_usgs()
            if fresh:
                self._cached     = fresh
                self._last_fetch = now
                self._history.append(fresh["sensors"])
                if len(self._history) > 200:
                    self._history.pop(0)

        if self._cached is None:
            return self._fallback_reading()

        # Add tiny realistic noise between 15-min updates so charts move
        noisy = self._add_noise(self._cached)
        return noisy

    def sensor_cols(self) -> List[str]:
        """Return list of all sensor IDs (populated after first fetch)."""
        if self._sensor_cols is None:
            r = self.fetch()
            self._sensor_cols = list(r["sensors"].keys())
        return self._sensor_cols

    def compute_stats(self) -> pd.DataFrame:
        """
        Compute mean/std/min/max from 7-day USGS historical data so thresholds
        are calibrated to each station's actual normal operating range.
        Falls back to rolling history if the historical fetch fails.
        """
        if self._stats_cache is not None:
            return self._stats_cache

        stats = self._fetch_historical_stats()
        if stats is not None:
            self._stats_cache = stats
            return stats

        # Last resort: use whatever live readings we have so far
        if len(self._history) >= 3:
            df = pd.DataFrame(self._history)
            stats = df.agg(["mean", "std", "min", "max"]).T
            stats["std"] = stats["std"].fillna(1.0).clip(lower=0.01)
            return stats

        # If nothing at all, return very permissive stats (huge std = no alerts)
        cols = self.sensor_cols()
        rows = [[0, 1e9, -1e9, 1e9] for _ in cols]
        return pd.DataFrame(rows, index=cols, columns=["mean", "std", "min", "max"])

    def _fetch_historical_stats(self) -> Optional[pd.DataFrame]:
        """
        Fetch last 7 days of hourly data from USGS for all stations,
        compute per-sensor mean/std/min/max as the normal baseline.
        """
        site_list  = ",".join(STATIONS.values())
        param_list = ",".join(PARAM_CODES.keys())
        params = {
            "format":      "json",
            "sites":       site_list,
            "parameterCd": param_list,
            "period":      "P7D",          # last 7 days
            "siteStatus":  "active",
        }
        try:
            resp = requests.get(USGS_BASE, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[USGS] Historical fetch error: {e}")
            return None

        site_to_name = {v: k for k, v in STATIONS.items()}
        series: Dict[str, List[float]] = {}

        try:
            for ts_data in data["value"]["timeSeries"]:
                site_no   = ts_data["sourceInfo"]["siteCode"][0]["value"]
                param_cd  = ts_data["variable"]["variableCode"][0]["value"]
                stat_name = site_to_name.get(site_no, "")
                param_name = PARAM_CODES.get(param_cd)
                if not stat_name or not param_name:
                    continue
                sensor_id = f"{stat_name}_{param_name}"
                values_raw = ts_data.get("values", [{}])[0].get("value", [])
                vals = []
                for entry in values_raw:
                    try:
                        v = float(entry["value"])
                        if v > -999990:
                            vals.append(v)
                    except (ValueError, TypeError, KeyError):
                        continue
                if vals:
                    series[sensor_id] = vals
        except (KeyError, IndexError, TypeError) as e:
            print(f"[USGS] Historical parse error: {e}")
            return None

        if not series:
            return None

        rows = []
        for sensor_id, vals in series.items():
            arr = np.array(vals)
            rows.append({
                "sensor_id": sensor_id,
                "mean": float(arr.mean()),
                "std":  max(float(arr.std()), float(arr.mean()) * 0.05 + 0.01),
                "min":  float(arr.min()),
                "max":  float(arr.max()),
            })

        df = pd.DataFrame(rows).set_index("sensor_id")
        print(f"[USGS] Historical stats computed for {len(df)} sensors (7-day baseline)")
        return df

    def seconds_until_refresh(self) -> int:
        if self._last_fetch is None:
            return 0
        remaining = self._cache_seconds - (time.time() - self._last_fetch)
        return max(0, int(remaining))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fetch_from_usgs(self) -> Optional[Dict]:
        site_list = ",".join(STATIONS.values())
        param_list = ",".join(PARAM_CODES.keys())
        params = {
            "format":      "json",
            "sites":       site_list,
            "parameterCd": param_list,
            "siteStatus":  "active",
        }
        try:
            resp = requests.get(USGS_BASE, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return self._parse_usgs_response(data)
        except Exception as e:
            print(f"[USGS] Fetch error: {e}")
            return None

    def _parse_usgs_response(self, data: dict) -> Dict[str, Any]:
        sensors: Dict[str, float] = {}
        ts_latest = datetime.now(timezone.utc)

        # Build reverse lookup: site_no → station_name
        site_to_name = {v: k for k, v in STATIONS.items()}

        try:
            for ts_data in data["value"]["timeSeries"]:
                site_no   = ts_data["sourceInfo"]["siteCode"][0]["value"]
                param_cd  = ts_data["variable"]["variableCode"][0]["value"]
                stat_name = STATIONS.get(site_to_name.get(site_no, ""), "")
                param_name = PARAM_CODES.get(param_cd)

                if not stat_name or not param_name:
                    continue

                values = ts_data.get("values", [{}])[0].get("value", [])
                if not values:
                    continue

                # Take the most recent non-null value
                for entry in reversed(values):
                    raw = entry.get("value", "-999999")
                    try:
                        val = float(raw)
                    except (ValueError, TypeError):
                        continue
                    if val > -999990:
                        sensor_id = f"{site_to_name[site_no]}_{param_name}"
                        sensors[sensor_id] = round(val, 4)
                        break
        except (KeyError, IndexError, TypeError) as e:
            print(f"[USGS] Parse error: {e}")

        if not sensors:
            return None

        return {
            "timestamp": ts_latest,
            "sensors":   sensors,
            "att_flag":  0,       # live data has no ground-truth labels
            "source":    "USGS_Live",
        }

    def _add_noise(self, cached: Dict) -> Dict:
        """Add ±0.5% Gaussian noise to simulate sub-15-min sensor updates."""
        rng = np.random.default_rng()
        noisy_sensors = {
            k: max(0.0, v * (1 + rng.normal(0, 0.005)))
            for k, v in cached["sensors"].items()
        }
        return {
            "timestamp": datetime.now(timezone.utc),
            "sensors":   noisy_sensors,
            "att_flag":  0,
            "source":    "USGS_Live",
        }

    def _fallback_reading(self) -> Dict:
        """Return zeroed reading if USGS is unreachable."""
        dummy_cols = [
            f"{name}_{param}"
            for name in STATIONS
            for param in PARAM_CODES.values()
        ]
        return {
            "timestamp": datetime.now(timezone.utc),
            "sensors":   {c: 0.0 for c in dummy_cols},
            "att_flag":  0,
            "source":    "USGS_Unavailable",
        }


# ── CLI test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Fetching live USGS water sensor data…")
    fetcher = USGSLiveFetcher()
    reading = fetcher.fetch()
    print(f"\nTimestamp : {reading['timestamp']}")
    print(f"Source    : {reading['source']}")
    print(f"Sensors   : {len(reading['sensors'])}")
    print("\nSample readings:")
    for k, v in list(reading["sensors"].items())[:10]:
        print(f"  {k:40s} = {v}")
    print(f"\nNext refresh in: {fetcher.seconds_until_refresh()}s")
