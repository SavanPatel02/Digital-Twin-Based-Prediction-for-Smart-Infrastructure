# Digital Twin Based Predictive Analytics for Smart Infrastructure

A live simulation system that ingests open-source infrastructure sensor data,
maintains a real-time digital twin state model, runs ML models to predict
failures and anomalies, and displays everything on an interactive Streamlit
dashboard.

---

## What is a Digital Twin?

A **Digital Twin** is a virtual representation of a physical system that
mirrors its real-time state, enabling monitoring, simulation, and predictive
analytics without touching the physical asset.  

In this project the physical system is a **water distribution network**
(BATADAL dataset).  The digital twin maintains the live state of 43 sensors
(tank levels, pump flows, pressures, valve statuses) and feeds three ML
models that detect anomalies, predict upcoming failures, and estimate
remaining useful life (RUL).

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1 — DATA INGESTION                               │
│  utils/data_loader.py      (Historical — BATADAL CSV)   │
│  • Load CSV  →  parse timestamps  →  clean              │
│  • stream_batadal() generator: one row per tick         │
│                                                         │
│  utils/live_data_fetcher.py  (Live — USGS NWIS API)     │
│  • Poll 8 US water stations every 15 min (no key)       │
│  • 40 real sensor streams (flow, level, temp, etc.)     │
│  • 7-day baseline for auto-threshold calibration        │
└───────────────────────┬─────────────────────────────────┘
                        │ {sensor_id: value, timestamp}
┌───────────────────────▼─────────────────────────────────┐
│  Layer 2 — DIGITAL TWIN STATE MODEL                     │
│  twin/digital_twin.py  ›  DigitalTwin class             │
│  • update()      — ingest snapshot, check thresholds    │
│  • flag_anomaly()— ML layer flags a sensor              │
│  • inject_fault()— what-if override (presentation demo) │
│  • get_state()   — JSON-ready current snapshot          │
│  • get_history() — per-sensor rolling buffer            │
└───────────────────────┬─────────────────────────────────┘
                        │ anomaly flags + state
┌───────────────────────▼─────────────────────────────────┐
│  Layer 3 — ML / PREDICTIVE ANALYTICS                    │
│  models/                                                │
│  ├── train_isolation_forest.py  (Model A — anomaly)     │
│  ├── train_lstm.py              (Model B — failure prob)│
│  └── train_rul.py               (Model C — RUL)         │
└───────────────────────┬─────────────────────────────────┘
                        │ anomaly_score, failure_prob, rul
┌───────────────────────▼─────────────────────────────────┐
│  Layer 4 — STREAMLIT DASHBOARD                          │
│  dashboard/app.py                                       │
│  • Live sensor charts (auto-refresh)                    │
│  • Failure probability gauges                           │
│  • Alert panel (RED / GREEN)                            │
│  • What-if fault injection sliders                      │
└─────────────────────────────────────────────────────────┘
```

---

## Data Sources

The system supports two data modes selectable from the dashboard sidebar.

---

### Mode 1 — Historical: BATADAL Dataset

| Property | Value |
|---|---|
| Name | BATADAL (Battle of the Attack Detection ALgorithms) |
| Domain | Water distribution network |
| Rows | ~8 700 (hourly readings, ~1 year) |
| Sensors | 43 (tank levels, pump flows, pressures, valve status) |
| Labels | `ATT_FLAG` — 1 during injected attack periods |
| Source | https://www.batadal.net/data.html |

Both labeled datasets are included in `data/`:

| File | Description |
|---|---|
| `training_dataset_1.csv` | 1 year of normal operations (labeled) |
| `training_dataset_2.csv` | 6 months with injected attack periods |

> **No dataset?** The system auto-generates synthetic BATADAL-like data so
> you can run the demo immediately without downloading anything.

---

### Mode 2 — Live: USGS National Water Information System (NWIS)

Real-time water sensor readings streamed directly from the **USGS Instantaneous Values API**.  
No API key required. Data updates every **15 minutes**.

**API:** https://waterservices.usgs.gov/rest/IV-Service.html

#### Monitoring Stations (8 geographically distributed across the US)

| Station Name | USGS ID | Location |
|---|---|---|
| Potomac River at Little Falls | 01646500 | Maryland |
| Neuse River at Kinston | 02087500 | North Carolina |
| Mississippi River at Baton Rouge | 07374000 | Louisiana |
| Colorado River at Lees Ferry | 09380000 | Arizona |
| Sacramento River at Sacramento | 11447650 | California |
| Tug Fork at Williamson | 03213700 | West Virginia |
| Mississippi River at Clinton | 05420500 | Iowa |
| South Platte River at Denver | 06710000 | Colorado |

#### Parameters Fetched per Station (5 sensors)

| USGS Code | Parameter | Unit |
|---|---|---|
| 00060 | Discharge / Flow rate | ft³/s |
| 00065 | Gage height / Water level | ft |
| 00010 | Water temperature | °C |
| 00095 | Specific conductance | µS/cm |
| 00300 | Dissolved oxygen | mg/L |

Total: **40 live sensor streams** (8 stations × 5 parameters).

Between 15-minute API refreshes, the dashboard animates readings with ±0.5% realistic noise so charts remain live.  
Thresholds are auto-calibrated from a **7-day historical baseline** fetched at startup.

---

## ML Models

| # | Model | Task | Algorithm |
|---|---|---|---|
| A | Isolation Forest | Real-time anomaly detection | scikit-learn |
| B | LSTM | Failure probability (next 6 steps) | Keras / TensorFlow |
| C | RUL Regressor | Remaining useful life (steps to failure) | XGBoost |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Download BATADAL data

Place CSVs in `data/`.  Skip this step to use synthetic data.

### 3. Train the models

```bash
python train_all.py
```

This trains Isolation Forest → LSTM → RUL regressor and saves all artefacts
to `models/`.

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

---

## Dashboard Guide

| Control | What it does |
|---|---|
| ▶ Start | Begin streaming simulation |
| ⏹ Stop | Pause streaming |
| Speed slider | Seconds between timesteps (0.05–2 s) |
| Fault sensor dropdown | Pick a sensor to override |
| Fault value slider | Drag to inject a custom reading |
| Inject Fault button | Apply the override — watch alerts fire |
| Clear Fault button | Restore normal sensor feed |
| Model checkboxes | Toggle IF / LSTM / RUL on or off |
| Sensor group checkboxes | Show/hide chart panels |

---

## Project Structure

```
digital_twin_project/
├── data/
│   ├── training_dataset_1.csv     ← BATADAL Dataset 1 (1 year, labeled normal ops)
│   └── training_dataset_2.csv     ← BATADAL Dataset 2 (6 months, attack periods)
├── models/
│   ├── train_isolation_forest.py
│   ├── train_lstm.py
│   └── train_rul.py
├── twin/
│   └── digital_twin.py            ← DigitalTwin class
├── dashboard/
│   └── app.py                     ← Streamlit app (Historical + Live modes)
├── utils/
│   ├── data_loader.py             ← BATADAL streaming simulator
│   └── live_data_fetcher.py       ← USGS live API fetcher (40 sensors, 8 stations)
├── train_all.py                   ← one-command training
├── requirements.txt
└── README.md
```

---

## Key Research References

- Fuller A. et al. (2020) — *Digital Twin: Enabling Technologies, Challenges
  and Open Research*, IEEE Access
- Rathore M.M. et al. (2021) — *The Role of AI, ML, and Big Data in Digital
  Twinning*, IEEE Access
- Wei X. et al. (2022) — *Anomaly Detection for Water Treatment via Digital
  Twin*, Water (MDPI)
- Mücke T. et al. (2023) — *Leak Localization using Digital Twin + Deep
  Learning*, Sensors (MDPI)
- Armijo & Zamora-Sanchez (2024) — *Railway Bridge SHM with Digital Twin*,
  Sensors (MDPI)

---

## License

MIT — free to use and modify for academic and research purposes.
