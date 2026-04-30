"""
Digital Twin Dashboard — Streamlit front-end (Layer 4).

Run with:
    streamlit run dashboard/app.py

Controls
--------
  Sidebar → Speed slider (simulation delay)
  Sidebar → Sensor fault injection (what-if)
  Sidebar → Model toggles
  Main    → Live sensor charts, anomaly alerts, failure gauges, RUL display
"""

import os
import sys
import time
import threading
import queue
from collections import deque

# TF must be imported before scikit-learn to avoid Windows segfault
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    import tensorflow as _tf  # noqa: F401 — side-effect import order matters
except Exception:
    pass

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from utils.data_loader import (
    load_batadal,
    stream_batadal,
    train_test_split_batadal,
    compute_sensor_stats,
    SENSOR_GROUPS,
)
from utils.live_data_fetcher import USGSLiveFetcher, LIVE_SENSOR_GROUPS
from twin.digital_twin import DigitalTwin

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digital Twin — Smart Infrastructure",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .alert-red   { background:#ff4b4b22; border-left:4px solid #ff4b4b;
                 padding:10px; border-radius:4px; margin:4px 0; }
  .alert-green { background:#21c35422; border-left:4px solid #21c354;
                 padding:10px; border-radius:4px; margin:4px 0; }
  .metric-box  { background:#1e1e2e; border-radius:8px; padding:12px;
                 text-align:center; }
  .stMetric    { background:#1a1a2e; border-radius:8px; }
  h1           { color:#00d4ff; }
</style>
""", unsafe_allow_html=True)


# Bump this version string whenever DigitalTwin or data logic changes —
# it forces a clean session state reset on next browser load.
_STATE_VERSION = "v4"

# ── Session-state initialisation ───────────────────────────────────────────────
def _init_state():
    # Reset everything if the code version changed (stale session after hot-reload)
    if st.session_state.get("_state_version") != _STATE_VERSION:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state["_state_version"] = _STATE_VERSION

    defaults = {
        "running":          False,
        "data_mode":        "Historical (BATADAL)",   # or "Live (USGS)"
        "live_fetcher":     None,
        "twin":             None,
        "sensor_cols":      [],
        "df":               None,
        "stream_pos":       0,
        "history_buf":      {},    # sensor_id -> deque of recent values
        "window_buf":       [],    # list of sensor dicts for LSTM/RUL
        "if_model":         None,
        "if_scaler":        None,
        "lstm_model":       None,
        "lstm_scaler":      None,
        "rul_model":        None,
        "rul_scaler":       None,
        "rul_feat_cols":    None,
        "models_loaded":    False,
        "failure_prob":     0.0,
        "rul_value":        200.0,
        "anomaly_score":    0.0,
        "alert_log":        deque(maxlen=50),
        "total_readings":   0,
        "total_anomalies":  0,
        "last_att_flag":    0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Data + model loading ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading BATADAL dataset…")
def get_data():
    df, sensor_cols = load_batadal(DATA_DIR)
    _, stream_df = train_test_split_batadal(df, train_ratio=0.7)
    train_df, _ = train_test_split_batadal(df, train_ratio=0.7)
    stats = compute_sensor_stats(train_df, sensor_cols)
    return df, stream_df, sensor_cols, stats


@st.cache_resource(show_spinner="Loading ML models…")
def get_models():
    import joblib
    results = {}

    # Isolation Forest
    if_path = os.path.join(MODELS_DIR, "isolation_forest.pkl")
    if os.path.exists(if_path):
        results["if_model"]  = joblib.load(if_path)
        results["if_scaler"] = joblib.load(os.path.join(MODELS_DIR, "if_scaler.pkl"))
        results["if_cols"]   = joblib.load(os.path.join(MODELS_DIR, "if_sensor_cols.pkl"))
    else:
        results["if_model"] = None

    # LSTM
    lstm_path = os.path.join(MODELS_DIR, "lstm_model.keras")
    if os.path.exists(lstm_path):
        import tensorflow as tf
        results["lstm_model"]  = tf.keras.models.load_model(lstm_path)
        results["lstm_scaler"] = joblib.load(os.path.join(MODELS_DIR, "lstm_scaler.pkl"))
        results["lstm_cols"]   = joblib.load(os.path.join(MODELS_DIR, "lstm_sensor_cols.pkl"))
    else:
        results["lstm_model"] = None

    # RUL
    rul_path = os.path.join(MODELS_DIR, "rul_model.pkl")
    if os.path.exists(rul_path):
        results["rul_model"]     = joblib.load(rul_path)
        results["rul_scaler"]    = joblib.load(os.path.join(MODELS_DIR, "rul_scaler.pkl"))
        results["rul_cols"]      = joblib.load(os.path.join(MODELS_DIR, "rul_sensor_cols.pkl"))
        results["rul_feat_cols"] = joblib.load(os.path.join(MODELS_DIR, "rul_feat_cols.pkl"))
    else:
        results["rul_model"] = None

    return results


# ── Plotly helpers ─────────────────────────────────────────────────────────────

def make_gauge(value: float, title: str, max_val: float = 1.0,
               thresholds=(0.4, 0.7)) -> go.Figure:
    color = "#21c354" if value < thresholds[0] else ("#ff9800" if value < thresholds[1] else "#ff4b4b")
    display_val = value * 100 if max_val == 1.0 else value
    axis_max    = 100 if max_val == 1.0 else max_val
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_val,
        number={"suffix": "%" if max_val == 1.0 else "", "font": {"size": 26, "color": "white"}},
        title={"text": f"<b>{title}</b>", "font": {"size": 16, "color": "white"}},
        gauge={
            "axis":  {
                "range": [0, axis_max],
                "tickcolor": "white",
                "tickfont": {"color": "white", "size": 11},
            },
            "bar":   {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,          axis_max * 0.4], "color": "#1b4332"},
                {"range": [axis_max * 0.4, axis_max * 0.7], "color": "#7d4e00"},
                {"range": [axis_max * 0.7, axis_max],   "color": "#5c0a0a"},
            ],
            "threshold": {
                "line":  {"color": "white", "width": 3},
                "value": display_val,
            },
        },
    ))
    fig.update_layout(
        height=200,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
    )
    return fig


def make_sensor_chart(timestamps: list, values: list, anomalies: list,
                      sensor_id: str, injected: bool = False) -> go.Figure:
    colors = ["#ff4b4b" if a else "#00d4ff" for a in anomalies]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=values,
        mode="lines",
        line=dict(color="#00d4ff" if not injected else "#ff9800", width=2),
        name=sensor_id,
    ))
    # Mark anomaly points
    anom_ts = [t for t, a in zip(timestamps, anomalies) if a]
    anom_vs = [v for v, a in zip(values, anomalies) if a]
    if anom_ts:
        fig.add_trace(go.Scatter(
            x=anom_ts, y=anom_vs,
            mode="markers",
            marker=dict(color="#ff4b4b", size=8, symbol="x"),
            name="Anomaly",
        ))
    fig.update_layout(
        height=180, margin=dict(t=20, b=20, l=40, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.2)",
        font_color="white", showlegend=False,
        xaxis=dict(showgrid=False, color="gray"),
        yaxis=dict(showgrid=True,  gridcolor="rgba(255,255,255,0.1)", color="gray"),
        title=dict(text=f"{'⚡ ' if injected else ''}{sensor_id}", font=dict(size=13)),
    )
    return fig


def make_alert_history_chart(alert_log: deque) -> go.Figure:
    if not alert_log:
        return go.Figure()
    rows = list(alert_log)
    df_log = pd.DataFrame(rows)
    fig = px.scatter(
        df_log, x="timestamp", y="sensor_id", color="score",
        color_continuous_scale="RdYlGn_r",
        title="Anomaly Event Log",
        labels={"score": "Anomaly Score"},
    )
    fig.update_layout(
        height=220, margin=dict(t=35, b=20, l=100, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.2)",
        font_color="white",
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar(sensor_cols, twin):
    with st.sidebar:
        st.title("Control Panel")
        st.divider()

        # ── Data source mode ─────────────────────────────────────────────────
        st.subheader("Data Source")
        mode = st.radio(
            "Select data source",
            ["Historical (BATADAL)", "Live (USGS)"],
            index=0 if st.session_state.data_mode == "Historical (BATADAL)" else 1,
            key="mode_radio",
            help="Historical: replays BATADAL CSV\nLive: real US water sensors via USGS API",
        )
        if mode != st.session_state.data_mode:
            # Reset twin when switching modes
            st.session_state.data_mode  = mode
            st.session_state.twin       = None
            st.session_state.sensor_cols = []
            st.session_state.history_buf = {}
            st.session_state.window_buf  = []
            st.session_state.stream_pos  = 0
            st.session_state.running     = False
            st.rerun()

        if mode == "Live (USGS)":
            fetcher = st.session_state.get("live_fetcher")
            if fetcher is not None:
                secs = fetcher.seconds_until_refresh()
                mins, s = divmod(secs, 60)
                st.caption(f"🌊 Live USGS data — next API refresh in **{mins}m {s:02d}s**")
                st.caption("Between refreshes values animate with ±0.5% noise")
            else:
                st.caption("🌊 Will connect to USGS on Start")

        st.divider()

        # Run / Stop
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ Start", type="primary", use_container_width=True,
                          disabled=st.session_state.running):
                st.session_state.running = True
                st.rerun()
        with col2:
            if st.button("⏹ Stop", use_container_width=True,
                          disabled=not st.session_state.running):
                st.session_state.running = False
                st.rerun()

        st.divider()
        st.subheader("Simulation Speed")
        speed_label = "Refresh interval (s)" if mode == "Live (USGS)" else "Delay between readings (s)"
        speed = st.slider(speed_label, 0.05, 2.0, 1.0 if mode == "Live (USGS)" else 0.3, 0.05,
                          key="speed_slider")

        st.divider()
        st.subheader("What-If Fault Injection")
        st.caption("Drag a slider to override a sensor value and watch the system react.")

        fault_sensor = st.selectbox("Sensor to fault", ["— None —"] + sensor_cols,
                                    key="fault_sensor_select")
        if fault_sensor != "— None —" and twin is not None:
            cur_state = twin.get_state()
            cur_val   = cur_state.get(fault_sensor, {}).get("value", 50.0)
            injected  = cur_state.get(fault_sensor, {}).get("injected", False)

            fault_val = st.slider(
                f"Override value for {fault_sensor}",
                min_value=0.0, max_value=cur_val * 3 + 10,
                value=cur_val, step=0.5,
                key=f"fault_val_{fault_sensor}",
            )
            cols2 = st.columns(2)
            with cols2[0]:
                if st.button("Inject Fault", use_container_width=True, type="primary"):
                    twin.inject_fault(fault_sensor, fault_val)
                    st.success(f"Injected {fault_val:.2f} → {fault_sensor}")
            with cols2[1]:
                if st.button("Clear Fault", use_container_width=True):
                    twin.clear_fault(fault_sensor)
                    st.info(f"Cleared fault on {fault_sensor}")

        if twin is not None and st.button("Clear ALL Faults"):
            twin.clear_all_faults()

        st.divider()
        st.subheader("Model Settings")
        use_if   = st.checkbox("Isolation Forest (anomaly)",  True, key="use_if")
        use_lstm = st.checkbox("LSTM (failure prediction)",   True, key="use_lstm")
        use_rul  = st.checkbox("RUL Regression (XGBoost)",    True, key="use_rul")

        st.divider()
        st.subheader("Display")
        active_groups = LIVE_SENSOR_GROUPS if mode == "Live (USGS)" else SENSOR_GROUPS
        show_groups = {}
        for group_name in active_groups:
            show_groups[group_name] = st.checkbox(group_name, True, key=f"grp_{group_name}")

        return speed, use_if, use_lstm, use_rul, show_groups, mode


# ── Main rendering loop ────────────────────────────────────────────────────────

def render_main(twin, sensor_cols, models, speed, use_if, use_lstm, use_rul, show_groups, mode):
    is_live = (mode == "Live (USGS)")
    st.title("🏗️ Digital Twin — Smart Infrastructure Monitor")
    if is_live:
        st.caption("🌊 **Live Mode** — Real-time sensor data from USGS US water monitoring stations")
    else:
        st.caption("📂 **Historical Mode** — Replaying BATADAL water distribution network dataset")

    if not st.session_state.running:
        if is_live:
            st.info("Press ▶ Start to connect to live USGS water sensor data.")
        else:
            st.info("Press ▶ Start in the sidebar to begin the simulation.")
        return

    # ── KPI row ─────────────────────────────────────────────────────────────
    summary = twin.get_summary()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("System Health", f"{summary['health_pct']}%")
    k2.metric("Active Alerts",   summary["active_alerts"])
    k3.metric("Total Readings",  st.session_state.total_readings)
    k4.metric("Total Anomalies", st.session_state.total_anomalies)
    k5.metric("Data Source", "🌊 USGS Live" if is_live else "📂 BATADAL")

    st.divider()

    # ── Gauge row ───────────────────────────────────────────────────────────
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(
            make_gauge(st.session_state.anomaly_score, "Anomaly Score (Isolation Forest)"),
            use_container_width=True, key="gauge_if"
        )
        score = st.session_state.anomaly_score
        label = "🔴 Anomaly detected" if score > 0.7 else ("🟡 Suspicious" if score > 0.4 else "🟢 Normal")
        st.markdown(f"<div style='text-align:center;font-size:13px;margin-top:-10px'>"
                    f"How abnormal the current sensor readings are<br><b>{label}</b></div>",
                    unsafe_allow_html=True)
    with g2:
        st.plotly_chart(
            make_gauge(st.session_state.failure_prob, "Failure Probability (LSTM)"),
            use_container_width=True, key="gauge_lstm"
        )
        prob = st.session_state.failure_prob
        label = "🔴 Failure imminent" if prob > 0.7 else ("🟡 Elevated risk" if prob > 0.4 else "🟢 Low risk")
        st.markdown(f"<div style='text-align:center;font-size:13px;margin-top:-10px'>"
                    f"Probability of failure in next 6 readings<br><b>{label}</b></div>",
                    unsafe_allow_html=True)
    with g3:
        rul = st.session_state.rul_value
        st.plotly_chart(
            make_gauge(rul, "Remaining Useful Life (XGBoost)", max_val=200, thresholds=(50, 20)),
            use_container_width=True, key="gauge_rul"
        )
        label = "🔴 Failure soon" if rul < 20 else ("🟡 Plan maintenance" if rul < 50 else "🟢 Healthy")
        st.markdown(f"<div style='text-align:center;font-size:13px;margin-top:-10px'>"
                    f"Estimated timesteps until next failure event<br><b>{label}</b></div>",
                    unsafe_allow_html=True)

    st.divider()

    # ── Alert panel ─────────────────────────────────────────────────────────
    alerts = twin.get_alerts()
    if alerts:
        st.subheader(f"🚨 Active Alerts ({len(alerts)})")
        for sid, info in alerts.items():
            st.markdown(
                f'<div class="alert-red">⚠️ <b>{sid}</b> — value: '
                f'<b>{info["value"]:.3f}</b> | score: {info["score"]:.3f} | '
                f'{info["timestamp"]}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="alert-green">✅ All sensors operating within normal range</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Sensor charts ────────────────────────────────────────────────────────
    state = twin.get_state()
    buf   = st.session_state.history_buf

    active_groups = LIVE_SENSOR_GROUPS if is_live else SENSOR_GROUPS
    first_group   = next(iter(active_groups))
    for group_name, sensor_list in active_groups.items():
        if not show_groups.get(group_name, True):
            continue
        present = [s for s in sensor_list if s in state]
        if not present:
            continue

        with st.expander(f"📡 {group_name} ({len(present)} sensors)", expanded=(group_name == first_group)):
            cols = st.columns(min(len(present), 3))
            for i, sid in enumerate(present):
                hist = list(buf.get(sid, deque()))
                ts   = [r["timestamp"] for r in hist]
                vs   = [r["value"]     for r in hist]
                an   = [r["anomaly"]   for r in hist]
                inj  = state.get(sid, {}).get("injected", False)
                with cols[i % 3]:
                    st.plotly_chart(
                        make_sensor_chart(ts, vs, an, sid, injected=inj),
                        use_container_width=True,
                        key=f"chart_{sid}",
                    )

    # ── Alert history ────────────────────────────────────────────────────────
    if st.session_state.alert_log:
        st.divider()
        st.subheader("Anomaly Event Timeline")
        st.plotly_chart(
            make_alert_history_chart(st.session_state.alert_log),
            use_container_width=True, key="alert_timeline"
        )

    # ── Ground truth / source info ───────────────────────────────────────────
    if is_live:
        fetcher = st.session_state.get("live_fetcher")
        if fetcher:
            secs = fetcher.seconds_until_refresh()
            mins, s = divmod(secs, 60)
            st.info(f"🌊 Live USGS data — next API refresh in **{mins}m {s:02d}s** "
                    f"| Stations: Potomac MD, Neuse NC, Mississippi LA/IA, Colorado AZ, Sacramento CA, TugFork WV, SouthPlatte CO")
    else:
        if st.session_state.last_att_flag == 1:
            st.warning("⚠️ Ground truth: ATTACK period in dataset (ATT_FLAG=1)")
        else:
            st.success("Ground truth: Normal operation (ATT_FLAG=0)")


# ── Simulation step ────────────────────────────────────────────────────────────

def simulation_step(reading, twin, models, use_if, use_lstm, use_rul):
    """Process one streaming reading through the twin and ML models."""
    sensors = reading["sensors"]
    ts      = reading["timestamp"]
    att     = reading["att_flag"]

    # Update digital twin state
    twin.update(sensors, ts)
    st.session_state.total_readings += 1
    st.session_state.last_att_flag   = att

    # ── Isolation Forest scoring ─────────────────────────────────────────────
    if use_if and models.get("if_model") is not None:
        from models.train_isolation_forest import predict as if_predict
        is_anom, score = if_predict(
            models["if_model"], models["if_scaler"],
            models["if_cols"], sensors
        )
        st.session_state.anomaly_score = score
        if is_anom:
            # Flag per-sensor based on IF result for top deviating sensors
            for sid in list(sensors.keys())[:3]:
                twin.flag_anomaly(sid, score)
            st.session_state.total_anomalies += 1
            st.session_state.alert_log.append({
                "timestamp": str(ts),
                "sensor_id": "IF-composite",
                "score":     score,
            })

    # ── LSTM failure probability ─────────────────────────────────────────────
    if use_lstm and models.get("lstm_model") is not None:
        from models.train_lstm import predict_proba
        prob = predict_proba(
            models["lstm_model"], models["lstm_scaler"],
            models["lstm_cols"], st.session_state.window_buf
        )
        st.session_state.failure_prob = prob

    # ── RUL prediction ───────────────────────────────────────────────────────
    if use_rul and models.get("rul_model") is not None:
        from models.train_rul import predict_rul
        rul = predict_rul(
            models["rul_model"], models["rul_scaler"],
            models["rul_cols"], models["rul_feat_cols"],
            st.session_state.window_buf
        )
        st.session_state.rul_value = rul

    # ── Update per-sensor history buffer ────────────────────────────────────
    state = twin.get_state()
    buf   = st.session_state.history_buf
    for sid, info in state.items():
        if sid not in buf:
            buf[sid] = deque(maxlen=100)
        buf[sid].append({
            "timestamp": str(ts),
            "value":     info["value"],
            "anomaly":   info["anomaly"],
        })

    # ── Rolling window buffer for sequence models ────────────────────────────
    st.session_state.window_buf.append(sensors)
    if len(st.session_state.window_buf) > 30:
        st.session_state.window_buf.pop(0)


# ── App entry-point ────────────────────────────────────────────────────────────

def main():
    mode    = st.session_state.data_mode
    is_live = (mode == "Live (USGS)")
    models  = get_models()

    # ── Initialise data source + DigitalTwin ─────────────────────────────────
    if is_live:
        # Lazy-init live fetcher
        if st.session_state.live_fetcher is None:
            st.session_state.live_fetcher = USGSLiveFetcher(cache_seconds=900)
        fetcher = st.session_state.live_fetcher

        if st.session_state.twin is None:
            # Do an initial fetch to discover sensor columns
            init_reading = fetcher.fetch()
            sensor_cols  = list(init_reading["sensors"].keys())
            stats        = fetcher.compute_stats()
            twin         = DigitalTwin(sensor_cols, history_len=200)
            twin.set_thresholds(stats)
            st.session_state.twin        = twin
            st.session_state.sensor_cols = sensor_cols
        else:
            twin        = st.session_state.twin
            sensor_cols = st.session_state.sensor_cols
            fetcher     = st.session_state.live_fetcher
    else:
        # Historical BATADAL mode
        full_df, stream_df, sensor_cols, stats = get_data()
        if st.session_state.twin is None:
            twin = DigitalTwin(sensor_cols, history_len=200)
            twin.set_thresholds(stats)
            st.session_state.twin        = twin
            st.session_state.sensor_cols = sensor_cols
            st.session_state.df          = stream_df
        else:
            twin        = st.session_state.twin
            sensor_cols = st.session_state.sensor_cols
            stream_df   = st.session_state.df

    # Render sidebar
    speed, use_if, use_lstm, use_rul, show_groups, mode = render_sidebar(sensor_cols, twin)

    # ── One simulation step per rerun ─────────────────────────────────────────
    if st.session_state.running:
        if is_live:
            reading = st.session_state.live_fetcher.fetch()
            simulation_step(reading, twin, models, use_if, use_lstm, use_rul)
        else:
            pos = st.session_state.stream_pos
            if pos < len(stream_df):
                reading = next(stream_batadal(stream_df, sensor_cols, speed=0, start_row=pos))
                simulation_step(reading, twin, models, use_if, use_lstm, use_rul)
                st.session_state.stream_pos = pos + 1
            else:
                st.session_state.stream_pos = 0

    # Render main panel
    render_main(twin, sensor_cols, models, speed, use_if, use_lstm, use_rul, show_groups, mode)

    # Auto-refresh
    if st.session_state.running:
        time.sleep(speed)
        st.rerun()


if __name__ == "__main__":
    main()
