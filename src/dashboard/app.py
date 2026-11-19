"""SIH 2026 Problem Statement 26153 SOC dashboard."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from src.contracts import (
    FORECAST_STEPS,
    MITRE_STAGE_MAP,
    WINDOW_SIZE,
    get_mock_forecast,
)
from src.model.baseline import BaselineAttackClassifier, evaluate_models
from src.model.world_model import AttackForecasterNet, predict_window
from src.pipeline.preprocessor import FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[2]
DEMO_PATH = ROOT / "data" / "sample_demo.csv"
MODEL_PATH = ROOT / "artifacts" / "world_model.pt"
SCALER_PATH = ROOT / "artifacts" / "scaler.pkl"
BASELINE_PATH = ROOT / "artifacts" / "baseline_rf.pkl"


st.set_page_config(
    page_title="SIH 26153 SOC Attack Forecaster",
    page_icon="🛡️",
    layout="wide",
)
st.markdown(
    """
    <style>
    .stApp { background: #07111f; color: #e6edf7; }
    [data-testid="stMetric"] {
        background: #101d30; border: 1px solid #1e3a5f;
        border-radius: 12px; padding: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("SIH 26153 | SOC Attack Forecaster")
st.caption("MITRE ATT&CK escalation intelligence • World Model + Random Forest")

uploaded = st.sidebar.file_uploader("Upload network flow CSV", type=["csv"])
if uploaded is not None:
    data = pd.read_csv(uploaded)
else:
    data = pd.read_csv(DEMO_PATH) if DEMO_PATH.exists() else pd.DataFrame()

if data.empty:
    st.error("No demo data or uploaded CSV is available.")
    st.stop()

missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
if missing:
    st.error(f"CSV is missing required network features: {', '.join(missing)}")
    st.stop()

numeric = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
attack_cat = (
    data["attack_cat"].fillna("Normal").astype(str)
    if "attack_cat" in data.columns
    else pd.Series(["Normal"] * len(data))
)

try:
    scaler = joblib.load(SCALER_PATH)
    scaled = scaler.transform(numeric).astype(np.float32)
except Exception as exc:
    st.warning(f"Scaler unavailable; using live fallback forecast ({exc}).")
    scaled = numeric.to_numpy(dtype=np.float32)

total_windows = max(1, len(scaled) - WINDOW_SIZE + 1)
window_index = st.slider(
    "Traffic window replay",
    min_value=1,
    max_value=total_windows,
    value=total_windows,
)

col_play, col_next = st.columns([1, 1])
if col_play.button("▶ Play"):
    st.session_state["play_window"] = min(
        total_windows, st.session_state.get("play_window", window_index) + 1
    )
    window_index = st.session_state["play_window"]
    st.rerun()
if col_next.button("Next"):
    window_index = min(total_windows, window_index + 1)

start = min(window_index - 1, len(scaled) - WINDOW_SIZE)
window = scaled[start : start + WINDOW_SIZE]
forecast = None

try:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = AttackForecasterNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    forecast = predict_window(model, window, FEATURE_COLUMNS)
except Exception as exc:
    st.caption(f"Using fallback forecast: {exc}")
    forecast = get_mock_forecast()

kpi1, kpi2, kpi3 = st.columns(3)
current_stage = (
    str(attack_cat.iloc[start + WINDOW_SIZE - 1])
    if start + WINDOW_SIZE - 1 < len(attack_cat)
    else "Normal"
)
kpi1.metric("Current Network State", current_stage)
kpi2.metric("Predicted Next MITRE ATT&CK Stage", forecast.predicted_next_stage)
kpi3.metric(
    "Peak Future Compromise Probability",
    f"{100.0 * max(forecast.forecast_probs):.1f}%",
)

progress_col, feature_col = st.columns([2, 1])
with progress_col:
    horizon = [f"T+{step}" for step in range(1, FORECAST_STEPS + 1)]
    probabilities = np.asarray(forecast.forecast_probs[:FORECAST_STEPS]) * 100.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=horizon,
        y=probabilities,
        mode="lines+markers",
        name="Compromise Probability",
        line=dict(color="#35b9ff", width=4),
        marker=dict(size=10),
    ))
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="#ff6b6b",
        annotation_text="70% Critical",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Attack Progression Forecast",
        template="plotly_dark",
        yaxis=dict(title="Probability (%)", range=[0, 100]),
        xaxis=dict(title="Forecast Horizon"),
        paper_bgcolor="#07111f",
        plot_bgcolor="#101d30",
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

with feature_col:
    names = [item["feature"] for item in forecast.top_features[:4]]
    importance = [float(item["importance"]) for item in forecast.top_features[:4]]
    importance = np.asarray(importance)
    if importance.sum() > 0:
        importance = importance / importance.sum()
    importance *= 100.0
    fig_features = go.Figure(
        go.Bar(
            x=importance,
            y=names,
            orientation="h",
            marker=dict(
                color=importance,
                colorscale=[[0, "#17324d"], [1, "#35b9ff"]],
            ),
        )
    )
    fig_features.update_layout(
        title="Top 4 Driving Features",
        template="plotly_dark",
        xaxis=dict(title="Relative Importance (%)"),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="#07111f",
        plot_bgcolor="#101d30",
        height=380,
    )
    st.plotly_chart(fig_features, use_container_width=True)

st.subheader("World Model vs. Random Forest Baseline")
try:
    baseline = BaselineAttackClassifier.load(str(BASELINE_PATH))
    if len(scaled) >= WINDOW_SIZE:
        sequence_features = np.stack(
            [scaled[index : index + WINDOW_SIZE] for index in range(len(scaled) - WINDOW_SIZE + 1)]
        )
        baseline_probs = baseline.predict_attack_proba(sequence_features)
        if "attack_cat" in data.columns:
            true_labels = (
                data["attack_cat"].fillna("Normal").astype(str).str.lower()
                .isin(["normal", "benign"])
            )
            y_true = (
                ~true_labels.iloc[WINDOW_SIZE - 1 :].to_numpy()
            ).astype(int)
            y_true = y_true[: len(baseline_probs)]
            checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            world_model = AttackForecasterNet()
            world_model.load_state_dict(checkpoint["model_state_dict"])
            world_model.eval()
            with torch.no_grad():
                world_probs = world_model(
                    torch.from_numpy(sequence_features.astype(np.float32))
                )[0].cpu().numpy()[:, 0]
            world_probs = world_probs[: len(y_true)]
            baseline_probs = baseline_probs[: len(y_true)]
            _, benchmark = evaluate_models(world_probs, baseline_probs, y_true)
        else:
            raise ValueError("attack_cat labels unavailable")
    else:
        raise ValueError("Not enough rows for benchmark")
except Exception as exc:
    st.info(f"Baseline benchmark unavailable: {exc}")
    benchmark = pd.DataFrame(
        {
            "Accuracy": [0.0],
            "Precision": [0.0],
            "Recall": [0.0],
            "F1-Score": [0.0],
            "ROC-AUC": [0.0],
        },
        index=["World Model", "Baseline RF"],
    )

st.dataframe(
    benchmark.style.format("{:.4f}") if hasattr(benchmark, "style") else benchmark,
    use_container_width=True,
)
