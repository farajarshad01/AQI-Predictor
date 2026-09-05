import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import LATITUDE, LONGITUDE
from src.prediction_pipeline import predict

st.set_page_config(page_title="Gujranwala Air Quality", layout="wide")

TEXT_COLOR = "#1c1d1f"
CARD_BACKGROUND = "#F3FDFE"
BACKGROUND = "#f7f7f7"
BORDER = "#e5e5e5"
GRID_COLOR = "#e5e5e5"
GOOD_COLOR = "#00E400"
MODERATE_COLOR = "#FFFF00"
SENSITIVE_COLOR = "#FF7E00"
UNHEALTHY_COLOR = "#FF0000"
VERY_UNHEALTHY_COLOR = "#8F3F97"
HAZARDOUS_COLOR = "#7E0023"
LOCAL_TZ = ZoneInfo("Asia/Karachi")
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

st.markdown(
    f"""
    <style>
    .stApp,.main{{background-color:{BACKGROUND};color:{TEXT_COLOR};}}
    h1,h2,h3,h4,h5,h6,p,span,label{{color:{TEXT_COLOR}!important;}}
    [data-testid="stHeader"]{{background:transparent;}}
    [data-testid="stSidebar"]{{background-color:{CARD_BACKGROUND};border-right:1px solid {BORDER};}}
    [data-testid="stSidebar"] *{{color:{TEXT_COLOR}!important;}}
    .dashboard-title{{font-size:2.2rem;font-weight:700;margin-bottom:.15rem;}}
    .dashboard-location{{font-size:1rem;font-weight:500;margin-bottom:.25rem;}}
    .dashboard-subtitle{{font-size:.9rem;color:#555!important;margin-bottom:1.5rem;}}
    .section-title{{font-size:1.25rem;font-weight:700;margin:1.5rem 0 .8rem;}}
    .forecast-card,.current-aqi-card,.pollutant-card{{
        background:{CARD_BACKGROUND};border:1px solid {BORDER};
        box-shadow:0 2px 8px rgba(0,0,0,.04);
    }}
    .forecast-card{{border-radius:14px;padding:1.25rem;min-height:150px;}}
    .forecast-horizon,.current-aqi-label{{font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em;}}
    .forecast-value{{font-size:2.8rem;font-weight:750;margin:.65rem 0 .55rem;}}
    .forecast-time,.current-aqi-fetched{{font-size:.72rem;color:#777!important;margin-top:.75rem;}}
    .current-aqi-card{{border-radius:14px;padding:1.25rem 1.5rem;display:flex;align-items:center;justify-content:space-between;gap:1rem;min-height:135px;}}
    .current-aqi-value{{font-size:3.2rem;font-weight:750;line-height:1.05;margin-top:.35rem;}}
    .current-aqi-side{{text-align:right;max-width:420px;}}
    .pollutant-card{{border-radius:12px;padding:.9rem .8rem;min-height:95px;}}
    .pollutant-name{{font-size:.72rem;font-weight:700;}}
    .pollutant-value{{font-size:1.35rem;font-weight:750;margin-top:.35rem;}}
    .pollutant-unit{{font-size:.68rem;color:#777!important;}}
    .footer{{text-align:center;color:#777!important;font-size:.72rem;padding:2rem 0 1rem;}}
    @media(max-width:768px){{
        .dashboard-title{{font-size:1.8rem;}}
        .current-aqi-card{{align-items:flex-start;flex-direction:column;}}
        .current-aqi-side{{text-align:left;max-width:none;}}
        .current-aqi-value{{font-size:2.6rem;}}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", GOOD_COLOR, "Air quality is considered satisfactory."
    if aqi <= 100:
        return "Moderate", MODERATE_COLOR, "Air quality is acceptable; unusually sensitive people may experience minor effects."
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups", SENSITIVE_COLOR, "Sensitive groups may experience health effects."
    if aqi <= 200:
        return "Unhealthy", UNHEALTHY_COLOR, "Everyone may begin to experience health effects."
    if aqi <= 300:
        return "Very Unhealthy", VERY_UNHEALTHY_COLOR, "Health alert: the risk of health effects is increased."
    return "Hazardous", HAZARDOUS_COLOR, "Health warning of emergency conditions."

def aqi_color(value):
    return get_aqi_category(float(value))[1]

FEATURE_LABELS = {
    "temperature_2m":"Temperature","relative_humidity_2m":"Relative Humidity",
    "wind_speed_10m":"Wind Speed","surface_pressure":"Surface Pressure",
    "precipitation":"Precipitation","cloud_cover":"Cloud Cover","pm10":"PM10",
    "pm2_5":"PM2.5","carbon_monoxide":"Carbon Monoxide",
    "nitrogen_dioxide":"Nitrogen Dioxide","sulphur_dioxide":"Sulphur Dioxide",
    "ozone":"Ozone","us_aqi":"Previous AQI","year":"Year","month":"Month",
    "day":"Day","day_of_week":"Day of Week","hour":"Hour","aqi_change":"AQI Change",
    "aqi_change_rate":"AQI Change Rate","aqi_lag_1":"AQI — 1 Hour Lag",
    "aqi_lag_12":"AQI — 12 Hour Lag","aqi_lag_24":"AQI — 24 Hour Lag",
    "aqi_rolling_6":"AQI — 6 Hour Average","aqi_rolling_12":"AQI — 12 Hour Average",
    "aqi_rolling_24":"AQI — 24 Hour Average",
}

def readable_feature(name):
    return FEATURE_LABELS.get(name, name.replace("_"," ").title())

@st.cache_data(ttl=900, show_spinner=False)
def fetch_dashboard_air_quality():
    now = datetime.now(LOCAL_TZ)
    params = {
        "latitude": float(LATITUDE),
        "longitude": float(LONGITUDE),
        "timezone": "Asia/Karachi",
        "current": "us_aqi,pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,ozone",
        "hourly": "us_aqi",
        "start_date": f"{now.year}-01-01",
        "end_date": now.date().isoformat(),
    }
    r = requests.get(AIR_QUALITY_URL, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    current = payload.get("current", {})
    hourly = payload.get("hourly", {})
    if current.get("us_aqi") is None:
        raise ValueError("Open-Meteo did not return a current AQI value.")

    history = pd.DataFrame({
        "datetime": hourly.get("time", []),
        "aqi": hourly.get("us_aqi", []),
    })
    if not history.empty:
        history["datetime"] = pd.to_datetime(history["datetime"])
        history["aqi"] = pd.to_numeric(history["aqi"], errors="coerce")
        history = history.dropna().sort_values("datetime")
        history = history[history["datetime"] <= now.replace(tzinfo=None)]

    return {
        "current_aqi": float(current["us_aqi"]),
        "pollutants": {
            "PM2.5": current.get("pm2_5"),
            "PM10": current.get("pm10"),
            "NO₂": current.get("nitrogen_dioxide"),
            "SO₂": current.get("sulphur_dioxide"),
            "O₃": current.get("ozone"),
        },
        "fetched_at": datetime.now(LOCAL_TZ),
        "history": history,
    }

@st.cache_data(ttl=1800, show_spinner=False)
def get_forecast():
    return predict()

with st.sidebar:
    st.markdown(f'<div style="font-size:1.15rem;font-weight:700;margin-bottom:1.8rem;">AQI Forecast</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:.72rem;color:#777;margin-bottom:.4rem;text-transform:uppercase;letter-spacing:.06em;">Location</div>'
        f'<div style="font-size:1rem;font-weight:600;margin-bottom:1.5rem;">Gujranwala, Punjab</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:.8rem;color:#555;line-height:1.5;">Air quality forecasts generated using weather observations, air-quality data, CatBoost machine-learning models and SHAP.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="dashboard-title">Air Quality Forecast</div>'
    '<div class="dashboard-location">Gujranwala, Punjab, Pakistan</div>'
    '<div class="dashboard-subtitle">Machine-learning forecast for the next 72 hours.</div>',
    unsafe_allow_html=True,
)

try:
    dashboard_air = fetch_dashboard_air_quality()
    current_aqi = dashboard_air["current_aqi"]
    category, category_color, health_message = get_aqi_category(current_aqi)

    st.markdown('<div class="section-title">Current Air Quality</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="current-aqi-card">
            <div>
                <div class="current-aqi-label">Current AQI</div>
                <div class="current-aqi-value">{current_aqi:.1f}</div>
                <div style="font-size:.9rem;font-weight:700;color:{category_color};">{category}</div>
                <div class="current-aqi-fetched">Last fetched: {dashboard_air["fetched_at"].strftime("%d %b %Y, %H:%M:%S PKT")}</div>
            </div>
            <div class="current-aqi-side">
                <div style="font-size:.78rem;color:#777;margin-bottom:.35rem;">Health status</div>
                <div style="font-size:.9rem;line-height:1.45;">{health_message}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Latest Pollutants</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for col, (name, value) in zip(cols, dashboard_air["pollutants"].items()):
        with col:
            value_text = "—" if value is None else f"{float(value):.1f}"
            st.markdown(
                f'<div class="pollutant-card"><div class="pollutant-name">{name}</div>'
                f'<div class="pollutant-value">{value_text}</div><div class="pollutant-unit">μg/m³</div></div>',
                unsafe_allow_html=True,
            )
except Exception as exc:
    dashboard_air = None
    st.error(f"Unable to load current air-quality data: {exc}")

if "forecast_result" not in st.session_state:
    with st.spinner("Loading latest AQI forecast..."):
        try:
            st.session_state["forecast_result"] = get_forecast()
        except Exception as exc:
            st.error(f"Unable to generate forecast: {exc}")
            st.stop()

result = st.session_state["forecast_result"]
if isinstance(result, dict):
    predictions = result["predictions"]
    explanations = result.get("shap_explanations", result.get("explanations", {}))
elif isinstance(result, tuple):
    predictions, explanations = result
else:
    st.error("Unexpected prediction result format.")
    st.stop()

predictions = predictions.copy()
predictions["forecast_time"] = pd.to_datetime(predictions["forecast_time"], utc=True)

st.markdown('<div class="section-title">Forecast</div>', unsafe_allow_html=True)
cards = st.columns(3)
for i, horizon in enumerate([24, 48, 72]):
    row = predictions[predictions["horizon_hours"] == horizon]
    if row.empty:
        continue
    value = float(row["predicted_aqi"].iloc[0])
    forecast_time = row["forecast_time"].iloc[0]
    category, category_color, _ = get_aqi_category(value)
    text_color = "#000000" if category in ["Good", "Moderate"] else "#ffffff"
    with cards[i]:
        st.markdown(
            f"""
            <div class="forecast-card">
                <div class="forecast-horizon">{horizon}-Hour Forecast</div>
                <div class="forecast-value">{value:.1f}</div>
                <div style="background:{category_color};color:{text_color};padding:.5rem .75rem;border-radius:8px;font-weight:600;font-size:.85rem;text-align:center;">{category}</div>
                <div class="forecast-time">{forecast_time.strftime("%d %b %Y, %H:%M UTC")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

left, right = st.columns(2)

with left:
    st.markdown('<div class="section-title">Forecast Trend</div>', unsafe_allow_html=True)
    fig = go.Figure(go.Scatter(
        x=predictions["forecast_time"],
        y=predictions["predicted_aqi"],
        mode="lines+markers",
        line=dict(color=TEXT_COLOR, width=3),
        marker=dict(
            color=[aqi_color(v) for v in predictions["predicted_aqi"]],
            size=10,
            line=dict(color=TEXT_COLOR, width=1),
        ),
        hovertemplate="<b>%{y:.1f} AQI</b><br>%{x}<extra></extra>",
    ))
    fig.update_layout(
        height=360, margin=dict(l=20,r=20,t=15,b=20),
        plot_bgcolor=CARD_BACKGROUND, paper_bgcolor=CARD_BACKGROUND,
        font=dict(color=TEXT_COLOR), showlegend=False,
        xaxis=dict(title="Forecast Time", showgrid=False, linecolor=TEXT_COLOR, tickcolor=TEXT_COLOR),
        yaxis=dict(title="AQI", gridcolor=GRID_COLOR, zerolinecolor=TEXT_COLOR, linecolor=TEXT_COLOR),
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    year = datetime.now(LOCAL_TZ).year
    st.markdown(f'<div class="section-title">AQI Trend — {year}</div>', unsafe_allow_html=True)
    if dashboard_air is not None and not dashboard_air["history"].empty:
        history = dashboard_air["history"]
        fig_year = go.Figure(go.Scatter(
            x=history["datetime"],
            y=history["aqi"],
            mode="lines+markers",
            line=dict(color=TEXT_COLOR, width=2),
            marker=dict(
                color=[aqi_color(v) for v in history["aqi"]],
                size=4,
            ),
            hovertemplate="<b>%{y:.1f} AQI</b><br>%{x}<extra></extra>",
        ))
        fig_year.update_layout(
            height=360, margin=dict(l=20,r=20,t=15,b=20),
            plot_bgcolor=CARD_BACKGROUND, paper_bgcolor=CARD_BACKGROUND,
            font=dict(color=TEXT_COLOR), showlegend=False,
            xaxis=dict(title="Date", showgrid=False, linecolor=TEXT_COLOR, tickcolor=TEXT_COLOR),
            yaxis=dict(title="AQI", gridcolor=GRID_COLOR, zerolinecolor=TEXT_COLOR, linecolor=TEXT_COLOR),
        )
        st.plotly_chart(fig_year, use_container_width=True)
    else:
        st.info("No current-year AQI history was returned.")

st.markdown('<div class="section-title">Model Explanation</div>', unsafe_allow_html=True)
st.caption("SHAP shows which input features contributed most to each AQI forecast. Positive values push the prediction higher; negative values push it lower.")
selected_horizon = st.radio(
    "Forecast horizon", [24,48,72], horizontal=True,
    format_func=lambda value: f"{value}-hour forecast",
)
explanation = explanations.get(selected_horizon)

if explanation is not None:
    explanation = explanation.copy()
    if "feature" not in explanation.columns or "shap_value" not in explanation.columns:
        st.error("SHAP explanation is missing required columns.")
        st.stop()

    explanation["feature"] = explanation["feature"].apply(readable_feature)
    explanation["abs_shap"] = explanation["shap_value"].abs()
    explanation = explanation.nlargest(10, "abs_shap").sort_values("shap_value")

    fig_shap = go.Figure(go.Bar(
        x=explanation["shap_value"],
        y=explanation["feature"],
        orientation="h",
        marker=dict(color="#02A4D3"),
        hovertemplate="<b>%{y}</b><br>SHAP contribution: %{x:.3f}<extra></extra>",
    ))
    fig_shap.update_layout(
        height=430, margin=dict(l=20,r=20,t=25,b=20),
        plot_bgcolor=CARD_BACKGROUND, paper_bgcolor=CARD_BACKGROUND,
        font=dict(color=TEXT_COLOR), showlegend=False,
        xaxis=dict(title="SHAP Contribution", gridcolor=GRID_COLOR, zeroline=True, zerolinecolor=TEXT_COLOR, linecolor=TEXT_COLOR),
        yaxis=dict(title="", linecolor=TEXT_COLOR),
    )
    st.plotly_chart(fig_shap, use_container_width=True)
else:
    st.info("SHAP explanation is not available for this forecast.")

with st.expander("Forecast data"):
    display_predictions = predictions.copy()
    display_predictions["forecast_time"] = display_predictions["forecast_time"].dt.strftime("%d %b %Y, %H:%M UTC")
    display_predictions["prediction_created_at"] = pd.to_datetime(display_predictions["prediction_created_at"], utc=True).dt.strftime("%d %b %Y, %H:%M UTC")
    st.dataframe(display_predictions, use_container_width=True, hide_index=True)

st.markdown('<div class="footer">Gujranwala Air Quality Forecast · CatBoost · Hopsworks · SHAP</div>', unsafe_allow_html=True)
