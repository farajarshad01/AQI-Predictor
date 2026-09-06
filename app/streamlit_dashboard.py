import os
import sys
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from zoneinfo import ZoneInfo


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.prediction_pipeline import predict


st.set_page_config(
    page_title="Gujranwala AQI Forecast",
    page_icon="🌫️",
    layout="wide"
)


TEXT_COLOR = "#1c1d1f"
BACKGROUND = "#ffffff"
CARD_BACKGROUND = "#f4f5f7"
BORDER = "#d4d4d4"
GRID_COLOR = "#e5e5e5"

CHART_BLUE = "#02A4D3"

GOOD_COLOR = "#00E400"
MODERATE_COLOR = "#C9A500"
SENSITIVE_COLOR = "#FF7E00"
UNHEALTHY_COLOR = "#FF0000"
VERY_UNHEALTHY_COLOR = "#8F3F97"
HAZARDOUS_COLOR = "#7E0023"

LATITUDE = 32.1617
LONGITUDE = 74.1883

LOCAL_TZ = ZoneInfo("Asia/Karachi")

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {BACKGROUND};
        color: {TEXT_COLOR};
    }}

    .main {{
        background-color: {BACKGROUND};
    }}

    .block-container {{
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT_COLOR} !important;
    }}

    p, span, label {{
        color: {TEXT_COLOR};
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stSidebar"] {{
        background-color: #ffffff;
        border-right: 2px solid {BORDER};
    }}

    [data-testid="stSidebar"] * {{
        color: {TEXT_COLOR} !important;
    }}

    .dashboard-title {{
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: {TEXT_COLOR};
        margin-bottom: 0.2rem;
    }}

    .dashboard-location {{
        font-size: 1rem;
        font-weight: 600;
        color: {TEXT_COLOR};
        margin-bottom: 0.35rem;
    }}

    .dashboard-subtitle {{
        font-size: 0.92rem;
        color: #555555 !important;
        margin-bottom: 1.8rem;
    }}

    .section-title {{
        font-size: 1.3rem;
        font-weight: 750;
        color: {TEXT_COLOR};
        margin-top: 1.8rem;
        margin-bottom: 0.8rem;
    }}

    .current-aqi-card {{
        background-color: {CARD_BACKGROUND};
        border: 1.5px solid {BORDER};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }}

    .current-aqi-title {{
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {TEXT_COLOR};
    }}

    .current-aqi-value {{
        font-size: 3rem;
        font-weight: 800;
        color: {TEXT_COLOR};
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
        line-height: 1;
    }}

    .current-aqi-time {{
        font-size: 0.75rem;
        color: #555555 !important;
        margin-top: 0.6rem;
    }}

    .forecast-card {{
        background-color: {CARD_BACKGROUND};
        border: 1.5px solid {BORDER};
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        height: 100%;
    }}

    .forecast-card-horizon {{
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #555555;
    }}

    .forecast-card-val {{
        font-size: 2.4rem;
        font-weight: 800;
        color: {TEXT_COLOR};
        margin-top: 0.3rem;
        margin-bottom: 0.1rem;
        line-height: 1;
    }}

    .forecast-card-category {{
        font-weight: 750;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }}

    .forecast-card-time {{
        font-size: 0.72rem;
        color: #666666;
        margin-top: 0.6rem;
    }}

    .pollutant-card {{
        background-color: {CARD_BACKGROUND};
        border: 1.5px solid {BORDER};
        border-radius: 10px;
        padding: 0.9rem 1rem;
        height: 100%;
    }}

    .pollutant-name {{
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: {TEXT_COLOR};
    }}

    .pollutant-value {{
        font-size: 1.4rem;
        font-weight: 750;
        color: {TEXT_COLOR};
        margin-top: 0.3rem;
    }}

    .footer {{
        text-align: center;
        color: #777777 !important;
        font-size: 0.72rem;
        padding: 2.5rem 0 1rem 0;
        margin-top: 2rem;
        border-top: 1px solid {BORDER};
    }}

    </style>
    """,
    unsafe_allow_html=True
)


def get_aqi_category(aqi):

    if aqi <= 50:
        return (
            "Good",
            GOOD_COLOR,
            "Air quality is considered satisfactory."
        )

    if aqi <= 100:
        return (
            "Moderate",
            MODERATE_COLOR,
            "Air quality is acceptable."
        )

    if aqi <= 150:
        return (
            "Unhealthy for Sensitive Groups",
            SENSITIVE_COLOR,
            "Sensitive groups may experience health effects."
        )

    if aqi <= 200:
        return (
            "Unhealthy",
            UNHEALTHY_COLOR,
            "Everyone may begin to experience health effects."
        )

    if aqi <= 300:
        return (
            "Very Unhealthy",
            VERY_UNHEALTHY_COLOR,
            "Health alert: increased risk of health effects."
        )

    return (
        "Hazardous",
        HAZARDOUS_COLOR,
        "Health warning of emergency conditions."
    )


FEATURE_LABELS = {
    "temperature_2m": "Temperature",
    "relative_humidity_2m": "Relative Humidity",
    "wind_speed_10m": "Wind Speed",
    "surface_pressure": "Surface Pressure",
    "precipitation": "Precipitation",
    "cloud_cover": "Cloud Cover",

    "pm10": "PM10",
    "pm2_5": "PM2.5",
    "carbon_monoxide": "Carbon Monoxide",
    "nitrogen_dioxide": "Nitrogen Dioxide",
    "sulphur_dioxide": "Sulphur Dioxide",
    "ozone": "Ozone",

    "us_aqi": "Previous AQI",

    "year": "Year",
    "month": "Month",
    "day": "Day",
    "day_of_week": "Day of Week",
    "hour": "Hour",

    "aqi_change": "AQI Change",
    "aqi_change_rate": "AQI Change Rate",

    "aqi_lag_1": "AQI — 1 Hour Lag",
    "aqi_lag_12": "AQI — 12 Hour Lag",
    "aqi_lag_24": "AQI — 24 Hour Lag",

    "aqi_rolling_6": "AQI — 6 Hour Average",
    "aqi_rolling_12": "AQI — 12 Hour Average",
    "aqi_rolling_24": "AQI — 24 Hour Average",
}


def readable_feature(name):
    return FEATURE_LABELS.get(
        name,
        name.replace("_", " ").title()
    )


@st.cache_data(ttl=900, show_spinner=False)
def fetch_current_air_quality():

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": "Asia/Karachi",
        "current": (
            "us_aqi,"
            "pm10,"
            "pm2_5,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone"
        ),
    }

    response = requests.get(
        AIR_QUALITY_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()
    payload = response.json()
    current = payload.get("current", {})

    if current.get("us_aqi") is None:
        raise ValueError("Current AQI data was not returned.")

    return {
        "aqi": float(current["us_aqi"]),
        "pollutants": {
            "PM2.5": current.get("pm2_5"),
            "PM10": current.get("pm10"),
            "NO₂": current.get("nitrogen_dioxide"),
            "SO₂": current.get("sulphur_dioxide"),
            "O₃": current.get("ozone"),
        },
        "fetched_at": datetime.now(LOCAL_TZ)
    }


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_aqi_history(year):

    year = int(year)
    now = datetime.now(LOCAL_TZ)
    start_date = f"{year}-01-01"

    if year == now.year:
        end_date = now.date().isoformat()
    else:
        end_date = f"{year}-12-31"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": "Asia/Karachi",
        "hourly": "us_aqi",
        "start_date": start_date,
        "end_date": end_date,
    }

    response = requests.get(
        AIR_QUALITY_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})

    history = pd.DataFrame({
        "datetime": hourly.get("time", []),
        "aqi": hourly.get("us_aqi", []),
    })

    if history.empty:
        return history

    history["datetime"] = pd.to_datetime(history["datetime"])
    history["aqi"] = pd.to_numeric(history["aqi"], errors="coerce")
    history = history.dropna()
    history = history.sort_values("datetime")

    return history


with st.sidebar:

    st.markdown(
        """
        <div style="font-size:1.2rem; font-weight:800; margin-bottom:2rem;">
            AQI Forecast
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="font-size:0.72rem; color:#777777; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;">
            Location
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="font-size:1rem; font-weight:650; margin-bottom:1.5rem;">
            Gujranwala, Punjab
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="font-size:0.82rem; color:#555555; line-height:1.6;">
            Air quality forecasts generated using
            weather observations, air-quality data,
            CatBoost machine-learning models and SHAP.
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    """
    <div class="dashboard-title">
        Gujranwala AQI Forecast
    </div>

    <div class="dashboard-location">
        Air Quality Index Dashboard · Punjab, Pakistan
    </div>

    <div class="dashboard-subtitle">
        Live air-quality observations, machine-learning forecasts and model explainability.
    </div>
    """,
    unsafe_allow_html=True
)


try:
    current_data = fetch_current_air_quality()
    current_aqi = current_data["aqi"]
    (
        current_category,
        current_color,
        current_message
    ) = get_aqi_category(current_aqi)

    st.markdown(
        f"""
        <div class="current-aqi-card">
            <div class="current-aqi-title">Current AQI</div>
            <div class="current-aqi-value">{current_aqi:.1f}</div>
            <div style="color:{current_color}; font-weight:800; font-size:0.95rem;">
                {current_category}
            </div>
            <div class="current-aqi-time">
                Last fetched: {current_data["fetched_at"].strftime("%d %b %Y, %H:%M PKT")}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Latest Pollutants</div>',
        unsafe_allow_html=True
    )

    pollutant_columns = st.columns(5)

    for index, (pollutant, value) in enumerate(current_data["pollutants"].items()):
        display_value = "—" if value is None else f"{float(value):.1f}"

        with pollutant_columns[index]:
            st.markdown(
                f"""
                <div class="pollutant-card">
                    <div class="pollutant-name">{pollutant}</div>
                    <div class="pollutant-value">{display_value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

except Exception as exc:
    st.warning(f"Unable to load current AQI data: {exc}")


if "forecast_result" not in st.session_state:
    with st.spinner("Generating AQI forecast..."):
        try:
            st.session_state["forecast_result"] = predict()
        except Exception as exc:
            st.error(f"Unable to generate forecast: {exc}")

if "forecast_result" in st.session_state:

    result = st.session_state["forecast_result"]

    if isinstance(result, dict):
        predictions = result["predictions"]
        explanations = result.get("shap_explanations", result.get("explanations", {}))
    elif isinstance(result, tuple):
        predictions = result[0]
        explanations = result[1]
    else:
        st.error("Unexpected prediction result format.")
        st.stop()

    predictions = predictions.copy()
    predictions["forecast_time"] = pd.to_datetime(
        predictions["forecast_time"],
        utc=True
    )

    st.markdown(
        '<div class="section-title">Forecast</div>',
        unsafe_allow_html=True
    )

    cards = st.columns(3)

    for index, horizon in enumerate([24, 48, 72]):
        row = predictions[predictions["horizon_hours"] == horizon]

        if row.empty:
            continue

        value = float(row["predicted_aqi"].iloc[0])
        forecast_time = row["forecast_time"].iloc[0]
        (
            category,
            category_color,
            message
        ) = get_aqi_category(value)

        with cards[index]:
            st.markdown(
                f"""
                <div class="forecast-card">
                    <div class="forecast-card-horizon">{horizon}-Hour Forecast</div>
                    <div class="forecast-card-val">{value:.1f}</div>
                    <div class="forecast-card-category" style="color:{category_color};">
                        {category}
                    </div>
                    <div class="forecast-card-time">
                        {forecast_time.strftime("%d %b %Y, %H:%M UTC")}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    left_chart, right_chart = st.columns(2)

    with left_chart:
        st.markdown(
            '<div class="section-title">Forecast Trend</div>',
            unsafe_allow_html=True
        )

        forecast_fig = go.Figure()

        forecast_fig.add_trace(
            go.Scatter(
                x=predictions["forecast_time"],
                y=predictions["predicted_aqi"],
                mode="lines+markers",
                line=dict(color=CHART_BLUE, width=3),
                marker=dict(size=6, color=CHART_BLUE),
                hovertemplate="<b>%{y:.1f} AQI</b><br>%{x}<extra></extra>"
            )
        )

        forecast_fig.update_layout(
            height=360,
            autosize=True,
            margin=dict(l=50, r=20, t=10, b=40),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(color=TEXT_COLOR, size=12),
            showlegend=False,
            xaxis=dict(
                title="Forecast Time",
                showgrid=False,
                linecolor=TEXT_COLOR,
                tickcolor=TEXT_COLOR,
                title_font=dict(color=TEXT_COLOR),
                tickfont=dict(color=TEXT_COLOR)
            ),
            yaxis=dict(
                title="AQI",
                gridcolor=GRID_COLOR,
                linecolor=TEXT_COLOR,
                tickcolor=TEXT_COLOR,
                title_font=dict(color=TEXT_COLOR),
                tickfont=dict(color=TEXT_COLOR)
            )
        )

        st.plotly_chart(forecast_fig, use_container_width=True)

    with right_chart:
        st.markdown(
            '<div class="section-title">AQI Trend</div>',
            unsafe_allow_html=True
        )

        available_years = [2026, 2025, 2024]
        selected_year = st.radio(
            "Select year",
            available_years,
            horizontal=True,
            index=0,
            label_visibility="collapsed"
        )

        try:
            history = fetch_aqi_history(selected_year)

            if history.empty:
                st.info(f"No AQI data available for {selected_year}.")
            else:
                historical_fig = go.Figure()

                historical_fig.add_trace(
                    go.Scatter(
                        x=history["datetime"],
                        y=history["aqi"],
                        mode="lines",
                        line=dict(color=CHART_BLUE, width=2),
                        hovertemplate="<b>%{y:.1f} AQI</b><br>%{x}<extra></extra>"
                    )
                )

                historical_fig.update_layout(
                    height=360,
                    autosize=True,
                    margin=dict(l=50, r=20, t=10, b=40),
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                    font=dict(color=TEXT_COLOR, size=12),
                    showlegend=False,
                    xaxis=dict(
                        title="Date",
                        showgrid=False,
                        linecolor=TEXT_COLOR,
                        tickcolor=TEXT_COLOR,
                        title_font=dict(color=TEXT_COLOR),
                        tickfont=dict(color=TEXT_COLOR)
                    ),
                    yaxis=dict(
                        title="AQI",
                        gridcolor=GRID_COLOR,
                        linecolor=TEXT_COLOR,
                        tickcolor=TEXT_COLOR,
                        title_font=dict(color=TEXT_COLOR),
                        tickfont=dict(color=TEXT_COLOR)
                    )
                )

                st.plotly_chart(historical_fig, use_container_width=True)

        except Exception as exc:
            st.warning(f"Unable to load AQI history: {exc}")

    st.markdown(
        '<div class="section-title">Model Explanation</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "SHAP shows the ten features with the strongest influence "
        "on the selected AQI forecast. Positive values push the "
        "prediction higher, while negative values push it lower."
    )

    selected_horizon = st.radio(
        "Forecast horizon",
        [24, 48, 72],
        horizontal=True,
        format_func=lambda value: f"{value}-hour forecast"
    )

    explanation = explanations.get(selected_horizon)

    if explanation is not None:
        explanation = explanation.copy()

        if "feature" not in explanation.columns or "shap_value" not in explanation.columns:
            st.error("SHAP explanation dataset is missing required columns.")
            st.stop()

        explanation["feature"] = explanation["feature"].apply(readable_feature)
        explanation["absolute_shap"] = explanation["shap_value"].abs()

        top_10 = (
            explanation
            .sort_values("absolute_shap", ascending=False)
            .head(10)
        )

        explanation = top_10.sort_values("shap_value", ascending=True)

        shap_fig = go.Figure()

        shap_fig.add_trace(
            go.Bar(
                x=explanation["shap_value"],
                y=explanation["feature"],
                orientation="h",
                marker=dict(color=CHART_BLUE),
                hovertemplate="<b>%{y}</b><br>SHAP contribution: %{x:.3f}<extra></extra>"
            )
        )

        shap_fig.update_layout(
            height=420,
            autosize=True,
            margin=dict(l=140, r=20, t=10, b=40),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(color=TEXT_COLOR, size=12),
            showlegend=False,
            xaxis=dict(
                title="SHAP Contribution",
                gridcolor=GRID_COLOR,
                zeroline=True,
                zerolinecolor=TEXT_COLOR,
                linecolor=TEXT_COLOR,
                tickcolor=TEXT_COLOR,
                title_font=dict(color=TEXT_COLOR),
                tickfont=dict(color=TEXT_COLOR)
            ),
            yaxis=dict(
                title="",
                linecolor=TEXT_COLOR,
                tickcolor=TEXT_COLOR,
                tickfont=dict(color=TEXT_COLOR)
            )
        )

        st.plotly_chart(shap_fig, use_container_width=True)

    else:
        st.info("SHAP explanation is not available for this forecast.")


st.markdown(
    """
    <div class="footer">
        Gujranwala AQI Forecast · CatBoost · Hopsworks · SHAP · Open-Meteo
    </div>
    """,
    unsafe_allow_html=True
)
