import os
import sys
from datetime import datetime
import calendar

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.prediction_pipeline import predict
from src.hopsworksclients import get_feature_group

st.set_page_config(
    page_title="Gujranwala Air Quality",
    layout="wide"
)

# Constants

TEXT_COLOR = "#1c1d1f"
WHITE = "#ffffff"
BACKGROUND = "#f7f7f7"
BORDER = "#e5e5e5"
GRID_COLOR = "#e5e5e5"

CHART_BLUE = "#02A4D3"

GOOD_COLOR = "#00E400"
MODERATE_COLOR = "#FFFF00"
SENSITIVE_COLOR = "#FF7E00"
UNHEALTHY_COLOR = "#FF0000"
VERY_UNHEALTHY_COLOR = "#8F3F97"
HAZARDOUS_COLOR = "#7E0023"

# Location (used for historical API calls)
LATITUDE = 32.1617
LONGITUDE = 74.1883
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Pollutant units
POLLUTANT_UNITS = {
    "pm2_5": "µg/m³",
    "pm10": "µg/m³",
    "nitrogen_dioxide": "µg/m³",
    "sulphur_dioxide": "µg/m³",
    "ozone": "µg/m³",
}

# Page styling
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

    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT_COLOR} !important;
    }}

    p, span, label, div {{
        color: {TEXT_COLOR};
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    /* Sidebar: light background (#cbcbcb), larger corner radius, slightly narrower */
    [data-testid="stSidebar"] {{
        background-color: #cbcbcb !important;
        border-right: 1px solid rgba(0,0,0,0.06);
        border-top-right-radius: 28px;
        border-bottom-right-radius: 28px;
        padding-top: 1.4rem;
        padding-left: 1.4rem;
        padding-right: 1.4rem;
        width: 300px !important;
        min-width: 260px !important;
        max-width: 320px !important;
    }}

    /* Ensure sidebar text is readable (dark) on the light background */
    [data-testid="stSidebar"] * {{
        color: {TEXT_COLOR} !important;
    }}

    .dashboard-title {{
        font-size: 2.2rem;
        font-weight: 700;
        color: {TEXT_COLOR};
        margin-bottom: 0.15rem;
    }}

    .dashboard-location {{
        font-size: 1rem;
        color: {TEXT_COLOR};
        font-weight: 500;
        margin-bottom: 0.25rem;
    }}

    .dashboard-subtitle {{
        font-size: 0.9rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }}

    .section-title {{
        font-size: 1.25rem;
        font-weight: 700;
        color: {TEXT_COLOR};
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }}

    .forecast-card {{
        background-color: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 1.35rem;
        min-height: 215px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }}

    .forecast-horizon {{
        font-size: 0.78rem;
        font-weight: 700;
        color: {TEXT_COLOR};
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}

    .forecast-value {{
        font-size: 2.8rem;
        font-weight: 750;
        color: {TEXT_COLOR};
        margin-top: 0.65rem;
        margin-bottom: 0.55rem;
    }}

    .health-message {{
        font-size: 0.82rem;
        line-height: 1.45;
        color: #555555;
        margin-top: 0.7rem;
    }}

    .forecast-time {{
        font-size: 0.72rem;
        color: #777777;
        margin-top: 0.75rem;
    }}

    .info-card {{
        background-color: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }}

    .pollutant-card {{
        background-color: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 0.6rem 0.75rem;
        text-align: center;
    }}

    .pollutant-name {{
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: {TEXT_COLOR};
    }}

    .pollutant-value {{
        font-size: 1.2rem;
        font-weight: 700;
        color: {TEXT_COLOR};
        margin-top: 0.25rem;
    }}

    .pollutant-unit {{
        font-size: 0.72rem;
        color: #666666;
        margin-top: 0.15rem;
    }}

    .aqi-current-card {{
        background-color: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1rem;
    }}

    .aqi-current-left {{
        display:flex;
        flex-direction:column;
        gap:0.35rem;
    }}

    .aqi-current-value {{
        font-size: 2.2rem;
        font-weight: 800;
        color: {TEXT_COLOR};
    }}

    .aqi-current-label {{
        font-size: 0.95rem;
        font-weight: 700;
        display:inline-block;
    }}

    .aqi-current-desc {{
        font-size: 0.9rem;
        color: #555555;
    }}

    .footer {{
        text-align: center;
        color: #777777;
        font-size: 0.72rem;
        padding: 2rem 0 1rem 0;
    }}

    /* Darker spinner for visibility */
    .stSpinner, .stSpinner * {{
        color: #111 !important;
        stroke: #111 !important;
        fill: #111 !important;
    }}

    /* Selectbox styling: light background (#cbcbcb) with dark text */
    .stSelectbox > div[role="button"], .stSelectbox > div[role="button"] * {{
        background-color: #cbcbcb !important;
        color: {TEXT_COLOR} !important;
        border-radius: 8px !important;
        padding-left: 12px !important;
    }}

    select,
    .stSelectbox select {{
        background-color: #cbcbcb !important;
        color: {TEXT_COLOR} !important;
        border-radius: 8px !important;
    }}

    div[role="listbox"] {{
        background-color: #ffffff !important;
        color: {TEXT_COLOR} !important;
        border-radius: 10px !important;
        padding: 8px !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08) !important;
        max-width: 520px !important;
    }}

    div[role="listbox"] > div[role="option"] {{
        background-color: transparent !important;
        color: {TEXT_COLOR} !important;
        padding: 8px 12px !important;
        border-radius: 6px !important;
    }}

    div[role="listbox"] > div[role="option"]:hover {{
        background-color: #e6e6e6 !important;
        color: {TEXT_COLOR} !important;
    }}

    div[role="listbox"] > div[role="option"][aria-selected="true"] {{
        background-color: #cbcbcb !important;
        color: {TEXT_COLOR} !important;
    }}

    option {{
        background-color: #cbcbcb !important;
        color: {TEXT_COLOR} !important;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# AQI classification
def get_aqi_category(aqi):
    if aqi <= 50:
        return ("Good", GOOD_COLOR, "#dcfce7", "Air quality is considered satisfactory.")
    if aqi <= 100:
        return ("Moderate", MODERATE_COLOR, "#fef9c3",
                "Air quality is acceptable; unusually sensitive people may experience minor effects.")
    if aqi <= 150:
        return ("Unhealthy for Sensitive Groups", SENSITIVE_COLOR, "#ffedd5",
                "Sensitive groups may experience health effects.")
    if aqi <= 200:
        return ("Unhealthy", UNHEALTHY_COLOR, "#fee2e2", "Everyone may begin to experience health effects.")
    if aqi <= 300:
        return ("Very Unhealthy", VERY_UNHEALTHY_COLOR, "#fecaca",
                "Health alert: the risk of health effects is increased.")
    return ("Hazardous", HAZARDOUS_COLOR, "#fca5a5", "Health warning of emergency conditions.")


def get_aqi_audience(aqi):
    """Return a short sentence stating who is at risk for the provided AQI."""
    if aqi is None:
        return ""
    if aqi <= 50:
        return "No special precautions required for the general population."
    if aqi <= 100:
        return "Unusually sensitive people should consider limiting prolonged outdoor exertion."
    if aqi <= 150:
        return "Sensitive groups (children, elderly, people with lung disease, and asthmatics) may be affected."
    if aqi <= 200:
        return "People with heart or lung disease, older adults, and children should avoid prolonged outdoor exertion."
    if aqi <= 300:
        return "Everyone may experience more serious health effects; avoid outdoor activity if possible."
    return "Health alert: everyone should avoid all outdoor exertion and follow local guidance."


# Feature labels / readable names
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
    return FEATURE_LABELS.get(name, name.replace("_", " ").title())


# Helper: fetch hourly AQI for a month from Open-Meteo (fallback)
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_monthly_aqi(year: int, month: int) -> pd.DataFrame:
    year = int(year)
    month = int(month)
    _, last_day = calendar.monthrange(year, month)
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-{last_day:02d}"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": "Asia/Karachi",
        "hourly": "us_aqi",
        "start_date": start_date,
        "end_date": end_date,
    }
    response = requests.get(AIR_QUALITY_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    df = pd.DataFrame({"datetime": hourly.get("time", []), "aqi": hourly.get("us_aqi", [])})
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
    df = df.dropna()
    df = df.sort_values("datetime")
    return df


# Helper: fetch monthly AQI from Hopsworks feature group
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_monthly_aqi_from_hopsworks(year: int, month: int) -> pd.DataFrame:
    try:
        fg = get_feature_group()
        df = fg.read()
    except Exception as exc:
        raise RuntimeError(f"Unable to read feature group from Hopsworks: {exc}")
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "datetime" not in df.columns:
        if "date_time" in df.columns:
            df["datetime"] = df["date_time"]
        elif "timestamp" in df.columns:
            df["datetime"] = df["timestamp"]
        else:
            raise RuntimeError("Feature group does not contain 'datetime' column")
    df["datetime"] = pd.to_datetime(df["datetime"])
    mask = (df["datetime"].dt.year == int(year)) & (df["datetime"].dt.month == int(month))
    month_df = df.loc[mask, ["datetime", "us_aqi"]].rename(columns={"us_aqi": "aqi"}).copy()
    if month_df.empty:
        return pd.DataFrame()
    month_df = month_df.sort_values("datetime").reset_index(drop=True)
    month_df["aqi"] = pd.to_numeric(month_df["aqi"], errors="coerce")
    month_df = month_df.dropna()
    return month_df


# Helper: fetch latest pollutant measurements from Hopsworks
@st.cache_data(ttl=300, show_spinner=False)
def fetch_latest_pollutants_from_hopsworks():
    try:
        fg = get_feature_group()
        df = fg.read()
    except Exception as exc:
        raise RuntimeError(f"Unable to read feature group from Hopsworks: {exc}")
    if df is None or df.empty:
        raise RuntimeError("Feature group is empty")
    df = df.copy()
    # normalize datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.rename(columns={df.columns[0]: "datetime"})
    latest_row = df.sort_values("datetime").tail(1).iloc[0]
    pollutants = {}
    for col in ["pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]:
        if col in df.columns:
            val = latest_row.get(col, None)
            pollutants[col] = None if pd.isna(val) else float(val)
        else:
            pollutants[col] = None
    return pollutants


# Helper: fetch current AQI (latest) from Hopsworks
@st.cache_data(ttl=300, show_spinner=False)
def fetch_latest_aqi_from_hopsworks():
    """
    Returns a dict: {"aqi": float or None, "datetime": pd.Timestamp or None}
    """
    try:
        fg = get_feature_group()
        df = fg.read()
    except Exception as exc:
        raise RuntimeError(f"Unable to read feature group from Hopsworks: {exc}")
    if df is None or df.empty:
        return {"aqi": None, "datetime": None}
    df = df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        if "date_time" in df.columns:
            df["datetime"] = pd.to_datetime(df["date_time"])
        elif "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"])
        else:
            return {"aqi": None, "datetime": None}
    if "us_aqi" not in df.columns:
        return {"aqi": None, "datetime": None}
    latest = df.sort_values("datetime").tail(1).iloc[0]
    aqi_val = latest.get("us_aqi", None)
    if pd.isna(aqi_val):
        return {"aqi": None, "datetime": latest.get("datetime", None)}
    try:
        return {"aqi": float(aqi_val), "datetime": pd.to_datetime(latest.get("datetime"))}
    except Exception:
        return {"aqi": None, "datetime": latest.get("datetime", None)}


# Sidebar
with st.sidebar:
    st.markdown(
        f"""
        <div style="
            font-size:1.15rem;
            font-weight:700;
            color:{TEXT_COLOR};
            margin-bottom:1.2rem;
        ">
            AQI Forecast
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="
            font-size:0.72rem;
            color:{TEXT_COLOR};
            text-transform:uppercase;
            letter-spacing:0.06em;
            margin-bottom:0.6rem;
        ">
            Location
        </div>

        <div style="
            font-size:1rem;
            font-weight:600;
            color:{TEXT_COLOR};
            margin-bottom:1.2rem;
        ">
            Gujranwala, Punjab
        </div>
        """,
        unsafe_allow_html=True
    )

    # extended description (slightly smaller)
    st.markdown(
        f"""
        <div style="
            font-size:0.78rem;
            color:{TEXT_COLOR};
            line-height:1.45;
            margin-top:0.6rem;
        ">
            Real-time air quality forecasts generated using live weather observations and measured air-quality features.
            Features and historical records are stored in Hopsworks; forecasts are produced by CatBoost models and
            explained with SHAP to highlight the drivers behind each prediction. Data is refreshed regularly and the
            dashboard shows the latest available measurements and 24/48/72-hour model forecasts.
        </div>
        """,
        unsafe_allow_html=True
    )


# Header
st.markdown(
    f"""
    <div class="dashboard-title">
        Air Quality Forecast
    </div>

    <div class="dashboard-location">
        Gujranwala, Punjab, Pakistan
    </div>
    """,
    unsafe_allow_html=True
)


# Current AQI card (below title, above generate button)
try:
    current_aqi = fetch_latest_aqi_from_hopsworks()
except Exception as exc:
    current_aqi = {"aqi": None, "datetime": None}
    st.info(f"Unable to load current AQI from Hopsworks: {exc}")

if current_aqi.get("aqi") is not None:
    aqi_value = current_aqi["aqi"]
    aqi_dt = current_aqi["datetime"]
    category, category_color, _bg, message = get_aqi_category(aqi_value)
    audience = get_aqi_audience(aqi_value)

    st.markdown(
        f"""
        <div class="aqi-current-card" style="width:100%;">
            <div class="aqi-current-left">
                <div style="font-size:0.85rem;color:#666666;">Current AQI</div>
                <div class="aqi-current-value">{aqi_value:.1f}</div>
                <div style="margin-top:6px;">
                    <span class="aqi-current-label" style="color:{category_color};">
                        {category}
                    </span>
                </div>
            </div>

            <div style="flex:1; padding-left:1rem;">
                <div class="aqi-current-desc">
                    {message}{" " if message else ""}{"• " + audience if audience else ""}{" " if (aqi_dt is not None and (message or audience)) else ""}{"Measured at " + pd.to_datetime(aqi_dt).strftime('%d %b %Y, %H:%M') if aqi_dt is not None else ""}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div class="aqi-current-card" style="width:100%;">
            <div class="aqi-current-left">
                <div style="font-size:0.85rem;color:#666666;">Current AQI</div>
                <div class="aqi-current-value">—</div>
                <div style="margin-top:6px;">
                    <span class="aqi-current-label" style="color:#666666;">
                        N/A
                    </span>
                </div>
            </div>

            <div style="flex:1; padding-left:1rem;">
                <div class="aqi-current-desc">
                    Current AQI measurement is not available.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Subtitle and Generate button
st.markdown(
    f"""
    <div class="dashboard-subtitle">
        Machine-learning forecast for the next 72 hours.
    </div>
    """,
    unsafe_allow_html=True
)

generate = st.button(
    "Generate AQI Forecast",
    type="primary"
)

if generate:
    with st.spinner("Generating AQI forecast..."):
        try:
            result = predict()
            st.session_state["forecast_result"] = result
        except Exception as exc:
            st.error(f"Unable to generate forecast: {exc}")


# Display forecast
if "forecast_result" in st.session_state:
    result = st.session_state["forecast_result"]

    # Supports the current dictionary return format.
    if isinstance(result, dict):
        predictions = result["predictions"]
        explanations = result.get("shap_explanations", result.get("explanations", {}))

    # Also supports the older tuple format.
    elif isinstance(result, tuple):
        predictions = result[0]
        explanations = result[1]

    else:
        st.error("Unexpected prediction result format.")
        st.stop()

    predictions = predictions.copy()
    predictions["forecast_time"] = pd.to_datetime(predictions["forecast_time"], utc=True)

    # Forecast cards
    st.markdown('<div class="section-title">Forecast</div>', unsafe_allow_html=True)
    cards = st.columns(3)

    for index, horizon in enumerate([24, 48, 72]):
        row = predictions[predictions["horizon_hours"] == horizon]
        if row.empty:
            continue
        value = float(row["predicted_aqi"].iloc[0])
        forecast_time = row["forecast_time"].iloc[0]
        category, category_color, category_background, message = get_aqi_category(value)
        with cards[index]:
            st.markdown(
                f"""<div class="forecast-card"><div class="forecast-horizon">{horizon}-Hour Forecast</div><div class="forecast-value">{value:.1f}</div></div>""",
                unsafe_allow_html=True
            )
            st.markdown(
                f"""<div style="
                    color: {"#000000" if category in ["Good", "Moderate"] else "#ffffff"};
                    padding: 0.5rem 0.75rem;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 0.85rem;
                    text-align: center;
                    margin-top: 0.6rem;
                "><span style="color:{category_color};">{category}</span></div>""",
                unsafe_allow_html=True
            )

    # Top pollutants
    st.markdown('<div class="section-title">Top Pollutants</div>', unsafe_allow_html=True)
    pollutant_order = ["pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]

    # Try to fetch from Hopsworks; fallback to predictions
    try:
        pollutants = fetch_latest_pollutants_from_hopsworks()
    except Exception:
        pollutants = None

    ref_row = predictions[predictions["horizon_hours"] == 24]
    if ref_row.empty and not predictions.empty:
        ref_row = predictions.iloc[[0]]

    # Inline summary
    inline_items = []
    for p in pollutant_order:
        if pollutants is not None:
            val = pollutants.get(p)
            display = "—" if val is None else f"{val:.1f}"
        else:
            if not ref_row.empty and p in ref_row.columns:
                try:
                    raw = ref_row[p].iloc[0]
                    display = "—" if pd.isna(raw) else f"{float(raw):.1f}"
                except Exception:
                    display = "—"
            else:
                display = "—"
        inline_items.append(f"{readable_feature(p)}: {display}")
    st.markdown(" • ".join(inline_items))

    # Small cards
    cols = st.columns(len(pollutant_order))
    for idx, p in enumerate(pollutant_order):
        with cols[idx]:
            if pollutants is not None:
                val = pollutants.get(p)
                display = "—" if val is None else f"{val:.1f}"
            else:
                if not ref_row.empty and p in ref_row.columns:
                    try:
                        raw = ref_row[p].iloc[0]
                        display = "—" if pd.isna(raw) else f"{float(raw):.1f}"
                    except Exception:
                        display = "—"
                else:
                    display = "—"
            unit = POLLUTANT_UNITS.get(p, "")
            st.markdown(
                f"""
                <div class="pollutant-card">
                    <div class="pollutant-name">{readable_feature(p)}</div>
                    <div class="pollutant-value">{display}</div>
                    <div class="pollutant-unit">{unit}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Forecast trend
    st.markdown('<div class="section-title">Forecast Trend</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=predictions["forecast_time"],
            y=predictions["predicted_aqi"],
            mode="lines+markers",
            line=dict(color=CHART_BLUE, width=3),
            marker=dict(color=CHART_BLUE, size=9),
            hovertemplate=("<b>%{y:.1f} AQI</b><br>%{x}<extra></extra>")
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=25, b=20),
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(title="Forecast Time", title_font=dict(color=TEXT_COLOR), tickfont=dict(color=TEXT_COLOR),
                   showgrid=False, linecolor=TEXT_COLOR, tickcolor=TEXT_COLOR),
        yaxis=dict(title="AQI", title_font=dict(color=TEXT_COLOR), tickfont=dict(color=TEXT_COLOR),
                   gridcolor=GRID_COLOR, zerolinecolor=TEXT_COLOR, linecolor=TEXT_COLOR),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # AQI Trend (monthly view)
    st.markdown('<div class="section-title">AQI Trend</div>', unsafe_allow_html=True)
    now = datetime.now()
    years = list(range(2024, now.year + 1))[::-1]
    selected_year = st.selectbox("Year", years, index=0)
    months = list(range(1, 13))
    default_month_index = now.month - 1 if selected_year == now.year else 0
    selected_month = st.selectbox("Month", months, index=default_month_index,
                                  format_func=lambda m: datetime(selected_year, m, 1).strftime("%B"))

    if selected_year > now.year or (selected_year == now.year and selected_month > now.month):
        st.error("Data not available for future months")
    else:
        try:
            month_df = fetch_monthly_aqi_from_hopsworks(selected_year, selected_month)
        except Exception:
            # fallback to external API if Hopsworks read fails
            try:
                month_df = fetch_monthly_aqi(selected_year, selected_month)
            except Exception as exc:
                month_df = pd.DataFrame()
                st.warning(f"Unable to load monthly AQI data: {exc}")

        if month_df.empty:
            st.info("No AQI data available for the selected month.")
        else:
            month_df = month_df.set_index("datetime")
            daily = month_df["aqi"].resample("D").mean().dropna()
            aqi_fig = go.Figure()
            aqi_fig.add_trace(
                go.Scatter(
                    x=daily.index,
                    y=daily.values,
                    mode="lines+markers",
                    line=dict(color=CHART_BLUE, width=2),
                    marker=dict(color=CHART_BLUE, size=6),
                    hovertemplate="<b>%{y:.1f} AQI</b><br>%{x}<extra></extra>"
                )
            )
            aqi_fig.update_layout(
                height=360,
                autosize=True,
                margin=dict(l=20, r=20, t=25, b=20),
                plot_bgcolor=WHITE,
                paper_bgcolor=WHITE,
                font=dict(color=TEXT_COLOR, size=12),
                showlegend=False,
                xaxis=dict(title="Date", showgrid=False, linecolor=TEXT_COLOR, tickcolor=TEXT_COLOR,
                           title_font=dict(color=TEXT_COLOR), tickfont=dict(color=TEXT_COLOR)),
                yaxis=dict(title="AQI", gridcolor=GRID_COLOR, linecolor=TEXT_COLOR, tickcolor=TEXT_COLOR,
                           title_font=dict(color=TEXT_COLOR), tickfont=dict(color=TEXT_COLOR))
            )
            st.plotly_chart(aqi_fig, use_container_width=True)

    # Model explanation (SHAP)
    st.markdown('<div class="section-title">Model Explanation</div>', unsafe_allow_html=True)
    st.caption(
        "SHAP shows which input features contributed most to each AQI forecast. Positive values push the prediction higher; negative values push it lower.")
    selected_horizon = st.radio("Forecast horizon", [24, 48, 72], horizontal=True,
                               format_func=lambda value: f"{value}-hour forecast")
    explanation = explanations.get(selected_horizon)

    if explanation is not None:
        explanation = explanation.copy()
        if "feature" not in explanation.columns:
            st.error("SHAP explanation does not contain feature names.")
            st.stop()
        if "shap_value" not in explanation.columns:
            st.error("SHAP explanation does not contain SHAP values.")
            st.stop()
        explanation["feature"] = explanation["feature"].apply(readable_feature)
        explanation = explanation.sort_values("shap_value")
        fig_shap = go.Figure()
        fig_shap.add_trace(
            go.Bar(
                x=explanation["shap_value"],
                y=explanation["feature"],
                orientation="h",
                marker=dict(color=CHART_BLUE),
                hovertemplate=("<b>%{y}</b><br>SHAP contribution: %{x:.3f}<extra></extra>")
            )
        )
        fig_shap.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=25, b=20),
            plot_bgcolor=WHITE,
            paper_bgcolor=WHITE,
            font=dict(color=TEXT_COLOR),
            xaxis=dict(title="SHAP Contribution", title_font=dict(color=TEXT_COLOR),
                       tickfont=dict(color=TEXT_COLOR), gridcolor=GRID_COLOR, zeroline=True,
                       zerolinecolor=TEXT_COLOR, linecolor=TEXT_COLOR, tickcolor=TEXT_COLOR),
            yaxis=dict(title="", tickfont=dict(color=TEXT_COLOR), linecolor=TEXT_COLOR, tickcolor=TEXT_COLOR),
            showlegend=False
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info("SHAP explanation is not available for this forecast.")


# Initial state when no forecast generated yet
else:
    st.info("Select 'Generate AQI Forecast' to retrieve the latest forecast.")


# Footer
st.markdown(
    """
    <div class="footer">
        Gujranwala Air Quality Forecast ·
        CatBoost · Hopsworks · SHAP
    </div>
    """,
    unsafe_allow_html=True
)
