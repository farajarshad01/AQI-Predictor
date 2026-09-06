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

    [data-testid="stSidebar"] {{
        background-color: {WHITE};
        border-right: 1px solid {BORDER};
    }}

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

    .footer {{
        text-align: center;
        color: #777777;
        font-size: 0.72rem;
        padding: 2rem 0 1rem 0;
    }}

    /* Darker spinner for visibility (attempt to override Streamlit spinner) */
    .stSpinner, .stSpinner * {{
        color: #111 !important;
        stroke: #111 !important;
        fill: #111 !important;
    }}

    /* Selectbox styling: make closed select/light label area use light background (#cbcbcb) with dark text */
    .stSelectbox > div[role="button"], .stSelectbox > div[role="button"] * {{
        background-color: #cbcbcb !important;
        color: {TEXT_COLOR} !important;
        border-radius: 8px !important;
        padding-left: 12px !important;
    }}

    /* Native select fallback */
    select,
    .stSelectbox select {{
        background-color: #cbcbcb !important;
        color: {TEXT_COLOR} !important;
        border-radius: 8px !important;
    }}

    /* Opened listbox (Streamlit's custom dropdown) - set container background to light and options dark text */
    div[role="listbox"] {{
        background-color: #ffffff !important;
        color: {TEXT_COLOR} !important;
        border-radius: 8px !important;
        padding: 8px !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08) !important;
    }}

    /* Individual option entries */
    div[role="listbox"] > div[role="option"] {{
        background-color: transparent !important;
        color: {TEXT_COLOR} !important;
        padding: 8px 12px !important;
        border-radius: 6px !important;
    }}

    /* Hover and focus styles for options */
    div[role="listbox"] > div[role="option"]:hover {{
        background-color: #e6e6e6 !important;
        color: {TEXT_COLOR} !important;
    }}

    /* Selected option should use the #cbcbcb light background with dark text */
    div[role="listbox"] > div[role="option"][aria-selected="true"] {{
        background-color: #cbcbcb !important;
        color: {TEXT_COLOR} !important;
    }}

    /* Option element fallback in some browsers */
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
        return (
            "Good",
            GOOD_COLOR,
            "#dcfce7",
            "Air quality is considered satisfactory."
        )

    if aqi <= 100:
        return (
            "Moderate",
            MODERATE_COLOR,
            "#fef9c3",
            "Air quality is acceptable; unusually sensitive people may experience minor effects."
        )

    if aqi <= 150:
        return (
            "Unhealthy for Sensitive Groups",
            SENSITIVE_COLOR,
            "#ffedd5",
            "Sensitive groups may experience health effects."
        )

    if aqi <= 200:
        return (
            "Unhealthy",
            UNHEALTHY_COLOR,
            "#fee2e2",
            "Everyone may begin to experience health effects."
        )

    if aqi <= 300:
        return (
            "Very Unhealthy",
            VERY_UNHEALTHY_COLOR,
            "#fecaca",
            "Health alert: the risk of health effects is increased."
        )

    return (
        "Hazardous",
        HAZARDOUS_COLOR,
        "#fca5a5",
        "Health warning of emergency conditions."
    )


# Feature names

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


# Helper: fetch hourly AQI for a month from Hopsworks
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_monthly_aqi_from_hopsworks(year: int, month: int) -> pd.DataFrame:
    """Fetch hourly AQI for a given year/month from the Hopsworks feature group."""
    try:
        fg = get_feature_group()
        df = fg.read()  # hsfs feature group read()
    except Exception as exc:
        raise RuntimeError(f"Unable to read feature group from Hopsworks: {exc}")

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    if "datetime" not in df.columns:
        # try alternate names
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
        # propagate error so caller can fall back
        raise RuntimeError(f"Unable to read feature group from Hopsworks: {exc}")

    if df is None or df.empty:
        raise RuntimeError("Feature group is empty")

    df = df.copy()

    # normalize datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        # try first column as datetime
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


# Helper: fetch latest AQI (from Hopsworks hourly pipeline)
@st.cache_data(ttl=300, show_spinner=False)
def fetch_latest_aqi_from_hopsworks():
    """
    Return dict: {"aqi": float or None, "datetime": pd.Timestamp or None}
    """
    try:
        fg = get_feature_group()
        df = fg.read()
    except Exception as exc:
        raise RuntimeError(f"Unable to read feature group from Hopsworks: {exc}")

    if df is None or df.empty:
        return {"aqi": None, "datetime": None}

    df = df.copy()

    # normalize datetime column
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif "date_time" in df.columns:
        df["datetime"] = pd.to_datetime(df["date_time"])
    elif "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"])
    else:
        return {"aqi": None, "datetime": None}

    if "us_aqi" not in df.columns:
        return {"aqi": None, "datetime": None}

    latest_row = df.sort_values("datetime").tail(1).iloc[0]
    aqi_val = latest_row.get("us_aqi", None)

    if pd.isna(aqi_val):
        return {"aqi": None, "datetime": latest_row.get("datetime", None)}

    try:
        return {"aqi": float(aqi_val), "datetime": pd.to_datetime(latest_row.get("datetime"))}
    except Exception:
        return {"aqi": None, "datetime": latest_row.get("datetime", None)}


def get_aqi_audience(aqi):
    """Short sentence describing who is at risk for this AQI."""
    if aqi is None:
        return ""
    if aqi <= 50:
        return "No special precautions required for the general population."
    if aqi <= 100:
        return "Unusually sensitive people may consider limiting prolonged outdoor exertion."
    if aqi <= 150:
        return "Sensitive groups (children, elderly, people with lung disease, asthmatics) may be affected."
    if aqi <= 200:
        return "People with heart/lung disease, older adults and children should avoid prolonged outdoor exertion."
    if aqi <= 300:
        return "Everyone may experience more serious health effects; avoid outdoor activity if possible."
    return "Health alert: everyone should avoid outdoor exertion and follow local guidance."


# Sidebar
with st.sidebar:

    st.markdown(
        f"""
        <div style="
            font-size:1.15rem;
            font-weight:700;
            color:{TEXT_COLOR};
            margin-bottom:1.8rem;
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
            color:#777777;
            text-transform:uppercase;
            letter-spacing:0.06em;
            margin-bottom:0.4rem;
        ">
            Location
        </div>

        <div style="
            font-size:1rem;
            font-weight:600;
            color:{TEXT_COLOR};
            margin-bottom:1.5rem;
        ">
            Gujranwala, Punjab
        </div>
        """,
        unsafe_allow_html=True
    )

    # updated description block with bullets (inside the sidebar)
    st.markdown(
        f"""
        <div style="
            font-size:0.78rem;
            color:{TEXT_COLOR};
            line-height:1.45;
            margin-top:0.6rem;
        ">
            Air Quality Forecasts generated using weather observations, air quality data, machine learning models for 24hr / 48hr / 72hr forecasts.
            <ul style="margin-top:8px; padding-left:20px; color:{TEXT_COLOR};">
                <li>Open Meteo</li>
                <li>CatBoost Regressor</li>
                <li>Hopsworks</li>
                <li>GitHub Actions</li>
                <li>Streamlit</li>
                <li>SHAP</li>
            </ul>
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

    <div class="dashboard-subtitle">
        Machine-learning forecast for the next 72 hours.
    </div>
    """,
    unsafe_allow_html=True
)


# Current AQI card (split into 30/70 value + Health Advisory) — below title/subtext and above the Generate button
try:
    _current = fetch_latest_aqi_from_hopsworks()
except Exception as exc:
    _current = {"aqi": None, "datetime": None}
    st.info(f"Unable to load current AQI from Hopsworks: {exc}")

cols = st.columns([3, 7])

if _current.get("aqi") is not None:
    aqi_value = _current["aqi"]
    aqi_dt = _current["datetime"]
    category, category_color, _bg, message = get_aqi_category(aqi_value)
    audience = get_aqi_audience(aqi_value)

    with cols[0]:
        st.markdown(
            f"""
            <div class="info-card" style="text-align:center;">
                <div style="font-size:0.85rem; color:#666666;">Current AQI</div>
                <div style="font-size:2.4rem; font-weight:800; color:{TEXT_COLOR}; margin-top:0.25rem;">{aqi_value:.1f}</div>
                <div style="margin-top:6px; color:{category_color}; font-weight:700;">{category}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with cols[1]:
        # Title now matches Current AQI title (size, color) and is center-aligned.
        # Message and audience use the same font-size and weight; audience colored but not bold.
        st.markdown(
            f"""
            <div class="info-card">
                <div style="font-size:0.85rem; color:#666666; font-weight:400; margin-bottom:8px; text-align:center;">Health Advisory</div>
                <div style="text-align:center; color:#555555; font-size:1rem; font-weight:400;">{message if message else ''}</div>
                <div style="text-align:center; margin-top:6px; color:{category_color}; font-weight:400; font-size:1rem;">{audience if audience else ''}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    with cols[0]:
        st.markdown(
            f"""
            <div class="info-card" style="text-align:center;">
                <div style="font-size:0.85rem; color:#666666;">Current AQI</div>
                <div style="font-size:2.4rem; font-weight:800; color:{TEXT_COLOR}; margin-top:0.25rem;">—</div>
                <div style="margin-top:6px; color:#666666; font-weight:700;">N/A</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with cols[1]:
        st.markdown(
            f"""
            <div class="info-card">
                <div style="font-size:0.85rem; color:#666666; font-weight:400; margin-bottom:8px; text-align:center;">Health Advisory</div>
                <div style="text-align:center; color:#555555; font-size:1rem; font-weight:400;">Current AQI measurement is not available.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Add spacing between the cards and the Generate button
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# Generate forecast

generate = st.button(
    "Generate AQI Forecast",
    type="primary"
)


if generate:

    with st.spinner(
        "Generating AQI forecast..."
    ):

        try:

            result = predict()

            st.session_state[
                "forecast_result"
            ] = result

        except Exception as exc:

            st.error(
                f"Unable to generate forecast: {exc}"
            )


# Display forecast

if "forecast_result" in st.session_state:

    result = st.session_state[
        "forecast_result"
    ]

    # Supports the current dictionary return format.
    if isinstance(result, dict):

        predictions = result[
            "predictions"
        ]

        explanations = result.get(
            "shap_explanations",
            result.get(
                "explanations",
                {}
            )
        )

    # Also supports the older tuple format.
    elif isinstance(result, tuple):

        predictions = result[0]

        explanations = result[1]

    else:

        st.error(
            "Unexpected prediction result format."
        )

        st.stop()


    predictions = predictions.copy()

    predictions["forecast_time"] = pd.to_datetime(
        predictions["forecast_time"],
        utc=True
    )


    # Forecast cards

    st.markdown(
        '<div class="section-title">Forecast</div>',
        unsafe_allow_html=True
    )

    cards = st.columns(3)

    for index, horizon in enumerate(
        [24, 48, 72]
    ):

        row = predictions[
            predictions["horizon_hours"] == horizon
        ]

        if row.empty:
            continue

        value = float(
            row["predicted_aqi"].iloc[0]
        )

        forecast_time = row[
            "forecast_time"
        ].iloc[0]

        (
            category,
            category_color,
            category_background,
            message
        ) = get_aqi_category(
            value
        )

        with cards[index]:

            st.markdown(
                f"""<div class="forecast-card"><div class="forecast-horizon">{horizon}-Hour Forecast</div><div class="forecast-value">{value:.1f}</div></div>""",
                unsafe_allow_html=True
            )

            # Native Streamlit warning box.
            # This guarantees the label renders correctly.
            st.markdown(
                f"""<div style="
                    background-color: {category_color};
                    color: {"#000000" if category in ["Good", "Moderate"] else "#ffffff"};
                    padding: 0.5rem 0.75rem;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 0.85rem;
                    text-align: center;
                    margin-top: 0.6rem;
                ">
                    {category}
                </div>""",
                unsafe_allow_html=True
            )

    # Top pollutants (line and small cards)
    st.markdown(
        '<div class="section-title">Top Pollutants</div>',
        unsafe_allow_html=True
    )

    pollutant_order = ["pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]

    # Try to fetch real pollutant values from Hopsworks; fall back to prediction row
    try:
        pollutants = fetch_latest_pollutants_from_hopsworks()
    except Exception as exc:
        pollutants = None
        st.info(f"Could not load latest pollutants from Hopsworks: {exc}")

    # Choose a reference row (prefer 24h horizon)
    ref_row = predictions[predictions["horizon_hours"] == 24]
    if ref_row.empty and not predictions.empty:
        ref_row = predictions.iloc[[0]]

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

    st.markdown(
        '<div class="section-title">Forecast Trend</div>',
        unsafe_allow_html=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=predictions["forecast_time"],
            y=predictions["predicted_aqi"],
            mode="lines+markers",
            line=dict(
                color=CHART_BLUE,
                width=3
            ),
            marker=dict(
                color=CHART_BLUE,
                size=9
            ),
            hovertemplate=(
                "<b>%{y:.1f} AQI</b>"
                "<br>%{x}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        height=360,

        margin=dict(
            l=20,
            r=20,
            t=25,
            b=20
        ),

        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,

        font=dict(
            color=TEXT_COLOR
        ),

        xaxis=dict(
            title="Forecast Time",

            title_font=dict(
                color=TEXT_COLOR
            ),

            tickfont=dict(
                color=TEXT_COLOR
            ),

            showgrid=False,

            linecolor=TEXT_COLOR,

            tickcolor=TEXT_COLOR
        ),

        yaxis=dict(
            title="AQI",

            title_font=dict(
                color=TEXT_COLOR
            ),

            tickfont=dict(
                color=TEXT_COLOR
            ),

            gridcolor=GRID_COLOR,

            zerolinecolor=TEXT_COLOR,

            linecolor=TEXT_COLOR
        ),

        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # AQI Trend (monthly view) — fetch from Hopsworks
    st.markdown(
        '<div class="section-title">AQI Trend</div>',
        unsafe_allow_html=True
    )

    now = datetime.now()
    # Data available since Jan 2024
    years = list(range(2024, now.year + 1))[::-1]
    selected_year = st.selectbox("Year", years, index=0)

    months = list(range(1, 13))
    # default month selection: current month if current year else January
    default_month_index = now.month - 1 if selected_year == now.year else 0
    selected_month = st.selectbox("Month", months, index=default_month_index, format_func=lambda m: datetime(selected_year, m, 1).strftime("%B"))

    # Prevent future month selection
    if selected_year > now.year or (selected_year == now.year and selected_month > now.month):
        st.error("Data not available for future months")
    else:
        try:
            month_df = fetch_monthly_aqi_from_hopsworks(selected_year, selected_month)

            if month_df.empty:
                st.info("No AQI data available for the selected month.")
            else:
                # aggregate daily mean for the selected month
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

                st.plotly_chart(aqi_fig, use_container_width=True)

        except Exception as exc:
            st.warning(f"Unable to load monthly AQI data from Hopsworks: {exc}")

    # Model explanation

    st.markdown(
        '<div class="section-title">Model Explanation</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "SHAP shows which input features contributed most to each AQI forecast. "
        "Positive values push the prediction higher; negative values push it lower."
    )

    # Horizon selector

    selected_horizon = st.radio(
        "Forecast horizon",
        [24, 48, 72],
        horizontal=True,
        format_func=lambda value:
            f"{value}-hour forecast"
    )


    # Retrieve explanation

    explanation = explanations.get(
        selected_horizon
    )


    if explanation is not None:

        explanation = explanation.copy()


        # Handle both possible SHAP column names.
        if "feature" not in explanation.columns:

            st.error(
                "SHAP explanation does not contain feature names."
            )

            st.stop()


        if "shap_value" not in explanation.columns:

            st.error(
                "SHAP explanation does not contain SHAP values."
            )

            st.stop()


        explanation["feature"] = (
            explanation["feature"]
            .apply(
                readable_feature
            )
        )


        explanation = (
            explanation
            .sort_values(
                "shap_value"
            )
        )


        # SHAP bar chart

        fig_shap = go.Figure()


        fig_shap.add_trace(
            go.Bar(
                x=explanation[
                    "shap_value"
                ],

                y=explanation[
                    "feature"
                ],

                orientation="h",

                marker=dict(
                    color=CHART_BLUE
                ),

                hovertemplate=(
                    "<b>%{y}</b>"
                    "<br>SHAP contribution: %{x:.3f}"
                    "<extra></extra>"
                )
            )
        )


        fig_shap.update_layout(
            height=380,

            margin=dict(
                l=20,
                r=20,
                t=25,
                b=20
            ),

            plot_bgcolor=WHITE,
            paper_bgcolor=WHITE,

            font=dict(
                color=TEXT_COLOR
            ),

            xaxis=dict(
                title="SHAP Contribution",

                title_font=dict(
                    color=TEXT_COLOR
                ),

                tickfont=dict(
                    color=TEXT_COLOR
                ),

                gridcolor=GRID_COLOR,

                zeroline=True,

                zerolinecolor=TEXT_COLOR,

                linecolor=TEXT_COLOR,

                tickcolor=TEXT_COLOR
            ),

            yaxis=dict(
                title="",

                tickfont=dict(
                    color=TEXT_COLOR
                ),

                linecolor=TEXT_COLOR,

                tickcolor=TEXT_COLOR
            ),

            showlegend=False
        )


        st.plotly_chart(
            fig_shap,
            use_container_width=True
        )


    else:

        st.info(
            "SHAP explanation is not available for this forecast."
        )


# Initial state

else:

    st.info(
        "Select 'Generate AQI Forecast' to retrieve the latest forecast."
    )


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
