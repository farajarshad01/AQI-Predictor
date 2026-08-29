import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.prediction_pipeline import predict


st.set_page_config(
    page_title="Gujranwala AQI Forecast",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>

    /* ---------- Global ---------- */

    .stApp {
        background: #f6f7f9;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    h1, h2, h3, h4 {
        color: #202124;
    }

    p {
        color: #5f6368;
    }


    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e6e8eb;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    .sidebar-logo {
        font-size: 18px;
        font-weight: 700;
        letter-spacing: -0.4px;
        color: #202124;
        padding: 0 10px 30px 10px;
    }

    .sidebar-logo span {
        color: #6b7280;
        font-weight: 400;
    }

    .sidebar-section {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #9aa0a6;
        padding: 0 10px 8px 10px;
    }

    .sidebar-info {
        margin: 20px 10px;
        padding: 16px;
        background: #f6f7f9;
        border-radius: 10px;
        border: 1px solid #eceef1;
    }

    .sidebar-info-title {
        font-size: 12px;
        font-weight: 600;
        color: #202124;
        margin-bottom: 5px;
    }

    .sidebar-info-text {
        font-size: 11px;
        line-height: 1.5;
        color: #777;
    }


    /* ---------- Header ---------- */

    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 2rem;
    }

    .header-title {
        font-size: 32px;
        font-weight: 700;
        letter-spacing: -1px;
        color: #202124;
        margin-bottom: 5px;
    }

    .header-location {
        font-size: 14px;
        color: #777;
    }

    .header-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #9aa0a6;
        margin-bottom: 4px;
    }


    /* ---------- Section ---------- */

    .section-title {
        font-size: 19px;
        font-weight: 650;
        color: #202124;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .section-description {
        font-size: 13px;
        color: #777;
        margin-bottom: 1.2rem;
    }


    /* ---------- AQI Cards ---------- */

    .aqi-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 22px;
        min-height: 185px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.025);
    }

    .aqi-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 18px;
    }

    .aqi-horizon {
        font-size: 12px;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .aqi-number {
        font-size: 42px;
        line-height: 1;
        font-weight: 700;
        letter-spacing: -1.5px;
        color: #202124;
        margin-bottom: 10px;
    }

    .aqi-category {
        display: inline-block;
        padding: 5px 9px;
        border-radius: 5px;
        font-size: 11px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .aqi-message {
        font-size: 12px;
        line-height: 1.45;
        color: #777;
    }

    .forecast-time {
        font-size: 11px;
        color: #9aa0a6;
        margin-top: 13px;
        padding-top: 12px;
        border-top: 1px solid #f0f1f3;
    }


    /* ---------- Status colors ---------- */

    .good {
        background: #edf8f0;
        color: #287a3d;
    }

    .moderate {
        background: #fff8df;
        color: #8a6d00;
    }

    .sensitive {
        background: #fff0e7;
        color: #a34d14;
    }

    .unhealthy {
        background: #fdebec;
        color: #a52a32;
    }

    .very-unhealthy {
        background: #f1eafa;
        color: #70449a;
    }

    .hazardous {
        background: #eeeeee;
        color: #333333;
    }


    /* ---------- Location / metadata ---------- */

    .location-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 18px 20px;
        margin-top: 1.5rem;
    }

    .location-label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #9aa0a6;
        margin-bottom: 5px;
    }

    .location-value {
        font-size: 15px;
        font-weight: 600;
        color: #202124;
    }


    /* ---------- SHAP ---------- */

    .shap-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }

    .shap-title {
        font-size: 15px;
        font-weight: 650;
        color: #202124;
        margin-bottom: 4px;
    }

    .shap-description {
        font-size: 12px;
        color: #777;
        margin-bottom: 15px;
    }


    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #9aa0a6;
        font-size: 11px;
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid #e5e7eb;
    }


    /* ---------- Buttons ---------- */

    div.stButton > button {
        border-radius: 7px;
        border: 1px solid #202124;
        background: #202124;
        color: #ffffff;
        font-weight: 600;
        font-size: 13px;
        padding: 0.55rem 1.1rem;
    }

    div.stButton > button:hover {
        background: #3b3d40;
        border-color: #3b3d40;
        color: #ffffff;
    }


    /* ---------- Hide Streamlit chrome ---------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

</style>
""",
    unsafe_allow_html=True,
)


def get_aqi_category(aqi):

    if aqi <= 50:
        return (
            "Good",
            "Air quality is satisfactory.",
            "good",
        )

    if aqi <= 100:
        return (
            "Moderate",
            "Air quality is acceptable; unusually sensitive people may be affected.",
            "moderate",
        )

    if aqi <= 150:
        return (
            "Unhealthy for Sensitive Groups",
            "Sensitive groups may experience health effects.",
            "sensitive",
        )

    if aqi <= 200:
        return (
            "Unhealthy",
            "Everyone may begin to experience health effects.",
            "unhealthy",
        )

    if aqi <= 300:
        return (
            "Very Unhealthy",
            "Health alert: increased risk of health effects.",
            "very-unhealthy",
        )

    return (
        "Hazardous",
        "Health warning: everyone is more likely to be affected.",
        "hazardous",
    )


FEATURE_LABELS = {

    "temperature_2m":
        "Temperature",

    "relative_humidity_2m":
        "Relative Humidity",

    "wind_speed_10m":
        "Wind Speed",

    "surface_pressure":
        "Surface Pressure",

    "precipitation":
        "Precipitation",

    "cloud_cover":
        "Cloud Cover",

    "pm10":
        "PM10",

    "pm2_5":
        "PM2.5",

    "carbon_monoxide":
        "Carbon Monoxide",

    "nitrogen_dioxide":
        "Nitrogen Dioxide",

    "sulphur_dioxide":
        "Sulphur Dioxide",

    "ozone":
        "Ozone",

    "us_aqi":
        "Previous AQI",

    "year":
        "Year",

    "month":
        "Month",

    "day":
        "Day",

    "day_of_week":
        "Day of Week",

    "hour":
        "Hour",

    "aqi_change":
        "AQI Change",

    "aqi_change_rate":
        "AQI Change Rate",

    "aqi_lag_1":
        "AQI Previous Hour",

    "aqi_lag_12":
        "AQI 12 Hours Ago",

    "aqi_lag_24":
        "AQI 24 Hours Ago",

    "aqi_rolling_6":
        "6-Hour AQI Average",

    "aqi_rolling_12":
        "12-Hour AQI Average",

    "aqi_rolling_24":
        "24-Hour AQI Average",
}


def format_feature_name(name):

    return FEATURE_LABELS.get(
        name,
        name.replace("_", " ").title()
    )


with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-logo">
            AQI <span>FORECAST</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.write("Overview")

    st.markdown(
        '<div class="sidebar-section" style="margin-top:25px;">Location</div>',
        unsafe_allow_html=True,
    )

    st.write("Gujranwala")

    st.markdown(
        """
        <div class="sidebar-info">
            <div class="sidebar-info-title">
                Forecast System
            </div>
            <div class="sidebar-info-text">
                CatBoost regression<br>
                Hopsworks model registry<br>
                SHAP explainability
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="dashboard-header">
        <div>
            <div class="header-label">
                Air Quality Monitoring
            </div>
            <div class="header-title">
                AQI Forecast
            </div>
            <div class="header-location">
                Gujranwala, Punjab, Pakistan
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    "Forecast air quality for the next 24, 48 and 72 hours."
)


if "predictions" not in st.session_state:
    st.session_state.predictions = None

if "explanations" not in st.session_state:
    st.session_state.explanations = None


button_col, _ = st.columns([1, 5])

with button_col:

    generate = st.button(
        "Generate Forecast",
        use_container_width=True,
    )


if generate:

    with st.spinner(
        "Generating forecast and explanations..."
    ):

        try:

            predictions, explanations = predict()

            st.session_state.predictions = predictions
            st.session_state.explanations = explanations

        except Exception as error:

            st.error(
                f"Unable to generate forecast: {error}"
            )


predictions = st.session_state.predictions
explanations = st.session_state.explanations


if predictions is not None:

    st.markdown(
        '<div class="section-title">Forecast Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">Predicted US AQI for Gujranwala.</div>',
        unsafe_allow_html=True,
    )

    columns = st.columns(3)

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

        forecast_time = (
            row["forecast_time"].iloc[0]
        )

        category, message, css_class = (
            get_aqi_category(value)
        )

        with columns[index]:

          

            card_html = f"""
<div class="aqi-card">
<div class="aqi-card-header">
<div class="aqi-horizon">{horizon}-Hour Forecast</div>
</div>
<div class="aqi-number">{value:.1f}</div>
<div class="aqi-category {css_class}">{category}</div>
<div class="aqi-message">{message}</div>
<div class="forecast-time">Forecast: {forecast_time}</div>
</div>
"""

            st.markdown(
                card_html,
                unsafe_allow_html=True,
            )



    st.markdown(
        """
        <div class="location-card">
            <div class="location-label">
                Forecast Location
            </div>
            <div class="location-value">
                Gujranwala, Punjab, Pakistan
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



    st.markdown(
        '<div class="section-title" style="margin-top:35px;">Model Explainability</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            SHAP identifies which features contributed most to each AQI prediction.
            Positive values increase the predicted AQI; negative values decrease it.
        </div>
        """,
        unsafe_allow_html=True,
    )


    for horizon in [24, 48, 72]:

        if horizon not in explanations:
            continue

        explanation = explanations[
            horizon
        ].copy()

        explanation["feature"] = (
            explanation["feature"]
            .apply(format_feature_name)
        )

        explanation = explanation[
            [
                "feature",
                "feature_value",
                "shap_value",
                "impact",
            ]
        ]

        explanation["feature_value"] = (
            explanation["feature_value"]
            .round(3)
        )

        explanation["shap_value"] = (
            explanation["shap_value"]
            .round(3)
        )

        st.markdown(
            f"""
            <div class="shap-card">
                <div class="shap-title">
                    {horizon}-Hour Forecast
                </div>
                <div class="shap-description">
                    Top factors influencing this prediction
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        chart_data = (
            explanation
            .head(8)
            .set_index("feature")["shap_value"]
        )

        st.bar_chart(
            chart_data,
            horizontal=True,
        )

        display_table = explanation.head(8).rename(
            columns={
                "feature": "Feature",
                "feature_value": "Value",
                "shap_value": "SHAP Impact",
                "impact": "Effect",
            }
        )

        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True,
        )


    st.markdown(
        '<div class="section-title" style="margin-top:35px;">Forecast System</div>',
        unsafe_allow_html=True,
    )

    info_columns = st.columns(4)

    with info_columns[0]:
        st.metric(
            "Location",
            "Gujranwala",
        )

    with info_columns[1]:
        st.metric(
            "Horizons",
            "24 / 48 / 72h",
        )

    with info_columns[2]:
        st.metric(
            "Model",
            "CatBoost",
        )

    with info_columns[3]:
        st.metric(
            "Explainability",
            "SHAP",
        )



else:

    st.markdown(
        """
        <div class="location-card">
            <div class="location-label">
                Forecast Location
            </div>
            <div class="location-value">
                Gujranwala, Punjab, Pakistan
            </div>
            <div style="font-size:12px;color:#777;margin-top:8px;">
                Generate a forecast to view predicted AQI values
                and model explanations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



st.markdown(
    """
    <div class="footer">
        AQI Forecasting System · CatBoost · Hopsworks · SHAP · Open-Meteo
    </div>
    """,
    unsafe_allow_html=True,
)
