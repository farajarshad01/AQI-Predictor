import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.prediction_pipeline import predict


st.set_page_config(
    page_title="Gujranwala AQI Forecast",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Page styling

st.markdown(
    """
    <style>

    .stApp {
        background: #f7f8fa;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1 {
        color: #172033 !important;
        font-weight: 700 !important;
    }

    h2, h3 {
        color: #172033 !important;
    }

    .location-text {
        color: #667085;
        font-size: 0.95rem;
        margin-top: -0.5rem;
        margin-bottom: 0.25rem;
    }

    .description-text {
        color: #667085;
        font-size: 0.85rem;
        margin-bottom: 1.5rem;
    }

    .forecast-card {
        background: white;
        border: 1px solid #e4e7ec;
        border-radius: 14px;
        padding: 22px;
        min-height: 210px;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
    }

    .forecast-horizon {
        color: #667085;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .forecast-value {
        color: #172033;
        font-size: 2.5rem;
        font-weight: 750;
        margin: 12px 0 8px 0;
    }

    .forecast-time {
        color: #98a2b3;
        font-size: 0.72rem;
        margin-top: 14px;
    }

    .health-box {
        border-radius: 8px;
        padding: 8px 10px;
        font-size: 0.74rem;
        font-weight: 700;
        display: inline-block;
    }

    .health-message {
        color: #667085;
        font-size: 0.78rem;
        line-height: 1.4;
        margin-top: 10px;
    }

    .section-label {
        color: #172033;
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 0.8rem;
    }

    .explanation-box {
        background: white;
        border: 1px solid #e4e7ec;
        border-radius: 12px;
        padding: 18px;
    }

    .footer {
        color: #98a2b3;
        text-align: center;
        font-size: 0.7rem;
        margin-top: 3rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# AQI classification

def get_aqi_category(aqi):

    if aqi <= 50:
        return (
            "Good",
            "#027A48",
            "#ECFDF3",
            "Air quality is satisfactory."
        )

    if aqi <= 100:
        return (
            "Moderate",
            "#B54708",
            "#FFFAEB",
            "Air quality is acceptable. Sensitive individuals may experience minor effects."
        )

    if aqi <= 150:
        return (
            "Unhealthy for Sensitive Groups",
            "#C4320A",
            "#FFF4ED",
            "Sensitive groups may experience health effects."
        )

    if aqi <= 200:
        return (
            "Unhealthy",
            "#D92D20",
            "#FEF3F2",
            "Everyone may begin to experience health effects."
        )

    if aqi <= 300:
        return (
            "Very Unhealthy",
            "#B42318",
            "#FEE4E2",
            "Health alert: the risk of health effects is increased."
        )

    return (
        "Hazardous",
        "#7A271A",
        "#FDEAD7",
        "Health warning of emergency conditions."
    )


# Feature display names

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

    "aqi_change": "AQI Change",

    "aqi_change_rate": "AQI Change Rate",

    "aqi_lag_1": "AQI — 1 Hour Lag",

    "aqi_lag_12": "AQI — 12 Hour Lag",

    "aqi_lag_24": "AQI — 24 Hour Lag",

    "aqi_rolling_6": "AQI — 6 Hour Average",

    "aqi_rolling_12": "AQI — 12 Hour Average",

    "aqi_rolling_24": "AQI — 24 Hour Average",

    "year": "Year",

    "month": "Month",

    "day": "Day",

    "day_of_week": "Day of Week",

    "hour": "Hour",
}


def readable_feature(name):

    return FEATURE_LABELS.get(
        name,
        name.replace("_", " ").title()
    )


# Header

st.title(
    "Air Quality Forecast"
)

st.markdown(
    '<div class="location-text">Gujranwala, Punjab, Pakistan</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="description-text">Machine-learning AQI forecasts for the next 24, 48 and 72 hours.</div>',
    unsafe_allow_html=True
)


# Generate forecast

generate = st.button(
    "Generate AQI Forecast",
    type="primary"
)


if generate:

    with st.spinner("Generating forecast..."):

        try:

            result = predict()

            st.session_state["forecast_result"] = result

        except Exception as exc:

            st.error(
                f"Unable to generate forecast: {exc}"
            )


# Dashboard

if "forecast_result" not in st.session_state:

    st.info(
        "Select Generate AQI Forecast to retrieve the latest forecast."
    )

else:

    result = st.session_state[
        "forecast_result"
    ]

    predictions = result[0]

    explanations = result[1]


    # Forecast cards

    st.markdown(
        '<div class="section-label">Forecast</div>',
        unsafe_allow_html=True
    )

    card_columns = st.columns(
        3,
        gap="medium"
    )


    for column, horizon in zip(
        card_columns,
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

        forecast_time = pd.to_datetime(
            row["forecast_time"].iloc[0]
        )

        category, text_color, background, message = (
            get_aqi_category(value)
        )


        with column:

            st.markdown(
                f"""
                <div class="forecast-card">

                    <div class="forecast-horizon">
                        {horizon}-Hour Forecast
                    </div>

                    <div class="forecast-value">
                        {value:.1f}
                    </div>

                    <span class="health-box"
                          style="
                              color:{text_color};
                              background:{background};
                          ">
                        {category}
                    </span>

                    <div class="health-message">
                        {message}
                    </div>

                    <div class="forecast-time">
                        Forecast:
                        {forecast_time.strftime("%d %b %Y, %H:%M UTC")}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # Forecast trend

    st.markdown(
        '<div class="section-label">Forecast Trend</div>',
        unsafe_allow_html=True
    )


    trend = go.Figure()


    trend.add_trace(
        go.Scatter(
            x=predictions["forecast_time"],
            y=predictions["predicted_aqi"],
            mode="lines+markers",
            line=dict(
                width=3
            ),
            marker=dict(
                size=9
            ),
            hovertemplate=(
                "AQI: %{y:.1f}"
                "<br>%{x}"
                "<extra></extra>"
            )
        )
    )


    trend.update_layout(
        height=360,
        margin=dict(
            l=30,
            r=20,
            t=20,
            b=30
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        xaxis=dict(
            title="Forecast Time",
            showgrid=False
        ),
        yaxis=dict(
            title="Predicted AQI",
            gridcolor="#eaecf0",
            zeroline=False
        )
    )


    st.plotly_chart(
        trend,
        use_container_width=True
    )


    # Forecast comparison

    st.markdown(
        '<div class="section-label">Forecast Summary</div>',
        unsafe_allow_html=True
    )


    summary = predictions[
        [
            "horizon_hours",
            "forecast_time",
            "predicted_aqi"
        ]
    ].copy()


    summary["horizon_hours"] = (
        summary["horizon_hours"]
        .astype(str)
        + " hours"
    )


    summary["forecast_time"] = (
        pd.to_datetime(
            summary["forecast_time"]
        )
        .dt.strftime(
            "%d %b %Y, %H:%M UTC"
        )
    )


    summary["predicted_aqi"] = (
        summary["predicted_aqi"]
        .round(1)
    )


    summary.columns = [
        "Forecast Horizon",
        "Forecast Time",
        "Predicted AQI"
    ]


    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


    # SHAP explanation

    st.markdown(
        '<div class="section-label">Model Explanation</div>',
        unsafe_allow_html=True
    )


    st.caption(
        "SHAP explains which features influenced each model prediction. "
        "Positive values increase the predicted AQI; negative values decrease it."
    )


    selected_horizon = st.selectbox(
        "Select forecast horizon",
        [24, 48, 72],
        format_func=lambda x: f"{x}-hour forecast"
    )


    explanation = explanations.get(
        selected_horizon
    )


    if explanation is not None:

        explanation = explanation.copy()


        explanation["feature"] = (
            explanation["feature"]
            .apply(readable_feature)
        )


        explanation = (
            explanation
            .sort_values(
                "shap_value"
            )
        )


        shap_chart = go.Figure()


        shap_chart.add_trace(
            go.Bar(
                x=explanation["shap_value"],
                y=explanation["feature"],
                orientation="h",
                hovertemplate=(
                    "%{y}"
                    "<br>SHAP: %{x:.3f}"
                    "<extra></extra>"
                )
            )
        )


        shap_chart.update_layout(
            height=380,
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=30
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
            xaxis=dict(
                title="SHAP Contribution",
                zeroline=True,
                gridcolor="#eaecf0"
            ),
            yaxis=dict(
                title=""
            )
        )


        st.plotly_chart(
            shap_chart,
            use_container_width=True
        )


        # SHAP table

        table = explanation.copy()


        table["feature"] = (
            table["feature"]
        )


        table["feature_value"] = (
            table["feature_value"]
            .round(2)
        )


        table["shap_value"] = (
            table["shap_value"]
            .round(3)
        )


        table["impact"] = table[
            "shap_value"
        ].apply(
            lambda value:
            "Increases AQI"
            if value > 0
            else "Decreases AQI"
        )


        table = table[
            [
                "feature",
                "feature_value",
                "shap_value",
                "impact"
            ]
        ]


        table.columns = [
            "Feature",
            "Current Value",
            "SHAP Contribution",
            "Impact"
        ]


        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True
        )


    else:

        st.info(
            "SHAP explanation is not available for this forecast."
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
