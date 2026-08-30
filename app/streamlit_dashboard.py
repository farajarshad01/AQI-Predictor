import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.prediction_pipeline import predict


st.set_page_config(
    page_title="Gujranwala AQI Forecast",
    layout="wide",
    initial_sidebar_state="collapsed",
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

    h1, h2, h3, p, div, span {
        font-family: Arial, sans-serif;
    }

    .page-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.2rem;
    }

    .page-subtitle {
        font-size: 0.95rem;
        color: #667085;
        margin-bottom: 1.8rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #111827;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    .forecast-card {
        background: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 14px;
        padding: 1.4rem;
        min-height: 220px;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
    }

    .forecast-horizon {
        font-size: 0.8rem;
        font-weight: 700;
        color: #667085;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .forecast-value {
        font-size: 2.7rem;
        font-weight: 750;
        color: #111827;
        margin-top: 0.8rem;
        margin-bottom: 0.6rem;
    }

    .health-box {
        display: inline-block;
        padding: 0.4rem 0.65rem;
        border-radius: 7px;
        font-size: 0.75rem;
        font-weight: 700;
        line-height: 1.2;
    }

    .health-message {
        color: #667085;
        font-size: 0.82rem;
        line-height: 1.45;
        margin-top: 0.85rem;
    }

    .forecast-time {
        color: #98a2b3;
        font-size: 0.72rem;
        margin-top: 0.9rem;
    }

    .info-card {
        background: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 14px;
        padding: 1.2rem;
    }

    .info-title {
        font-size: 1rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.8rem;
    }

    .info-text {
        color: #667085;
        font-size: 0.83rem;
        line-height: 1.5;
    }

    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 0.72rem;
        margin-top: 3rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# AQI classification

def get_aqi_category(aqi):

    if aqi <= 50:
        return (
            "Good",
            "#027A48",
            "#ECFDF3",
            "Air quality is considered satisfactory.",
        )

    if aqi <= 100:
        return (
            "Moderate",
            "#B54708",
            "#FFFAEB",
            "Air quality is acceptable; unusually sensitive people may experience minor effects.",
        )

    if aqi <= 150:
        return (
            "Unhealthy for Sensitive Groups",
            "#C4320A",
            "#FFF4ED",
            "Sensitive groups may experience health effects.",
        )

    if aqi <= 200:
        return (
            "Unhealthy",
            "#B42318",
            "#FEF3F2",
            "Everyone may begin to experience health effects.",
        )

    if aqi <= 300:
        return (
            "Very Unhealthy",
            "#912018",
            "#FEE4E2",
            "Health alert: the risk of health effects is increased.",
        )

    return (
        "Hazardous",
        "#7A271A",
        "#FDECEC",
        "Health warning of emergency conditions.",
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
    "aqi_change": "AQI Change",
    "aqi_change_rate": "AQI Change Rate",
    "aqi_lag_1": "AQI - 1 Hour Lag",
    "aqi_lag_12": "AQI - 12 Hour Lag",
    "aqi_lag_24": "AQI - 24 Hour Lag",
    "aqi_rolling_6": "AQI - 6 Hour Average",
    "aqi_rolling_12": "AQI - 12 Hour Average",
    "aqi_rolling_24": "AQI - 24 Hour Average",
    "year": "Year",
    "month": "Month",
    "day": "Day",
    "day_of_week": "Day of Week",
    "hour": "Hour",
}


def readable_feature(name):

    return FEATURE_LABELS.get(
        name,
        str(name).replace("_", " ").title(),
    )

# Header

st.markdown(
    '<div class="page-title">Air Quality Forecast</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="page-subtitle">
        Gujranwala, Punjab, Pakistan · Machine-learning AQI forecast
        for the next 72 hours
    </div>
    """,
    unsafe_allow_html=True,
)

# Generate forecast

generate = st.button(
    "Generate Forecast",
    type="primary",
    use_container_width=True,
)


if generate:

    with st.spinner(
        "Generating AQI forecast..."
    ):

        try:

            result = predict()

            st.session_state["forecast_result"] = result

            st.success(
                "Forecast generated successfully."
            )

        except Exception as exc:

            st.error(
                f"Unable to generate forecast: {exc}"
            )

# Forecast results

if "forecast_result" in st.session_state:

    result = st.session_state["forecast_result"]


    # Support both possible return formats

    if isinstance(result, dict):

        predictions = result.get(
            "predictions"
        )

        explanations = result.get(
            "shap_explanations",
            result.get(
                "explanations",
                {}
            ),
        )

    elif isinstance(result, (tuple, list)):

        predictions = result[0]

        explanations = (
            result[1]
            if len(result) > 1
            else {}
        )

    else:

        st.error(
            "Unexpected prediction result format."
        )

        st.stop()


    if predictions is None:

        st.error(
            "No predictions were returned."
        )

        st.stop()


    # Make sure predictions are a DataFrame

    if not isinstance(
        predictions,
        pd.DataFrame
    ):

        predictions = pd.DataFrame(
            predictions
        )

    # Forecast cards

    st.markdown(
    "## AQI Forecast"
)

cards = st.columns(3)

for index, horizon in enumerate([24, 48, 72]):

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

    with cards[index]:

        st.markdown(
            f"### {horizon}-Hour Forecast"
        )

        st.metric(
            label="Predicted AQI",
            value=f"{value:.1f}"
        )

        if category == "Good":

            st.success(
                f"{category}\n\n{message}"
            )

        elif category == "Moderate":

            st.warning(
                f"{category}\n\n{message}"
            )

        elif category == "Unhealthy for Sensitive Groups":

            st.warning(
                f"{category}\n\n{message}"
            )

        elif category == "Unhealthy":

            st.error(
                f"{category}\n\n{message}"
            )

        else:

            st.error(
                f"{category}\n\n{message}"
            )

        st.caption(
            "Forecast: "
            + forecast_time.strftime(
                "%d %b %Y, %H:%M UTC"
            )
        )


    # Forecast chart

    st.markdown(
        '<div class="section-title">Forecast Trend</div>',
        unsafe_allow_html=True,
    )


    chart_data = predictions.copy()

    chart_data["forecast_time"] = pd.to_datetime(
        chart_data["forecast_time"]
    )

    chart_data = chart_data.sort_values(
        "horizon_hours"
    )


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=chart_data["forecast_time"],
            y=chart_data["predicted_aqi"],
            mode="lines+markers",
            line=dict(
                width=3,
                color="#1c1d1f",
    ),
            marker=dict(
                size=9,
                color="#1c1d1f",
    ),
            hovertemplate=(
                "<b>%{y:.1f} AQI</b>"
                "<br>%{x}"
                "<extra></extra>"
            ),
        )
    )


    fig.update_layout(
        height=350,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis=dict(
            title="Forecast Time",
            showgrid=False,
        ),
        yaxis=dict(
            title="AQI",
            gridcolor="#eaecf0",
        ),
        showlegend=False,
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # Health outlook
    st.markdown(
    "## Health Outlook"
)

    health_columns = st.columns(3)

    for index, horizon in enumerate([24, 48, 72]):

        row = predictions[
        predictions["horizon_hours"] == horizon
    ]

    if row.empty:
        continue

    value = float(
        row["predicted_aqi"].iloc[0]
    )

    category, text_color, background, message = (
        get_aqi_category(value)
    )

    with health_columns[index]:

        st.markdown(
            f"### {horizon}-Hour Outlook"
        )

        st.metric(
            label="Predicted AQI",
            value=f"{value:.1f}"
        )

        if category == "Good":

            st.success(
                category
            )

        elif category == "Moderate":

            st.warning(
                category
            )

        elif category == "Unhealthy for Sensitive Groups":

            st.warning(
                category
            )

        else:

            st.error(
                category
            )

        st.caption(
            message
        )


    # SHAP explanation

    st.markdown(
        '<div class="section-title">Model Explanation</div>',
        unsafe_allow_html=True,
    )


    st.markdown(
        """
        <div class="info-text">
            SHAP explains which features contributed most to
            each model's AQI prediction. Positive values push
            the prediction higher, while negative values push
            it lower.
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.write("")


    selected_horizon = st.selectbox(
        "Forecast horizon",
        [24, 48, 72],
        format_func=lambda x: f"{x}-hour forecast",
    )


    explanation = explanations.get(
        selected_horizon
    )


    if explanation is not None:

        if not isinstance(
            explanation,
            pd.DataFrame
        ):

            explanation = pd.DataFrame(
                explanation
            )


        explanation = explanation.copy()


        if "feature" in explanation.columns:

            explanation["feature"] = (
                explanation["feature"]
                .apply(readable_feature)
            )


        if "shap_value" not in explanation.columns:

            st.info(
                "SHAP values are not available."
            )

        else:

            explanation = (
                explanation
                .sort_values(
                    "shap_value"
                )
            )


            fig_shap = go.Figure()


            fig_shap.add_trace(
                go.Bar(
                    x=explanation["shap_value"],
                    y=explanation["feature"],
                    orientation="h",
                    hovertemplate=(
                        "<b>%{y}</b>"
                        "<br>SHAP contribution: %{x:.3f}"
                        "<extra></extra>"
                    ),
                )
            )


            fig_shap.update_layout(
                height=400,
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20,
                ),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                xaxis=dict(
                    title="SHAP Contribution",
                    gridcolor="#eaecf0",
                    zeroline=True,
                ),
                yaxis=dict(
                    title="",
                ),
                showlegend=False,
            )


            st.plotly_chart(
                fig_shap,
                use_container_width=True,
            )


    else:

        st.info(
            "SHAP explanation is not available for this forecast."
        )

# Initial state

else:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-title">
                Gujranwala Air Quality
            </div>

            <div class="info-text">
                Generate a forecast to view the predicted AQI
                for 24, 48, and 72 hours, together with the
                model explanation and forecast trend.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

# Footer

st.markdown(
    """
    <div class="footer">
        Gujranwala AQI Forecast · CatBoost · Hopsworks · SHAP
    </div>
    """,
    unsafe_allow_html=True,
)
