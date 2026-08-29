import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from src.prediction_pipeline import predict


st.set_page_config(
    page_title="Gujranwala AQI Forecast",
    page_icon=None,
    layout="wide",
)


st.markdown(
    """
    <style>

    .block-container {
        max-width: 1200px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    .location {
        color: #666;
        font-size: 1rem;
        margin-top: -10px;
        margin-bottom: 2rem;
    }

    .forecast-card {
        padding: 1.4rem;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        background: #ffffff;
        min-height: 180px;
    }

    .forecast-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 0.4rem;
    }

    .aqi-number {
        font-size: 2.8rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 0.35rem;
    }

    .aqi-category {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }

    .health-message {
        font-size: 0.82rem;
        color: #666;
        line-height: 1.4;
    }

    .section-divider {
        margin-top: 2.5rem;
        margin-bottom: 2rem;
        border-top: 1px solid #e5e7eb;
    }

    .metadata {
        color: #6b7280;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


def get_aqi_category(aqi):

    if aqi <= 50:

        return (
            "Good",
            "Air quality is satisfactory."
        )

    if aqi <= 100:

        return (
            "Moderate",
            "Air quality is acceptable; unusually sensitive people may be affected."
        )

    if aqi <= 150:

        return (
            "Unhealthy for Sensitive Groups",
            "Sensitive groups may experience health effects."
        )

    if aqi <= 200:

        return (
            "Unhealthy",
            "Everyone may begin to experience health effects."
        )

    if aqi <= 300:

        return (
            "Very Unhealthy",
            "Health alert: increased risk of health effects."
        )

    return (
        "Hazardous",
        "Health warning: everyone is more likely to be affected."
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
        "AQI 6-Hour Average",

    "aqi_rolling_12":
        "AQI 12-Hour Average",

    "aqi_rolling_24":
        "AQI 24-Hour Average",
}


def format_feature_name(name):

    return FEATURE_LABELS.get(
        name,
        name.replace("_", " ").title()
    )


st.title(
    "Air Quality Forecast"
)

st.markdown(
    """
    <div class="location">
        Gujranwala, Punjab, Pakistan
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Machine-learning forecasts for the next 24, 48 and 72 hours."
)


if "predictions" not in st.session_state:

    st.session_state.predictions = None


if "explanations" not in st.session_state:

    st.session_state.explanations = None


if st.button(
    "Generate AQI Forecast",
    type="primary",
):

    with st.spinner(
        "Generating forecast and model explanations..."
    ):

        try:

            predictions, explanations = predict()

            st.session_state.predictions = (
                predictions
            )

            st.session_state.explanations = (
                explanations
            )

            st.success(
                "Forecast generated successfully."
            )

        except Exception as error:

            st.error(
                f"Unable to generate forecast: {error}"
            )


predictions = (
    st.session_state.predictions
)

explanations = (
    st.session_state.explanations
)


if predictions is not None:

    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True,
    )

    st.subheader(
        "Forecast"
    )

    columns = st.columns(3)

    for index, horizon in enumerate(
        [24, 48, 72]
    ):

        row = predictions[
            predictions[
                "horizon_hours"
            ] == horizon
        ]

        if row.empty:
            continue

        value = float(
            row[
                "predicted_aqi"
            ].iloc[0]
        )

        forecast_time = (
            row[
                "forecast_time"
            ].iloc[0]
        )

        category, message = (
            get_aqi_category(value)
        )

        with columns[index]:

            st.markdown(
                f"""
                <div class="forecast-card">

                    <div class="forecast-label">
                        {horizon}-Hour Forecast
                    </div>

                    <div class="aqi-number">
                        {value:.1f}
                    </div>

                    <div class="aqi-category">
                        {category}
                    </div>

                    <div class="health-message">
                        {message}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                f"Forecast time: {forecast_time}"
            )


    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True,
    )

    st.subheader(
        "AQI Health Categories"
    )

    st.dataframe(
        {
            "AQI": [
                "0–50",
                "51–100",
                "101–150",
                "151–200",
                "201–300",
                "301+",
            ],
            "Category": [
                "Good",
                "Moderate",
                "Unhealthy for Sensitive Groups",
                "Unhealthy",
                "Very Unhealthy",
                "Hazardous",
            ],
        },
        use_container_width=True,
        hide_index=True,
    )


    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True,
    )

    st.subheader(
        "Model Explainability"
    )

    st.write(
        "SHAP shows how the input features influenced "
        "each CatBoost forecast. Positive values push "
        "the prediction higher, while negative values "
        "push it lower."
    )


    for horizon in [
        24,
        48,
        72
    ]:

        explanation = explanations[
            horizon
        ].copy()

        explanation["feature"] = (
            explanation["feature"]
            .apply(
                format_feature_name
            )
        )

        explanation = explanation[
            [
                "feature",
                "feature_value",
                "shap_value",
                "impact",
            ]
        ]

        explanation[
            "feature_value"
        ] = explanation[
            "feature_value"
        ].round(3)

        explanation[
            "shap_value"
        ] = explanation[
            "shap_value"
        ].round(3)

        st.markdown(
            f"#### {horizon}-Hour Forecast"
        )

        chart_data = (
            explanation
            .head(8)
            .set_index(
                "feature"
            )[
                "shap_value"
            ]
        )

        st.bar_chart(
            chart_data,
            horizontal=True,
        )

        st.dataframe(
            explanation.head(8),
            use_container_width=True,
            hide_index=True,
        )


    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True,
    )

    st.subheader(
        "About this forecast"
    )

    st.markdown(
        """
        <div class="metadata">

        <strong>Location:</strong> Gujranwala, Punjab, Pakistan<br>
        <strong>Forecast horizons:</strong> 24, 48 and 72 hours<br>
        <strong>Model:</strong> CatBoost regression<br>
        <strong>Explainability:</strong> SHAP<br>
        <strong>Data source:</strong> Open-Meteo<br>
        <strong>Feature storage:</strong> Hopsworks Feature Store<br>
        <strong>Model registry:</strong> Hopsworks Model Registry

        </div>
        """,
        unsafe_allow_html=True,
    )
