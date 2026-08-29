import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from src.prediction_pipeline import predict


st.set_page_config(
    page_title="Gujranwala AQI Forecast",
    layout="wide"
)

st.markdown(
    """
    <style>

    /* Overall page */

    .stApp {
        background-color: #f7f8fa;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }


    /* Header */

    .page-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #172033;
        margin-bottom: 0.15rem;
    }

    .page-location {
        font-size: 0.95rem;
        color: #667085;
        margin-bottom: 0.35rem;
    }

    .page-description {
        font-size: 0.85rem;
        color: #667085;
        margin-bottom: 1.5rem;
    }


    /* Section headings */

    .section-heading {
        font-size: 1.15rem;
        font-weight: 700;
        color: #172033;
        margin-top: 1.8rem;
        margin-bottom: 0.8rem;
    }


    /* AQI value */

    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #172033;
    }

    [data-testid="stMetricLabel"] {
        color: #667085;
        font-weight: 600;
    }


    /* Buttons */

    .stButton > button {
        border-radius: 7px;
        font-weight: 600;
        min-height: 42px;
    }


    /* Divider */

    hr {
        border-color: #e4e7ec;
    }


    /* Small explanatory text */

    .muted-text {
        color: #667085;
        font-size: 0.82rem;
        line-height: 1.5;
    }


    /* Footer */

    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 0.7rem;
        margin-top: 3rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

def get_aqi_category(aqi):

    if aqi <= 50:
        return (
            "Good",
            "green",
            "Air quality is considered satisfactory."
        )

    if aqi <= 100:
        return (
            "Moderate",
            "orange",
            "Air quality is acceptable. Sensitive people may experience minor effects."
        )

    if aqi <= 150:
        return (
            "Unhealthy for Sensitive Groups",
            "orange",
            "Sensitive groups may experience health effects."
        )

    if aqi <= 200:
        return (
            "Unhealthy",
            "red",
            "Everyone may begin to experience health effects."
        )

    if aqi <= 300:
        return (
            "Very Unhealthy",
            "red",
            "Health alert: the risk of health effects is increased."
        )

    return (
        "Hazardous",
        "red",
        "Health warning of emergency conditions."
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
        "AQI — 1 Hour Lag",

    "aqi_lag_12":
        "AQI — 12 Hour Lag",

    "aqi_lag_24":
        "AQI — 24 Hour Lag",

    "aqi_rolling_6":
        "AQI — 6 Hour Average",

    "aqi_rolling_12":
        "AQI — 12 Hour Average",

    "aqi_rolling_24":
        "AQI — 24 Hour Average",
}


def readable_feature(name):

    return FEATURE_LABELS.get(
        name,
        name.replace("_", " ").title()
    )


st.markdown(
    '<div class="page-title">Air Quality Forecast</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="page-location">Gujranwala, Punjab, Pakistan</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="page-description">'
    'Machine-learning forecast for the next 24, 48 and 72 hours.'
    '</div>',
    unsafe_allow_html=True
)

if st.button(
    "Generate AQI Forecast",
    type="primary"
):

    try:

        # This displays Streamlit's loading spinner
        # while the complete prediction + SHAP process runs.

        with st.spinner(
            "Generating forecast and model explanations..."
        ):

            result = predict()

        st.session_state["forecast_result"] = result

        st.success(
            "Forecast generated successfully."
        )

    except Exception as exc:

        st.error(
            f"Unable to generate forecast: {exc}"
        )

if "forecast_result" in st.session_state:

    result = st.session_state[
        "forecast_result"
    ]

    predictions = result[
        "predictions"
    ]

    shap_explanations = result.get(
        "shap_explanations",
        {}
    )


    st.markdown(
        '<div class="section-heading">Forecast</div>',
        unsafe_allow_html=True
    )

    columns = st.columns(
        3,
        gap="medium"
    )


    for column, horizon in zip(
        columns,
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

        forecast_time = pd.to_datetime(
            row[
                "forecast_time"
            ].iloc[0]
        )

        category, category_color, health_message = (
            get_aqi_category(value)
        )


        with column:

            # Native Streamlit bordered card
            with st.container(
                border=True
            ):

                st.caption(
                    f"{horizon}-HOUR FORECAST"
                )

                st.metric(
                    "Predicted AQI",
                    f"{value:.1f}"
                )


                # Bright warning label.
                # Native Streamlit markdown renders this
                # as actual text, not a card made from HTML.

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


                st.write(
                    health_message
                )

                st.caption(
                    "Forecast time: "
                    + forecast_time.strftime(
                        "%d %b %Y, %H:%M UTC"
                    )
                )


    st.markdown(
        '<div class="section-heading">Forecast Outlook</div>',
        unsafe_allow_html=True
    )


    chart_column, summary_column = st.columns(
        [2, 1],
        gap="medium"
    )


    with chart_column:

        with st.container(
            border=True
        ):

            st.subheader(
                "AQI Trend"
            )

            chart = go.Figure()


            chart.add_trace(
                go.Scatter(
                    x=predictions[
                        "forecast_time"
                    ],
                    y=predictions[
                        "predicted_aqi"
                    ],
                    mode="lines+markers",
                    line=dict(
                        width=3
                    ),
                    marker=dict(
                        size=9
                    ),
                    hovertemplate=(
                        "<b>%{y:.1f} AQI</b>"
                        "<br>%{x}"
                        "<extra></extra>"
                    )
                )
            )


            chart.update_layout(

                height=330,

                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20
                ),

                paper_bgcolor="#ffffff",

                plot_bgcolor="#ffffff",

                xaxis=dict(
                    title="Forecast Time",
                    showgrid=False
                ),

                yaxis=dict(
                    title="AQI",
                    rangemode="tozero",
                    gridcolor="#eaecf0"
                ),

                showlegend=False
            )


            st.plotly_chart(
                chart,
                use_container_width=True
            )


    with summary_column:

        with st.container(
            border=True
        ):

            st.subheader(
                "Health Outlook"
            )


            for horizon in [
                24,
                48,
                72
            ]:

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

                category, _, _ = (
                    get_aqi_category(value)
                )


                st.markdown(
                    f"**{horizon}-hour forecast**"
                )

                st.write(
                    f"AQI: **{value:.1f}**"
                )

                if category == "Good":

                    st.success(
                        category
                    )

                elif category in [
                    "Moderate",
                    "Unhealthy for Sensitive Groups"
                ]:

                    st.warning(
                        category
                    )

                else:

                    st.error(
                        category
                    )


                if horizon != 72:

                    st.divider()


    st.markdown(
        '<div class="section-heading">Model Explanation</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="muted-text">
        SHAP explains which input features influenced each AQI
        forecast. Positive contributions push the prediction higher,
        while negative contributions push it lower.
        </div>
        """,
        unsafe_allow_html=True
    )


    selected_horizon = st.radio(
        "Forecast horizon",
        [24, 48, 72],
        horizontal=True,
        format_func=lambda x:
            f"{x}-hour forecast"
    )


    explanation = shap_explanations.get(
        selected_horizon
    )


    if explanation is not None:

        explanation = explanation.copy()


        explanation["Feature"] = (
            explanation[
                "feature"
            ]
            .apply(
                readable_feature
            )
        )


        explanation["Current Value"] = (
            explanation[
                "feature_value"
            ]
            .round(3)
        )


        explanation["SHAP Contribution"] = (
            explanation[
                "shap_value"
            ]
            .round(3)
        )


        explanation["Impact"] = (
            explanation[
                "shap_value"
            ]
            .apply(
                lambda value:
                "Increases AQI"
                if value > 0
                else "Decreases AQI"
            )
        )


        explanation = explanation[
            [
                "Feature",
                "Current Value",
                "SHAP Contribution",
                "Impact"
            ]
        ]

        chart_data = (
            explanation
            .sort_values(
                "SHAP Contribution"
            )
        )


        with st.container(
            border=True
        ):

            st.subheader(
                f"{selected_horizon}-Hour Forecast Drivers"
            )


            shap_chart = go.Figure()


            shap_chart.add_trace(
                go.Bar(
                    x=chart_data[
                        "SHAP Contribution"
                    ],
                    y=chart_data[
                        "Feature"
                    ],
                    orientation="h",
                    hovertemplate=(
                        "<b>%{y}</b>"
                        "<br>Contribution: %{x:.3f}"
                        "<extra></extra>"
                    )
                )
            )


            shap_chart.update_layout(

                height=360,

                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20
                ),

                paper_bgcolor="#ffffff",

                plot_bgcolor="#ffffff",

                xaxis=dict(
                    title="SHAP Contribution",
                    gridcolor="#eaecf0",
                    zeroline=True
                ),

                yaxis=dict(
                    title=""
                ),

                showlegend=False
            )


            st.plotly_chart(
                shap_chart,
                use_container_width=True
            )


            st.dataframe(
                explanation,
                use_container_width=True,
                hide_index=True
            )


    else:

        st.info(
            "SHAP explanation is not available for this forecast."
        )


st.markdown(
    """
    <div class="footer">
        Gujranwala Air Quality Forecast ·
        CatBoost · Hopsworks · SHAP
    </div>
    """,
    unsafe_allow_html=True
)
