import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.prediction_pipeline import predict


st.set_page_config(
    page_title="Gujranwala Air Quality",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f7f8fa;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2.2rem;
        padding-bottom: 2rem;
    }

    header {
        visibility: hidden;
    }

    /* Header */

    .main-title {
        font-size: 2.25rem;
        font-weight: 750;
        color: #101828;
        letter-spacing: -0.035em;
        margin-bottom: 0.15rem;
    }

    .location {
        font-size: 0.95rem;
        color: #667085;
        margin-bottom: 0.25rem;
    }

    .subtitle {
        font-size: 0.88rem;
        color: #98a2b3;
    }

    .live-container {
        text-align: right;
        padding-top: 0.5rem;
    }

    .live-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #ecfdf3;
        border: 1px solid #abefc6;
        color: #027a48;
        border-radius: 999px;
        padding: 0.38rem 0.72rem;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    .live-dot {
        width: 7px;
        height: 7px;
        background: #12b76a;
        border-radius: 50%;
    }


    /* Section headings */

    .section-title {
        margin-top: 2rem;
        margin-bottom: 0.9rem;
        font-size: 1.05rem;
        font-weight: 700;
        color: #101828;
    }


    /* Forecast cards */

    .forecast-card {
        background: #ffffff;
        border: 1px solid #eaecf0;
        border-radius: 16px;
        padding: 1.35rem;
        min-height: 218px;
        box-shadow: 0 3px 12px rgba(
            16,
            24,
            40,
            0.04
        );
    }

    .forecast-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #667085;
        text-transform: uppercase;
        letter-spacing: 0.055em;
    }

    .aqi-number {
        margin-top: 0.75rem;
        font-size: 2.9rem;
        line-height: 1;
        font-weight: 800;
        color: #101828;
        letter-spacing: -0.04em;
    }

    .category-pill {
        display: inline-block;
        margin-top: 0.75rem;
        padding: 0.38rem 0.62rem;
        border-radius: 7px;
        font-size: 0.69rem;
        font-weight: 750;
    }

    .health-text {
        margin-top: 0.7rem;
        color: #667085;
        font-size: 0.76rem;
        line-height: 1.45;
    }

    .forecast-time {
        margin-top: 0.7rem;
        color: #98a2b3;
        font-size: 0.68rem;
    }


    /* Information cards */

    .info-card {
        background: #ffffff;
        border: 1px solid #eaecf0;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 3px 12px rgba(
            16,
            24,
            40,
            0.035
        );
    }


    /* Explanation */

    .explanation-text {
        color: #667085;
        font-size: 0.82rem;
        line-height: 1.55;
        margin-bottom: 1rem;
    }


    /* Footer */

    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 0.7rem;
        padding-top: 2.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

def get_aqi_category(aqi):

    if aqi <= 50:

        return (
            "Good",
            "#027A48",
            "#ECFDF3",
            "Air quality is considered satisfactory."
        )

    if aqi <= 100:

        return (
            "Moderate",
            "#B54708",
            "#FFFAEB",
            "Air quality is acceptable. "
            "Unusually sensitive people may experience minor effects."
        )

    if aqi <= 150:

        return (
            "Unhealthy for Sensitive Groups",
            "#C4320A",
            "#FFF6ED",
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
        "#FEE4E2",
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
        name.replace(
            "_",
            " "
        ).title()
    )

header_left, header_right = st.columns(
    [5, 1]
)

with header_left:

    st.markdown(
        """
        <div class="main-title">
            Air Quality Forecast
        </div>

        <div class="location">
            Gujranwala, Punjab, Pakistan
        </div>

        <div class="subtitle">
            Machine-learning forecast for the next 72 hours
        </div>
        """,
        unsafe_allow_html=True
    )


with header_right:

    st.markdown(
        """
        <div class="live-container">
            <span class="live-pill">
                <span class="live-dot"></span>
                LIVE
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    '<div class="section-title">Forecast</div>',
    unsafe_allow_html=True
)

generate = st.button(
    "Generate AQI Forecast",
    type="primary"
)


if generate:

    with st.spinner(
        "Fetching current data and generating forecast..."
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


if (
    "forecast_result"
    in st.session_state
):

    result = st.session_state[
        "forecast_result"
    ]

    predictions = result[
        "predictions"
    ]

    explanations = result[
        "shap_explanations"
    ]

    cards = st.columns(3)

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

        forecast_time = pd.to_datetime(
            row[
                "forecast_time"
            ].iloc[0]
        )

        category, text_color, background, message = (
            get_aqi_category(value)
        )

        with cards[index]:

            st.markdown(
                f"""
                <div class="forecast-card">

                    <div class="forecast-label">
                        {horizon}-Hour Forecast
                    </div>

                    <div class="aqi-number">
                        {value:.1f}
                    </div>

                    <div class="category-pill"
                         style="
                            color:{text_color};
                            background:{background};
                         ">
                        {category}
                    </div>

                    <div class="health-text">
                        {message}
                    </div>

                    <div class="forecast-time">
                        Forecast:
                        {forecast_time.strftime(
                            "%d %b %Y, %H:%M UTC"
                        )}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    st.markdown(
        '<div class="section-title">Forecast Outlook</div>',
        unsafe_allow_html=True
    )

    chart_col, health_col = st.columns(
        [2, 1]
    )


    with chart_col:

        fig = go.Figure()

        fig.add_trace(
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

        fig.update_layout(
            height=350,

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            ),

            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",

            xaxis=dict(
                title="Forecast Time",
                showgrid=False
            ),

            yaxis=dict(
                title="AQI",
                gridcolor="#eef0f3",
                zeroline=False
            ),

            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with health_col:

        st.markdown(
            '<div class="info-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "**Health Outlook**"
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

            category, text_color, background, _ = (
                get_aqi_category(value)
            )

            st.markdown(
                f"""
                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                    gap:10px;
                    padding:0.8rem 0;
                    border-bottom:1px solid #eef0f3;
                ">

                    <span style="
                        color:#344054;
                        font-size:0.82rem;
                        font-weight:600;
                    ">
                        {horizon} hours
                    </span>

                    <span style="
                        color:{text_color};
                        background:{background};
                        padding:0.32rem 0.5rem;
                        border-radius:6px;
                        font-size:0.67rem;
                        font-weight:700;
                        text-align:right;
                    ">
                        {category}
                    </span>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    st.markdown(
        '<div class="section-title">Model Explanation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="explanation-text">

            SHAP explains how the input features influenced
            each forecast. Positive contributions push the
            predicted AQI higher, while negative contributions
            push it lower.

        </div>
        """,
        unsafe_allow_html=True
    )


    selected_horizon = st.radio(
        "Forecast horizon",
        [24, 48, 72],
        horizontal=True,
        format_func=lambda value:
            f"{value}-hour forecast",
        label_visibility="collapsed"
    )


    explanation = explanations.get(
        selected_horizon
    )


    if explanation is not None:

        explanation = explanation.copy()

        explanation[
            "feature"
        ] = (
            explanation[
                "feature"
            ]
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


        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=explanation[
                    "shap_value"
                ],

                y=explanation[
                    "feature"
                ],

                orientation="h",

                hovertemplate=(
                    "<b>%{y}</b>"
                    "<br>SHAP contribution: %{x:.3f}"
                    "<extra></extra>"
                )
            )
        )


        fig.update_layout(
            height=350,

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            ),

            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",

            xaxis=dict(
                title="SHAP Contribution",
                gridcolor="#eef0f3",
                zeroline=True
            ),

            yaxis=dict(
                title=""
            ),

            showlegend=False
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        display_explanation = (
            explanation.copy()
        )

        display_explanation[
            "feature"
        ] = display_explanation[
            "feature"
        ]

        display_explanation[
            "feature_value"
        ] = display_explanation[
            "feature_value"
        ].round(3)

        display_explanation[
            "shap_value"
        ] = display_explanation[
            "shap_value"
        ].round(3)

        display_explanation = (
            display_explanation.rename(
                columns={
                    "feature":
                        "Feature",

                    "feature_value":
                        "Current Value",

                    "shap_value":
                        "SHAP Contribution",

                    "impact":
                        "Impact"
                }
            )
        )

        st.dataframe(
            display_explanation[
                [
                    "Feature",
                    "Current Value",
                    "SHAP Contribution",
                    "Impact"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


st.markdown(
    """
    <div class="footer">
        Gujranwala Air Quality Forecast
        · CatBoost
        · Hopsworks
        · SHAP
    </div>
    """,
    unsafe_allow_html=True
)
