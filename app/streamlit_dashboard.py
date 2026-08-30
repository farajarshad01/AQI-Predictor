import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.prediction_pipeline import predict


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

GOOD_COLOR = "#00E400"
MODERATE_COLOR = "#FFFF00"
SENSITIVE_COLOR = "#FF7E00"
UNHEALTHY_COLOR = "#FF0000"
VERY_UNHEALTHY_COLOR = "#8F3F97"
HAZARDOUS_COLOR = "#7E0023"


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

    .footer {{
        text-align: center;
        color: #777777;
        font-size: 0.72rem;
        padding: 2rem 0 1rem 0;
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

    st.markdown(
        f"""
        <div style="
            font-size:0.8rem;
            color:#555555;
            line-height:1.5;
        ">
            Air quality forecasts generated using
            weather observations, air-quality data,
            CatBoost machine-learning models and SHAP.
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
                color=TEXT_COLOR,
                width=3
            ),
            marker=dict(
                color=TEXT_COLOR,
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

    # SHAP explanation

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
                    color=TEXT_COLOR
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


    # Forecast data

    with st.expander(
        "Forecast data"
    ):

        display_predictions = predictions.copy()

        display_predictions[
            "forecast_time"
        ] = display_predictions[
            "forecast_time"
        ].dt.strftime(
            "%d %b %Y, %H:%M UTC"
        )

        display_predictions[
            "prediction_created_at"
        ] = pd.to_datetime(
            display_predictions[
                "prediction_created_at"
            ],
            utc=True
        ).dt.strftime(
            "%d %b %Y, %H:%M UTC"
        )

        st.dataframe(
            display_predictions,
            use_container_width=True,
            hide_index=True
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
