import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.prediction_pipeline import predict


st.set_page_config(
    page_title="Gujranwala Air Quality",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(
    """
    <style>

    .stApp {
        background: #f5f6f8;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #171923;
        margin-bottom: 0.1rem;
    }

    .location {
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 0.4rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 650;
        color: #171923;
        margin-top: 1.6rem;
        margin-bottom: 0.8rem;
    }

    .forecast-card {
        background: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 14px;
        padding: 1.25rem;
        min-height: 205px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    .forecast-label {
        font-size: 0.78rem;
        font-weight: 650;
        color: #667085;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .aqi-number {
        font-size: 2.7rem;
        line-height: 1;
        font-weight: 750;
        color: #171923;
        margin-top: 0.8rem;
    }

    .aqi-category {
        display: inline-block;
        margin-top: 0.8rem;
        padding: 0.35rem 0.65rem;
        border-radius: 7px;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .health-message {
        color: #667085;
        font-size: 0.8rem;
        line-height: 1.4;
        margin-top: 0.8rem;
    }

    .forecast-time {
        color: #98a2b3;
        font-size: 0.72rem;
        margin-top: 0.7rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #ecfdf3;
        color: #027a48;
        border: 1px solid #abefc6;
        padding: 0.35rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 650;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #12b76a;
    }

    .info-card {
        background: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 14px;
        padding: 1.15rem;
    }

    .explanation-note {
        color: #667085;
        font-size: 0.82rem;
        margin-bottom: 1rem;
    }

    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 0.72rem;
        padding: 2rem 0 1rem 0;
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
            "Air quality is acceptable; unusually sensitive people may experience minor effects."
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


with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:1.1rem;
            font-weight:700;
            color:#171923;
            margin-bottom:1.8rem;
        ">
            AQI Forecast
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            font-size:0.72rem;
            color:#98a2b3;
            text-transform:uppercase;
            letter-spacing:.06em;
            margin-bottom:.5rem;
        ">
            Location
        </div>

        <div style="
            font-weight:600;
            color:#344054;
            margin-bottom:1.5rem;
        ">
            Gujranwala
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            font-size:.78rem;
            color:#667085;
            line-height:1.5;
        ">
            Machine-learning AQI forecasting using
            weather and air-quality observations.
        </div>
        """,
        unsafe_allow_html=True
    )


header_left, header_right = st.columns(
    [5, 1],
    vertical_alignment="center"
)

with header_left:

    st.markdown(
        '<div class="main-title">Air Quality Forecast</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="location">Gujranwala, Punjab, Pakistan</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Machine-learning forecast for the next 72 hours.</div>',
        unsafe_allow_html=True
    )


with header_right:

    st.markdown(
        """
        <div style="text-align:right;">
            <span class="status-pill">
                <span class="status-dot"></span>
                LIVE
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

if st.button(
    "Generate AQI Forecast",
    type="primary",
    use_container_width=False
):

    with st.spinner("Generating forecast..."):

        try:

            result = predict()

            st.session_state["forecast_result"] = result

        except Exception as exc:

            st.error(
                f"Unable to generate forecast: {exc}"
            )


if "forecast_result" in st.session_state:

    result = st.session_state[
        "forecast_result"
    ]

    predictions = result["predictions"]

    shap_explanations = result.get(
        "shap_explanations",
        {}
    )

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

        forecast_time = pd.to_datetime(
            row["forecast_time"].iloc[0]
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

                    <div class="aqi-category"
                         style="
                            color:{text_color};
                            background:{background};
                         ">
                        {category}
                    </div>

                    <div class="health-message">
                        {message}
                    </div>

                    <div class="forecast-time">
                        Forecast time:
                        {forecast_time.strftime("%d %b %Y, %H:%M UTC")}
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
                    "%{y:.1f} AQI"
                    "<br>%{x}<extra></extra>"
                )
            )
        )

        fig.add_hrect(
            y0=0,
            y1=50,
            fillcolor="rgba(18,183,106,0.08)",
            line_width=0
        )

        fig.add_hrect(
            y0=51,
            y1=100,
            fillcolor="rgba(234,179,8,0.08)",
            line_width=0
        )

        fig.add_hrect(
            y0=101,
            y1=150,
            fillcolor="rgba(249,115,22,0.08)",
            line_width=0
        )

        fig.update_layout(
            height=340,
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            ),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            xaxis=dict(
                showgrid=False
            ),
            yaxis=dict(
                title="AQI",
                gridcolor="#eef0f3"
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

        for horizon in [24, 48, 72]:

            row = predictions[
                predictions["horizon_hours"] == horizon
            ]

            if row.empty:
                continue

            value = float(
                row["predicted_aqi"].iloc[0]
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
                    padding:.75rem 0;
                    border-bottom:1px solid #eef0f3;
                ">
                    <span style="
                        font-size:.82rem;
                        font-weight:600;
                        color:#344054;
                    ">
                        {horizon} hours
                    </span>

                    <span style="
                        color:{text_color};
                        background:{background};
                        padding:.3rem .55rem;
                        border-radius:6px;
                        font-size:.7rem;
                        font-weight:700;
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
        <div class="explanation-note">
            SHAP shows which input features contributed most to each
            forecast. Positive values push the prediction higher;
            negative values push it lower.
        </div>
        """,
        unsafe_allow_html=True
    )

    selected_horizon = st.radio(
        "Forecast horizon",
        [24, 48, 72],
        horizontal=True,
        format_func=lambda x: f"{x}-hour forecast",
        label_visibility="collapsed"
    )

    explanation = shap_explanations.get(
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

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=explanation["shap_value"],
                y=explanation["feature"],
                orientation="h",
                hovertemplate=(
                    "%{y}"
                    "<br>Contribution: %{x:.3f}"
                    "<extra></extra>"
                )
            )
        )

        fig.update_layout(
            height=340,
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            ),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            xaxis=dict(
                title="SHAP contribution",
                zeroline=True,
                zerolinecolor="#98a2b3",
                gridcolor="#eef0f3"
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
