import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

APP_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

for path in (PROJECT_ROOT, APP_DIR):

    if path not in sys.path:
        sys.path.insert(0, path)

import pandas as pd
import streamlit as st

import dashboard_ui as ui

from src.data_fetch import fetch_air_quality_history
from src.prediction_pipeline import predict


st.set_page_config(
    page_title="Gujranwala Air Quality",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    ui.css(),
    unsafe_allow_html=True
)


# Data loading
#
# predict() downloads the models and fetches live Open-Meteo data, so it is
# cached. The forecast loads automatically; "Refresh" clears the caches.

@st.cache_data(ttl=1800, show_spinner=False)
def load_forecast():

    result = predict()

    result["fetched_at"] = pd.Timestamp.now(
        tz="UTC"
    )

    return result


@st.cache_data(ttl=900, show_spinner=False)
def load_history():

    return fetch_air_quality_history(
        past_days=90
    )


def refresh():

    load_forecast.clear()
    load_history.clear()


def sticky(selection, key, default):

    # Segmented controls can be un-selected by clicking the active option.
    # Keep showing the last valid choice instead of falling back.
    if selection is None:
        return st.session_state.get(key, default)

    st.session_state[key] = selection

    return selection


try:

    with st.spinner(
        "Loading the latest air-quality data and forecast models..."
    ):

        result = load_forecast()

except Exception as exc:

    st.error(
        f"Unable to generate the forecast: {exc}"
    )

    st.button(
        "Try again",
        on_click=refresh
    )

    st.stop()


predictions = result["predictions"].copy()

explanations = result.get(
    "shap_explanations",
    {}
)

current = result["current"]

predictions["forecast_time"] = pd.to_datetime(
    predictions["forecast_time"],
    utc=True
)

now = pd.Timestamp.now(
    tz="UTC"
)

current_time = pd.Timestamp(
    current["datetime"]
)

if current_time.tzinfo is None:
    current_time = current_time.tz_localize("UTC")

current_aqi = float(
    current["us_aqi"]
)

refreshed_minutes = int(
    (now - result["fetched_at"]).total_seconds() // 60
)

stale_hours = (
    (now - current_time).total_seconds() / 3600
)


# Header

title_col, status_col, button_col = st.columns(
    [2.4, 2.2, 1.2],
    vertical_alignment="center"
)

with title_col:

    st.markdown(
        """
        <div class="title">Gujranwala</div>
        <div class="m">Punjab, Pakistan · 72-hour machine-learning forecast · times in PKT</div>
        """,
        unsafe_allow_html=True
    )

with status_col:

    st.markdown(
        ui.status_html(
            current_time,
            refreshed_minutes,
            stale_hours
        ),
        unsafe_allow_html=True
    )

with button_col:

    st.button(
        "Refresh forecast",
        type="primary",
        on_click=refresh
    )


# Current AQI and forecast

hero_col, forecast_col = st.columns(
    [1.15, 1],
    gap="medium"
)

with hero_col:

    st.markdown(
        ui.hero_html(current_aqi),
        unsafe_allow_html=True
    )

with forecast_col:

    with st.container(key="forecast_card"):

        label_col, control_col = st.columns(
            [1, 1.3],
            vertical_alignment="center"
        )

        label_col.markdown(
            '<div class="t">Forecast</div>',
            unsafe_allow_html=True
        )

        with control_col:

            picked = st.segmented_control(
                "Forecast horizon",
                [24, 48, 72],
                default=24,
                format_func=lambda hours: f"{hours} h",
                key="horizon_control",
                label_visibility="collapsed"
            )

        horizon = sticky(
            picked,
            "horizon_last",
            24
        )

        row = predictions[
            predictions["horizon_hours"] == horizon
        ].iloc[0]

        value = float(
            row["predicted_aqi"]
        )

        st.markdown(
            ui.forecast_html(
                value,
                value - current_aqi,
                row["forecast_time"]
            ),
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="m" style="margin-top:6px">'
            "Pick a horizon to update the chart and the model "
            "explanation below."
            "</div>",
            unsafe_allow_html=True
        )


# Pollutants

st.markdown(
    '<div class="m" style="margin:6px 0 -6px">'
    "Pollutants now, compared with WHO guideline levels"
    "</div>",
    unsafe_allow_html=True
)

st.markdown(
    ui.pollutants_html(current),
    unsafe_allow_html=True
)


# Timeline and model explanation

timeline_col, drivers_col = st.columns(
    [1.6, 1],
    gap="medium"
)

with timeline_col:

    with st.container(key="timeline_card"):

        title_part, range_part = st.columns(
            [1.3, 1],
            vertical_alignment="center"
        )

        title_part.markdown(
            '<div class="t">AQI timeline</div>'
            '<div class="m">History and forecast on one axis</div>',
            unsafe_allow_html=True
        )

        with range_part:

            picked_range = st.segmented_control(
                "Range",
                ["7 d", "30 d", "90 d"],
                default="7 d",
                key="range_control",
                label_visibility="collapsed"
            )

        range_label = sticky(
            picked_range,
            "range_last",
            "7 d"
        )

        days = {
            "7 d": 7,
            "30 d": 30,
            "90 d": 90
        }[range_label]

        history = None

        try:

            history = load_history()

        except Exception:

            st.caption(
                "Past readings are unavailable right now, "
                "showing the forecast only."
            )

        st.plotly_chart(
            ui.build_timeline(
                history,
                current,
                predictions,
                horizon,
                days
            ),
            theme=None,
            config={"displayModeBar": False}
        )

        st.markdown(
            '<div class="m">Solid = measured · dashed = forecast · '
            "shaded bands = AQI categories"
            + (
                " · 30 d and 90 d show daily averages"
                if days > 7 else ""
            )
            + "</div>",
            unsafe_allow_html=True
        )

with drivers_col:

    with st.container(key="drivers_card"):

        st.markdown(
            '<div class="t">Why this forecast</div>'
            f'<div class="m" style="margin:2px 0 10px">'
            f"{horizon}-hour forecast · red pushes it up, "
            "blue pulls it down</div>",
            unsafe_allow_html=True
        )

        explanation = explanations.get(
            horizon
        )

        if (
            explanation is None
            or explanation.empty
            or "feature" not in explanation.columns
            or "shap_value" not in explanation.columns
        ):

            st.info(
                "Model explanation is not available "
                "for this forecast."
            )

        else:

            st.markdown(
                ui.drivers_html(explanation),
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="m" style="margin-top:8px;line-height:1.5">'
                "SHAP contribution of each input to the selected "
                "forecast. Hover a row to see the input value."
                "</div>",
                unsafe_allow_html=True
            )


# About

st.markdown(
    ui.about_html(),
    unsafe_allow_html=True
)

with st.expander("Forecast data"):

    table = predictions.copy()

    table["forecast_time"] = table[
        "forecast_time"
    ].apply(ui.fmt_day_time)

    table["prediction_created_at"] = pd.to_datetime(
        table["prediction_created_at"],
        utc=True
    ).apply(ui.fmt_day_time)

    st.dataframe(
        table,
        hide_index=True
    )

st.markdown(
    '<div class="foot">Gujranwala Air Quality Forecast · '
    "CatBoost · Hopsworks · SHAP</div>",
    unsafe_allow_html=True
)
