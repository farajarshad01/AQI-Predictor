import os

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="AQI Forecast",
    layout="wide"
)


st.title(
    "Air Quality Forecast"
)


prediction_file = "predictions.csv"


if not os.path.exists(
    prediction_file
):

    st.warning(
        "No predictions available yet."
    )

else:

    predictions = pd.read_csv(
        prediction_file
    )

    if "prediction_created_at" in predictions.columns:
        predictions["prediction_created_at"] = pd.to_datetime(
            predictions["prediction_created_at"]
        )

    if "forecast_time" in predictions.columns:
        predictions["forecast_time"] = pd.to_datetime(
            predictions["forecast_time"]
        )

    st.subheader(
        "AQI Predictions"
    )

    st.dataframe(
        predictions,
        use_container_width=True
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

        if not row.empty:

            value = row[
                "predicted_aqi"
            ].iloc[0]

            st.metric(
                f"{horizon}-Hour AQI",
                round(
                    value,
                    1
                )
            )