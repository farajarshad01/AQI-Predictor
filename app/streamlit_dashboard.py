import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.prediction_pipeline import predict


st.set_page_config(
    page_title="AQI Forecast",
    page_icon="🌍",
    layout="wide",
)


st.title("Air Quality Forecast")

st.write(
    "Generate an AQI forecast using the latest "
    "features and registered CatBoost models."
)


if st.button(
    "Generate AQI Forecast",
    type="primary",
):

    with st.spinner(
        "Generating forecast..."
    ):

        try:

            predictions = predict()

            st.success(
                "Forecast generated successfully."
            )

            st.subheader(
                "AQI Forecast"
            )

            cols = st.columns(3)

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

                value = row[
                    "predicted_aqi"
                ].iloc[0]

                forecast_time = row[
                    "forecast_time"
                ].iloc[0]

                with cols[index]:

                    st.metric(
                        f"{horizon}-Hour AQI",
                        f"{value:.1f}",
                    )

                    st.caption(
                        f"Forecast: {forecast_time}"
                    )

            st.subheader(
                "Forecast Details"
            )

            st.dataframe(
                predictions,
                use_container_width=True,
                hide_index=True,
            )

        except Exception as error:

            st.error(
                "Unable to generate the forecast."
            )

            st.exception(error)
