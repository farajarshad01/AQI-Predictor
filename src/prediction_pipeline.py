import os

import pandas as pd
from catboost import CatBoostRegressor

from .config import FEATURES, MODEL_CONFIGS
from .hopsworksclients import (
    get_feature_group,
    download_latest_model,
)


MODEL_DIRECTORY = "downloaded_models"


def load_latest_feature():
    """Load the most recent feature row from Hopsworks."""

    feature_group = get_feature_group()

    df = (
        feature_group
        .select_all()
        .read()
    )

    if df.empty:
        raise ValueError(
            "Feature Store contains no feature rows."
        )

    if "datetime" not in df.columns:
        raise ValueError(
            "Feature Store is missing datetime."
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
    )

    df = (
        df
        .sort_values("datetime")
        .drop_duplicates(
            "datetime",
            keep="last",
        )
        .reset_index(drop=True)
    )

    latest = (
        df
        .tail(1)
        .copy()
    )

    missing_features = (
        set(FEATURES)
        - set(latest.columns)
    )

    if missing_features:
        raise ValueError(
            "Latest feature row is missing "
            f"features: {missing_features}"
        )

    missing_values = (
        latest[FEATURES]
        .isnull()
        .any()
    )

    if missing_values.any():

        missing = (
            missing_values[
                missing_values
            ]
            .index
            .tolist()
        )

        raise ValueError(
            "Latest feature row contains "
            f"missing values: {missing}"
        )

    return latest[
        ["datetime"] + FEATURES
    ].copy()


def download_models():
    """Download the latest registered models."""

    os.makedirs(
        MODEL_DIRECTORY,
        exist_ok=True,
    )

    model_paths = {}

    for horizon, config in MODEL_CONFIGS.items():

        model_name = config["name"]

        model_directory = os.path.join(
            MODEL_DIRECTORY,
            f"model_{horizon}",
        )

        os.makedirs(
            model_directory,
            exist_ok=True,
        )

        model_path = os.path.join(
            model_directory,
            "model.cbm",
        )

        # Reuse the locally downloaded model if it already exists.
        # Streamlit can therefore avoid downloading models
        # repeatedly during the same deployment.
        if not os.path.exists(model_path):

            download_latest_model(
                model_name,
                model_directory,
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        model_paths[horizon] = model_path

    return model_paths


def load_models():
    """Load the latest registered models."""

    model_paths = download_models()

    models = {}

    for horizon, path in model_paths.items():

        model = CatBoostRegressor()

        model.load_model(
            path
        )

        models[horizon] = model

    return models


def make_predictions(
    latest_features,
    models,
):
    """Generate 24h, 48h and 72h AQI forecasts."""

    X = latest_features[
        FEATURES
    ]

    prediction_time = (
        latest_features[
            "datetime"
        ].iloc[0]
    )

    predictions = []

    for horizon in [
        "24h",
        "48h",
        "72h",
    ]:

        if horizon not in models:
            raise ValueError(
                f"Model not loaded: {horizon}"
            )

        model = models[
            horizon
        ]

        prediction = float(
            model.predict(X)[0]
        )

        # AQI cannot be negative.
        prediction = max(
            0.0,
            prediction,
        )

        horizon_hours = int(
            horizon.replace(
                "h",
                "",
            )
        )

        forecast_time = (
            prediction_time
            + pd.Timedelta(
                hours=horizon_hours,
            )
        )

        predictions.append(
            {
                "prediction_created_at": (
                    prediction_time
                ),
                "forecast_time": (
                    forecast_time
                ),
                "horizon_hours": (
                    horizon_hours
                ),
                "predicted_aqi": (
                    prediction
                ),
            }
        )

    return pd.DataFrame(
        predictions
    )


def predict():
    """
    Complete inference operation.

    Loads the latest feature row, loads the latest
    registered models, and returns forecasts.
    """

    latest_features = (
        load_latest_feature()
    )

    models = load_models()

    return make_predictions(
        latest_features,
        models,
    )


def run():
    """CLI entry point for local/manual testing."""

    print(
        "Loading latest feature row..."
    )

    latest_features = (
        load_latest_feature()
    )

    print(
        "Latest feature timestamp:",
        latest_features[
            "datetime"
        ].iloc[0],
    )

    print(
        "Loading registered models..."
    )

    models = load_models()

    print(
        "Generating predictions..."
    )

    predictions = make_predictions(
        latest_features,
        models,
    )

    print(
        "\nPredictions:"
    )

    print(
        predictions.to_string(
            index=False,
        )
    )

    return predictions


if __name__ == "__main__":
    run()
