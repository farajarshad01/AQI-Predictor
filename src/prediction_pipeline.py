import os

import pandas as pd
from catboost import CatBoostRegressor

from data_fetch import (
    fetch_recent_weather,
    fetch_recent_air_quality
)

from feature_engineering import (
    create_features
)

from config import (
    FEATURES,
    MODEL_24_NAME,
    MODEL_48_NAME,
    MODEL_72_NAME
)

from hopsworksclients import (
    download_latest_model
)


MODEL_DIRECTORY = "downloaded_models"


def download_models():

    os.makedirs(
        MODEL_DIRECTORY,
        exist_ok=True
    )

    model_names = {
        24: MODEL_24_NAME,
        48: MODEL_48_NAME,
        72: MODEL_72_NAME
    }

    model_paths = {}

    for horizon, model_name in model_names.items():

        model_directory = os.path.join(
            MODEL_DIRECTORY,
            f"model_{horizon}h"
        )

        os.makedirs(
            model_directory,
            exist_ok=True
        )

        download_latest_model(
            model_name,
            model_directory
        )

        model_path = os.path.join(
            model_directory,
            "model.cbm"
        )

        if not os.path.exists(
            model_path
        ):
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        model_paths[horizon] = model_path

    return model_paths


def load_models(model_paths):

    models = {}

    for horizon, path in model_paths.items():

        model = CatBoostRegressor()

        model.load_model(
            path
        )

        models[horizon] = model

    return models


def get_latest_features():

    weather = fetch_recent_weather(
        past_hours=48
    )

    air_quality = fetch_recent_air_quality(
        past_hours=48
    )

    df = weather.merge(
        air_quality,
        on="datetime",
        how="inner"
    )

    if df.empty:
        raise ValueError(
            "No recent weather and air-quality data returned."
        )

    df = (
        df
        .sort_values("datetime")
        .drop_duplicates(
            "datetime",
            keep="last"
        )
        .reset_index(
            drop=True
        )
    )

    df = create_features(
        df
    )

    latest = (
        df
        .sort_values("datetime")
        .tail(1)
        .copy()
    )

    if latest.empty:
        raise ValueError(
            "Could not create latest feature row."
        )

    missing_features = (
        set(FEATURES)
        - set(latest.columns)
    )

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    latest = latest[
        ["datetime"] + FEATURES
    ]

    if latest[FEATURES].isnull().any().any():

        missing = (
            latest[FEATURES]
            .columns[
                latest[FEATURES]
                .isnull()
                .any()
            ]
            .tolist()
        )

        raise ValueError(
            f"Latest feature row contains missing values: {missing}"
        )

    return latest


def make_predictions(
    latest_features,
    models
):

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
        24,
        48,
        72
    ]:

        model = models[
            horizon
        ]

        prediction = float(
            model.predict(X)[0]
        )

        prediction = max(
            0,
            prediction
        )

        target_time = (
            prediction_time
            + pd.Timedelta(
                hours=horizon
            )
        )

        predictions.append(
            {
                "prediction_created_at": prediction_time,
                "forecast_time": target_time,
                "horizon_hours": horizon,
                "predicted_aqi": prediction
            }
        )

    return pd.DataFrame(
        predictions
    )


def run():

    print(
        "Downloading latest models..."
    )

    model_paths = download_models()

    print(
        "Loading models..."
    )

    models = load_models(
        model_paths
    )

    print(
        "Fetching latest data..."
    )

    latest_features = (
        get_latest_features()
    )

    print(
        "Latest feature timestamp:",
        latest_features[
            "datetime"
        ].iloc[0]
    )

    print(
        "Generating predictions..."
    )

    predictions = make_predictions(
        latest_features,
        models
    )

    predictions.to_csv(
        "predictions.csv",
        index=False
    )

    print(
        "\nPredictions:"
    )

    print(
        predictions.to_string(
            index=False
        )
    )

    return predictions


if __name__ == "__main__":
    run()