import os

import pandas as pd
import shap
from catboost import CatBoostRegressor

from .config import (
    FEATURES,
    MODEL_24_NAME,
    MODEL_48_NAME,
    MODEL_72_NAME,
)

from .data_fetch import (
    fetch_recent_weather,
    fetch_recent_air_quality,
)

from .feature_engineering import (
    create_features,
)

from .hopsworksclients import (
    download_latest_model,
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
        72: MODEL_72_NAME,
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

        if not os.path.exists(model_path):

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
                "predicted_aqi": prediction,
            }
        )

    return pd.DataFrame(
        predictions
    )


def create_shap_explanations(
    latest_features,
    models
):

    X = latest_features[
        FEATURES
    ]

    explanations = {}

    for horizon in [
        24,
        48,
        72
    ]:

        model = models[
            horizon
        ]

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = (
            explainer.shap_values(X)
        )

        if isinstance(
            shap_values,
            list
        ):

            shap_values = shap_values[0]

        values = shap_values[0]

        explanation = pd.DataFrame(
            {
                "feature": FEATURES,
                "feature_value": X.iloc[0].values,
                "shap_value": values,
            }
        )

        explanation["impact"] = (
            explanation["shap_value"]
            .apply(
                lambda value:
                "Increases AQI"
                if value > 0
                else "Decreases AQI"
            )
        )

        explanation["absolute_impact"] = (
            explanation["shap_value"]
            .abs()
        )

        explanation = (
            explanation
            .sort_values(
                "absolute_impact",
                ascending=False
            )
            .drop(
                columns=["absolute_impact"]
            )
            .reset_index(
                drop=True
            )
        )

        explanations[horizon] = explanation

    return explanations


def predict():

    model_paths = download_models()

    models = load_models(
        model_paths
    )

    latest_features = (
        get_latest_features()
    )

    predictions = make_predictions(
        latest_features,
        models
    )

    explanations = create_shap_explanations(
        latest_features,
        models
    )

    return (
        predictions,
        explanations
    )


if __name__ == "__main__":

    predictions, explanations = predict()

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

    for horizon, explanation in explanations.items():

        print(
            f"\nTop SHAP features for {horizon}-hour forecast:"
        )

        print(
            explanation
            .head(10)
            .to_string(
                index=False
            )
        )
