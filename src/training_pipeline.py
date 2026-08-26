import json
import os
import tempfile

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from config import (
    FEATURES,
    MODEL_CONFIGS
)

from hopsworksclients import (
    get_feature_group,
    get_project
)


def load_features():

    feature_group = (
        get_feature_group()
    )

    df = (
        feature_group
        .select_all()
        .read()
    )

    if df.empty:
        raise ValueError(
            "Feature Store is empty."
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True
    )

    df = (
        df
        .sort_values("datetime")
        .drop_duplicates(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    return df


def create_targets(df):

    df = df.copy()

    df["target_24"] = (
        df["us_aqi"].shift(-24)
    )

    df["target_48"] = (
        df["us_aqi"].shift(-48)
    )

    df["target_72"] = (
        df["us_aqi"].shift(-72)
    )

    return df


def evaluate_model(
    X,
    y,
    config
):

    split_index = int(
        len(X) * 0.8
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    model = CatBoostRegressor(
        iterations=config["iterations"],
        learning_rate=config["learning_rate"],
        depth=config["depth"],
        random_seed=42,
        verbose=False
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = (
        model.predict(X_test)
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    }


def train_final_model(
    X,
    y,
    config
):

    model = CatBoostRegressor(
        iterations=config["iterations"],
        learning_rate=config["learning_rate"],
        depth=config["depth"],
        random_seed=42,
        verbose=False
    )

    model.fit(
        X,
        y
    )

    return model


def register_model(
    project,
    model,
    config,
    metrics
):

    with tempfile.TemporaryDirectory() as temp_dir:

        model_path = os.path.join(
            temp_dir,
            "model.cbm"
        )

        model.save_model(
            model_path
        )

        registry = (
            project
            .get_model_registry()
        )

        registered_model = (
            registry
            .python.create_model(
                name=config["name"],
                metrics=metrics,
                description=(
                    "CatBoost AQI forecasting model"
                )
            )
        )

        registered_model.save(
            model_path
        )


def train_one_model(
    project,
    df,
    config
):

    target = config["target"]

    data = (
        df[
            FEATURES + [target]
        ]
        .dropna()
        .copy()
    )

    if len(data) < 100:

        raise ValueError(
            f"Not enough data for {target}."
        )

    X = data[
        FEATURES
    ]

    y = data[
        target
    ]

    metrics = evaluate_model(
        X,
        y,
        config
    )

    print(
        f"{target} metrics:"
    )

    print(
        f"MAE: {metrics['mae']}"
    )

    print(
        f"RMSE: {metrics['rmse']}"
    )

    print(
        f"R²: {metrics['r2']}"
    )

    # Final production model:
    # train on 100% of available data.

    final_model = train_final_model(
        X,
        y,
        config
    )

    register_model(
        project,
        final_model,
        config,
        metrics
    )

    return metrics


def run():

    project = get_project()

    df = load_features()

    df = create_targets(
        df
    )

    results = {}

    for horizon, config in MODEL_CONFIGS.items():

        results[horizon] = train_one_model(
            project,
            df,
            config
        )

    with open(
        "training_metrics.json",
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print(
        "Training completed."
    )


if __name__ == "__main__":
    run()