import json
import os
import tempfile

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from .config import FEATURES, MODEL_CONFIGS

from .hopsworksclients import (
    get_feature_group,
    get_project
)


def load_features():

    feature_group = get_feature_group()

    df = (
        feature_group
        .select_all()
        .read()
    )

    if df.empty:
        raise ValueError(
            "Feature Store is empty."
        )

    required_columns = [
        "datetime",
        "us_aqi"
    ] + FEATURES

    missing_columns = (
        set(required_columns)
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Feature Store is missing required "
            f"columns: {missing_columns}"
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True
    )

    df = (
        df
        .sort_values("datetime")
        .drop_duplicates("datetime")
        .reset_index(drop=True)
    )

    return df


def create_targets(df):

    df = df.copy()

    df["target_aqi_24"] = (
        df["us_aqi"].shift(-24)
    )

    df["target_aqi_48"] = (
        df["us_aqi"].shift(-48)
    )

    df["target_aqi_72"] = (
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

    if split_index <= 0:
        raise ValueError(
            "Training dataset is too small."
        )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if X_test.empty:
        raise ValueError(
            "Test dataset is empty."
        )

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

    predictions = model.predict(
        X_test
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

        print(
            f"Registered model: "
            f"{config['name']}"
        )

        print(
            f"Model version: "
            f"{registered_model.version}"
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
            f"Not enough valid training rows "
            f"for {target}: {len(data)}"
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
        f"\n{target} metrics:"
    )

    print(
        f"MAE: {metrics['mae']:.4f}"
    )

    print(
        f"RMSE: {metrics['rmse']:.4f}"
    )

    print(
        f"R²: {metrics['r2']:.4f}"
    )

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

    print(
        "Starting training pipeline..."
    )

    project = get_project()

    print(
        "Loading feature data from Hopsworks..."
    )

    df = load_features()

    print(
        f"Loaded {len(df)} feature rows."
    )

    df = create_targets(
        df
    )

    results = {}

    for horizon, config in MODEL_CONFIGS.items():

        print(
            f"\nTraining {horizon} model..."
        )

        results[horizon] = train_one_model(
            project,
            df,
            config
        )

    metrics_path = os.path.join(
        os.getcwd(),
        "training_metrics.json"
    )

    with open(
        metrics_path,
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print(
        "\nTraining completed successfully."
    )


if __name__ == "__main__":
    run()
