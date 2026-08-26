import numpy as np
import pandas as pd

from config import FEATURES, RAW_FEATURES


TARGETS = [
    "target_aqi_24",
    "target_aqi_48",
    "target_aqi_72",
]


def create_features(df):

    df = df.copy()

    required_columns = ["datetime"] + RAW_FEATURES

    missing_columns = (
        set(required_columns)
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
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

    dt = df["datetime"]

    # Time features
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["day_of_week"] = dt.dt.dayofweek
    df["hour"] = dt.dt.hour

    # AQI change features
    df["aqi_change"] = (
        df["us_aqi"].diff(1)
    )

    df["aqi_change_rate"] = (
        df["us_aqi"]
        .pct_change(1)
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # Lag features
    df["aqi_lag_1"] = (
        df["us_aqi"].shift(1)
    )

    df["aqi_lag_12"] = (
        df["us_aqi"].shift(12)
    )

    df["aqi_lag_24"] = (
        df["us_aqi"].shift(24)
    )

    # Leakage-safe rolling features
    previous_aqi = (
        df["us_aqi"].shift(1)
    )

    df["aqi_rolling_6"] = (
        previous_aqi
        .rolling(
            window=6,
            min_periods=6
        )
        .mean()
    )

    df["aqi_rolling_12"] = (
        previous_aqi
        .rolling(
            window=12,
            min_periods=12
        )
        .mean()
    )

    df["aqi_rolling_24"] = (
        previous_aqi
        .rolling(
            window=24,
            min_periods=24
        )
        .mean()
    )

    # Future AQI targets
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


def validate_features(df):

    missing = (
        set(FEATURES)
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing engineered features: {missing}"
        )

    missing_targets = (
        set(TARGETS)
        - set(df.columns)
    )

    if missing_targets:
        raise ValueError(
            f"Missing target columns: {missing_targets}"
        )

    return True


def get_feature_data(df):

    validate_features(df)

    columns = (
        ["datetime"]
        + FEATURES
        + TARGETS
    )

    return df[columns].copy()