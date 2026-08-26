import os

import pandas as pd

from data_fetch import (
    fetch_historical_weather,
    fetch_historical_air_quality
)

from feature_engineering import (
    create_features,
    get_feature_data
)

from hopsworksclients import (
    get_or_create_feature_group
)


START_DATE = os.environ[
    "BACKFILL_START_DATE"
]

END_DATE = os.environ[
    "BACKFILL_END_DATE"
]


def run():

    start = pd.Timestamp(
        START_DATE,
        tz="UTC"
    )

    fetch_start = (
        start
        - pd.Timedelta(hours=48)
    ).strftime(
        "%Y-%m-%d"
    )

    print(
        f"Fetching historical data "
        f"from {fetch_start} to {END_DATE}"
    )

    weather = fetch_historical_weather(
        fetch_start,
        END_DATE
    )

    air_quality = (
        fetch_historical_air_quality(
            fetch_start,
            END_DATE
        )
    )

    df = weather.merge(
        air_quality,
        on="datetime",
        how="inner"
    )

    if df.empty:
        raise ValueError(
            "No historical data returned."
        )

    df = create_features(
        df
    )

    df = df[
        df["datetime"] >= start
    ].copy()

    df = df.dropna(
        subset=[
            "datetime"
        ]
    )

    df = get_feature_data(
        df
    )

    df = df.dropna()

    if df.empty:
        raise ValueError(
            "No valid feature rows remain."
        )

    feature_group = (
        get_or_create_feature_group()
    )

    feature_group.insert(
        df,
        wait=True
    )

    print(
        f"Inserted {len(df)} rows."
    )


if __name__ == "__main__":
    run()