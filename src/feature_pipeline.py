from data_fetch import (
    fetch_recent_weather,
    fetch_recent_air_quality
)

from feature_engineering import (
    create_features,
    get_feature_data
)

from hopsworksclients import (
    get_or_create_feature_group
)


def run():

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
            "No recent weather/AQI data returned "
            "after merging API responses."
        )

    df = create_features(df)

    df = (
        df
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    latest = df.tail(1).copy()

    if latest.empty:
        raise ValueError(
            "No latest row available."
        )

    latest = get_feature_data(
        latest
    )

    missing_features = [
        feature
        for feature in latest.columns
        if feature != "datetime"
        and latest[feature].isna().any()
    ]

    if missing_features:
        raise ValueError(
            "Latest feature row contains missing "
            "values: "
            + ", ".join(missing_features)
        )

    feature_group = (
        get_or_create_feature_group()
    )

    feature_group.insert(
        latest,
        wait=True
    )

    print(
        "Inserted feature row:"
    )

    print(
        latest[
            ["datetime", "us_aqi"]
        ]
    )


if __name__ == "__main__":
    run()
