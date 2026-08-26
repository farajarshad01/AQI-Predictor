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
            "No recent data returned."
        )

    df = create_features(
        df
    )

    df = df.dropna()

    df = get_feature_data(
        df
    )

    latest = (
        df
        .sort_values("datetime")
        .tail(1)
    )

    if latest.empty:
        raise ValueError(
            "No valid latest feature row."
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