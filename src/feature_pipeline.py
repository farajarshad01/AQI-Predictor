import pandas as pd

from .data_fetch import (
    fetch_recent_weather,
    fetch_recent_air_quality
)

from .feature_engineering import (
    create_features,
    get_feature_data,
    match_schema
)

from .hopsworksclients import (
    get_or_create_feature_group
)


# How far back every run looks. Rows that already exist are overwritten
# with identical values (the primary key is `datetime`), so a wide window
# means a skipped or failed scheduled run no longer leaves a permanent
# gap. 96 hours of data yields about 72 complete rows, which covers up
# to three days of missed runs.
WINDOW_HOURS = 96


def build_rows():

    weather = fetch_recent_weather(
        past_hours=WINDOW_HOURS
    )

    air_quality = fetch_recent_air_quality(
        past_hours=WINDOW_HOURS
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

    # Never store hours that have not happened yet.
    now_hour = pd.Timestamp.now(
        tz="UTC"
    ).floor("h")

    df = df[
        df["datetime"] <= now_hour
    ]

    rows = (
        get_feature_data(df)
        .dropna()
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    if rows.empty:
        raise ValueError(
            "No complete feature rows available."
        )

    newest_available = df["datetime"].max()

    if rows["datetime"].max() < newest_available:

        # The newest hour can still be missing values on the API side.
        # It is not an error: the next run fills it in.
        print(
            "Warning: newest hour "
            f"({newest_available}) is incomplete; "
            f"inserting up to {rows['datetime'].max()}."
        )

    return rows


def run():

    rows = build_rows()

    feature_group = (
        get_or_create_feature_group()
    )

    # Hopsworks' offline writer (Hudi) fails with a ClassCastException
    # (Instant cannot be cast to Long) whenever an insert updates a row
    # that already exists. Only rows that are not stored yet are inserted,
    # so every insert is a pure append. The wide window still heals gaps.
    if getattr(feature_group, "features", None):

        stored = feature_group.read()

        known = pd.to_datetime(
            stored["datetime"],
            utc=True
        )

        rows = rows[
            ~rows["datetime"].isin(known)
        ]

    if rows.empty:

        print(
            "No new rows to insert: "
            "the feature group is up to date."
        )

        return

    rows = match_schema(
        rows,
        feature_group
    )

    feature_group.insert(
        rows,
        wait=True
    )

    print(
        f"Inserted {len(rows)} feature rows "
        f"({rows['datetime'].min()} to "
        f"{rows['datetime'].max()})."
    )

    print(
        rows[
            ["datetime", "us_aqi"]
        ].tail(3)
    )


if __name__ == "__main__":
    run()
