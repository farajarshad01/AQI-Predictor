"""Find (and optionally fill) hours missing from the feature group.

Dry run, changes nothing and prints what is stored and what is missing:

    python -m src.gap_fill --days 45

Insert the missing rows:

    python -m src.gap_fill --days 45 --apply

Only hours that are missing are inserted, so it is safe to re-run.
Open-Meteo allows at most 92 past days, so --days is capped at 89.
"""

import argparse

import pandas as pd

from .data_fetch import (
    fetch_past_days_weather,
    fetch_past_days_air_quality
)

from .feature_engineering import (
    create_features,
    get_feature_data,
    match_schema
)

from .hopsworksclients import (
    get_feature_group,
    get_or_create_feature_group
)


def run(days=45, apply=False):

    days = min(days, 89)

    now_hour = pd.Timestamp.now(
        tz="UTC"
    ).floor("h")

    start = now_hour - pd.Timedelta(
        days=days
    )

    stored = get_feature_group().read()

    stored["datetime"] = pd.to_datetime(
        stored["datetime"],
        utc=True
    )

    print(
        f"Stored rows : {len(stored)}"
    )

    print(
        f"Oldest      : {stored['datetime'].min()}"
    )

    print(
        f"Newest      : {stored['datetime'].max()}"
    )

    print(
        "\nRows per day (last 14 days):"
    )

    per_day = (
        stored[
            stored["datetime"]
            >= now_hour - pd.Timedelta(days=14)
        ]
        .set_index("datetime")
        .resample("D")
        .size()
    )

    print(per_day.to_string())

    expected = pd.date_range(
        start,
        now_hour,
        freq="h"
    )

    missing = expected.difference(
        pd.DatetimeIndex(
            stored["datetime"]
        )
    )

    print(
        f"\nMissing hours in the last {days} days: "
        f"{len(missing)} of {len(expected)}"
    )

    if len(missing) == 0:
        print("Nothing to fill.")
        return

    # Extra days give the lag features enough history.
    past_days = min(92, days + 3)

    weather = fetch_past_days_weather(
        past_days=past_days
    )

    air_quality = fetch_past_days_air_quality(
        past_days=past_days
    )

    df = weather.merge(
        air_quality,
        on="datetime",
        how="inner"
    )

    df = create_features(df)

    rows = get_feature_data(df).dropna()

    rows = rows[
        rows["datetime"].isin(missing)
    ].sort_values("datetime")

    print(
        f"Fillable from Open-Meteo: {len(rows)}"
    )

    print(
        f"Not available / incomplete: "
        f"{len(missing) - len(rows)}"
    )

    if rows.empty:
        return

    if not apply:

        print(
            "\nDry run only. Re-run with --apply "
            "to insert these rows."
        )

        return

    feature_group = get_or_create_feature_group()

    rows = match_schema(
        rows,
        feature_group
    )

    feature_group.insert(
        rows,
        wait=True
    )

    print(
        f"\nInserted {len(rows)} rows "
        f"({rows['datetime'].min()} to "
        f"{rows['datetime'].max()})."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--days",
        type=int,
        default=45
    )

    parser.add_argument(
        "--apply",
        action="store_true"
    )

    args = parser.parse_args()

    run(
        days=args.days,
        apply=args.apply
    )
