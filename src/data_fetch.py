import time

import requests
import pandas as pd

from .config import LATITUDE, LONGITUDE


RETRY_STATUS = {429, 500, 502, 503, 504}

AIR_QUALITY_VARIABLES = (
    "pm10,"
    "pm2_5,"
    "carbon_monoxide,"
    "nitrogen_dioxide,"
    "sulphur_dioxide,"
    "ozone,"
    "us_aqi"
)

WEATHER_VARIABLES = (
    "temperature_2m,"
    "relative_humidity_2m,"
    "wind_speed_10m,"
    "surface_pressure,"
    "precipitation,"
    "cloud_cover"
)


def _get_json(url, params, retries=3):

    # Retries rate limits (429), server errors and network hiccups, which
    # are common when many GitHub Actions runners share one IP address.
    for attempt in range(1, retries + 1):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=120
            )

        except (
            requests.ConnectionError,
            requests.Timeout
        ):

            if attempt == retries:
                raise

        else:

            if response.status_code not in RETRY_STATUS:

                response.raise_for_status()

                return response.json()

            if attempt == retries:
                response.raise_for_status()

        time.sleep(2 * attempt)


def _request(url, params):

    data = _get_json(
        url,
        params
    )

    if "hourly" not in data:
        raise ValueError(
            "API response does not contain hourly data."
        )

    return pd.DataFrame(
        data["hourly"]
    )


def _prepare(df):

    if "time" not in df.columns:
        raise ValueError(
            "API response is missing time column."
        )

    df["datetime"] = pd.to_datetime(
        df["time"],
        utc=True
    )

    df.drop(
        columns=["time"],
        inplace=True
    )

    return df


def fetch_historical_weather(
    start_date,
    end_date
):

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m,"
            "surface_pressure,"
            "precipitation,"
            "cloud_cover"
        ),
        "timezone": "UTC"
    }

    df = _request(
        url,
        params
    )

    return _prepare(df)


def fetch_historical_air_quality(
    start_date,
    end_date
):

    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": (
            "pm10,"
            "pm2_5,"
            "carbon_monoxide,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone,"
            "us_aqi"
        ),
        "timezone": "UTC"
    }

    df = _request(
        url,
        params
    )

    return _prepare(df)


def fetch_recent_weather(
    past_hours=48
):

    url = (
        "https://api.open-meteo.com/v1/forecast"
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "past_hours": past_hours,
        "forecast_hours": 0,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m,"
            "surface_pressure,"
            "precipitation,"
            "cloud_cover"
        ),
        "timezone": "UTC"
    }

    df = _request(
        url,
        params
    )

    return _prepare(df)


def fetch_recent_air_quality(
    past_hours=48
):

    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "past_hours": past_hours,
        "forecast_hours": 0,
        "hourly": (
            "pm10,"
            "pm2_5,"
            "carbon_monoxide,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone,"
            "us_aqi"
        ),
        "timezone": "UTC"
    }

    df = _request(
        url,
        params
    )

    return _prepare(df)


def fetch_forecast_weather(
    forecast_hours=72
):

    url = (
        "https://api.open-meteo.com/v1/forecast"
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "forecast_hours": forecast_hours,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m,"
            "surface_pressure,"
            "precipitation,"
            "cloud_cover"
        ),
        "timezone": "UTC"
    }

    df = _request(
        url,
        params
    )

    return _prepare(df)


def fetch_forecast_air_quality(
    forecast_hours=72
):

    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "forecast_hours": forecast_hours,
        "hourly": (
            "pm10,"
            "pm2_5,"
            "carbon_monoxide,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone,"
            "us_aqi"
        ),
        "timezone": "UTC"
    }

    df = _request(
        url,
        params
    )

    return _prepare(df)


def _up_to_now(df):

    # Open-Meteo returns forecast hours after the past ones.
    # Keep only hours that have already happened.
    now_hour = pd.Timestamp.now(
        tz="UTC"
    ).floor("h")

    return df[
        df["datetime"] <= now_hour
    ].reset_index(drop=True)


def fetch_past_days_weather(
    past_days=92
):

    url = (
        "https://api.open-meteo.com/v1/forecast"
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "past_days": past_days,
        "hourly": WEATHER_VARIABLES,
        "timezone": "UTC"
    }

    df = _prepare(
        _request(
            url,
            params
        )
    )

    return _up_to_now(df)


def fetch_past_days_air_quality(
    past_days=92,
    variables=AIR_QUALITY_VARIABLES
):

    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "past_days": past_days,
        "hourly": variables,
        "timezone": "UTC"
    }

    df = _prepare(
        _request(
            url,
            params
        )
    )

    return _up_to_now(df)


def fetch_air_quality_history(
    past_days=90
):

    # Live AQI history for the dashboard timeline. It reads Open-Meteo
    # directly, so it is always current and does not depend on the
    # feature store being up to date.
    df = fetch_past_days_air_quality(
        past_days=past_days,
        variables="us_aqi"
    )

    return (
        df[["datetime", "us_aqi"]]
        .dropna()
        .sort_values("datetime")
        .reset_index(drop=True)
    )
