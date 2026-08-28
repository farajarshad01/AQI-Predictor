import requests
import pandas as pd

from .config import LATITUDE, LONGITUDE


def _request(url, params):

    response = requests.get(
        url,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

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
