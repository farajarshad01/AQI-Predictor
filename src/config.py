import os
from dotenv import load_dotenv

load_dotenv()

LATITUDE = float(
    os.getenv("LATITUDE", "0")
)

LONGITUDE = float(
    os.getenv("LONGITUDE", "0")
)

HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]


FEATURE_GROUP_NAME = "aqi_features_v2"
FEATURE_GROUP_VERSION = 1


MODEL_24_NAME = "aqi_model_24h"
MODEL_48_NAME = "aqi_model_48h"
MODEL_72_NAME = "aqi_model_72h"


FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
    "year",
    "month",
    "day",
    "day_of_week",
    "hour",
    "aqi_change",
    "aqi_change_rate",
    "aqi_lag_1",
    "aqi_lag_12",
    "aqi_lag_24",
    "aqi_rolling_6",
    "aqi_rolling_12",
    "aqi_rolling_24"
]


RAW_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi"
]


MODEL_CONFIGS = {
    "24h": {
        "name": "aqi_model_24h",
        "target": "target_aqi_24",
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 7
    },

    "48h": {
        "name": "aqi_model_48h",
        "target": "target_aqi_48",
        "iterations": 200,
        "learning_rate": 0.05,
        "depth": 6
    },

    "72h": {
        "name": "aqi_model_72h",
        "target": "target_aqi_72",
        "iterations": 200,
        "learning_rate": 0.05,
        "depth": 6
    }

}
