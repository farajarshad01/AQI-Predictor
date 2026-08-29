import os

import hopsworks

from .config import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION
)


def get_project():

    api_key = os.environ["HOPSWORKS_API_KEY"]

    return hopsworks.login(
        api_key_value=api_key
    )


def get_feature_store():

    project = get_project()

    return project.get_feature_store()


def get_feature_group():

    feature_store = (
        get_feature_store()
    )

    return feature_store.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION
    )


def get_or_create_feature_group():

    feature_store = (
        get_feature_store()
    )

    return (
        feature_store
        .get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            description=(
                "Hourly AQI forecasting features"
            ),
            primary_key=["datetime"],
            event_time="datetime",
            online_enabled=True,
            time_travel_format="HUDI"
        )
    )


def get_model_registry():

    project = get_project()

    return project.get_model_registry()


def get_registered_models(
    model_name
):

    registry = get_model_registry()

    models = registry.get_models(
        name=model_name
    )

    if not models:
        raise ValueError(
            f"No registered models found: {model_name}"
        )

    return models


def get_latest_model(
    model_name
):

    models = get_registered_models(
        model_name
    )

    return sorted(
        models,
        key=lambda model: model.version,
        reverse=True
    )[0]


def download_latest_model(
    model_name,
    local_path,
    overwrite=True
):

    model = get_latest_model(
        model_name
    )

    model.download(
    local_path=local_path,
    overwrite=overwrite
)

    return model
