# AQI Predictor

An automated machine learning system for forecasting Air Quality Index (AQI) for the next 24, 48, and 72 hours.

## Overview

The project:

- Fetches weather and air-quality data from APIs
- Creates time-based and AQI-based features
- Stores features in Hopsworks
- Trains CatBoost forecasting models
- Stores trained models in the model registry
- Generates 24, 48, and 72-hour AQI predictions
- Automates pipelines using GitHub Actions
- Displays predictions through a Streamlit dashboard

## Project Structure

AQI Forecasting/
│
├── .github/
│   └── workflows/
│       ├── hourly_features.yml
│       └── daily_training.yml
│
├── app/
│   └── streamlit_dashboard.py
│
├── src/
│   ├── __init__.py
│   ├── backfill.py
│   ├── config.py
│   ├── data_fetch.py
│   ├── feature_engineering.py
│   ├── feature_pipeline.py
│   ├── hopsworksclients.py
│   ├── prediction_pipeline.py
│   └── training_pipeline.py
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt

Setup

Install dependencies:

pip install -r requirements.txt

Create a .env file using .env.example and add the required API credentials.

Run

Feature pipeline:

python src/feature_pipeline.py

Training pipeline:

python src/training_pipeline.py

Prediction pipeline:

python src/prediction_pipeline.py

Dashboard:

streamlit run app/streamlit_dashboard.py
Automation

GitHub Actions runs:

Feature updates hourly
Model training daily
Models

The system uses CatBoost regression models for:

24-hour AQI forecasting
48-hour AQI forecasting
72-hour AQI forecasting

See the project report for detailed methodology, feature engineering, experiments, model evaluation, and results.