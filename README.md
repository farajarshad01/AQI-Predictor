# AQI Predictor

AQI Predictor is an end-to-end machine learning system that forecasts **Air Quality Index (AQI)** for **Gujranwala, Punjab, Pakistan**, with predictions at **24, 48, and 72 hours**.

The project combines real-time environmental data, automated ML pipelines, cloud-based feature and model management, and explainable AI into a single forecasting application.

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

## Tech Stack 
**Python**  | 
**Pandas**  | 
**NumPy**  | 
**Scikit-learn** |
**CatBoost** | **SHAP** |
**Hopsworks** |
**Streamlit** |
**Plotly** |
**GitHub Actions** |


## Project Architecture

```text
                Open-Meteo
                    │
                    ▼
             Data Collection
                    │
                    ▼
          Feature Engineering
                    │
                    ▼
          Hopsworks Feature Store
                    │
                    ▼
             CatBoost Models
                    │
             ┌──────┼──────┐
             ▼      ▼      ▼
            24h    48h    72h
             │      │      │
             └──────┼──────┘
                    ▼
             SHAP Explanations
                    │
                    ▼
            Streamlit Dashboard
```

## Machine Learning

Three CatBoost regression models are trained for different forecasting horizons:

- 24-hour
- 48-hour
- 72-hour

Models are evaluated using MAE, RMSE, and R², then trained on the available data and registered in the Hopsworks Model Registry.

SHAP provides feature-level explanations, showing which environmental and historical factors are contributing to each AQI prediction.

## MLOps

The project uses GitHub Actions to automate the ML workflow:

### Data Backfill → Feature Pipeline → Model Training → Model Registration

Hopsworks provides persistent storage for the feature data and production models, while the Streamlit application retrieves the latest data and registered models when a forecast is requested.

## Dashboard

The Streamlit application provides:

- 24/48/72-hour forecast cards
- AQI health-risk classification
- Forecast trend visualization
- SHAP-based model explanations
- On-demand prediction generation
- Location-specific forecasting for Gujranwala

## Setup

Install the dependencies:

```bash
git clone <repository-url>
cd aqi-predictor
pip install -r requirements.txt
streamlit run app/streamlit_dashboard.py
```
### Configure the required environment variables:
```
LATITUDE
LONGITUDE
HOPSWORKS_API_KEY
```
## Location

Gujranwala, Punjab, Pakistan

The system can be adapted to other locations by changing the configured coordinates and retraining the models with the corresponding environmental data.

See the project report for detailed methodology, feature engineering, experiments, model evaluation, and results.
