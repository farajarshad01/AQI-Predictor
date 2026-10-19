"""UI helpers for the AQI dashboard.

Everything here returns an HTML string or a Plotly figure. Nothing imports
Streamlit, so these functions can be tested on their own.
"""

from html import escape

import pandas as pd
import plotly.graph_objects as go


DISPLAY_TZ = "Asia/Karachi"
TZ_LABEL = "PKT"

INK = "#17181A"
MUTED = "#6B6F76"
GRID = "#EEECE7"


# AQI categories (US EPA breakpoints)

CATEGORIES = [
    {
        "max": 50,
        "label": "Good",
        "ring": "#8FD19E",
        "bg": "#E3F0E6",
        "fg": "#25673A",
        "headline": "Air quality is considered satisfactory.",
        "detail": "Enjoy outdoor activities as usual.",
        "actions": ["Outdoors: fine", "Windows: open"],
    },
    {
        "max": 100,
        "label": "Moderate",
        "ring": "#F2D66B",
        "bg": "#FBF3D0",
        "fg": "#7A5F00",
        "headline": "Air quality is acceptable.",
        "detail": "Unusually sensitive people may experience minor effects.",
        "actions": ["Sensitive people: take breaks", "Windows: open"],
    },
    {
        "max": 150,
        "label": "Unhealthy for Sensitive Groups",
        "ring": "#F4A261",
        "bg": "#FDEBD9",
        "fg": "#8A410A",
        "headline": "Sensitive groups may experience health effects.",
        "detail": (
            "Children, older adults and people with heart or lung "
            "disease should limit prolonged outdoor exertion."
        ),
        "actions": ["Mask: consider N95", "Exercise: keep it light", "Windows: limit"],
    },
    {
        "max": 200,
        "label": "Unhealthy",
        "ring": "#E5675C",
        "bg": "#FBE3E0",
        "fg": "#9B2C25",
        "headline": "Everyone may begin to experience health effects.",
        "detail": (
            "People with heart or lung disease, older adults and "
            "children should avoid prolonged outdoor exertion."
        ),
        "actions": ["Mask: N95", "Windows: closed", "Exercise: indoors"],
    },
    {
        "max": 300,
        "label": "Very Unhealthy",
        "ring": "#A16BB5",
        "bg": "#EEE0F3",
        "fg": "#5E2A74",
        "headline": "Health alert: the risk of health effects is increased.",
        "detail": (
            "Everyone should avoid outdoor exertion; sensitive groups "
            "should stay indoors."
        ),
        "actions": ["Mask: N95", "Windows: closed", "Stay indoors"],
    },
    {
        "max": float("inf"),
        "label": "Hazardous",
        "ring": "#8C2F4A",
        "bg": "#F4DCE3",
        "fg": "#6B1530",
        "headline": "Health warning of emergency conditions.",
        "detail": "Everyone should stay indoors with filtered air if possible.",
        "actions": ["Mask: N95", "Windows: closed", "Air purifier: on"],
    },
]


def category_for(aqi):

    for category in CATEGORIES:

        if aqi <= category["max"]:
            return category

    return CATEGORIES[-1]


def _bands():

    low = 0

    for category in CATEGORIES:

        high = category["max"]

        if high == float("inf"):
            high = 500

        yield low, high, category["ring"]

        low = high


# Pollutants: (column, label, WHO 2021 guideline in ug/m3).
# 24-hour guideline values; ozone uses the 8-hour value.

POLLUTANTS = [
    ("pm2_5", "PM2.5", 15.0),
    ("pm10", "PM10", 45.0),
    ("nitrogen_dioxide", "Nitrogen dioxide", 25.0),
    ("sulphur_dioxide", "Sulphur dioxide", 40.0),
    ("ozone", "Ozone", 100.0),
]


FEATURE_LABELS = {
    "temperature_2m": "Temperature",
    "relative_humidity_2m": "Relative Humidity",
    "wind_speed_10m": "Wind Speed",
    "surface_pressure": "Surface Pressure",
    "precipitation": "Precipitation",
    "cloud_cover": "Cloud Cover",
    "pm10": "PM10",
    "pm2_5": "PM2.5",
    "carbon_monoxide": "Carbon Monoxide",
    "nitrogen_dioxide": "Nitrogen Dioxide",
    "sulphur_dioxide": "Sulphur Dioxide",
    "ozone": "Ozone",
    "us_aqi": "Current AQI",
    "year": "Year",
    "month": "Month",
    "day": "Day",
    "day_of_week": "Day of Week",
    "hour": "Hour",
    "aqi_change": "AQI Change",
    "aqi_change_rate": "AQI Change Rate",
    "aqi_lag_1": "AQI, 1 Hour Ago",
    "aqi_lag_12": "AQI, 12 Hours Ago",
    "aqi_lag_24": "AQI, 24 Hours Ago",
    "aqi_rolling_6": "AQI, 6 Hour Average",
    "aqi_rolling_12": "AQI, 12 Hour Average",
    "aqi_rolling_24": "AQI, 24 Hour Average",
}


def readable_feature(name):

    return FEATURE_LABELS.get(
        name,
        name.replace("_", " ").title()
    )


# Time helpers (all display times are shown in local time)

def to_local(ts):

    ts = pd.Timestamp(ts)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    return ts.tz_convert(DISPLAY_TZ)


def fmt_clock(ts):

    return f"{to_local(ts):%H:%M} {TZ_LABEL}"


def fmt_day_time(ts):

    local = to_local(ts)

    return f"{local:%b} {local.day} · {local:%H:%M} {TZ_LABEL}"


def _naive_local(series):

    return (
        pd.to_datetime(series, utc=True)
        .dt.tz_convert(DISPLAY_TZ)
        .dt.tz_localize(None)
    )


def _h(html):

    # Streamlit's markdown parser treats blank lines and indented lines
    # specially, so the HTML is collapsed onto one line.
    return "".join(
        line.strip()
        for line in html.strip().splitlines()
    )


# HTML builders

def status_html(as_of, refreshed_minutes, stale_hours):

    if stale_hours > 3:

        dot = "#E0A030"

        text = (
            f"Latest reading is {stale_hours:.0f} h old "
            f"({fmt_clock(as_of)})"
        )

    else:

        dot = "#2E9E5B"

        text = (
            f"Live · data for {fmt_clock(as_of)} · "
            f"refreshed {refreshed_minutes} min ago"
        )

    return _h(f"""
    <div class="status">
        <span class="dot" style="background:{dot}"></span>{text}
    </div>
    """)


def hero_html(aqi):

    category = category_for(aqi)

    degrees = max(0.0, min(aqi / 300.0, 1.0)) * 360

    chips = "".join(
        f'<span class="ch sk">{escape(action)}</span>'
        for action in category["actions"]
    )

    return _h(f"""
    <div class="card hero">
        <div class="ring" style="background:conic-gradient({category['ring']} 0deg {degrees:.0f}deg,{GRID} {degrees:.0f}deg 360deg)">
            <div class="ring-in">
                <div class="ring-n">{aqi:.0f}</div>
                <div class="ring-l">Current AQI</div>
            </div>
        </div>
        <div class="hero-t">
            <span class="ch" style="background:{category['bg']};color:{category['fg']};align-self:flex-start">{category['label']}</span>
            <div class="hero-h">{category['headline']}</div>
            <div class="m hero-d">{category['detail']}</div>
            <div class="chips">{chips}</div>
        </div>
    </div>
    """)


def forecast_html(value, delta, when):

    category = category_for(value)

    if abs(delta) < 0.05:
        delta_chip = '<span class="ch sk">no change vs now</span>'

    elif delta < 0:
        delta_chip = f'<span class="ch gr">↓ {abs(delta):.1f} vs now</span>'

    else:
        delta_chip = f'<span class="ch red">↑ {delta:.1f} vs now</span>'

    return _h(f"""
    <div class="fv"><span class="fn">{value:.1f}</span>{delta_chip}</div>
    <div class="fm">
        <span class="ch" style="background:{category['bg']};color:{category['fg']}">{category['label']}</span>
        <span class="m">{fmt_day_time(when)}</span>
    </div>
    """)


def _ratio_color(ratio):

    if ratio <= 1:
        return "#8FD19E"

    if ratio <= 2:
        return "#F2D66B"

    if ratio <= 3:
        return "#F4A261"

    return "#E5675C"


def pollutants_html(current):

    tiles = []

    for key, label, reference in POLLUTANTS:

        value = current.get(key)

        if value is None or pd.isna(value):
            continue

        ratio = value / reference

        fill = min(ratio / 3.0, 1.0) * 100

        tiles.append(
            f'<div class="tile">'
            f'<div class="m">{label}</div>'
            f'<div class="tv">{value:.1f} <span class="u">µg/m³</span></div>'
            f'<div class="bar"><i style="width:{fill:.0f}%;background:{_ratio_color(ratio)}"></i></div>'
            f'<div class="m tg">{ratio:.1f}× WHO guideline</div>'
            f'</div>'
        )

    return '<div class="tiles">' + "".join(tiles) + "</div>"


def drivers_html(explanation):

    largest = explanation["shap_value"].abs().max()

    if not largest or pd.isna(largest):
        largest = 1.0

    rows = []

    for _, row in explanation.iterrows():

        value = float(row["shap_value"])

        label = readable_feature(row["feature"])

        width = abs(value) / largest * 48

        direction = "up" if value >= 0 else "dn"

        sign = "+" if value >= 0 else "−"

        feature_value = row.get("feature_value")

        tooltip = ""

        if feature_value is not None and pd.notna(feature_value):
            tooltip = (
                f' title="{escape(label)} = {float(feature_value):.1f}"'
            )

        rows.append(
            f'<div class="dr"{tooltip}>'
            f'<span>{escape(label)}</span>'
            f'<div class="tr {direction}"><i style="width:{width:.0f}%"></i></div>'
            f'<b>{sign}{abs(value):.1f}</b>'
            f'</div>'
        )

    return "".join(rows)


def about_html():

    chips = "".join(
        f'<span class="ch sk">{name}</span>'
        for name in [
            "Open-Meteo",
            "CatBoost",
            "Hopsworks",
            "GitHub Actions",
            "Streamlit",
            "SHAP",
        ]
    )

    return _h(f"""
    <div class="card about">
        <div style="max-width:520px">
            <div class="t">About this forecast</div>
            <div class="m" style="line-height:1.5;margin-top:4px">
                Air quality forecasts generated from weather observations,
                air-quality data, CatBoost machine-learning models and SHAP.
                AQI values are modelled estimates from Open-Meteo, not
                ground-station readings.
            </div>
        </div>
        <div class="chips" style="justify-content:flex-end">{chips}</div>
    </div>
    """)


# Timeline chart

def build_timeline(history, current, predictions, horizon, days):

    now = pd.Timestamp(current["datetime"])

    if now.tzinfo is None:
        now = now.tz_localize("UTC")

    current_aqi = float(current["us_aqi"])

    now_local = now.tz_convert(DISPLAY_TZ).tz_localize(None)

    fig = go.Figure()

    values = [current_aqi]

    # History (solid)

    if history is not None and not history.empty:

        past = history[
            (history["datetime"] > now - pd.Timedelta(days=days))
            & (history["datetime"] <= now)
        ].copy()

        past["local"] = _naive_local(
            past["datetime"]
        )

        if days > 7:

            # Daily averages, grouped by local (PKT) day.
            past = (
                past
                .set_index("local")["us_aqi"]
                .resample("D")
                .mean()
                .dropna()
                .reset_index()
            )

        if not past.empty:

            values += past["us_aqi"].tolist()

            fig.add_trace(
                go.Scatter(
                    x=past["local"],
                    y=past["us_aqi"],
                    mode="lines",
                    line=dict(color=INK, width=3),
                    hovertemplate=(
                        "<b>%{y:.0f} AQI</b>"
                        "<br>%{x|%a %d %b, %H:%M}"
                        f" {TZ_LABEL}<extra></extra>"
                    ),
                )
            )

    # Forecast (dashed line from now, one marker per horizon)

    forecast = predictions.sort_values("horizon_hours")

    forecast_x = _naive_local(forecast["forecast_time"])

    forecast_y = forecast["predicted_aqi"].astype(float)

    values += forecast_y.tolist()

    fig.add_trace(
        go.Scatter(
            x=[now_local] + forecast_x.tolist(),
            y=[current_aqi] + forecast_y.tolist(),
            mode="lines",
            line=dict(color=INK, width=3, dash="dash"),
            hoverinfo="skip",
        )
    )

    selected = [
        int(h) == int(horizon)
        for h in forecast["horizon_hours"]
    ]

    fig.add_trace(
        go.Scatter(
            x=forecast_x,
            y=forecast_y,
            mode="markers+text",
            text=[f"{v:.1f}" for v in forecast_y],
            textposition="top center",
            textfont=dict(color=INK, size=13),
            marker=dict(
                size=13,
                color=[INK if s else "#FFFFFF" for s in selected],
                line=dict(color=INK, width=3),
            ),
            customdata=forecast["horizon_hours"],
            hovertemplate=(
                "<b>%{y:.1f} AQI</b>"
                "<br>%{customdata}-hour forecast"
                "<br>%{x|%a %d %b, %H:%M}"
                f" {TZ_LABEL}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[now_local],
            y=[current_aqi],
            mode="markers",
            marker=dict(size=13, color=INK),
            hovertemplate=(
                "<b>%{y:.0f} AQI</b><br>Now<extra></extra>"
            ),
        )
    )

    # AQI category bands

    y_max = max(160.0, max(values) * 1.18)

    for low, high, color in _bands():

        if low >= y_max:
            break

        fig.add_hrect(
            y0=low,
            y1=min(high, y_max),
            fillcolor=color,
            opacity=0.17,
            line_width=0,
            layer="below",
        )

    # "Now" marker

    fig.add_shape(
        type="line",
        x0=now_local,
        x1=now_local,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color=INK, width=1.5, dash="dot"),
    )

    fig.add_annotation(
        x=now_local,
        y=1,
        yref="paper",
        text="Now",
        showarrow=False,
        yanchor="bottom",
        font=dict(color=INK, size=12),
    )

    fig.update_layout(
        height=360,
        margin=dict(l=44, r=20, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Figtree, system-ui, sans-serif",
            color=INK,
            size=12,
        ),
        showlegend=False,
        hovermode="closest",
        xaxis=dict(
            showgrid=False,
            fixedrange=True,
            range=[
                now_local - pd.Timedelta(days=days),
                now_local + pd.Timedelta(hours=84),
            ],
            tickfont=dict(color=MUTED),
        ),
        yaxis=dict(
            range=[0, y_max],
            gridcolor=GRID,
            zeroline=False,
            fixedrange=True,
            tickfont=dict(color=MUTED),
            title=dict(text="AQI", font=dict(color=MUTED)),
        ),
    )

    return fig


# Page CSS

def css():

    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700;800&display=swap');

    html, body, .stApp, [data-testid="stMarkdownContainer"] {
        font-family: 'Figtree', system-ui, sans-serif;
    }
    .stApp { background: #F6F5F2; color: #17181A; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] { display: none; }
    footer { visibility: hidden; }
    .block-container { max-width: 1240px; padding: 2.2rem 2.4rem 3rem; }

    .card,
    .st-key-forecast_card,
    .st-key-timeline_card,
    .st-key-drivers_card {
        background: #FFFFFF;
        border: 1px solid #E8E6E1;
        border-radius: 24px;
        padding: 28px;
        min-width: 0;
    }
    .st-key-forecast_card { min-height: 100%; }

    .m { font-size: 13px; color: #6B6F76; }
    .u { font-size: 13px; font-weight: 500; color: #6B6F76; }
    .t { font-weight: 700; font-size: 18px; color: #17181A; }

    .ch {
        display: inline-block;
        border-radius: 999px;
        padding: 5px 12px;
        font-size: 13px;
        font-weight: 600;
    }
    .red { background: #FBE3E0; color: #9B2C25; }
    .gr  { background: #E3F0E6; color: #25673A; }
    .sk  { background: #F0EEE9; color: #4A4E55; }
    .chips { display: flex; gap: 8px; flex-wrap: wrap; }

    .title { font-size: 30px; font-weight: 800; line-height: 1.15; }
    .status { display: flex; align-items: center; gap: 8px;
              font-size: 13px; color: #6B6F76; }
    .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }

    .hero { display: flex; align-items: center; gap: 30px;
            flex-wrap: wrap; min-height: 100%; box-sizing: border-box; }
    .ring { width: 200px; height: 200px; border-radius: 50%; flex: none;
            display: flex; align-items: center; justify-content: center; }
    .ring-in { width: 168px; height: 168px; border-radius: 50%;
               background: #FFFFFF; display: flex; flex-direction: column;
               align-items: center; justify-content: center; }
    .ring-n { font-size: 56px; font-weight: 800; line-height: 1; }
    .ring-l { font-size: 13px; color: #6B6F76; margin-top: 4px; }
    .hero-t { display: flex; flex-direction: column; gap: 12px;
              flex: 1; min-width: 220px; }
    .hero-h { font-size: 20px; font-weight: 700; line-height: 1.3; }
    .hero-d { font-size: 14px; line-height: 1.5; }

    .fv { display: flex; align-items: baseline; gap: 12px;
          flex-wrap: wrap; margin-top: 18px; }
    .fn { font-size: 60px; font-weight: 800; line-height: 1; }
    .fm { margin-top: 16px; display: flex; align-items: center;
          gap: 10px; flex-wrap: wrap; }

    .tiles { display: grid; gap: 16px;
             grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }
    .tile { background: #FFFFFF; border: 1px solid #E8E6E1;
            border-radius: 18px; padding: 18px;
            transition: transform .15s, box-shadow .15s; }
    .tile:hover { transform: translateY(-3px);
                  box-shadow: 0 8px 20px rgba(23, 24, 26, .08); }
    .tv { font-size: 26px; font-weight: 700; margin: 4px 0 10px; }
    .tg { margin-top: 8px; }
    .bar { height: 6px; background: #EEECE7; border-radius: 3px; }
    .bar i { display: block; height: 6px; border-radius: 3px; }

    .dr { display: grid; grid-template-columns: minmax(0, 1fr) 130px 48px;
          gap: 12px; align-items: center; font-size: 14px; padding: 9px 0; }
    .dr b { text-align: right; }
    .tr { position: relative; height: 8px; background: #F0EEE9;
          border-radius: 4px; }
    .tr i { position: absolute; height: 8px; }
    .tr.up i { left: 50%; background: #C23A32; border-radius: 0 4px 4px 0; }
    .tr.dn i { right: 50%; background: #2F6FB5; border-radius: 4px 0 0 4px; }

    .about { display: flex; justify-content: space-between;
             align-items: center; gap: 24px; flex-wrap: wrap; }
    .foot { text-align: center; font-size: 13px; color: #6B6F76;
            padding-top: 8px; }
    </style>
    """
