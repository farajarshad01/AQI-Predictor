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
        "ring": "#3DDC84",
        "bg": "#3DDC84",
        "fg": "#17181A",
        "headline": "Air quality is considered satisfactory.",
        "detail": "Enjoy outdoor activities as usual.",
        "actions": ["Outdoors: fine", "Windows: open"],
    },
    {
        "max": 100,
        "label": "Moderate",
        "ring": "#FFD60A",
        "bg": "#FFD60A",
        "fg": "#17181A",
        "headline": "Air quality is acceptable.",
        "detail": "Unusually sensitive people may experience minor effects.",
        "actions": ["Sensitive people: take breaks", "Windows: open"],
    },
    {
        "max": 150,
        "label": "Unhealthy for Sensitive Groups",
        "ring": "#FF9500",
        "bg": "#FF9500",
        "fg": "#17181A",
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
        "ring": "#E5251F",
        "bg": "#E5251F",
        "fg": "#FFFFFF",
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
        "ring": "#8E24F5",
        "bg": "#8E24F5",
        "fg": "#FFFFFF",
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
        "ring": "#B0003A",
        "bg": "#B0003A",
        "fg": "#FFFFFF",
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
    return " ".join(
        line.strip()
        for line in html.strip().splitlines()
    )


# HTML builders

def title_html():

    return _h("""
    <div class="head">
        <div class="title">AQI Gujranwala</div>
        <div class="head-s">Punjab, Pakistan</div>
        <div class="head-d">72-hour machine-learning forecast</div>
    </div>
    """)


def section_html(title, subtitle):

    return _h(f"""
    <div class="sec">
        <div class="sec-t">{title}</div>
        <div class="sec-s">{subtitle}</div>
    </div>
    """)


def status_html(as_of, refreshed_minutes, stale_hours):

    if stale_hours > 3:

        dot = "#F59E0B"

        headline = (
            f"Latest reading is {stale_hours:.0f} h old "
            f"({fmt_clock(as_of)})"
        )

    else:

        dot = "#22C55E"

        headline = f"Live · data for {fmt_clock(as_of)}"

    return _h(f"""
    <div class="status">
        <div class="st-l"><span class="dot" style="background:{dot}"></span>{headline}</div>
        <div class="st-s">Refreshed {refreshed_minutes} min ago</div>
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
    <div class="fc-main">
        <div class="fv"><span class="fn">{value:.1f}</span>{delta_chip}</div>
        <div class="fm">
            <span class="ch" style="background:{category['bg']};color:{category['fg']}">{category['label']}</span>
            <span class="m">{fmt_day_time(when)}</span>
        </div>
    </div>
    """)


def forecast_hint_html():

    return (
        '<div class="fc-foot">Pick a horizon to update the chart '
        "and the model explanation below.</div>"
    )


def _ratio_color(ratio):

    if ratio <= 1:
        return "#3DDC84"

    if ratio <= 2:
        return "#FFD60A"

    if ratio <= 3:
        return "#FF9500"

    return "#E5251F"


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

    text = (
        "Air quality forecasts are generated from weather observations, "
        "air-quality data, CatBoost machine-learning models and SHAP. "
        "AQI values are modelled estimates from Open-Meteo, "
        "not ground-station readings."
    )

    return _h(f"""
    <div class="card about">
        <div class="t">About this forecast</div>
        <div class="about-p">{text}</div>
        <div class="built">
            <div class="m">Built with</div>
            <div class="chips">{chips}</div>
        </div>
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
            opacity=0.15,
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

    /* The CSS block itself must not take up a gap in the layout. */
    [data-testid="stElementContainer"]:has(style) { display: none; }

    /* Spacing system: 3rem between sections, 1.5rem inside a section,
       1.25rem inside a card. */
    .block-container {
        max-width: 1280px;
        padding: 3rem 3rem 4rem;
    }
    .block-container > [data-testid="stVerticalBlock"] { gap: 3rem; }
    .st-key-sec_now,
    .st-key-sec_pollutants,
    .st-key-sec_trends,
    .st-key-sec_about { gap: 1.5rem !important; }

    .card,
    .st-key-forecast_card,
    .st-key-timeline_card,
    .st-key-drivers_card {
        background: #FFFFFF;
        border: 1px solid #E8E6E1;
        border-radius: 24px;
        padding: 36px 40px;
        min-width: 0;
        box-sizing: border-box;
    }
    .st-key-forecast_card,
    .st-key-timeline_card,
    .st-key-drivers_card { gap: 1.25rem !important; }
    .st-key-forecast_card { min-height: 360px; justify-content: space-between; }
    /* Streamlit gives every markdown block a negative bottom margin, which
       squeezes the gaps and pushes headings below their neighbours. */
    .st-key-forecast_card [data-testid="stMarkdownContainer"],
    .st-key-forecast_card [data-testid="stMarkdown"] { margin-bottom: 0 !important; }
    .st-key-horizon_control { display: flex; justify-content: flex-end; width: 100%; }
    .st-key-timeline_card,
    .st-key-drivers_card { min-height: 590px; }

    .m { font-size: 14px; color: #6B6F76; line-height: 1.5; }
    .u { font-size: 13px; font-weight: 500; color: #6B6F76; }
    .t { font-weight: 700; font-size: 19px; color: #17181A; }

    .ch {
        display: inline-block;
        border-radius: 999px;
        padding: 6px 14px;
        font-size: 13px;
        font-weight: 700;
    }
    .red { background: #FFDAD6; color: #B3140D; }
    .gr  { background: #C9F5DA; color: #0A7A3A; }
    .sk  { background: #F0EEE9; color: #4A4E55; font-weight: 600; }
    .chips { display: flex; gap: 10px; flex-wrap: wrap; }

    /* Header */
    .head { display: flex; flex-direction: column; gap: 6px; }
    .title { font-size: 34px; font-weight: 800; line-height: 1.1; }
    .head-s { font-size: 17px; font-weight: 600; color: #17181A; }
    .head-d { font-size: 14px; color: #6B6F76; }
    .st-key-refresh_btn { display: flex; justify-content: flex-end; width: 100%; }
    .st-key-refresh_btn button {
        border-radius: 999px; min-height: 48px;
        padding: 0 28px; font-weight: 700;
    }
    .status { display: flex; flex-direction: column; align-items: flex-end;
              gap: 4px; text-align: right; }
    .st-l { display: flex; align-items: center; gap: 8px;
            font-size: 14px; font-weight: 600; }
    .st-s { font-size: 13px; color: #6B6F76; }
    .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }

    /* Section headers */
    .sec { display: flex; flex-direction: column; gap: 4px; }
    .sec-t { font-size: 22px; font-weight: 800; }
    .sec-s { font-size: 14px; color: #6B6F76; }

    /* Hero */
    .hero { display: flex; align-items: center; gap: 40px; flex-wrap: wrap;
            min-height: 360px; }
    .ring { width: 220px; height: 220px; border-radius: 50%; flex: none;
            display: flex; align-items: center; justify-content: center; }
    .ring-in { width: 184px; height: 184px; border-radius: 50%;
               background: #FFFFFF; display: flex; flex-direction: column;
               align-items: center; justify-content: center; gap: 4px; }
    .ring-n { font-size: 60px; font-weight: 800; line-height: 1; }
    .ring-l { font-size: 14px; color: #6B6F76; }
    .hero-t { display: flex; flex-direction: column; gap: 18px;
              flex: 1; min-width: 240px; }
    .hero-h { font-size: 22px; font-weight: 700; line-height: 1.3; }
    .hero-d { font-size: 15px; line-height: 1.6; }

    /* Forecast card */
    .fc-main { display: flex; flex-direction: column; gap: 22px; }
    .fv { display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }
    .fn { font-size: 84px; font-weight: 800; line-height: 1; letter-spacing: -0.02em; }
    .fm { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
    .fm .ch { padding: 8px 18px; font-size: 14px; }
    .fm .m { font-size: 15px; }
    .fc-foot { border-top: 1px solid #F0EEE9; padding-top: 18px;
               font-size: 14px; line-height: 1.5; color: #6B6F76; }

    /* Pollutant tiles */
    .tiles { display: grid; gap: 24px;
             grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
    .tile { background: #FFFFFF; border: 1px solid #E8E6E1;
            border-radius: 20px; padding: 24px;
            display: flex; flex-direction: column; gap: 10px;
            transition: transform .15s, box-shadow .15s; }
    .tile:hover { transform: translateY(-3px);
                  box-shadow: 0 10px 24px rgba(23, 24, 26, .09); }
    .tv { font-size: 28px; font-weight: 700; line-height: 1.1; }
    .bar { height: 8px; background: #EEECE7; border-radius: 4px; }
    .bar i { display: block; height: 8px; border-radius: 4px; }

    /* Model explanation rows */
    .dr { display: grid; grid-template-columns: minmax(0, 1fr) 130px 52px;
          gap: 14px; align-items: center; font-size: 15px; padding: 14px 0;
          border-bottom: 1px solid #F0EEE9; }
    .dr:last-of-type { border-bottom: 0; }
    .dr b { text-align: right; }
    .tr { position: relative; height: 10px; background: #F0EEE9;
          border-radius: 5px; }
    .tr i { position: absolute; height: 10px; }
    .tr.up i { left: 50%; background: #E5251F; border-radius: 0 5px 5px 0; }
    .tr.dn i { right: 50%; background: #1E7BE8; border-radius: 5px 0 0 5px; }

    /* About */
    .about { display: flex; flex-direction: column; gap: 20px; }
    .about-p { font-size: 15px; line-height: 1.7; color: #4A4E55;
               max-width: 760px; }
    .built { display: flex; flex-direction: column; gap: 12px; }
    .foot { text-align: center; font-size: 13px; color: #6B6F76; }
    </style>
    """
