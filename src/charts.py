from typing import List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_categorical_dtype(df[c]) or df[c].dtype == "object"]


def get_datetime_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def histogram(df: pd.DataFrame, column: str):
    fig = px.histogram(df, x=column, nbins=50, title=f"Distribution of {column}")
    fig.update_layout(bargap=0.05)
    return fig


def bar_agg(df: pd.DataFrame, category_col: str, metric_col: str, agg: str = "sum", top_n: int = 20):
    if agg not in {"sum", "mean", "count", "median"}:
        agg = "sum"
    grouped = (df.groupby(category_col)[metric_col].agg(agg).reset_index().sort_values(metric_col, ascending=False).head(top_n))
    title = f"{agg.title()} of {metric_col} by {category_col}"
    fig = px.bar(grouped, x=category_col, y=metric_col, title=title)
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def timeseries(df: pd.DataFrame, date_col: str, metric_col: Optional[str] = None, agg: str = "sum", freq: str = "D"):
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col])
    temp = temp.set_index(date_col)
    if metric_col and metric_col in temp.columns and pd.api.types.is_numeric_dtype(temp[metric_col]):
        res = temp[[metric_col]].resample(freq).agg(agg).reset_index()
        y = metric_col
        title = f"{agg.title()} of {metric_col} over time ({freq})"
    else:
        res = temp.resample(freq).size().reset_index(name="count")
        y = "count"
        title = f"Count over time ({freq})"
    fig = px.line(res, x=date_col, y=y, title=title)
    return fig


def gauge(value: float, title: str, suffix: str = "", min_val: float = 0, max_val: float = 100):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix},
            title={"text": title},
            gauge={"axis": {"range": [min_val, max_val]}},
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def geo_map(df: pd.DataFrame, country_col: str, metric_col: str, top_n: int = 30):
    # If lat/lon are available, prefer a marker map (more reliable than country name matching).
    lat_col = None
    lon_col = None
    for c in df.columns:
        cl = c.lower()
        if lat_col is None and cl in {"lat", "latitude"}:
            lat_col = c
        if lon_col is None and cl in {"lon", "lng", "longitude"}:
            lon_col = c
    for c in df.columns:
        cl = c.lower()
        if lat_col is None and "latitude" in cl:
            lat_col = c
        if lon_col is None and ("longitude" in cl or cl == "lon" or cl == "lng"):
            lon_col = c

    if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
        tmp = df[[lat_col, lon_col, metric_col]].copy()
        tmp[lat_col] = pd.to_numeric(tmp[lat_col], errors="coerce")
        tmp[lon_col] = pd.to_numeric(tmp[lon_col], errors="coerce")
        tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
        tmp = tmp.dropna(subset=[lat_col, lon_col, metric_col])
        if not tmp.empty:
            agg = tmp.groupby([lat_col, lon_col], as_index=False)[metric_col].sum()
            fig = px.scatter_geo(
                agg,
                lat=lat_col,
                lon=lon_col,
                size=metric_col,
                color=metric_col,
                hover_name=country_col if country_col in df.columns else None,
                hover_data={metric_col: ":,.0f", lat_col: False, lon_col: False},
                color_continuous_scale="Viridis",
                title=f"{metric_col} by Location",
                projection="natural earth",
            )
            fig.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10))
            return fig

    # Choropleth fallback: auto-detect country identifier format
    agg = df.groupby(country_col)[metric_col].sum().reset_index()
    agg[metric_col] = pd.to_numeric(agg[metric_col], errors="coerce")
    agg = agg.dropna(subset=[metric_col])
    if agg.empty:
        return px.scatter(title="No data available for geo map")

    agg = agg.sort_values(metric_col, ascending=False).head(top_n)

    sample = agg[country_col].astype(str).head(25)
    is_iso2 = sample.str.fullmatch(r"[A-Za-z]{2}").mean() >= 0.8
    is_iso3 = sample.str.fullmatch(r"[A-Za-z]{3}").mean() >= 0.8

    if is_iso3:
        locmode = "ISO-3"
        locations = agg[country_col].astype(str).str.upper()
    elif is_iso2:
        locmode = "ISO-2"
        locations = agg[country_col].astype(str).str.upper()
    else:
        locmode = "country names"
        locations = agg[country_col].astype(str)

    agg = agg.copy()
    agg["_locations"] = locations

    fig = px.choropleth(
        agg,
        locations="_locations",
        locationmode=locmode,
        color=metric_col,
        color_continuous_scale="Blues",
        hover_name=country_col,
        hover_data={metric_col: ":,.0f", "_locations": False},
        title=f"{metric_col} by Country (Top {top_n})",
    )
    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=50, b=10),
        geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
        coloraxis_colorbar_title=metric_col,
    )
    return fig


def funnel_chart(stages: List[Tuple[str, float]], title: str = "Funnel"):
    names = [s for s, _ in stages]
    values = [v for _, v in stages]
    fig = go.Figure(go.Funnel(y=names, x=values, textinfo="value+percent previous"))
    fig.update_layout(title=title, height=400)
    return fig


def heatmap(df: pd.DataFrame, title: str = "Heatmap"):
    arr = df.to_numpy(dtype=np.float64)
    fig = px.imshow(arr, x=list(df.columns), y=list(df.index), color_continuous_scale="Blues", aspect="auto")
    fig.update_layout(title=title, height=500, margin=dict(l=10, r=10, t=40, b=10))
    return fig
