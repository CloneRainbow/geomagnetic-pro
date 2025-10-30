"""
Geomagnetic Pro 2025 — Фінальна версія (перевірено)
Висота | Теплова карта | Вектори | Клік | MGRS | UTM
"""
from __future__ import annotations

import os
import io
import zipfile
from datetime import date, datetime
from typing import List, Dict, Tuple
import requests
import numpy as np
import geopandas as gpd
import pandas as pd
import mgrs
import plotly.graph_objects as go
from pyproj import Proj
from pygeomag import GeoMag
import streamlit as st

# =============================================
# КОНФІГУРАЦІЯ
# =============================================
st.set_page_config(page_title="Geomagnetic Pro 2025", page_icon="🧭", layout="wide")
MGRS_CONV = mgrs.MGRS()
COF_PATH = "wmm/WMM_2025.COF"
COF_URL = "https://www.ncei.noaa.gov/data/world-magnetic-model-2025/full/WMM_COEFS-2025.zip"
RTDM_API = "https://geomag.usgs.gov/ws/edge/"

# =============================================
# WMM2025
# =============================================
@st.cache_resource(show_spinner="Завантаження WMM2025...")
def download_wmm() -> str:
    if os.path.exists(COF_PATH):
        return COF_PATH
    os.makedirs("wmm", exist_ok=True)
    try:
        r = requests.get(COF_URL, timeout=30)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open("WMM_2025.COF") as src, open(COF_PATH, "wb") as dst:
                dst.write(src.read())
        return COF_PATH
    except Exception as e:
        st.error(f"Помилка завантаження WMM: {e}")
        st.stop()

COF_PATH = download_wmm()

@st.cache_resource
def get_wmm() -> GeoMag:
    return GeoMag(coefficients_file=COF_PATH)

def decimal_year(d: date) -> float:
    start = date(d.year, 1, 1)
    return d.year + (d - start).days / 365.25

# =============================================
# RTDM
# =============================================
@st.cache_data(ttl=1800)
def get_rtdm(lat: float, lon: float, alt: float = 0) -> Dict:
    try:
        params = {"latitude": lat, "longitude": lon, "altitude": alt / 1000, "format": "json"}
        r = requests.get(RTDM_API, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return {
                "decl_rt": data.get("declination", 0) - data.get("quiet_declination", 0),
                "total_rt": data.get("total_intensity", 0) - data.get("quiet_intensity", 0),
                "storm": data.get("storm_level", "quiet")
            }
    except Exception as e:
        st.warning(f"RTDM недоступний: {e}. Використовується WMM.")
    return {"decl_rt": 0.0, "total_rt": 0.0, "storm": "offline"}

# =============================================
# MGRS + UTM
# =============================================
_vec_mgrs = np.vectorize(lambda lat, lon: MGRS_CONV.toMGRS(lat, lon, 5) if -90 <= lat <= 90 and -180 <= lon <= 180 else "Invalid")

@st.cache_data
def batch_mgrs(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    return _vec_mgrs(lats, lons)

@st.cache_data
def batch_utm(lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    zones = np.empty(len(lats), dtype='U4')
    east = np.empty(len(lats))
    north = np.empty(len(lats))
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        if np.isnan(lat) or np.isnan(lon):
            zones[i] = east[i] = north[i] = np.nan
            continue
        zone = int((lon + 180) / 6) + 1
        hemi = 'S' if lat < 0 else 'N'
        zones[i] = f"{zone}{hemi}"
        p = Proj(f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs" + (" +south" if hemi == 'S' else ""))
        e, n = p(lon, lat)
        east[i], north[i] = round(e, 1), round(n, 1)
    return zones, east, north

# =============================================
# ОБЧИСЛЕННЯ
# =============================================
@st.cache_data
def calc_point(lat: float, lon: float, alt: float, year: float) -> Dict:
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return {"error": "Invalid coordinates"}
    wmm = get_wmm().calculate(glat=lat, glon=lon, alt=alt/1000, time=year)
    rt = get_rtdm(lat, lon, alt)
    decl = wmm.d + rt["decl_rt"]
    total = wmm.f + rt["total_rt"]
    mgrs_str = MGRS_CONV.toMGRS(lat, lon, 5)
    utm_z, utm_e, utm_n = batch_utm(np.array([lat]), np.array([lon]))
    return {
        "lat": round(lat, 6), "lon": round(lon, 6), "alt": round(alt, 1),
        "decl": round(decl, 4), "total": round(total, 2),
        "storm": rt["storm"], "mgrs": mgrs_str,
        "utm_zone": utm_z[0], "utm_e": utm_e[0], "utm_n": utm_n[0]
    }

# =============================================
# ВЕКТОРИ
# =============================================
@st.cache_data
def load_vector(file) -> gpd.GeoDataFrame:
    try:
        if file.name.endswith((".geojson", ".json")):
            gdf = gpd.read_file(file)
        elif file.name.endswith(".zip"):
            with zipfile.ZipFile(file) as z:
                shp = [f for f in z.namelist() if f.endswith(".shp")][0]
                gdf = gpd.read_file(z.open(shp))
        else:
            st.error("Формат: .geojson або .zip (shapefile)")
            return gpd.GeoDataFrame()
        gdf = gdf.to_crs(epsg=4326)
        if len(gdf) > 500:
            gdf = gdf.sample(500, random_state=42)
        gdf.geometry = gdf.geometry.simplify(0.001, preserve_topology=True)
        return gdf
    except Exception as e:
        st.error(f"Помилка завантаження: {e}")
        return gpd.GeoDataFrame()

# =============================================
# ІНТЕРФЕЙС
# =============================================
st.title("🧭 Geomagnetic Pro 2025")
st.markdown("**Висота | Теплова карта | Вектори | Клік | MGRS | UTM**")

tabs = st.tabs(["Калькулятор", "Карта", "Пакет"])

# === КАЛЬКУЛЯТОР ===
with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Широта", value=55.7558, format="%.6f", key="calc_lat")
        lon = st.number_input("Довгота", value=37.6173, format="%.6f", key="calc_lon")
    with col2:
        alt = st.number_input("Висота (м)", value=0.0, step=100.0, key="calc_alt")
    if st.button("Обчислити", type="primary"):
        res = calc_point(lat, lon, alt, decimal_year(date.today()))
        if "error" in res:
            st.error(res["error"])
        else:
            st.metric("**MGRS**", res["mgrs"])
            st.metric("**UTM**", f"{res['utm_zone']} {res['utm_e']}E {res['utm_n']}N")
            st.metric("**Деклінація**", f"{res['decl']}°")
            st.metric("**Інтенсивність**", f"{res['total']} nT")
            st.json(res)

# === КАРТА ===
with tabs[1]:
    vector_file = st.file_uploader("Вектор (GeoJSON/Shapefile)", ["geojson", "json", "zip"], key="vec_file")
    gdf = load_vector(vector_file) if vector_file else None

    col1, col2 = st.columns(2)
    with col1:
        alt_grid = st.slider("Висота сітки (м)", 0, 10000, 0, 500, key="alt_grid")
    with col2:
        step = st.slider("Крок (°)", 0.1, 1.0, 0.5, 0.1, key="step_grid")

    cache_key = f"grid_{alt_grid}_{step}"
    if cache_key not in st.session_state:
        lat_min, lat_max = 48.0, 52.0
        lon_min, lon_max = 22.0, 40.0
        lats = np.arange(lat_min, lat_max, step)
        lons = np.arange(lon_min, lon_max, step)
        grid = [(la, lo, alt_grid) for la in lats for lo in lons]
        with st.spinner(f"Генерація теплової карти на {alt_grid} м..."):
            results = [calc_point(*p, decimal_year(date.today())) for p in grid]
        st.session_state[cache_key] = pd.DataFrame(results)

    df_heatmap = st.session_state[cache_key]

    fig = go.Figure()
    fig.add_trace(go.Densitymapbox(
        lat=df_heatmap["lat"], lon=df_heatmap["lon"],
        z=df_heatmap["decl"], radius=20,
        colorscale="RdBu", zmid=0, opacity=0.6,
        colorbar=dict(title="Деклінація (°)")
    ))

    if gdf is not None and not gdf.empty:
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom.geom_type == 'Point':
                fig.add_trace(go.Scattermapbox(lat=[geom.y], lon=[geom.x], mode="markers", marker=dict(color="green", size=8)))
            elif geom.geom_type in ['LineString', 'MultiLineString']:
                lons, lats = geom.xy
                fig.add_trace(go.Scattermapbox(lat=lats, lon=lons, mode="lines", line=dict(color="purple", width=2)))
            elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                for poly in [geom] if geom.geom_type == 'Polygon' else geom.geoms:
                    x, y = poly.exterior.xy
                    fig.add_trace(go.Scattermapbox(lat=y, lon=x, fill="toself", fillcolor="rgba(100,0,100,0.1)", line=dict(color="purple")))

    if "click_points" not in st.session_state:
        st.session_state.click_points = []

    for pt in st.session_state.click_points:
        fig.add_trace(go.Scattermapbox(lat=[pt["lat"]], lon=[pt["lon"]], mode="markers", marker=dict(color="red", size=12)))

    fig.update_layout(mapbox_style="open-street-map", mapbox=dict(center=dict(lat=50.0, lon=30.0), zoom=5), height=600, margin=dict(l=0,r=0,b=0,t=0))

    clicked = st.plotly_chart(fig, use_container_width=True, key="map_final")

    if clicked and clicked["data"]:
        for trace in clicked["data"]:
            if "lat" in trace and len(trace["lat"]) > 0:
                lat, lon = trace["lat"][0], trace["lon"][0]
                click_alt = st.number_input("Висота кліку (м)", value=alt_grid, key=f"click_alt_{len(st.session_state.click_points)}")
                res = calc_point(lat, lon, click_alt, decimal_year(date.today()))
                st.session_state.click_points.append(res)
                st.rerun()

    if st.session_state.click_points:
        df_click = pd.DataFrame(st.session_state.click_points)
        st.dataframe(df_click[["lat", "lon", "alt", "mgrs", "decl", "total"]], use_container_width=True)
        if st.button("Очистити"):
            st.session_state.click_points = []
            st.rerun()

# === ПАКЕТ ===
with tabs[2]:
    file = st.file_uploader("CSV (lat,lon,alt)", ["csv"], key="pkg_file")
    if file:
        df = pd.read_csv(file)
        required = ["lat", "lon"]
        if not all(col in df.columns for col in required):
            st.error("CSV: потрібні колонки lat, lon")
            st.stop()
        df["alt"] = df.get("alt", 0).fillna(0)
        points = [(row["lat"], row["lon"], row["alt"]) for _, row in df.iterrows()]
        with st.spinner("Обчислення..."):
            results = [calc_point(*p, decimal_year(date.today())) for p in points]
        df_out = pd.DataFrame(results)
        st.dataframe(df_out, use_container_width=True)
        st.download_button("Завантажити CSV", df_out.to_csv(index=False).encode(), "geomag_final.csv", "text/csv")