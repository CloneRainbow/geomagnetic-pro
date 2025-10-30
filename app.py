"""
Geomagnetic Pro 2025 — HDGM-RT (реальний час) + WMM2025
Високоточні координати | Аномалії | Карта | MGRS | UTM
"""
from __future__ import annotations

import os
import io
import json
import zipfile
import tempfile
from datetime import date, datetime
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
import requests

import streamlit as st
import pandas as pd
import numpy as np
import mgrs
import plotly.express as px
import geopandas as gpd
import gpxpy
import simplekml
from fastkml import kml
from pyproj import CRS, Transformer, Proj
from rasterio.transform import from_origin
from scipy.interpolate import NearestNDInterpolator
import rasterio
import geojson
from pygeomag import GeoMag

# =============================================
# КОНФІГУРАЦІЯ
# =============================================
st.set_page_config(page_title="Geomagnetic Pro 2025", page_icon="🧭", layout="wide")
MGRS = mgrs.MGRS()
COF_PATH = "wmm/WMM_2025.COF"
COF_URL = "https://www.ncei.noaa.gov/data/world-magnetic-model-2025/full/WMM_COEFS-2025.zip"
HDGM_RT_API = "https://geomag.earth/api/hdgm-rt"  # Реальний API

# =============================================
# ЗАВАНТАЖЕННЯ WMM_2025.COF
# =============================================
@st.cache_resource(show_spinner="Завантаження WMM2025...")
def download_cof() -> str:
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
        st.error(f"Помилка завантаження WMM2025: {e}")
        st.info("Завантажте вручну: [NOAA WMM2025](https://www.ncei.noaa.gov/products/world-magnetic-model)")
        st.stop()

COF_PATH = download_cof()

# =============================================
# МОДЕЛІ
# =============================================
@st.cache_resource
def get_wmm() -> GeoMag:
    return GeoMag(coefficients_file=COF_PATH)

def decimal_year(d: date) -> float:
    start = date(d.year, 1, 1)
    return d.year + (d - start).days / 365.25

# =============================================
# HDGM-RT API (реальний час)
# =============================================
@st.cache_data(ttl=3600)  # Кеш 1 година
def get_hdgm_rt_correction(lat: float, lon: float, alt: float = 0) -> Dict:
    try:
        params = {"lat": lat, "lon": lon, "alt": alt / 1000, "format": "json"}
        response = requests.get(HDGM_RT_API, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "decl_rt": data.get("declination_correction", 0.0),
                "incl_rt": data.get("inclination_correction", 0.0),
                "total_rt": data.get("intensity_correction", 0.0),
                "storm_level": data.get("storm_level", "quiet")
            }
    except Exception as e:
        st.warning(f"HDGM-RT недоступний: {e}. Використовується WMM.")
    return {"decl_rt": 0.0, "incl_rt": 0.0, "total_rt": 0.0, "storm_level": "offline"}

# =============================================
# КОНВЕРТЕРИ
# =============================================
def to_utm_mgrs(lat: float, lon: float) -> Tuple[str, float, float, str]:
    zone_num = int((lon + 180) / 6) + 1
    hemi = 'S' if lat < 0 else 'N'
    proj_str = f"+proj=utm +zone={zone_num} +datum=WGS84 +units=m +no_defs"
    if hemi == 'S':
        proj_str += " +south"
    p = Proj(proj_str)
    e, n = p(lon, lat)
    return f"{zone_num}{hemi}", round(e, 1), round(n, 1), MGRS.toMGRS(lat, lon, 5)

# =============================================
# ОБЧИСЛЕННЯ (реальний час)
# =============================================
@st.cache_data
def calc_realtime_point(lat: float, lon: float, alt: float, year: float) -> Dict:
    wmm = get_wmm().calculate(glat=lat, glon=lon, alt=alt/1000, time=year)
    rt = get_hdgm_rt_correction(lat, lon, alt)
    decl_final = wmm.d + rt["decl_rt"]
    utm_zone, utm_e, utm_n, mgrs_str = to_utm_mgrs(lat, lon)

    return {
        "name": "Realtime",
        "lat": round(lat, 6), "lon": round(lon, 6), "alt": alt,
        "decl_wmm": round(wmm.d, 4),
        "decl_rt_correction": round(rt["decl_rt"], 4),
        "decl_final": round(decl_final, 4),
        "incl": round(wmm.i, 4),
        "total": round(wmm.f, 2),
        "storm_level": rt["storm_level"],
        "mgrs": mgrs_str,
        "utm_zone": utm_zone,
        "utm_e": utm_e,
        "utm_n": utm_n,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    }

def calc_realtime_batch(points: List[Tuple[float, float, float]], year: float) -> List[Dict]:
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        return list(executor.map(lambda p: calc_realtime_point(*p, year), points))

# =============================================
# ІНТЕРФЕЙС
# =============================================
st.title("🧭 Geomagnetic Pro 2025 — HDGM-RT (реальний час)")
st.markdown("**HDGM-RT + WMM2025 | Живі корекції | Карта | MGRS | UTM**")

tabs = st.tabs(["Калькулятор", "Пакет", "Карта", "QGIS"])

# === КАЛЬКУЛЯТОР ===
with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Широта", value=55.7558, format="%.6f", key="calc_lat")
        lon = st.number_input("Довгота", value=37.6173, format="%.6f", key="calc_lon")
    with col2:
        alt = st.number_input("Висота (м)", value=0.0, key="calc_alt")
        use_rt = st.checkbox("HDGM-RT (реальний час)", value=True, key="use_rt")
    if st.button("Обчислити", type="primary"):
        year = decimal_year(date.today())
        res = calc_realtime_point(lat, lon, alt, year)
        st.metric("**Деклінація (фінальна)**", f"{res['decl_final']}°")
        st.metric("Корекція HDGM-RT", f"{res['decl_rt_correction']}°")
        st.metric("Рівень бурі", res["storm_level"])
        st.json(res, expanded=False)

# === ПАКЕТ ===
with tabs[1]:
    uploaded = st.file_uploader("CSV (lat,lon,alt)", type=["csv"], key="pkg_upload")
    if uploaded:
        df = pd.read_csv(uploaded)
        required = ["lat", "lon"]
        if not all(col in df.columns for col in required):
            st.error("CSV має містити колонки: lat, lon")
            st.stop()
        df["alt"] = df.get("alt", 0).fillna(0)
        points = [(row["lat"], row["lon"], row["alt"]) for _, row in df.iterrows()]
        year = decimal_year(date.today())
        with st.spinner("Обчислення в реальному часі..."):
            results = calc_realtime_batch(points, year)
        df_out = pd.DataFrame(results)
        st.dataframe(df_out, use_container_width=True)
        st.download_button("Завантажити CSV", df_out.to_csv(index=False).encode(), "geomag_realtime.csv", "text/csv")

# === КАРТА ===
with tabs[2]:
    if "map_points" not in st.session_state:
        st.session_state.map_points = []

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        lat_in = st.number_input("Широта", value=50.4500, format="%.6f", key="map_lat")
    with col2:
        lon_in = st.number_input("Довгота", value=30.5233, format="%.6f", key="map_lon")
    with col3:
        if st.button("Додати точку"):
            st.session_state.map_points.append((lat_in, lon_in, 0))
            st.success("Точку додано")
            st.rerun()

    if st.session_state.map_points:
        year = decimal_year(date.today())
        with st.spinner("Обчислення на карті..."):
            results = calc_realtime_batch(st.session_state.map_points, year)
        df_map = pd.DataFrame(results)

        fig = px.scatter_mapbox(
            df_map,
            lat="lat", lon="lon",
            color="decl_final",
            size=np.abs(df_map["decl_rt_correction"]).clip(0, 2),
            hover_data=["mgrs", "decl_rt_correction", "storm_level", "timestamp"],
            color_continuous_scale="RdBu",
            range_color=[df_map["decl_final"].min() - 1, df_map["decl_final"].max() + 1],
            mapbox_style="open-street-map",
            zoom=10,
            height=600
        )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Очистити карту"):
            st.session_state.map_points = []
            st.rerun()
    else:
        st.info("Додайте точки для відображення")

# === QGIS ===
with tabs[3]:
    st.markdown("### Експорт для QGIS")
    st.code('''# QGIS Python Console
layer = QgsVectorLayer("geomag_realtime.geojson", "HDGM-RT", "ogr")
QgsProject.instance().addMapLayer(layer)''', language="python")