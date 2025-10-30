# =============================================
# 🧭 Geomagnetic Pro 2025 — Фінальна стабільна версія (оптимізовано)
# =============================================

from __future__ import annotations
import os, io, zipfile, requests
from datetime import date, datetime
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Proj
from pygeomag import GeoMag
import mgrs
import plotly.graph_objects as go
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
# ЗАВАНТАЖЕННЯ WMM
# =============================================
@st.cache_resource(show_spinner="Завантаження WMM2025...")
def download_wmm() -> str:
    os.makedirs("wmm", exist_ok=True)
    if os.path.exists(COF_PATH):
        return COF_PATH
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
# REAL-TIME DATA MODEL (RTDM)
# =============================================
@st.cache_data(ttl=1800)
def get_rtdm(lat: float, lon: float, alt: float = 0) -> Dict:
    try:
        params = {"latitude": lat, "longitude": lon, "altitude": alt / 1000, "format": "json"}
        r = requests.get(RTDM_API, params=params, timeout=10)
        if r.status_code == 200:
            d = r.json()
            return {
                "decl_rt": d.get("declination", 0) - d.get("quiet_declination", 0),
                "total_rt": d.get("total_intensity", 0) - d.get("quiet_intensity", 0),
                "storm": d.get("storm_level", "quiet")
            }
    except Exception as e:
        st.warning(f"RTDM недоступний: {e}. Використовується WMM.")
    return {"decl_rt": 0.0, "total_rt": 0.0, "storm": "offline"}

# =============================================
# MGRS + UTM
# =============================================
_vec_mgrs = np.vectorize(lambda lat, lon: MGRS_CONV.toMGRS(lat, lon, 5)
                         if -90 <= lat <= 90 and -180 <= lon <= 180 else "Invalid")

@st.cache_data
def batch_mgrs(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    return _vec_mgrs(lats, lons)

@st.cache_data
def batch_utm(lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    zones, east, north = [], [], []
    for lat, lon in zip(lats, lons):
        if np.isnan(lat) or np.isnan(lon):
            zones.append(np.nan); east.append(np.nan); north.append(np.nan); continue
        zone = int((lon + 180) / 6) + 1
        hemi = 'S' if lat < 0 else 'N'
        p = Proj(f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs" + (" +south" if hemi == 'S' else ""))
        e, n = p(lon, lat)
        zones.append(f"{zone}{hemi}"); east.append(round(e, 1)); north.append(round(n, 1))
    return np.array(zones), np.array(east), np.array(north)

# =============================================
# ОБЧИСЛЕННЯ
# =============================================
@st.cache_data
def calc_point(lat: float, lon: float, alt: float, year: float) -> Dict:
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return {"error": "Invalid coordinates"}
    wmm = get_wmm().calculate(glat=lat, glon=lon, alt=alt/1000, time=year)
    rt = get_rtdm(lat, lon, alt)
    decl, total = wmm.d + rt["decl_rt"], wmm.f + rt["total_rt"]
    mgrs_str = MGRS_CONV.toMGRS(lat, lon, 5)
    utm_z, utm_e, utm_n = batch_utm(np.array([lat]), np.array([lon]))
    return {
        "lat": round(lat, 6), "lon": round(lon, 6), "alt": round(alt, 1),
        "decl": round(decl, 4), "total": round(total, 2),
        "storm": rt["storm"], "mgrs": mgrs_str,
        "utm_zone": utm_z[0], "utm_e": utm_e[0], "utm_n": utm_n[0]
    }

# =============================================
# ЗАВАНТАЖЕННЯ ВЕКТОРНИХ ДАНИХ
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
    lat = col1.number_input("Широта", value=55,819666, format="%.6f")
    lon = col1.number_input("Довгота", value=37,611481, format="%.6f")
    alt = col2.number_input("Висота (м)", value=20.0, step=5.0)

    if st.button("Обчислити", type="primary"):
        res = calc_point(lat, lon, alt, decimal_year(date.today()))
        if "error" in res:
            st.error(res["error"])
        else:
            st.metric("MGRS", res["mgrs"])
            st.metric("UTM", f"{res['utm_zone']} {res['utm_e']}E {res['utm_n']}N")
            st.metric("Деклінація", f"{res['decl']}°")
            st.metric("Інтенсивність", f"{res['total']} nT")
            st.json(res)

# === КАРТА ===
with tabs[1]:
    vector_file = st.file_uploader("Вектор (GeoJSON/Shapefile)", ["geojson", "json", "zip"])
    gdf = load_vector(vector_file) if vector_file else None

    col1, col2 = st.columns(2)
    alt_grid = col1.slider("Висота сітки (м)", 0, 10000, 0, 100)
    step = col2.slider("Крок (°)", 0.1, 1.0, 0.5, 0.1)

    cache_key = f"grid_{alt_grid}_{step}"
    if cache_key not in st.session_state:
        lat_min, lat_max = 48.0, 52.0
        lon_min, lon_max = 22.0, 40.0
        lats, lons = np.arange(lat_min, lat_max, step), np.arange(lon_min, lon_max, step)
        grid = [(la, lo, alt_grid) for la in lats for lo in lons]
        with st.spinner(f"Генерація теплової карти на {alt_grid} м..."):
            results = [calc_point(*p, decimal_year(date.today())) for p in grid]
        st.session_state[cache_key] = pd.DataFrame(results)

    df_heatmap = st.session_state[cache_key]

    fig = go.Figure()
    fig.add_trace(go.Densitymap(
        lat=df_heatmap["lat"], lon=df_heatmap["lon"], z=df_heatmap["decl"],
        radius=40, colorscale="RdBu", zmid=0, opacity=0.8,
        colorbar=dict(title="Деклінація (°)")
    ))

    if gdf is not None and not gdf.empty:
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom.geom_type == 'Point':
                fig.add_trace(go.Scattermapbox(lat=[geom.y], lon=[geom.x],
                                               mode="markers", marker=dict(color="green", size=8)))
            elif geom.geom_type in ['LineString', 'MultiLineString']:
                x, y = geom.xy
                fig.add_trace(go.Scattermapbox(lat=y, lon=x, mode="lines",
                                               line=dict(color="purple", width=2)))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=50.0, lon=30.0), zoom=5),
        height=600, margin=dict(l=0, r=0, b=0, t=0)
    )

    st.plotly_chart(fig, width="stretch", key="map_final")

# === ПАКЕТ ===
with tabs[2]:
    file = st.file_uploader("CSV (lat,lon,alt)", ["csv"])
    if file:
        df = pd.read_csv(file)
        if not all(c in df.columns for c in ["lat", "lon"]):
            st.error("CSV: потрібні колонки lat, lon")
            st.stop()
        df["alt"] = df.get("alt", 0).fillna(0)
        pts = [(r["lat"], r["lon"], r["alt"]) for _, r in df.iterrows()]
        with st.spinner("Обчислення..."):
            res = [calc_point(*p, decimal_year(date.today())) for p in pts]
        df_out = pd.DataFrame(res)
        st.dataframe(df_out, width="stretch")
        st.download_button("⬇️ Завантажити CSV", df_out.to_csv(index=False).encode(),
                           "geomag_final.csv", "text/csv")