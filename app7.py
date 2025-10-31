"""
Geomagnetic Pro 2025 — Професійний UI/UX | Точність 1 м | Автовисота | WGS84
"""
from __future__ import annotations

import os
import io
import zipfile
from datetime import date
from concurrent.futures import ThreadPoolExecutor
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
# СТИЛЬ ТА КОНФІГУРАЦІЯ
# =============================================
st.set_page_config(
    page_title="Geomagnetic Pro 2025",
    page_icon="compass",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомний CSS
st.markdown("""
<style>
    .main > div {padding-top: 2rem;}
    .stMetric {font-weight: bold; color: #1E90FF;}
    .stSuccess {background: #e6f7ff; border-left: 5px solid #1890ff;}
    .stInfo {background: #f0f2f6; border-left: 5px solid #909399;}
    .stButton>button {background: #1E90FF; color: white; border-radius: 8px;}
    .stTextInput>div>input {border-radius: 8px;}
    .stNumberInput>div>input {border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

# =============================================
# ДАТУМ ТА КОНСТАНТИ
# =============================================
DATUM = "WGS84"
MGRS_CONV = mgrs.MGRS()
COF_PATH = "wmm/WMM_2025.COF"
COF_URL = "https://www.ncei.noaa.gov/data/world-magnetic-model-2025/full/WMM_COEFS-2025.zip"
RTDM_API = "https://geomag.usgs.gov/ws/edge/"

# =============================================
# ЗАВАНТАЖЕННЯ WMM2025
# =============================================
@st.cache_resource(show_spinner="Завантаження WMM2025...")
def download_wmm() -> str:
    if os.path.exists(COF_PATH) and os.path.getsize(COF_PATH) > 1000:
        return COF_PATH
    os.makedirs("wmm", exist_ok=True)
    try:
        r = requests.get(COF_URL, timeout=30)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            cof_files = [n for n in z.namelist() if n.lower().endswith('.cof')]
            if not cof_files:
                raise FileNotFoundError("COF файл не знайдено")
            with z.open(cof_files[0]) as src, open(COF_PATH, "wb") as dst:
                dst.write(src.read())
        return COF_PATH
    except Exception as e:
        st.error(f"Помилка завантаження WMM2025: {e}")
        st.info("Завантажте вручну: [NOAA WMM2025](https://www.ncei.noaa.gov/products/world-magnetic-model/wmm-coefficients)")
        st.stop()

COF_PATH = download_wmm()

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
# RTDM
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
    except:
        pass
    return {"decl_rt": 0.0, "total_rt": 0.0, "storm": "offline"}
# =============================================
# АВТОВИСОТА (DEM)
# =============================================
@st.cache_data(ttl=3600)
def get_elevation(lat: float, lon: float) -> float:
    try:
        url = f"https://api.opentopodata.org/v1/srtm30m?locations={lat},{lon}"
        r = requests.get(url, timeout=10).json()
        elev = r["results"][0]["elevation"]
        return round(elev, 1) if elev is not None else 0.0
    except:
        return 0.0

# =============================================
# ТОЧНІ КОНВЕРТОРИ
# =============================================
@st.cache_data
def latlon_to_mgrs(lat: float, lon: float, precision: int = 5) -> str:
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return "Invalid"
    return MGRS_CONV.toMGRS(lat, lon, precision)

@st.cache_data
def mgrs_to_latlon(mgrs_str: str) -> Tuple[float | None, float | None, int]:
    try:
        mgrs_str = mgrs_str.strip().upper()
        digits = ''.join(c for c in mgrs_str if c.isdigit())
        if len(digits) % 2 != 0 or len(digits) > 10:
            return None, None, 0
        precision = len(digits) // 2
        if precision < 1 or precision > 5:
            return None, None, 0
        lat, lon = MGRS_CONV.fromMGRS(mgrs_str)
        return round(lat, 8), round(lon, 8), precision
    except:
        return None, None, 0

@st.cache_data
def latlon_to_utm(lat: float, lon: float) -> Tuple[str, float, float]:
    zone = int((lon + 180) / 6) + 1
    hemi = 'S' if lat < 0 else 'N'
    p = Proj(f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs" + (" +south" if hemi == 'S' else ""))
    e, n = p(lon, lat)
    return f"{zone}{hemi}", round(e, 3), round(n, 3)

@st.cache_data
def utm_to_latlon(zone: int, east: float, north: float, south: bool = False) -> Tuple[float, float]:
    p = Proj(f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs" + (" +south" if south else ""))
    lon, lat = p(east, north, inverse=True)
    return round(lat, 8), round(lon, 8)

# =============================================
# ОБЧИСЛЕННЯ
# =============================================
@st.cache_data
def calc_point(lat: float, lon: float, alt: float, year: float) -> Dict:
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return {"error": "Invalid coordinates"}
    final_alt = alt if alt > 0 else get_elevation(lat, lon)
    wmm = get_wmm().calculate(glat=lat, glon=lon, alt=final_alt/1000, time=year)
    rt = get_rtdm(lat, lon, final_alt)
    decl = wmm.d + rt["decl_rt"]
    total = wmm.f + rt["total_rt"]
    mgrs_str = latlon_to_mgrs(lat, lon)
    utm_zone, utm_e, utm_n = latlon_to_utm(lat, lon)
    return {
        "datum": DATUM,
        "lat": round(lat, 8), "lon": round(lon, 8), "alt": round(final_alt, 1),
        "decl": round(decl, 4), "total": round(total, 2),
        "storm": rt["storm"], "mgrs": mgrs_str,
        "utm_zone": utm_zone, "utm_e": utm_e, "utm_n": utm_n
    }
def calc_batch(points: List[Tuple[float, float, float]], year: float) -> List[Dict]:
    with ThreadPoolExecutor() as executor:
        return list(executor.map(lambda p: calc_point(*p, year), points))

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
                shp = next((f for f in z.namelist() if f.endswith(".shp")), None)
                if not shp:
                    st.error("Shapefile не знайдено")
                    return gpd.GeoDataFrame()
                gdf = gpd.read_file(z.open(shp))
        else:
            st.error("Формат: .geojson або .zip")
            return gpd.GeoDataFrame()
        gdf = gdf.to_crs(epsg=4326)
        if len(gdf) > 500:
            gdf = gdf.sample(500, random_state=42)
        gdf.geometry = gdf.geometry.simplify(0.001)
        return gdf
    except Exception as e:
        st.error(f"Помилка: {e}")
        return gpd.GeoDataFrame()

# =============================================
# ІНТЕРФЕЙС
# =============================================
st.title("Geomagnetic Pro 2025")
st.markdown(f"**Датум: {DATUM}** | Точність 1 м | Автовисота | WMM2025 + RTDM")

tab_conv, tab_calc, tab_map, tab_pkg = st.tabs(["Конвертер", "Калькулятор", "Карта", "Пакет"])

# === КОНВЕРТЕР ===
with tab_conv:
    st.subheader("Точний конвертер координат")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**MGRS → Lat/Lon**")
        mgrs_input = st.text_input("MGRS", "36TWN1234567890", help="5 пар = 1 м")
        if st.button("Конвертувати", key="mgrs_to_ll"):
            lat, lon, prec = mgrs_to_latlon(mgrs_input)
            if lat:
                st.success(f"**Lat:** {lat}° | **Lon:** {lon}°")
                st.info(f"**Точність:** {10**(5-prec)} м")
            else:
                st.error("Невірний MGRS")

    with col2:
        st.markdown("**Lat/Lon → MGRS**")
        lat_in = st.number_input("Широта", value=50.4501, format="%.8f", key="ll_to_mgrs_lat")
        lon_in = st.number_input("Довгота", value=30.5234, format="%.8f", key="ll_to_mgrs_lon")
        precision = st.select_slider("Точність", [1,2,3,4,5], 5,
                                    format_func=lambda x: f"{10**(5-x)} м")
        if st.button("Конвертувати", key="ll_to_mgrs"):
            mgrs_out = latlon_to_mgrs(lat_in, lon_in, precision)
            st.success(f"**MGRS:** `{mgrs_out}`")
            st.info(f"**Точність:** {10**(5-precision)} м")

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**UTM → Lat/Lon**")
        zone_in = st.number_input("Зона", 1, 60, 36, key="utm_zone")
        east_in = st.number_input("Easting (м)", value=500000.000, format="%.3f", key="utm_e")
        north_in = st.number_input("Northing (м)", value=5500000.000, format="%.3f", key="utm_n")
        south_hem = st.checkbox("Південна півкуля", key="utm_south")
        if st.button("Конвертувати", key="utm_to_ll"):
            lat, lon = utm_to_latlon(zone_in, east_in, north_in, south_hem)
            st.success(f"**Lat:** {lat}° | **Lon:** {lon}°")
            st.info("**Точність:** ~0.1 м")

    with col4:
        st.markdown("**Lat/Lon → UTM**")
        lat_u = st.number_input("Широта", value=50.4501, format="%.8f", key="ll_to_utm_lat")
        lon_u = st.number_input("Довгота", value=30.5234, format="%.8f", key="ll_to_utm_lon")
        if st.button("Конвертувати", key="ll_to_utm"):
            zone, e, n = latlon_to_utm(lat_u, lon_u)
            st.success(f"**Зона:** {zone} | **E:** {e} м | **N:** {n} м")
            st.info("**Точність:** ~1 мм")

# === КАЛЬКУЛЯТОР ===
with tab_calc:
    st.subheader("Калькулятор магнітного поля")
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Широта", value=50.4501, format="%.8f", key="calc_lat")
        lon = st.number_input("Довгота", value=30.5234, format="%.8f", key="calc_lon")
    with col2:
        auto_alt = st.checkbox("Автоматична висота (DEM)", value=True)
        if auto_alt:
            with st.spinner("Визначення висоти..."):
                elevation = get_elevation(lat, lon)
            alt = elevation
            st.info(f"Висота над рівнем моря: **{elevation} м**")
        else:
            alt = st.number_input("Висота (м)", value=0.0, step=100.0, key="manual_alt")

    if st.button("Обчислити", type="primary"):
        res = calc_point(lat, lon, alt, decimal_year(date.today()))
        if "error" in res:
            st.error(res["error"])
        else:
            cols = st.columns(3)
            with cols[0]:
                st.metric("MGRS", res["mgrs"])
            with cols[1]:
                st.metric("UTM", f"{res['utm_zone']} {res['utm_e']}E")
            with cols[2]:
                st.metric("Деклінація", f"{res['decl']}°")
            st.metric("Інтенсивність", f"{res['total']} nT")
            st.json(res)

# === КАРТА ===
with tab_map:
    st.subheader("Теплова карта деклінації — Росія")

    # === НАЛАШТУВАННЯ СІТКИ ===
    col1, col2 = st.columns(2)
    with col1:
        alt_min = st.slider("Мін. висота (м)", 0, 10000, 500, 500, key="alt_min")
        alt_max = st.slider("Макс. висота (м)", 0, 10000, 5000, 500, key="alt_max")
        alt_step = st.slider("Крок висоти (м)", 100, 2000, 500, 100, key="alt_step")
    with col2:
        lat_step = st.slider("Крок широти (°)", 0.1, 2.0, 1.0, 0.1, key="lat_step")
        lon_step = st.slider("Крок довготи (°)", 0.1, 2.0, 1.0, 0.1, key="lon_step")

    # === СІТКА ПО РОСІЇ ===
    cache_key = f"grid_ru_{alt_min}_{alt_max}_{alt_step}_{lat_step}_{lon_step}"
    if cache_key not in st.session_state:
        lats = np.arange(41.0, 82.0, lat_step)
        lons = np.arange(19.0, 180.0, lon_step)
        alts = np.arange(alt_min, alt_max + alt_step, alt_step)
        grid = [(la, lo, alt) for la in lats for lo in lons for alt in alts]
        with st.spinner("Генерація теплової карти (Росія)..."):
            results = calc_batch(grid, decimal_year(date.today()))
        st.session_state[cache_key] = pd.DataFrame(results)

    df_heatmap = st.session_state[cache_key]

    # === ФІГУРА ===
    fig = go.Figure()
    fig.add_trace(go.Densitymap(
        lat=df_heatmap["lat"], lon=df_heatmap["lon"],
        z=df_heatmap["decl"], radius=15,
        colorscale="RdBu", zmid=0, opacity=0.7,
        colorbar=dict(title="Деклінація (°)")
    ))

    # Вектори
    vector_file = st.file_uploader("Вектор (GeoJSON/Shapefile)", ["geojson", "json", "zip"], key="vector_map")
    gdf = load_vector(vector_file) if vector_file else None
    if gdf is not None and not gdf.empty:
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom.geom_type == 'Point':
                fig.add_trace(go.Scattermapbox(lat=[geom.y], lon=[geom.x], mode="markers", marker=dict(color="green", size=8)))
            elif geom.geom_type in ['LineString', 'MultiLineString']:
                lons, lats = geom.xy
                fig.add_trace(go.Scattermapbox(lat=lats, lon=lons, mode="lines", line=dict(color="purple", width=2)))

    # === ІНІЦІАЛІЗАЦІЯ ІСТОРІЇ ===
    if "history_points" not in st.session_state:
        st.session_state.history_points = []

    # === ДОДАВАННЯ ТОЧОК ===
    def add_point(lat: float, lon: float, alt: float, source: str):
        res = calc_point(lat, lon, alt, decimal_year(date.today()))
        res["source"] = source
        st.session_state.history_points.append(res)
        st.session_state.click_points.append(res)
        # Автоцентрування
        fig.update_layout(mapbox=dict(center=dict(lat=lat, lon=lon), zoom=10))

    # === ВВЕДЕННЯ ТОЧКИ ЧЕРЕЗ MGRS/UTM ===
    st.divider()
    st.subheader("Додати точку через MGRS або UTM")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Через MGRS**")
        mgrs_input = st.text_input("MGRS", "49TDG1234567890", key="mgrs_input")
        mgrs_prec = st.select_slider("Точність", [1,2,3,4,5], 5, format_func=lambda x: f"{10**(5-x)} м", key="mgrs_prec")
        if st.button("Додати точку", key="add_mgrs"):
            lat, lon, _ = mgrs_to_latlon(mgrs_input)
            if lat:
                elev = get_elevation(lat, lon)
                add_point(lat, lon, elev, "MGRS")
                st.success(f"Додано: {lat:.6f}°, {lon:.6f}° | Висота: {elev:.1f} м")
                st.rerun()
            else:
                st.error("Невірний MGRS")

    with col_b:
        st.markdown("**Через UTM**")
        zone_in = st.number_input("Зона", 1, 60, 49, key="utm_zone_in")
        east_in = st.number_input("Easting (м)", value=300000.000, format="%.3f", key="utm_e_in")
        north_in = st.number_input("Northing (м)", value=6500000.000, format="%.3f", key="utm_n_in")
        south_hem = st.checkbox("Південна півкуля", key="utm_south_in")
        if st.button("Додати точку", key="add_utm"):
            lat, lon = utm_to_latlon(zone_in, east_in, north_in, south_hem)
            elev = get_elevation(lat, lon)
            add_point(lat, lon, elev, "UTM")
            st.success(f"Додано: {lat:.6f}°, {lon:.6f}° | Висота: {elev:.1f} м")
            st.rerun()

    # === КЛІК ПО КАРТІ ===
    if st.session_state.get("plotly_events"):
        event = st.session_state.plotly_events[0]
        if event["type"] == "click":
            lat, lon = event["points"][0]["lat"], event["points"][0]["lon"]
            elev = get_elevation(lat, lon)
            add_point(lat, lon, elev, "Клік")
            st.rerun()

    # === ВІДОБРАЖЕННЯ ТОЧОК НА КАРТІ ===
    for pt in st.session_state.history_points:
        color = {"Клік": "red", "MGRS": "blue", "UTM": "orange"}.get(pt["source"], "black")
        fig.add_trace(go.Scattermapbox(
            lat=[pt["lat"]], lon=[pt["lon"]],
            mode="markers",
            marker=dict(color=color, size=12),
            text=f"{pt['mgrs']}<br>{pt['decl']}°<br>{pt['alt']} м",
            hoverinfo="text"
        ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=61.0, lon=100.0), zoom=3),
        height=600, margin=dict(l=0,r=0,b=0,t=0)
    )

    st.plotly_chart(fig, width="stretch", key="map_ru", on_select="rerun")

    # === РОЗДІЛ ІСТОРІЯ ===
    st.divider()
    st.subheader("Історія точок")

    if st.session_state.history_points:
        df_history = pd.DataFrame(st.session_state.history_points)
        df_display = df_history[["source", "lat", "lon", "alt", "mgrs", "utm_zone", "decl", "total"]].copy()
        df_display.columns = ["Джерело", "Широта", "Довгота", "Висота (м)", "MGRS", "UTM Зона", "Деклінація (°)", "Інтенсивність (nT)"]

        st.dataframe(
            df_display,
            use_container_width=True,
            column_config={
                "Широта": st.column_config.NumberColumn(format="%.6f"),
                "Довгота": st.column_config.NumberColumn(format="%.6f"),
                "Висота (м)": st.column_config.NumberColumn(format="%.1f"),
                "Деклінація (°)": st.column_config.NumberColumn(format="%.3f"),
                "Інтенсивність (nT)": st.column_config.NumberColumn(format="%.0f"),
            }
        )

        # ЕКСПОРТ
        csv = df_history.to_csv(index=False).encode()
        st.download_button(
            label="Експорт історії в CSV",
            data=csv,
            file_name=f"geomag_history_{date.today()}.csv",
            mime="text/csv"
        )

        # ОЧИСТКА
        if st.button("Очистити історію та карту", type="secondary"):
            st.session_state.history_points = []
            st.session_state.click_points = []
            st.rerun()
    else:
        st.info("Історія порожня. Додайте точки через клік, MGRS або UTM.")
# === ПАКЕТ ===
with tab_pkg:
    st.subheader("Пакетне обчислення")
    file = st.file_uploader("CSV (lat,lon,alt)", ["csv"])
    if file:
        df = pd.read_csv(file)
        df["alt"] = df.get("alt", 0).fillna(0)
        points = [(r["lat"], r["lon"], r["alt"]) for _, r in df.iterrows()]
        with st.spinner("Обчислення..."):
            results = calc_batch(points, decimal_year(date.today()))
        df_out = pd.DataFrame(results)
        st.dataframe(df_out, use_container_width=True)
        st.download_button("Експорт CSV", df_out.to_csv(index=False).encode(), "geomag_results.csv", "text/csv")
