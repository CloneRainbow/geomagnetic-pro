# =============================================
# 🧭 Geomagnetic Pro 2025 — Виправлена версія
# =============================================

from __future__ import annotations
import os, io, zipfile, requests
from datetime import date, datetime
from typing import Dict, Tuple, List, Optional
import json

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Proj, Transformer
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
def safe_mgrs_conversion(lat: float, lon: float) -> str:
    """Безпечна конвертація в MGRS з валідацією"""
    try:
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return MGRS_CONV.toMGRS(lat, lon, 5)
        else:
            return "Invalid coordinates"
    except Exception as e:
        return f"Error: {str(e)}"

_vec_mgrs = np.vectorize(safe_mgrs_conversion)

@st.cache_data
def batch_mgrs(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    return _vec_mgrs(lats, lons)

@st.cache_data
def batch_utm(lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    zones, east, north = [], [], []
    for lat, lon in zip(lats, lons):
        if np.isnan(lat) or np.isnan(lon) or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            zones.append("Invalid")
            east.append(np.nan)
            north.append(np.nan)
            continue
        
        try:
            zone = int((lon + 180) / 6) + 1
            hemi = 'S' if lat < 0 else 'N'
            proj_string = f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs"
            if hemi == 'S':
                proj_string += " +south"
            
            transformer = Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)
            e, n = transformer.transform(lon, lat)
            zones.append(f"{zone}{hemi}")
            east.append(round(e, 1))
            north.append(round(n, 1))
        except Exception as e:
            zones.append(f"Error: {str(e)}")
            east.append(np.nan)
            north.append(np.nan)
    
    return np.array(zones), np.array(east), np.array(north)

# =============================================
# РОЗШИРЕНІ ФУНКЦІЇ КООРДИНАТ
# =============================================
def get_mgrs_details(mgrs_str: str) -> Dict:
    """Детальна інформація MGRS"""
    try:
        if not mgrs_str or "Error" in mgrs_str or "Invalid" in mgrs_str:
            return {"full": mgrs_str}
        
        # Базова валідація MGRS формату
        if len(mgrs_str) < 5:
            return {"full": mgrs_str}
            
        # Спрощений парсинг MGRS
        zone = mgrs_str[:2] if mgrs_str[:2].isdigit() else ""
        band = mgrs_str[2] if len(mgrs_str) > 2 else ""
        square = mgrs_str[3:5] if len(mgrs_str) > 4 else ""
        
        return {
            "zone": zone,
            "band": band,
            "square": square,
            "full": mgrs_str
        }
    except Exception:
        return {"full": mgrs_str}

def convert_mgrs_to_latlon(mgrs_str: str) -> Tuple[Optional[float], Optional[float]]:
    """Конвертація MGRS в географічні координати"""
    try:
        if not mgrs_str or len(mgrs_str) < 5:
            return None, None
        lat, lon = MGRS_CONV.toLatLon(mgrs_str)
        return float(lat), float(lon)
    except Exception as e:
        st.error(f"Помилка конвертації MGRS '{mgrs_str}': {e}")
        return None, None

# =============================================
# ВАЛІДАЦІЯ ТА УТІЛІТИ
# =============================================
def validate_coordinates(lat: float, lon: float, alt: float = 0) -> bool:
    """Валідація вхідних координат"""
    if not (-90 <= lat <= 90):
        st.error(f"Некоректна широта: {lat}. Має бути в діапазоні -90 до 90.")
        return False
    if not (-180 <= lon <= 180):
        st.error(f"Некоректна довгота: {lon}. Має бути в діапазоні -180 до 180.")
        return False
    if not (-10000 <= alt <= 100000):
        st.error(f"Некоректна висота: {alt}. Має бути в діапазоні -10,000 до 100,000 м.")
        return False
    return True

# =============================================
# ОБЧИСЛЕННЯ
# =============================================
@st.cache_data
def calc_point(lat: float, lon: float, alt: float, year: float) -> Dict:
    if not validate_coordinates(lat, lon, alt):
        return {"error": "Invalid coordinates"}
    
    try:
        wmm = get_wmm().calculate(glat=lat, glon=lon, alt=alt/1000, time=year)
        rt = get_rtdm(lat, lon, alt)
        decl = wmm.d + rt["decl_rt"]
        total = wmm.f + rt["total_rt"]
        mgrs_str = safe_mgrs_conversion(lat, lon)
        mgrs_details = get_mgrs_details(mgrs_str)
        utm_z, utm_e, utm_n = batch_utm(np.array([lat]), np.array([lon]))
        
        return {
            "lat": round(lat, 6), 
            "lon": round(lon, 6), 
            "alt": round(alt, 1),
            "decl": round(decl, 4), 
            "total": round(total, 2),
            "storm": rt["storm"], 
            "mgrs": mgrs_str,
            "mgrs_details": mgrs_details,
            "utm_zone": utm_z[0] if len(utm_z) > 0 else "Error",
            "utm_e": utm_e[0] if len(utm_e) > 0 else np.nan,
            "utm_n": utm_n[0] if len(utm_n) > 0 else np.nan,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Calculation error: {str(e)}"}

# =============================================
# ЗАВАНТАЖЕННЯ ВЕКТОРНИХ ДАНИХ
# =============================================
@st.cache_data
def load_vector(file) -> gpd.GeoDataFrame:
    try:
        # Перевірка розміру файлу (макс 50MB)
        if file.size > 50 * 1024 * 1024:
            st.error("Файл занадто великий. Максимальний розмір: 50MB")
            return gpd.GeoDataFrame()
        
        if file.name.endswith((".geojson", ".json")):
            gdf = gpd.read_file(file)
        elif file.name.endswith(".zip"):
            with zipfile.ZipFile(file) as z:
                # Пошук shapefile у архіві
                shp_files = [f for f in z.namelist() if f.endswith(".shp")]
                if not shp_files:
                    st.error("Shapefile не знайдено в архіві")
                    return gpd.GeoDataFrame()
                shp = shp_files[0]
                gdf = gpd.read_file(z.open(shp))
        else:
            st.error("Підтримувані формати: .geojson, .json або .zip (shapefile)")
            return gpd.GeoDataFrame()
        
        # Конвертація в WGS84
        gdf = gdf.to_crs(epsg=4326)
        
        # Обмеження кількості точок
        if len(gdf) > 500:
            st.warning(f"Файл містить {len(gdf)} об'єктів. Буде показано лише 500.")
            gdf = gdf.sample(500, random_state=42)
        
        # Спрощення геометрії для продуктивності
        gdf.geometry = gdf.geometry.simplify(0.001, preserve_topology=True)
        return gdf
        
    except Exception as e:
        st.error(f"Помилка завантаження файлу: {e}")
        return gpd.GeoDataFrame()

# =============================================
# ІНТЕРФЕЙС
# =============================================
st.title("🧭 Geomagnetic Pro 2025")
st.markdown("**Висота | Теплова карта | Вектори | Клік | MGRS | UTM**")

# Ініціалізація сесії для зберігання вибраних точок
if "selected_points" not in st.session_state:
    st.session_state.selected_points = []
if "last_click" not in st.session_state:
    st.session_state.last_click = None

tabs = st.tabs(["Калькулятор", "Карта", "Пакет", "Історія вибору"])

# === КАЛЬКУЛЯТОР ===
with tabs[0]:
    st.subheader("🧮 Калькулятор геомагнітних параметрів")
    
    col1, col2 = st.columns(2)
    
    with col1:
        coord_type = st.radio("Тип координат", ["Географічні", "MGRS"], horizontal=True, key="coord_type")
        
        if coord_type == "Географічні":
            lat = st.number_input("Широта (°)", min_value=-90.0, max_value=90.0, value=50.4501, format="%.6f", key="calc_lat")
            lon = st.number_input("Довгота (°)", min_value=-180.0, max_value=180.0, value=30.5234, format="%.6f", key="calc_lon")
            mgrs_input = None
        else:
            mgrs_input = st.text_input("MGRS координати", value="36TWN1234567890", key="calc_mgrs")
            lat, lon = 50.4501, 30.5234  # Значення за замовчуванням
            
            if mgrs_input and mgrs_input.strip():
                lat_conv, lon_conv = convert_mgrs_to_latlon(mgrs_input.strip())
                if lat_conv is not None and lon_conv is not None:
                    lat, lon = lat_conv, lon_conv
                    st.success(f"Конвертовано: {lat:.6f}°, {lon:.6f}°")
                else:
                    st.error("Не вдалося конвертувати MGRS координати")
    
    with col2:
        alt = st.number_input("Висота (м)", min_value=-10000.0, max_value=100000.0, value=0.0, step=100.0, key="calc_alt")
        calc_date = st.date_input("Дата розрахунку", value=date.today(), key="calc_date")
        
        st.markdown("---")
        st.markdown("**Діапазони:**")
        st.markdown("- Широта: -90° до 90°")
        st.markdown("- Довгота: -180° до 180°") 
        st.markdown("- Висота: -10,000 до 100,000 м")

    if st.button("🔄 Обчислити", type="primary", key="calc_btn"):
        if coord_type == "MGRS" and (not mgrs_input or not mgrs_input.strip()):
            st.error("Будь ласка, введіть MGRS координати")
        else:
            with st.spinner("Виконуються розрахунки..."):
                res = calc_point(lat, lon, alt, decimal_year(calc_date))
            
            if "error" in res:
                st.error(f"Помилка розрахунку: {res['error']}")
            else:
                # Відображення результатів у стовпцях
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.metric("📍 MGRS", res["mgrs"])
                    if res["mgrs_details"] and res["mgrs_details"].get("zone"):
                        with st.expander("Деталі MGRS"):
                            details = res["mgrs_details"]
                            st.write(f"**Зона:** {details.get('zone', '')}")
                            st.write(f"**Банда:** {details.get('band', '')}")
                            st.write(f"**Квадрат:** {details.get('square', '')}")
                            st.write(f"**Повний код:** {details.get('full', '')}")
                
                with col_res2:
                    st.metric("🗺️ UTM Зона", res["utm_zone"])
                    if not np.isnan(res["utm_e"]):
                        st.metric("Easting", f"{res['utm_e']:,.1f} m")
                    if not np.isnan(res["utm_n"]):
                        st.metric("Northing", f"{res['utm_n']:,.1f} m")
                
                with col_res3:
                    st.metric("🧭 Деклінація", f"{res['decl']}°")
                    st.metric("⚡ Інтенсивність", f"{res['total']:,.1f} nT")
                    storm_status = "🟢 Тихий" if res["storm"] == "quiet" else "🟡 Помірний" if res["storm"] == "moderate" else "🔢 Офлайн"
                    st.metric("🌪️ Магнітна буря", storm_status)
                
                # Збереження в історію
                if len(st.session_state.selected_points) >= 10:
                    st.session_state.selected_points.pop(0)
                st.session_state.selected_points.append(res)
                
                with st.expander("📊 Детальна інформація"):
                    st.json({k: v for k, v in res.items() if k != 'mgrs_details'})

# === КАРТА ===
with tabs[1]:
    st.subheader("🗺️ Інтерактивна карта магнітного поля")
    
    col_map1, col_map2 = st.columns([3, 1])
    
    with col_map2:
        st.markdown("### Налаштування")
        
        vector_file = st.file_uploader(
            "Векторні дані", 
            type=["geojson", "json", "zip"],
            key="map_vector",
            help="Завантажте GeoJSON або Shapefile (ZIP)"
        )
        gdf = load_vector(vector_file) if vector_file else None
        
        st.markdown("---")
        alt_grid = st.slider("Висота сітки (м)", 0, 10000, 0, 500, key="map_alt")
        step = st.slider("Крок сітки (°)", 0.1, 2.0, 0.5, 0.1, key="map_step")
        
        # Вибір регіону
        region = st.selectbox("Регіон", ["Україна", "Європа", "Світ", "Кастомний"], key="map_region")
        
        if region == "Україна":
            lat_min, lat_max = 44.0, 52.5
            lon_min, lon_max = 22.0, 40.0
        elif region == "Європа":
            lat_min, lat_max = 35.0, 70.0
            lon_min, lon_max = -10.0, 40.0
        elif region == "Світ":
            lat_min, lat_max = -90.0, 90.0
            lon_min, lon_max = -180.0, 180.0
            step = min(step, 1.0)  # Обмеження кроку для світу
        else:  # Кастомний
            col_reg1, col_reg2 = st.columns(2)
            with col_reg1:
                lat_min = st.number_input("Мін. широта", min_value=-90.0, max_value=90.0, value=44.0, key="lat_min")
                lat_max = st.number_input("Макс. широта", min_value=-90.0, max_value=90.0, value=52.5, key="lat_max")
            with col_reg2:
                lon_min = st.number_input("Мін. довгота", min_value=-180.0, max_value=180.0, value=22.0, key="lon_min")
                lon_max = st.number_input("Макс. довгота", min_value=-180.0, max_value=180.0, value=40.0, key="lon_max")

    with col_map1:
        # Генерація теплової карти
        cache_key = f"grid_{alt_grid}_{step}_{region}"
        if cache_key not in st.session_state:
            if lat_max > lat_min and lon_max > lon_min:
                lats = np.arange(lat_min, lat_max, step)
                lons = np.arange(lon_min, lon_max, step)
                
                if len(lats) > 0 and len(lons) > 0:
                    with st.spinner(f"Генерація теплової карти ({len(lats)}x{len(lons)} точок)..."):
                        grid = [(la, lo, alt_grid) for la in lats for lo in lons]
                        results = []
                        for i, point in enumerate(grid):
                            res = calc_point(*point, decimal_year(date.today()))
                            if "error" not in res:
                                results.append(res)
                            if i % 100 == 0:
                                st.progress(min((i + 1) / len(grid), 1.0))
                        
                        st.session_state[cache_key] = pd.DataFrame(results)
                else:
                    st.session_state[cache_key] = pd.DataFrame()
            else:
                st.session_state[cache_key] = pd.DataFrame()

        df_heatmap = st.session_state.get(cache_key, pd.DataFrame())

        # Створення карти
        fig = go.Figure()

        # Теплова карта
        if not df_heatmap.empty:
            fig.add_trace(go.Densitymapbox(
                lat=df_heatmap["lat"], 
                lon=df_heatmap["lon"], 
                z=df_heatmap["decl"],
                radius=15, 
                colorscale="RdBu", 
                zmid=0, 
                opacity=0.7,
                colorbar=dict(title="Деклінація (°)", titleside="right")
            ))

        # Векторні дані
        if gdf is not None and not gdf.empty:
            for idx, row in gdf.iterrows():
                geom = row.geometry
                if geom.geom_type == 'Point':
                    fig.add_trace(go.Scattermapbox(
                        lat=[geom.y], 
                        lon=[geom.x],
                        mode="markers", 
                        marker=dict(color="green", size=8),
                        name=f"Точка {idx}",
                        text=[f"Об'єкт {idx}: {geom.y:.4f}°, {geom.x:.4f}°"],
                        hoverinfo="text"
                    ))
                elif geom.geom_type in ['LineString', 'MultiLineString']:
                    if geom.geom_type == 'LineString':
                        coords = list(geom.coords)
                    else:
                        coords = []
                        for line in geom.geoms:
                            coords.extend(list(line.coords))
                    
                    lons_line = [coord[0] for coord in coords]
                    lats_line = [coord[1] for coord in coords]
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=lats_line, 
                        lon=lons_line,
                        mode="lines", 
                        line=dict(color="purple", width=2),
                        name=f"Лінія {idx}",
                        text=f"Лінія {idx}",
                        hoverinfo="text"
                    ))

        # Вибрані точки з історії
        if st.session_state.selected_points:
            selected_lats = [p["lat"] for p in st.session_state.selected_points]
            selected_lons = [p["lon"] for p in st.session_state.selected_points]
            selected_texts = [
                f"Точка: {p['lat']:.4f}°, {p['lon']:.4f}°<br>"
                f"MGRS: {p['mgrs']}<br>"
                f"Деклінація: {p['decl']}°<br>"
                f"Висота: {p['alt']} м"
                for p in st.session_state.selected_points
            ]
            
            fig.add_trace(go.Scattermapbox(
                lat=selected_lats, 
                lon=selected_lons,
                mode="markers",
                marker=dict(color="red", size=10, symbol="circle"),
                name="Історія точок",
                text=selected_texts,
                hoverinfo="text"
            ))

        # Останній клік
        if st.session_state.last_click:
            lc = st.session_state.last_click
            fig.add_trace(go.Scattermapbox(
                lat=[lc["lat"]], 
                lon=[lc["lon"]],
                mode="markers",
                marker=dict(color="blue", size=15, symbol="star"),
                name="Останній вибір",
                text=[f"Останній вибір: {lc['lat']:.4f}°, {lc['lon']:.4f}°"],
                hoverinfo="text"
            ))

        # Налаштування макету карти
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        
        zoom_levels = {
            "Україна": 5,
            "Європа": 3,
            "Світ": 1,
            "Кастомний": 4
        }
        
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=dict(lat=center_lat, lon=center_lon),
                zoom=zoom_levels.get(region, 4)
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=0),
            showlegend=True,
            legend=dict(x=0, y=1)
        )

        # Відображення карти
        st.plotly_chart(fig, use_container_width=True, key="main_map")
        
        # Інструкція для вибору точок
        st.info("💡 **Користувацький вибір:** Натисніть на будь-яку точку карти, щоб отримати координати та геомагнітні параметри")

        # Обробка подій карти через clickData
        map_click = st.session_state.get("main_map", None)
        
        # Альтернативний спосіб вибору через координати
        st.markdown("---")
        st.subheader("Ручний ввід координат з карти")
        col_manual1, col_manual2, col_manual3 = st.columns(3)
        with col_manual1:
            manual_lat = st.number_input("Широта", min_value=-90.0, max_value=90.0, value=center_lat, key="manual_lat")
        with col_manual2:
            manual_lon = st.number_input("Довгота", min_value=-180.0, max_value=180.0, value=center_lon, key="manual_lon")
        with col_manual3:
            if st.button("Додати точку", key="add_manual"):
                result = calc_point(manual_lat, manual_lon, alt_grid, decimal_year(date.today()))
                if "error" not in result:
                    st.session_state.last_click = result
                    if len(st.session_state.selected_points) >= 10:
                        st.session_state.selected_points.pop(0)
                    st.session_state.selected_points.append(result)
                    st.success(f"Додано точку: {manual_lat:.4f}°, {manual_lon:.4f}°")

        # Інформація про останній вибір
        if st.session_state.last_click:
            lc = st.session_state.last_click
            with st.expander("📋 Інформація про останню обрану точку", expanded=True):
                col_lc1, col_lc2, col_lc3 = st.columns(3)
                with col_lc1:
                    st.write(f"**📍 Координати:**")
                    st.write(f"Широта: {lc['lat']:.6f}°")
                    st.write(f"Довгота: {lc['lon']:.6f}°")
                    st.write(f"Висота: {lc['alt']} м")
                with col_lc2:
                    st.write(f"**🧭 Системи координат:**")
                    st.write(f"MGRS: {lc['mgrs']}")
                    st.write(f"UTM: {lc['utm_zone']}")
                    if not np.isnan(lc['utm_e']):
                        st.write(f"Easting: {lc['utm_e']:,.1f} m")
                    if not np.isnan(lc['utm_n']):
                        st.write(f"Northing: {lc['utm_n']:,.1f} m")
                with col_lc3:
                    st.write(f"**⚡ Геомагнітні параметри:**")
                    st.write(f"Деклінація: {lc['decl']}°")
                    st.write(f"Інтенсивність: {lc['total']:,.1f} nT")
                    st.write(f"Статус: {lc['storm']}")

# === ПАКЕТ ===
with tabs[2]:
    st.subheader("📦 Пакетна обробка даних")
    
    file = st.file_uploader("Завантажте CSV файл", type=["csv"], key="batch_csv")
    
    if file:
        try:
            df = pd.read_csv(file)
            st.success(f"Файл успішно завантажено: {len(df)} рядків")
            
            # Перевірка обов'язкових колонок
            if not all(c in df.columns for c in ["lat", "lon"]):
                st.error("CSV файл повинен містити колонки 'lat' та 'lon'")
                st.stop()
            
            # Заповнення відсутніх висот
            df["alt"] = df.get("alt", 0).fillna(0)
            
            # Статистика даних
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric("Кількість точок", len(df))
            with col_stats2:
                valid_lats = df[(df['lat'] >= -90) & (df['lat'] <= 90)]
                st.metric("Валідні широти", len(valid_lats))
            with col_stats3:
                valid_lons = df[(df['lon'] >= -180) & (df['lon'] <= 180)]
                st.metric("Валідні довготи", len(valid_lons))
            
            # Додаткові опції
            col_batch1, col_batch2 = st.columns(2)
            with col_batch1:
                batch_date = st.date_input("Дата для розрахунків", value=date.today(), key="batch_date")
            with col_batch2:
                include_mgrs_details = st.checkbox("Включити деталі MGRS", value=True, key="include_mgrs")
            
            pts = [(r["lat"], r["lon"], r["alt"]) for _, r in df.iterrows()]
            
            if st.button("🚀 Запустити пакетну обробку", type="primary", key="batch_btn"):
                with st.spinner(f"Обчислення для {len(pts)} точок..."):
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, point in enumerate(pts):
                        lat, lon, alt = point
                        if validate_coordinates(lat, lon, alt):
                            res = calc_point(lat, lon, alt, decimal_year(batch_date))
                            if "error" not in res and include_mgrs_details:
                                res.update(res.get("mgrs_details", {}))
                            results.append(res)
                        else:
                            results.append({"error": "Invalid coordinates", "lat": lat, "lon": lon, "alt": alt})
                        
                        progress = (i + 1) / len(pts)
                        progress_bar.progress(progress)
                        status_text.text(f"Оброблено {i + 1}/{len(pts)} точок")
                    
                    df_out = pd.DataFrame(results)
                    
                    # Відображення результатів
                    st.subheader("📊 Результати обробки")
                    st.dataframe(df_out, use_container_width=True, height=400)
                    
                    # Статистика результатів
                    st.subheader("📈 Статистика")
                    successful = len([r for r in results if "error" not in r])
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    with col_stat1:
                        st.metric("Успішні розрахунки", successful)
                    with col_stat2:
                        st.metric("Помилки", len(results) - successful)
                    with col_stat3:
                        if successful > 0:
                            decl_mean = df_out[df_out['decl'].notna()]['decl'].mean()
                            st.metric("Середня деклінація", f"{decl_mean:.2f}°")
                    with col_stat4:
                        if successful > 0:
                            total_mean = df_out[df_out['total'].notna()]['total'].mean()
                            st.metric("Середня інтенсивність", f"{total_mean:,.0f} nT")
                    
                    # Завантаження результатів
                    st.subheader("💾 Експорт результатів")
                    csv_data = df_out.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                    st.download_button(
                        "⬇️ Завантажити CSV", 
                        csv_data,
                        f"geomag_batch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", 
                        "text/csv",
                        key="download_batch"
                    )
                    
        except Exception as e:
            st.error(f"Помилка читання CSV файлу: {e}")

# === ІСТОРІЯ ВИБОРУ ===
with tabs[3]:
    st.subheader("📚 Історія обраних точок")
    
    if not st.session_state.selected_points:
        st.info("ℹ️ Ще немає обраних точок. Використовуйте карту або калькулятор для додавання точок.")
    else:
        # Відображення історії у таблиці
        history_df = pd.DataFrame(st.session_state.selected_points)
        
        # Спрощена таблиця для відображення
        display_cols = ['lat', 'lon', 'alt', 'mgrs', 'decl', 'total', 'storm', 'timestamp']
        available_cols = [col for col in display_cols if col in history_df.columns]
        
        st.dataframe(history_df[available_cols], use_container_width=True, height=300)
        
        # Керування історією
        col_hist1, col_hist2, col_hist3 = st.columns(3)
        with col_hist1:
            if st.button("🗑️ Очистити історію", type="secondary"):
                st.session_state.selected_points = []
                st.session_state.last_click = None
                st.rerun()
        with col_hist2:
            if st.button("💾 Експортувати історію", type="primary"):
                hist_csv = history_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button(
                    "⬇️ Завантажити історію", 
                    hist_csv,
                    f"geomag_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", 
                    "text/csv",
                    key="download_history"
                )
        with col_hist3:
            st.metric("Кількість точок", len(history_df))
        
        # Швидкий перехід до точок
        st.subheader("🔍 Швидкий перехід")
        for i, point in enumerate(reversed(st.session_state.selected_points)):
            point_idx = len(st.session_state.selected_points) - i
            with st.expander(f"Точка {point_idx}: {point['lat']:.4f}°, {point['lon']:.4f}"):
                col_quick1, col_quick2 = st.columns(2)
                with col_quick1:
                    st.write(f"**📍 Координати:**")
                    st.write(f"Широта: {point['lat']:.6f}°")
                    st.write(f"Довгота: {point['lon']:.6f}°")
                    st.write(f"Висота: {point['alt']} м")
                    st.write(f"**🗺️ MGRS:** {point['mgrs']}")
                    st.write(f"**🧭 UTM:** {point.get('utm_zone', 'N/A')}")
                with col_quick2:
                    st.write(f"**⚡ Геомагнітні параметри:**")
                    st.write(f"Деклінація: {point['decl']}°")
                    st.write(f"Інтенсивність: {point['total']:,.1f} nT")
                    st.write(f"Статус: {point['storm']}")
                    st.write(f"**⏰ Час:** {point.get('timestamp', 'N/A')}")
                
                if st.button(f"📍 Перейти до точки {point_idx}", key=f"goto_{i}"):
                    st.session_state.last_click = point
                    st.rerun()

# === ФУТЕР ===
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <strong>Geomagnetic Pro 2025</strong> • WMM 2025 • Real-time магнітні дані USGS<br>
        <small>Розроблено для геомагнітних досліджень та навігації</small>
    </div>
    """,
    unsafe_allow_html=True
)
