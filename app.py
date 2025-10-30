# =============================================
# 🧭 Geomagnetic Pro 2025 — Виправлена версія
# =============================================

from __future__ import annotations
import os, io, zipfile, requests
from datetime import date, datetime
from typing import Dict, Tuple, List, Optional, Any
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
        
        if len(mgrs_str) < 5:
            return {"full": mgrs_str}
            
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
        if file.size > 50 * 1024 * 1024:
            st.error("Файл занадто великий. Максимальний розмір: 50MB")
            return gpd.GeoDataFrame()
        
        if file.name.endswith((".geojson", ".json")):
            gdf = gpd.read_file(file)
        elif file.name.endswith(".zip"):
            with zipfile.ZipFile(file) as z:
                shp_files = [f for f in z.namelist() if f.endswith(".shp")]
                if not shp_files:
                    st.error("Shapefile не знайдено в архіві")
                    return gpd.GeoDataFrame()
                shp = shp_files[0]
                gdf = gpd.read_file(z.open(shp))
        else:
            st.error("Підтримувані формати: .geojson, .json або .zip (shapefile)")
            return gpd.GeoDataFrame()
        
        gdf = gdf.to_crs(epsg=4326)
        
        if len(gdf) > 500:
            st.warning(f"Файл містить {len(gdf)} об'єктів. Буде показано лише 500.")
            gdf = gdf.sample(500, random_state=42)
        
        gdf.geometry = gdf.geometry.simplify(0.001, preserve_topology=True)
        return gdf
        
    except Exception as e:
        st.error(f"Помилка завантаження файлу: {e}")
        return gpd.GeoDataFrame()

# =============================================
# ФУНКЦІЇ ДЛЯ КАРТИ
# =============================================
def create_heatmap_data(region: str, alt_grid: float, step: float) -> pd.DataFrame:
    """Створення даних для теплової карти"""
    if region == "Україна":
        lat_min, lat_max = 44.0, 52.5
        lon_min, lon_max = 22.0, 40.0
    elif region == "Європа":
        lat_min, lat_max = 35.0, 70.0
        lon_min, lon_max = -10.0, 40.0
    elif region == "Світ":
        lat_min, lat_max = -90.0, 90.0
        lon_min, lon_max = -180.0, 180.0
        step = min(step, 1.0)
    else:
        lat_min, lat_max = 44.0, 52.5
        lon_min, lon_max = 22.0, 40.0
    
    lats = np.arange(lat_min, lat_max, step)
    lons = np.arange(lon_min, lon_max, step)
    
    if len(lats) == 0 or len(lons) == 0:
        return pd.DataFrame()
    
    grid = [(la, lo, alt_grid) for la in lats for lo in lons]
    results = []
    
    max_points = 1000
    if len(grid) > max_points:
        grid = grid[:max_points]
        st.warning(f"Для продуктивності обмежено до {max_points} точок")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, point in enumerate(grid):
        res = calc_point(*point, decimal_year(date.today()))
        if "error" not in res:
            results.append(res)
        
        if i % 10 == 0:
            progress_bar.progress((i + 1) / len(grid))
            status_text.text(f"Генерація теплової карти: {i + 1}/{len(grid)}")
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

def create_map_figure(heatmap_df: pd.DataFrame, gdf: Optional[gpd.GeoDataFrame], 
                     selected_points: List[Dict], last_click: Optional[Dict],
                     region: str) -> go.Figure:
    """Створення інтерактивної карти"""
    fig = go.Figure()
    
    # Визначення центру карти залежно від регіону
    if region == "Україна":
        center_lat, center_lon = 48.5, 31.0
        zoom = 5
    elif region == "Європа":
        center_lat, center_lon = 52.5, 15.0
        zoom = 3
    elif region == "Світ":
        center_lat, center_lon = 0, 0
        zoom = 1
    else:
        center_lat, center_lon = 48.5, 31.0
        zoom = 5
    
    # Додавання теплової карти
    if not heatmap_df.empty:
        fig.add_trace(go.Densitymapbox(
            lat=heatmap_df["lat"], 
            lon=heatmap_df["lon"], 
            z=heatmap_df["decl"],
            radius=20, 
            colorscale="RdBu", 
            zmid=0, 
            opacity=0.7,
            colorbar=dict(
                title="Деклінація (°)",
                x=0.95
            ),
            hoverinfo="none",
            name="Магнітне поле"
        ))
    
    # Додавання векторних даних
    if gdf is not None and not gdf.empty:
        for idx, row in gdf.iterrows():
            geom = row.geometry
            
            if geom.geom_type == 'Point':
                fig.add_trace(go.Scattermapbox(
                    lat=[geom.y], 
                    lon=[geom.x],
                    mode="markers",
                    marker=dict(size=10, color="green"),
                    name=f"Векторна точка {idx+1}",
                    text=f"Точка {idx+1}",
                    hoverinfo="text"
                ))
                
            elif geom.geom_type == 'LineString':
                coords = list(geom.coords)
                lons_line = [coord[0] for coord in coords]
                lats_line = [coord[1] for coord in coords]
                
                fig.add_trace(go.Scattermapbox(
                    lat=lats_line, 
                    lon=lons_line,
                    mode="lines",
                    line=dict(width=3, color="purple"),
                    name=f"Лінія {idx+1}",
                    text=f"Лінія {idx+1}",
                    hoverinfo="text"
                ))
    
    # Додавання історичних точок
    if selected_points:
        selected_lats = [p["lat"] for p in selected_points]
        selected_lons = [p["lon"] for p in selected_points]
        selected_texts = [
            f"Точка {i+1}: {p['lat']:.4f}°, {p['lon']:.4f}°<br>"
            f"MGRS: {p['mgrs']}<br>"
            f"Деклінація: {p['decl']}°<br>"
            f"Висота: {p['alt']} м"
            for i, p in enumerate(selected_points)
        ]
        
        fig.add_trace(go.Scattermapbox(
            lat=selected_lats, 
            lon=selected_lons,
            mode="markers+text",
            marker=dict(size=12, color="red", symbol="circle"),
            text=[f"{i+1}" for i in range(len(selected_points))],
            textposition="top center",
            name="Історія точок",
            hovertext=selected_texts,
            hoverinfo="text"
        ))
    
    # Додавання останньої обраної точки
    if last_click:
        fig.add_trace(go.Scattermapbox(
            lat=[last_click["lat"]], 
            lon=[last_click["lon"]],
            mode="markers+text",
            marker=dict(size=16, color="blue", symbol="star"),
            text=["★"],
            textposition="top center",
            name="Останній вибір",
            hovertext=[
                f"Останній вибір:<br>"
                f"Координати: {last_click['lat']:.4f}°, {last_click['lon']:.4f}°<br>"
                f"MGRS: {last_click['mgrs']}<br>"
                f"Деклінація: {last_click['decl']}°"
            ],
            hoverinfo="text"
        ))
    
    # Налаштування макету карти
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=True,
        legend=dict(
            x=0,
            y=1,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    return fig

# =============================================
# ІНТЕРФЕЙС
# =============================================
st.title("🧭 Geomagnetic Pro 2025")
st.markdown("**Висота | Теплова карта | Вектори | Клік | MGRS | UTM**")

# Ініціалізація сесії
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
            lat, lon = 50.4501, 30.5234
            
            if mgrs_input and mgrs_input.strip():
                lat_conv, lon_conv = convert_mgrs_to_latlon(mgrs_input.strip())
                if lat_conv is not None and lon_conv is not None:
                    lat, lon = lat_conv, lon_conv
                    st.success(f"Конвертовано: {lat:.6f}°, {lon:.6f}°")
    
    with col2:
        alt = st.number_input("Висота (м)", min_value=-10000.0, max_value=100000.0, value=0.0, step=100.0, key="calc_alt")
        calc_date = st.date_input("Дата розрахунку", value=date.today(), key="calc_date")

    if st.button("🔄 Обчислити", type="primary", key="calc_btn"):
        if coord_type == "MGRS" and (not mgrs_input or not mgrs_input.strip()):
            st.error("Будь ласка, введіть MGRS координати")
        else:
            with st.spinner("Виконуються розрахунки..."):
                res = calc_point(lat, lon, alt, decimal_year(calc_date))
            
            if "error" in res:
                st.error(f"Помилка розрахунку: {res['error']}")
            else:
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.metric("📍 MGRS", res["mgrs"])
                    if res["mgrs_details"] and res["mgrs_details"].get("zone"):
                        with st.expander("Деталі MGRS"):
                            details = res["mgrs_details"]
                            st.write(f"**Зона:** {details.get('zone', '')}")
                            st.write(f"**Банда:** {details.get('band', '')}")
                            st.write(f"**Квадрат:** {details.get('square', '')}")
                
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
            key="map_vector"
        )
        gdf = load_vector(vector_file) if vector_file else None
        
        st.markdown("---")
        alt_grid = st.slider("Висота сітки (м)", 0, 10000, 0, 500, key="map_alt")
        step = st.slider("Крок сітки (°)", 0.1, 2.0, 0.5, 0.1, key="map_step")
        
        region = st.selectbox("Регіон", ["Україна", "Європа", "Світ", "Кастомний"], key="map_region")
        
        if st.button("🔄 Оновити теплову карту", key="update_heatmap"):
            # Очищаємо кеш для генерації нової теплової карти
            for key in list(st.session_state.keys()):
                if key.startswith("heatmap_"):
                    del st.session_state[key]
            st.rerun()

    with col_map1:
        # Генерація або завантаження даних теплової карти
        cache_key = f"heatmap_{region}_{alt_grid}_{step}"
        
        if cache_key not in st.session_state:
            with st.spinner("Генерація теплової карти..."):
                heatmap_data = create_heatmap_data(region, alt_grid, step)
                st.session_state[cache_key] = heatmap_data
        else:
            heatmap_data = st.session_state[cache_key]
        
        # Створення карти
        fig = create_map_figure(
            heatmap_data, 
            gdf, 
            st.session_state.selected_points,
            st.session_state.last_click,
            region
        )
        
        # Відображення карти
        st.plotly_chart(fig, use_container_width=True, key="main_map")
        
        # Інформація про взаємодію з картою
        st.info("""
        **💡 Інструкція:**
        - **Теплова карта**: Показує магнітну деклінацію (червоний - північ, синій - південь)
        - **Червоні точки**: Історія ваших виборів
        - **Сині зірки**: Останній вибір
        - **Зелені маркери**: Векторні дані
        """)
        
        # Ручний ввід координат для додавання точок
        with st.form("map_click_form"):
            st.subheader("Додати точку вручну")
            col_click1, col_click2 = st.columns(2)
            with col_click1:
                click_lat = st.number_input("Широта (°)", min_value=-90.0, max_value=90.0, value=50.4501, key="click_lat")
            with col_click2:
                click_lon = st.number_input("Довгота (°)", min_value=-180.0, max_value=180.0, value=30.5234, key="click_lon")
            
            if st.form_submit_button("✅ Додати точку"):
                result = calc_point(click_lat, click_lon, alt_grid, decimal_year(date.today()))
                if "error" not in result:
                    st.session_state.last_click = result
                    if len(st.session_state.selected_points) >= 10:
                        st.session_state.selected_points.pop(0)
                    st.session_state.selected_points.append(result)
                    st.success(f"Додано точку: {click_lat:.4f}°, {click_lon:.4f}°")
                    st.rerun()

        # Інформація про останній вибір
        if st.session_state.last_click:
            lc = st.session_state.last_click
            with st.expander("📋 Інформація про останню обрану точку", expanded=True):
                col_lc1, col_lc2 = st.columns(2)
                with col_lc1:
                    st.write(f"**📍 Координати:**")
                    st.write(f"Широта: {lc['lat']:.6f}°")
                    st.write(f"Довгота: {lc['lon']:.6f}°")
                    st.write(f"Висота: {lc['alt']} м")
                    st.write(f"**🗺️ MGRS:** {lc['mgrs']}")
                with col_lc2:
                    st.write(f"**⚡ Геомагнітні параметри:**")
                    st.write(f"Деклінація: {lc['decl']}°")
                    st.write(f"Інтенсивність: {lc['total']:,.1f} nT")
                    st.write(f"Статус: {lc['storm']}")
                    st.write(f"**🧭 UTM:** {lc.get('utm_zone', 'N/A')}")

# === ПАКЕТНА ОБРОБКА ===
with tabs[2]:
    st.subheader("📦 Пакетна обробка даних")
    
    file = st.file_uploader("Завантажте CSV файл", type=["csv"], key="batch_csv")
    
    if file:
        try:
            df = pd.read_csv(file)
            st.success(f"Файл успішно завантажено: {len(df)} рядків")
            
            if not all(c in df.columns for c in ["lat", "lon"]):
                st.error("CSV файл повинен містити колонки 'lat' та 'lon'")
                st.stop()
            
            df["alt"] = df.get("alt", 0).fillna(0)
            
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric("Кількість точок", len(df))
            with col_stats2:
                valid_lats = df[(df['lat'] >= -90) & (df['lat'] <= 90)]
                st.metric("Валідні широти", len(valid_lats))
            with col_stats3:
                valid_lons = df[(df['lon'] >= -180) & (df['lon'] <= 180)]
                st.metric("Валідні довготи", len(valid_lons))
            
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
                    
                    for i, point in enumerate(pts):
                        lat, lon, alt = point
                        if validate_coordinates(lat, lon, alt):
                            res = calc_point(lat, lon, alt, decimal_year(batch_date))
                            if "error" not in res and include_mgrs_details:
                                res.update(res.get("mgrs_details", {}))
                            results.append(res)
                        else:
                            results.append({"error": "Invalid coordinates", "lat": lat, "lon": lon, "alt": alt})
                        
                        progress_bar.progress((i + 1) / len(pts))
                    
                    df_out = pd.DataFrame(results)
                    
                    st.subheader("📊 Результати обробки")
                    st.dataframe(df_out, use_container_width=True, height=400)
                    
                    successful = len([r for r in results if "error" not in r])
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("Успішні розрахунки", successful)
                    with col_stat2:
                        st.metric("Помилки", len(results) - successful)
                    
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
        history_df = pd.DataFrame(st.session_state.selected_points)
        display_cols = ['lat', 'lon', 'alt', 'mgrs', 'decl', 'total', 'storm', 'timestamp']
        available_cols = [col for col in display_cols if col in history_df.columns]
        
        st.dataframe(history_df[available_cols], use_container_width=True, height=300)
        
        col_hist1, col_hist2 = st.columns(2)
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