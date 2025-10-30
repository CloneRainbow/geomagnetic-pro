# =============================================
# 🧭 Geomagnetic Pro 2025 — З Leaflet картою
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
import streamlit as st
import streamlit.components.v1 as components

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
# REAL-TIME DATA MODEL (RTDM) - ПОКРАЩЕНА ВЕРСІЯ
# =============================================
@st.cache_data(ttl=3600)
def get_rtdm(lat: float, lon: float, alt: float = 0) -> Dict:
    """
    Спрощена версія RTDM, яка повертає нульові корекції при недоступності сервісу
    """
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return {"decl_rt": 0.0, "total_rt": 0.0, "storm": "invalid_coords", "available": False}
    
    try:
        params = {
            "latitude": lat, 
            "longitude": lon, 
            "altitude": alt / 1000, 
            "format": "json"
        }
        
        r = requests.get(RTDM_API, params=params, timeout=5)
        
        if r.status_code == 200:
            d = r.json()
            return {
                "decl_rt": d.get("declination", 0) - d.get("quiet_declination", 0),
                "total_rt": d.get("total_intensity", 0) - d.get("quiet_intensity", 0),
                "storm": d.get("storm_level", "quiet"),
                "available": True
            }
    except Exception:
        pass
    
    return {"decl_rt": 0.0, "total_rt": 0.0, "storm": "offline", "available": False}

# =============================================
# MGRS + UTM - РОЗШИРЕНІ ФУНКЦІЇ
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

def validate_mgrs(mgrs_str: str) -> bool:
    """Валідація MGRS координат"""
    try:
        if not mgrs_str or len(mgrs_str) < 5:
            return False
        parts = mgrs_str.strip()
        if len(parts) < 5:
            return False
        lat, lon = MGRS_CONV.toLatLon(parts)
        return lat is not None and lon is not None
    except Exception:
        return False

def get_mgrs_precision(mgrs_str: str) -> str:
    """Визначення точності MGRS координат"""
    length = len(mgrs_str)
    if length <= 5:
        return "Низька (100 км)"
    elif length <= 7:
        return "Середня (10 км)"
    elif length <= 9:
        return "Висока (1 км)"
    elif length <= 11:
        return "Дуже висока (100 м)"
    else:
        return "Найвища (10 м)"

# =============================================
# LEAFLET КАРТА
# =============================================
def create_leaflet_map(center_lat: float, center_lon: float, zoom: int, 
                      points: List[Dict], search_point: Optional[Dict] = None) -> str:
    """Створення Leaflet карти з точками"""
    
    # Підготовка даних для точок
    points_data = []
    for i, point in enumerate(points):
        points_data.append({
            'lat': point['lat'],
            'lon': point['lon'],
            'title': f"Точка {i+1}",
            'color': 'red',
            'icon': 'circle',
            'description': f"""
                <b>Точка {i+1}:</b><br>
                Координати: {point['lat']:.6f}°, {point['lon']:.6f}°<br>
                MGRS: {point.get('mgrs', 'N/A')}<br>
                Деклінація: {point.get('decl', 'N/A')}°<br>
                Висота: {point.get('alt', 'N/A')} м
            """
        })
    
    # Додавання точки пошуку
    if search_point:
        points_data.append({
            'lat': search_point['lat'],
            'lon': search_point['lon'],
            'title': "Пошук MGRS",
            'color': 'orange',
            'icon': 'star',
            'description': f"""
                <b>Пошук MGRS:</b><br>
                MGRS: {search_point['mgrs']}<br>
                Координати: {search_point['lat']:.6f}°, {search_point['lon']:.6f}°<br>
                Точність: {search_point.get('precision', 'Невідомо')}
            """
        })
    
    # Створення HTML з Leaflet
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Geomagnetic Pro 2025</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            #map {{ height: 600px; width: 100%; }}
            .leaflet-popup-content {{ font-family: Arial, sans-serif; }}
            .custom-popup {{ min-width: 250px; }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            // Ініціалізація карти
            var map = L.map('map').setView([{center_lat}, {center_lon}], {zoom});
            
            // Додавання базових шарів
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '© OpenStreetMap contributors',
                maxZoom: 19
            }}).addTo(map);
            
            // Додавання точок
            var points = {json.dumps(points_data)};
            
            points.forEach(function(point) {{
                var icon = L.divIcon({{
                    className: 'custom-icon',
                    html: '<div style="background-color: ' + point.color + '; width: 12px; height: 12px; border-radius: 50%; border: 2px solid white;"></div>',
                    iconSize: [16, 16],
                    iconAnchor: [8, 8]
                }});
                
                if (point.icon === 'star') {{
                    icon = L.divIcon({{
                        className: 'custom-icon',
                        html: '<div style="background-color: ' + point.color + '; width: 16px; height: 16px; transform: rotate(45deg); border: 2px solid white;"></div>',
                        iconSize: [20, 20],
                        iconAnchor: [10, 10]
                    }});
                }}
                
                var marker = L.marker([point.lat, point.lon], {{icon: icon}}).addTo(map);
                
                marker.bindPopup(
                    '<div class="custom-popup">' +
                    '<h4>' + point.title + '</h4>' +
                    point.description +
                    '</div>'
                );
            }});
            
            // Додавання контролу масштабу
            L.control.scale({{imperial: false}}).addTo(map);
        </script>
    </body>
    </html>
    """
    return html

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
            "full": mgrs_str,
            "precision": get_mgrs_precision(mgrs_str)
        }
    except Exception:
        return {"full": mgrs_str}

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
        
        # Обчислення UTM
        try:
            zone = int((lon + 180) / 6) + 1
            hemi = 'S' if lat < 0 else 'N'
            proj_string = f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs"
            if hemi == 'S':
                proj_string += " +south"
            
            transformer = Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)
            e, n = transformer.transform(lon, lat)
            utm_zone = f"{zone}{hemi}"
            utm_e = round(e, 1)
            utm_n = round(n, 1)
        except Exception:
            utm_zone = "Error"
            utm_e = np.nan
            utm_n = np.nan
        
        return {
            "lat": round(lat, 6), 
            "lon": round(lon, 6), 
            "alt": round(alt, 1),
            "decl": round(decl, 4), 
            "total": round(total, 2),
            "storm": rt["storm"], 
            "rtdm_available": rt["available"],
            "mgrs": mgrs_str,
            "mgrs_details": mgrs_details,
            "utm_zone": utm_zone,
            "utm_e": utm_e,
            "utm_n": utm_n,
            "timestamp": datetime.now().isoformat(),
            "model": "WMM2025" + (" + RTDM" if rt["available"] else "")
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
        
        return gdf
        
    except Exception as e:
        st.error(f"Помилка завантаження файлу: {e}")
        return gpd.GeoDataFrame()

# =============================================
# ІНТЕРФЕЙС
# =============================================
st.title("🧭 Geomagnetic Pro 2025")
st.markdown("**Точна карта | MGRS пошук | Геомагнітні параметри**")

# Інформація про статус RTDM
rtdm_status = get_rtdm(50.45, 30.52, 0)["available"]
if not rtdm_status:
    st.info("""
    **ℹ️ Інформація про сервіс:** 
    Real-time магнітні корекції (RTDM) тимчасово недоступні. 
    Використовуються точні дані WMM 2025 моделі.
    """)

# Ініціалізація сесії
if "selected_points" not in st.session_state:
    st.session_state.selected_points = []
if "last_click" not in st.session_state:
    st.session_state.last_click = None
if "mgrs_search" not in st.session_state:
    st.session_state.mgrs_search = None

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
            lat, lon = 50, 30.5234
            
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
                            st.write(f"**Точність:** {details.get('precision', 'Невідомо')}")
                
                with col_res2:
                    st.metric("🗺️ UTM Зона", res["utm_zone"])
                    if not np.isnan(res["utm_e"]):
                        st.metric("Easting", f"{res['utm_e']:,.1f} m")
                    if not np.isnan(res["utm_n"]):
                        st.metric("Northing", f"{res['utm_n']:,.1f} m")
                
                with col_res3:
                    st.metric("🧭 Деклінація", f"{res['decl']}°")
                    st.metric("⚡ Інтенсивність", f"{res['total']:,.1f} nT")
                    storm_status = "🟢 Тихий" if res["storm"] == "quiet" else "🟡 Помірний" if res["storm"] == "moderate" else "⚫ Офлайн"
                    st.metric("🌪️ Статус", storm_status)
                    st.metric("📊 Модель", res.get("model", "WMM2025"))
                
                # Збереження в історію
                if len(st.session_state.selected_points) >= 10:
                    st.session_state.selected_points.pop(0)
                st.session_state.selected_points.append(res)
                st.session_state.last_click = res
                
                with st.expander("📊 Детальна інформація"):
                    st.json({k: v for k, v in res.items() if k != 'mgrs_details'})

# === КАРТА ===
with tabs[1]:
    st.subheader("🗺️ Точна карта з Leaflet")
    
    col_map1, col_map2 = st.columns([3, 1])
    
    with col_map2:
        st.markdown("### Налаштування карти")
        
        # Налаштування центру карти
        st.markdown("**Центр карти:**")
        col_center1, col_center2 = st.columns(2)
        with col_center1:
            center_lat = st.number_input("Широта центру", min_value=-90.0, max_value=90.0, value=50.4501, key="center_lat")
        with col_center2:
            center_lon = st.number_input("Довгота центру", min_value=-180.0, max_value=180.0, value=30.5234, key="center_lon")
        
        zoom_level = st.slider("Масштаб", min_value=1, max_value=18, value=10, key="zoom_level")
        
        st.markdown("---")
        st.markdown("### 🔍 Пошук за MGRS")
        
        # Пошук за MGRS координатами
        mgrs_search_input = st.text_input(
            "MGRS координати для пошуку", 
            placeholder="Наприклад: 36TWN1234567890",
            key="mgrs_search_input"
        )
        
        col_search1, col_search2 = st.columns(2)
        with col_search1:
            if st.button("🔎 Знайти на карті", key="search_mgrs"):
                if mgrs_search_input and mgrs_search_input.strip():
                    if validate_mgrs(mgrs_search_input.strip()):
                        lat, lon = convert_mgrs_to_latlon(mgrs_search_input.strip())
                        if lat is not None and lon is not None:
                            # Створюємо точку для пошуку
                            search_result = {
                                "lat": lat,
                                "lon": lon,
                                "mgrs": mgrs_search_input.strip(),
                                "precision": get_mgrs_precision(mgrs_search_input.strip()),
                                "type": "search"}

    st.session_state.mgrs_search = search_result
                            # Оновлюємо центр карти
                            st.session_state.last_click = None
                            st.success(f"Знайдено координати: {lat:.6f}°, {lon:.6f}°")
                            st.rerun()
                        else:
                            st.error("Не вдалося конвертувати MGRS координати")
                    else:
                        st.error("Некоректний формат MGRS координат")
                else:
                    st.error("Будь ласка, введіть MGRS координати")
        
        with col_search2:
            if st.button("🗑️ Очистити пошук", key="clear_search"):
                st.session_state.mgrs_search = None
                st.rerun()
        
        if st.session_state.mgrs_search:
            st.info(f"**Пошук:** {st.session_state.mgrs_search['mgrs']}")
        
        st.markdown("---")
        st.markdown("### Додати точку")
        
        # Ручний ввід координат для додавання точок
        col_click1, col_click2 = st.columns(2)
        with col_click1:
            click_lat = st.number_input("Широта (°)", min_value=-90.0, max_value=90.0, value=50.4501, key="click_lat")
        with col_click2:
            click_lon = st.number_input("Довгота (°)", min_value=-180.0, max_value=180.0, value=30.5234, key="click_lon")
        
        click_alt = st.number_input("Висота (м)", min_value=-10000.0, max_value=100000.0, value=0.0, key="click_alt")
        
        if st.button("✅ Додати точку на карту", key="add_point"):
            result = calc_point(click_lat, click_lon, click_alt, decimal_year(date.today()))
            if "error" not in result:
                st.session_state.last_click = result
                if len(st.session_state.selected_points) >= 10:
                    st.session_state.selected_points.pop(0)
                st.session_state.selected_points.append(result)
                st.success(f"Додано точку: {click_lat:.4f}°, {click_lon:.4f}°")
                st.rerun()

    with col_map1:
        # Визначення центру карти
        if st.session_state.mgrs_search:
            # Якщо є пошук - центруємо на ньому
            map_center_lat = st.session_state.mgrs_search['lat']
            map_center_lon = st.session_state.mgrs_search['lon']
            map_zoom = 14  # Збільшений масштаб для пошуку
        elif st.session_state.last_click:
            # Якщо є остання точка - центруємо на ній
            map_center_lat = st.session_state.last_click['lat']
            map_center_lon = st.session_state.last_click['lon']
            map_zoom = 12
        else:
            # Інакше використовуємо налаштування користувача
            map_center_lat = center_lat
            map_center_lon = center_lon
            map_zoom = zoom_level
        
        # Підготовка точок для карти
        all_points = st.session_state.selected_points.copy()
        
        # Створення Leaflet карти
        map_html = create_leaflet_map(
            center_lat=map_center_lat,
            center_lon=map_center_lon,
            zoom=map_zoom,
            points=all_points,
            search_point=st.session_state.mgrs_search
        )
        
        # Відображення карти
        components.html(map_html, height=600)
        
        # Інформація про взаємодію з картою
        st.info("""
        **💡 Інструкція:**
        - **🔴 Червоні точки**: Історія ваших виборів (натисніть для деталей)
        - **🟠 Помаранчеві зірки**: Точки пошуку MGRS
        - **📏 Масштаб**: Використовуйте кнопки +/- або прокрутку миші
        - **🗺️ Переміщення**: Перетягніть карту для навігації
        """)
        
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
# Інформація про точку пошуку MGRS
        if st.session_state.mgrs_search:
            search = st.session_state.mgrs_search
            with st.expander("🔍 Інформація про точку пошуку MGRS", expanded=True):
                col_search1, col_search2 = st.columns(2)
                with col_search1:
                    st.write(f"**📍 MGRS координати:**")
                    st.write(f"**Код:** {search['mgrs']}")
                    st.write(f"**Точність:** {search['precision']}")
                    st.write(f"**Широта:** {search['lat']:.6f}°")
                    st.write(f"**Довгота:** {search['lon']:.6f}°")
                
                with col_search2:
                    # Розрахунок параметрів для точки пошуку
                    search_result = calc_point(search['lat'], search['lon'], 0, decimal_year(date.today()))
                    if "error" not in search_result:
                        st.write(f"**⚡ Геомагнітні параметри:**")
                        st.write(f"Деклінація: {search_result['decl']}°")
                        st.write(f"Інтенсивність: {search_result['total']:,.1f} nT")
                        st.write(f"UTM: {search_result['utm_zone']}")
                    
                    if st.button("➕ Додати до історії", key="add_search_to_history"):
                        if "error" not in search_result:
                            st.session_state.last_click = search_result
                            if len(st.session_state.selected_points) >= 10:
                                st.session_state.selected_points.pop(0)
                            st.session_state.selected_points.append(search_result)
                            st.success("Точку додано до історії!")
                            st.rerun()

