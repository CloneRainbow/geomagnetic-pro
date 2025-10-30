# =============================================
# 🧭 Geomagnetic Pro 2025 — Покращена версія
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
from plotly.events import click
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
# РОЗШИРЕНІ ФУНКЦІЇ КООРДИНАТ
# =============================================
def get_mgrs_details(mgrs_str: str) -> Dict:
    """Детальна інформація MGRS"""
    try:
        # Розбиваємо MGRS на складові
        if len(mgrs_str) >= 3:
            zone = mgrs_str[:2]
            band = mgrs_str[2]
            square = mgrs_str[3:5]
            easting = mgrs_str[5:10] if len(mgrs_str) > 10 else mgrs_str[5:]
            northing = mgrs_str[10:] if len(mgrs_str) > 10 else ""
            
            return {
                "zone": zone,
                "band": band,
                "square": square,
                "easting": easting,
                "northing": northing,
                "full": mgrs_str
            }
    except:
        pass
    return {"full": mgrs_str}

def convert_mgrs_to_latlon(mgrs_str: str) -> Tuple[float, float]:
    """Конвертація MGRS в географічні координати"""
    try:
        lat, lon = MGRS_CONV.toLatLon(mgrs_str)
        return lat, lon
    except Exception as e:
        st.error(f"Помилка конвертації MGRS: {e}")
        return None, None

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
    mgrs_details = get_mgrs_details(mgrs_str)
    utm_z, utm_e, utm_n = batch_utm(np.array([lat]), np.array([lon]))
    return {
        "lat": round(lat, 6), "lon": round(lon, 6), "alt": round(alt, 1),
        "decl": round(decl, 4), "total": round(total, 2),
        "storm": rt["storm"], 
        "mgrs": mgrs_str,
        "mgrs_details": mgrs_details,
        "utm_zone": utm_z[0], "utm_e": utm_e[0], "utm_n": utm_n[0],
        "timestamp": datetime.now().isoformat()
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

# Ініціалізація сесії для зберігання вибраних точок
if "selected_points" not in st.session_state:
    st.session_state.selected_points = []
if "last_click" not in st.session_state:
    st.session_state.last_click = None

tabs = st.tabs(["Калькулятор", "Карта", "Пакет", "Історія вибору"])

# === КАЛЬКУЛЯТОР ===
with tabs[0]:
    col1, col2 = st.columns(2)
    
    with col1:
        coord_type = st.radio("Тип координат", ["Географічні", "MGRS"], horizontal=True)
        
        if coord_type == "Географічні":
            lat = st.number_input("Широта", value=50.4501, format="%.6f", key="calc_lat")
            lon = st.number_input("Довгота", value=30.5234, format="%.6f", key="calc_lon")
            mgrs_input = None
        else:
            mgrs_input = st.text_input("MGRS координати", value="36TWN1234567890", key="calc_mgrs")
            if mgrs_input:
                lat, lon = convert_mgrs_to_latlon(mgrs_input)
                if lat is not None and lon is not None:
                    st.info(f"Географічні координати: {lat:.6f}, {lon:.6f}")
                else:
                    lat, lon = 50.4501, 30.5234
            else:
                lat, lon = 50.4501, 30.5234
    
    with col2:
        alt = st.number_input("Висота (м)", value=0.0, step=100.0, key="calc_alt")
        calc_date = st.date_input("Дата розрахунку", value=date.today(), key="calc_date")

    if st.button("Обчислити", type="primary", key="calc_btn"):
        res = calc_point(lat, lon, alt, decimal_year(calc_date))
        if "error" in res:
            st.error(res["error"])
        else:
            # Відображення результатів у стовпцях
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.metric("MGRS", res["mgrs"])
                if res["mgrs_details"]:
                    with st.expander("Деталі MGRS"):
                        st.write(f"Зона: {res['mgrs_details'].get('zone', '')}")
                        st.write(f"Банда: {res['mgrs_details'].get('band', '')}")
                        st.write(f"Квадрат: {res['mgrs_details'].get('square', '')}")
                        st.write(f"Easting: {res['mgrs_details'].get('easting', '')}")
                        st.write(f"Northing: {res['mgrs_details'].get('northing', '')}")
            
            with col_res2:
                st.metric("UTM", f"{res['utm_zone']}")
                st.metric("Easting", f"{res['utm_e']} m")
                st.metric("Northing", f"{res['utm_n']} m")
            
            with col_res3:
                st.metric("Деклінація", f"{res['decl']}°")
                st.metric("Інтенсивність", f"{res['total']} nT")
                st.metric("Статорм", res["storm"])
            
            # Збереження в історію
            if len(st.session_state.selected_points) >= 10:
                st.session_state.selected_points.pop(0)
            st.session_state.selected_points.append(res)
            
            with st.expander("Детальна інформація"):
                st.json(res)

# === КАРТА ===
with tabs[1]:
    col_map1, col_map2 = st.columns([3, 1])
    
    with col_map2:
        st.subheader("Налаштування карти")
        vector_file = st.file_uploader("Вектор (GeoJSON/Shapefile)", ["geojson", "json", "zip"], key="map_vector")
        gdf = load_vector(vector_file) if vector_file else None
        
        alt_grid = st.slider("Висота сітки (м)", 0, 10000, 0, 500, key="map_alt")
        step = st.slider("Крок (°)", 0.1, 1.0, 0.5, 0.1, key="map_step")
        
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
        else:
            col_reg1, col_reg2 = st.columns(2)
            with col_reg1:
                lat_min = st.number_input("Мін. широта", value=44.0, key="lat_min")
                lat_max = st.number_input("Макс. широта", value=52.5, key="lat_max")
            with col_reg2:
                lon_min = st.number_input("Мін. довгота", value=22.0, key="lon_min")
                lon_max = st.number_input("Макс. довгота", value=40.0, key="lon_max")

    with col_map1:
        # Генерація теплової карти
        cache_key = f"grid_{alt_grid}_{step}_{region}_{lat_min}_{lat_max}_{lon_min}_{lon_max}"
        if cache_key not in st.session_state:
            lats, lons = np.arange(lat_min, lat_max, step), np.arange(lon_min, lon_max, step)
            if len(lats) > 0 and len(lons) > 0:
                grid = [(la, lo, alt_grid) for la in lats for lo in lons]
                with st.spinner(f"Генерація теплової карти на {alt_grid} м..."):
                    results = [calc_point(*p, decimal_year(date.today())) for p in grid]
                st.session_state[cache_key] = pd.DataFrame(results)
            else:
                st.session_state[cache_key] = pd.DataFrame()

        df_heatmap = st.session_state[cache_key]

        # Створення карти
        fig = go.Figure()

        # Теплова карта
        if not df_heatmap.empty:
            fig.add_trace(go.Densitymapbox(
                lat=df_heatmap["lat"], lon=df_heatmap["lon"], z=df_heatmap["decl"],
                radius=20, colorscale="RdBu", zmid=0, opacity=0.6,
                colorbar=dict(title="Деклінація (°)")
            ))

        # Векторні дані
        if gdf is not None and not gdf.empty:
            for idx, row in gdf.iterrows():
                geom = row.geometry
                if geom.geom_type == 'Point':
                    fig.add_trace(go.Scattermapbox(
                        lat=[geom.y], lon=[geom.x],
                        mode="markers", 
                        marker=dict(color="green", size=10),
                        name=f"Точка {idx}",
                        text=[f"Об'єкт {idx}: {geom.y:.4f}, {geom.x:.4f}"],
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
                        lat=lats_line, lon=lons_line,
                        mode="lines", 
                        line=dict(color="purple", width=3),
                        name=f"Лінія {idx}",
                        text=f"Лінія {idx}",
                        hoverinfo="text"
                    ))

        # Вибрані точки
        if st.session_state.selected_points:
            selected_lats = [p["lat"] for p in st.session_state.selected_points]
            selected_lons = [p["lon"] for p in st.session_state.selected_points]
            selected_texts = [f"Вибрана точка: {p['lat']:.4f}, {p['lon']:.4f}<br>MGRS: {p['mgrs']}<br>Деклінація: {p['decl']}°" 
                             for p in st.session_state.selected_points]
            
            fig.add_trace(go.Scattermapbox(
                lat=selected_lats, lon=selected_lons,
                mode="markers",
                marker=dict(color="red", size=12, symbol="circle"),
                name="Вибрані точки",
                text=selected_texts,
                hoverinfo="text"
            ))

        # Останній клік
        if st.session_state.last_click:
            fig.add_trace(go.Scattermapbox(
                lat=[st.session_state.last_click["lat"]], 
                lon=[st.session_state.last_click["lon"]],
                mode="markers",
                marker=dict(color="blue", size=15, symbol="star"),
                name="Останній клік",
                text=[f"Клік: {st.session_state.last_click['lat']:.4f}, {st.session_state.last_click['lon']:.4f}"],
                hoverinfo="text"
            ))

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(center=dict(lat=(lat_min+lat_max)/2, lon=(lon_min+lon_max)/2), 
                       zoom=5 if region == "Україна" else 3),
            height=600, 
            margin=dict(l=0, r=0, b=0, t=0),
            showlegend=True
        )

        # Відображення карти з обробкою кліків
        map_click = st.plotly_chart(fig, width="stretch", key="interactive_map", on_select="rerun")
        
        # Обробка вибору на карті
        if map_click.selection:
            try:
                point_data = map_click.selection["points"][0]
                clicked_lat = point_data["lat"]
                clicked_lon = point_data["lon"]
                
                # Розрахунок параметрів для вибраної точки
                result = calc_point(clicked_lat, clicked_lon, alt_grid, decimal_year(date.today()))
                st.session_state.last_click = result
                
                # Додавання в історію
                if len(st.session_state.selected_points) >= 10:
                    st.session_state.selected_points.pop(0)
                st.session_state.selected_points.append(result)
                
                st.success(f"Обрано точку: {clicked_lat:.4f}, {clicked_lon:.4f}")
                
            except Exception as e:
                st.error(f"Помилка обробки вибору: {e}")

        # Інформація про останній клік
        if st.session_state.last_click:
            with st.expander("Інформація про обрану точку"):
                lc = st.session_state.last_click
                col_lc1, col_lc2 = st.columns(2)
                with col_lc1:
                    st.write(f"**Координати:** {lc['lat']:.6f}, {lc['lon']:.6f}")
                    st.write(f"**MGRS:** {lc['mgrs']}")
                    st.write(f"**UTM:** {lc['utm_zone']}")
                with col_lc2:
                    st.write(f"**Деклінація:** {lc['decl']}°")
                    st.write(f"**Інтенсивність:** {lc['total']} nT")
                    st.write(f"**Висота:** {lc['alt']} m")

# === ПАКЕТ ===
with tabs[2]:
    st.subheader("Пакетна обробка даних")
    
    file = st.file_uploader("CSV (lat,lon,alt)", ["csv"], key="batch_csv")
    if file:
        df = pd.read_csv(file)
        if not all(c in df.columns for c in ["lat", "lon"]):
            st.error("CSV: потрібні колонки lat, lon")
            st.stop()
        
        df["alt"] = df.get("alt", 0).fillna(0)
        
        # Додаткові опції
        col_batch1, col_batch2 = st.columns(2)
        with col_batch1:
            batch_date = st.date_input("Дата для розрахунків", value=date.today(), key="batch_date")
        with col_batch2:
            include_mgrs_details = st.checkbox("Включити деталі MGRS", value=True)
        
        pts = [(r["lat"], r["lon"], r["alt"]) for _, r in df.iterrows()]
        
        if st.button("Запустити пакетну обробку", type="primary", key="batch_btn"):
            with st.spinner("Обчислення..."):
                results = []
                progress_bar = st.progress(0)
                for i, point in enumerate(pts):
                    res = calc_point(*point, decimal_year(batch_date))
                    if include_mgrs_details and "mgrs_details" in res:
                        res.update(res["mgrs_details"])
                    results.append(res)
                    progress_bar.progress((i + 1) / len(pts))
                
                df_out = pd.DataFrame(results)
                
                # Відображення результатів
                st.dataframe(df_out, width="stretch", height=400)
                
                # Статистика
                st.subheader("Статистика")
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Кількість точок", len(df_out))
                    st.metric("Мін. деклінація", f"{df_out['decl'].min():.2f}°")
                with col_stat2:
                    st.metric("Успішні розрахунки", len(df_out[df_out['decl'].notna()]))
                    st.metric("Макс. деклінація", f"{df_out['decl'].max():.2f}°")
                with col_stat3:
                    st.metric("Середня деклінація", f"{df_out['decl'].mean():.2f}°")
                    st.metric("Storm рівні", df_out['storm'].value_counts().to_dict())
                
                # Завантаження
                csv_data = df_out.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Завантажити CSV", 
                    csv_data,
          