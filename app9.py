"""
Geomagnetic Pro 2025 — Оптимізована повна версія
Підтримка: Висота | Теплова карта | Вектори | Клік | MGRS | UTM
Примітка: для інтерактивних кліків на plotly потрібна бібліотека `streamlit-plotly-events`.
Якщо її немає, працює fallback: ручний ввід координат після кліку (через інтерфейс).
"""

from __future__ import annotations

import os
import io
import zipfile
import tempfile
from datetime import date, datetime
from typing import List, Dict, Tuple, Optional

import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import mgrs
import plotly.graph_objects as go
from pyproj import Proj
from pygeomag import GeoMag
import streamlit as st
from shapely.geometry import Point, LineString, Polygon, MultiLineString, MultiPolygon

# ---------- Конфіг ----------
st.set_page_config(page_title="Geomagnetic Pro 2025", page_icon="🧭", layout="wide")
MGRS_CONV = mgrs.MGRS()
COF_DIR = "wmm"
COF_FILENAME = "WMM_2025.COF"
COF_PATH = os.path.join(COF_DIR, COF_FILENAME)
COF_URL = "https://www.ncei.noaa.gov/data/world-magnetic-model-2025/full/WMM_COEFS-2025.zip"
RTDM_API = "https://geomag.usgs.gov/ws/edge/"

# ---------- Допоміжні: завантаження WMM (ліниво) ----------
@st.cache_resource(show_spinner="Завантаження WMM2025...")
def download_wmm(cof_path: str = COF_PATH, cof_url: str = COF_URL) -> str:
    """Завантажити файл WMM_2025.COF у локальну теку (ліниво). Повертає шлях."""
    if os.path.exists(cof_path):
        return cof_path
    os.makedirs(os.path.dirname(cof_path) or ".", exist_ok=True)
    try:
        r = requests.get(cof_url, timeout=30)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            # знайдемо файл з потрібною назвою і витягнемо
            target = None
            for name in z.namelist():
                if name.endswith(COF_FILENAME) or name.endswith(".COF"):
                    target = name
                    break
            if target is None:
                # якщо немає файлу з точною назвою, спробуємо знайти перший .COF
                for name in z.namelist():
                    if name.lower().endswith(".cof"):
                        target = name
                        break
            if target is None:
                raise RuntimeError("WMM COF не знайдено у zip-архіві.")
            with z.open(target) as src, open(cof_path, "wb") as dst:
                dst.write(src.read())
        return cof_path
    except Exception as e:
        st.error(f"Помилка завантаження WMM: {e}")
        st.stop()

@st.cache_resource
def get_wmm() -> GeoMag:
    """Повертає ініціалізований GeoMag (завантажить COF якщо потрібно)."""
    if not os.path.exists(COF_PATH):
        download_wmm()
    # GeoMag приймає шлях до файлу коефіцієнтів
    return GeoMag(coefficients_file=COF_PATH)

# ---------- Дата у десятковому році ----------
def decimal_year(d: date | datetime) -> float:
    """Повертає десятковий рік (враховує час доби, якщо datetime)."""
    if isinstance(d, date) and not isinstance(d, datetime):
        d = datetime(d.year, d.month, d.day)
    start = datetime(d.year, 1, 1)
    next_year = datetime(d.year + 1, 1, 1)
    year_length = (next_year - start).total_seconds()
    elapsed = (d - start).total_seconds()
    return d.year + elapsed / year_length

# ---------- RTDM (on-demand, кешований) ----------
@st.cache_data(ttl=1800)
def get_rtdm(lat: float, lon: float, alt_m: float = 0.0) -> Dict:
    """Отримати оперативні RTDM-дані (USGS). Повертає словник з доповненнями."""
    try:
        params = {"latitude": float(lat), "longitude": float(lon), "altitude": float(alt_m) / 1000.0, "format": "json"}
        r = requests.get(RTDM_API, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        # захищене зчитування значень — якщо ключа немає, повернемо 0 або 'unknown'
        decl = data.get("declination")
        quiet_decl = data.get("quiet_declination")
        total = data.get("total_intensity")
        quiet_total = data.get("quiet_intensity")
        storm = data.get("storm_level", data.get("magnetic_storm_level", "quiet"))
        return {
            "decl_rt": (decl - quiet_decl) if decl is not None and quiet_decl is not None else 0.0,
            "total_rt": (total - quiet_total) if total is not None and quiet_total is not None else 0.0,
            "storm": storm
        }
    except Exception as e:
        st.warning(f"RTDM недоступний: {e}. Використовується лише WMM.")
        return {"decl_rt": 0.0, "total_rt": 0.0, "storm": "offline"}

# ---------- MGRS та UTM батчі ----------
# стабільні функції з обробкою ошибок/невалідних значень

def _safe_to_mgrs(lat: float, lon: float, prec: int = 5) -> str:
    """Безпечна конвертація у MGRS: пробуємо (lat,lon), якщо помилка — пробуємо (lon,lat)."""
    try:
        return MGRS_CONV.toMGRS(lat, lon, prec)
    except Exception:
        try:
            return MGRS_CONV.toMGRS(lon, lat, prec)
        except Exception:
            return "Invalid"

_vec_mgrs = np.vectorize(lambda la, lo: _safe_to_mgrs(float(la), float(lo), 5) if not (np.isnan(la) or np.isnan(lo)) else "Invalid")

@st.cache_data
def batch_mgrs(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Повертає масив MGRS-строк (векторизовано)."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    return _vec_mgrs(lats, lons)

@st.cache_data
def batch_utm(lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Повертає (zones, east, north) для масивів lat, lon.
    zones — dtype=object (щоб підтримувати np.nan), east/north — float масиви з np.nan для невалідних.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    n = len(lats)
    zones = np.empty(n, dtype=object)
    east = np.full(n, np.nan, dtype=float)
    north = np.full(n, np.nan, dtype=float)

    for i, (lat, lon) in enumerate(zip(lats, lons)):
        if np.isnan(lat) or np.isnan(lon):
            zones[i] = np.nan
            continue
        # zone від 1 до 60
        zone = int((lon + 180.0) / 6.0) + 1
        zone = max(1, min(60, zone))
        hemi = 'S' if lat < 0 else 'N'
        zones[i] = f"{zone}{hemi}"
        proj_str = f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs"
        if hemi == 'S':
            proj_str += " +south"
        try:
            p = Proj(proj_str)
            e, n_ = p(lon, lat)
            east[i], north[i] = round(float(e), 1), round(float(n_), 1)
        except Exception:
            zones[i] = np.nan
            east[i] = north[i] = np.nan
    return zones, east, north

# ---------- Обчислення для однієї точки ----------
@st.cache_data
def calc_point(lat: float, lon: float, alt_m: float, year: float) -> Dict:
    """Обчислює WMM+RTDM показники для однієї точки."""
    try:
        lat = float(lat); lon = float(lon); alt_m = float(alt_m)
    except Exception:
        return {"error": "Invalid numeric inputs"}

    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return {"error": "Invalid coordinates"}

    try:
        wmm = get_wmm().calculate(glat=lat, glon=lon, alt=alt_m / 1000.0, time=year)
    except Exception as e:
        return {"error": f"WMM calculation failed: {e}"}

    rt = get_rtdm(lat, lon, alt_m)
    decl = (getattr(wmm, "d", 0.0) or 0.0) + float(rt.get("decl_rt", 0.0))
    total = (getattr(wmm, "f", 0.0) or 0.0) + float(rt.get("total_rt", 0.0))
    mgrs_str = _safe_to_mgrs(lat, lon, 5)

    utm_zone, utm_e, utm_n = None, None, None
    try:
        zones, easts, norths = batch_utm(np.array([lat]), np.array([lon]))
        utm_zone = zones[0]
        utm_e = easts[0]
        utm_n = norths[0]
    except Exception:
        utm_zone = np.nan; utm_e = np.nan; utm_n = np.nan

    return {
        "lat": round(lat, 6), "lon": round(lon, 6), "alt": round(alt_m, 1),
        "decl": round(float(decl), 4), "total": round(float(total), 2),
        "storm": rt.get("storm", "offline"), "mgrs": mgrs_str,
        "utm_zone": utm_zone, "utm_e": utm_e, "utm_n": utm_n
    }

# ---------- Завантаження векторів (GeoJSON / shapefile ZIP) ----------
@st.cache_data
def load_vector(uploaded_file) -> gpd.GeoDataFrame:
    """
    Приймає UploadedFile від streamlit.file_uploader або шлях/буфер.
    Підтримує geojson/json або zip (shapefile всередині).
    Повертає GeoDataFrame в EPSG:4326.
    """
    if uploaded_file is None:
        return gpd.GeoDataFrame()

    try:
        name_lower = uploaded_file.name.lower()
        # GeoJSON / JSON
        if name_lower.endswith((".geojson", ".json")):
            # geopandas може читати file-like об'єкт, тому просто використаємо BytesIO
            data_bytes = uploaded_file.read()
            gdf = gpd.read_file(io.BytesIO(data_bytes))
        elif name_lower.endswith(".zip"):
            # розпаковуємо в тимчасову теку, щоб geopandas міг правильно знайти .shp + .dbf + .shx
            with tempfile.TemporaryDirectory() as td:
                with zipfile.ZipFile(uploaded_file) as z:
                    z.extractall(td)
                # знайдемо перший .shp
                shp_files = []
                for root, _, files in os.walk(td):
                    for f in files:
                        if f.lower().endswith(".shp"):
                            shp_files.append(os.path.join(root, f))
                if not shp_files:
                    st.error("У zip немає .shp файлу")
                    return gpd.GeoDataFrame()
                gdf = gpd.read_file(shp_files[0])
        else:
            st.error("Підтримувані формати: .geojson, .json або .zip (shapefile)")
            return gpd.GeoDataFrame()

        # привести до WGS84
        gdf = gdf.to_crs(epsg=4326)
        if len(gdf) > 500:
            gdf = gdf.sample(500, random_state=42)
        # спростити геометрію (невеликий tolerance)
        gdf["geometry"] = gdf.geometry.simplify(0.001, preserve_topology=True)
        return gdf
    except Exception as e:
        st.error(f"Помилка завантаження вектору: {e}")
        return gpd.GeoDataFrame()

# ---------- Підтримка кліків на plotly (опціонально) ----------
_HAS_PLOTLY_EVENTS = False
try:
    from streamlit_plotly_events import plotly_events
    _HAS_PLOTLY_EVENTS = True
except Exception:
    _HAS_PLOTLY_EVENTS = False
    # не фатально — буде fallback (ручний ввід координат)

# ---------- Інтерфейс ----------
st.title("🧭 Geomagnetic Pro 2025")
st.markdown("**Висота | Теплова карта | Вектори | Клік | MGRS | UTM**")

tabs = st.tabs(["Калькулятор", "Карта", "Пакет"])

# === КАЛЬКУЛЯТОР ===
with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Широта", value=55.755800, format="%.6f", key="calc_lat")
        lon = st.number_input("Довгота", value=37.617300, format="%.6f", key="calc_lon")
    with col2:
        alt = st.number_input("Висота (м)", value=0.0, step=10.0, format="%.1f", key="calc_alt")
    if st.button("Обчислити", type="primary"):
        res = calc_point(lat, lon, alt, decimal_year(datetime.utcnow()))
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
    vector_file = st.file_uploader("Вектор (GeoJSON/Shapefile .zip)", ["geojson", "json", "zip"], key="vec_file")
    gdf: Optional[gpd.GeoDataFrame] = None
    if vector_file:
        with st.spinner("Завантаження вектору..."):
            gdf = load_vector(vector_file)

    col1, col2 = st.columns(2)
    with col1:
        alt_grid = st.slider("Висота сітки (м)", 0, 10000, 0, step=100, key="alt_grid")
    with col2:
        step = st.slider("Крок (°)", 0.1, 1.0, 0.5, step=0.1, key="step_grid")

    # кешування гріду у сесії
    cache_key = f"grid_{alt_grid}_{step}"
    if cache_key not in st.session_state:
        lat_min, lat_max = 48.0, 52.0
        lon_min, lon_max = 22.0, 40.0
        lats = np.arange(lat_min, lat_max + 1e-9, step)
        lons = np.arange(lon_min, lon_max + 1e-9, step)
        grid = [(float(la), float(lo), float(alt_grid)) for la in lats for lo in lons]
        with st.spinner(f"Генерація теплової карти ({len(grid)} точок)..."):
            # обчислюємо серією (кешується в calc_point)
            results = [calc_point(la, lo, al, decimal_year(datetime.utcnow())) for la, lo, al in grid]
        st.session_state[cache_key] = pd.DataFrame(results)

    df_heatmap = st.session_state[cache_key]

    # Побудова карти plotly
    fig = go.Figure()
    # densitymapbox потребує non-null значень
    if "decl" in df_heatmap.columns:
        fig.add_trace(go.Densitymapbox(
            lat=df_heatmap["lat"], lon=df_heatmap["lon"],
            z=df_heatmap["decl"].astype(float).fillna(0.0),
            radius=20,
            colorscale="RdBu", zmid=0, opacity=0.6,
            colorbar=dict(title="Деклінація (°)")
        ))

    # Додаємо вектори, якщо є
    if gdf is not None and not gdf.empty:
        for _, row in gdf.iterrows():
            geom = row.geometry
            try:
                if isinstance(geom, Point):
                    fig.add_trace(go.Scattermapbox(lat=[geom.y], lon=[geom.x], mode="markers", marker=dict(size=8), name="point"))
                elif isinstance(geom, LineString):
                    xs, ys = zip(*list(geom.coords))
                    fig.add_trace(go.Scattermapbox(lat=list(ys), lon=list(xs), mode="lines", line=dict(width=2), name="line"))
                elif isinstance(geom, MultiLineString):
                    for part in geom.geoms:
                        xs, ys = zip(*list(part.coords))
                        fig.add_trace(go.Scattermapbox(lat=list(ys), lon=list(xs), mode="lines", line=dict(width=2)))
                elif isinstance(geom, Polygon):
                    xs, ys = zip(*list(geom.exterior.coords))
                    fig.add_trace(go.Scattermapbox(lat=list(ys), lon=list(xs), fill="toself", fillcolor="rgba(100,0,100,0.1)", line=dict(width=1)))
                elif isinstance(geom, MultiPolygon):
                    for poly in geom.geoms:
                        xs, ys = zip(*list(poly.exterior.coords))
                        fig.add_trace(go.Scattermapbox(lat=list(ys), lon=list(xs), fill="toself", fillcolor="rgba(100,0,100,0.1)", line=dict(width=1)))
            except Exception:
                # якщо якась геометрія нестандартна — пропускаємо
                continue

    # Відмалюємо попередні кліки (збережені в сесії)
    if "click_points" not in st.session_state:
        st.session_state.click_points = []

    for pt in st.session_state.click_points:
        try:
            fig.add_trace(go.Scattermapbox(lat=[pt["lat"]], lon=[pt["lon"]], mode="markers", marker=dict(color="red", size=12)))
        except Exception:
            continue

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=50.0, lon=30.0), zoom=5),
        height=650,
        margin=dict(l=0, r=0, b=0, t=0)
    )

    # Відображення карти та захоплення кліків
    st.write("Клікніть по карті, щоб додати точку. Якщо `streamlit-plotly-events` встановлено — клік буде оброблено автоматично.")
    if _HAS_PLOTLY_EVENTS:
        # використовуємо plotly_events для обробки кліків
        plotted = st.plotly_chart(fig, use_container_width=True, key="map_final")
        events = plotly_events(fig, click_event=True, hover_event=False, key="plotly_events")
        if events:
            # перший клік беремо
            ev = events[0]
            # plotly_events повертає 'lat'/'lon' або 'points' з координатами
            lat_clicked = ev.get("lat") or (ev.get("point", {}).get("lat"))
            lon_clicked = ev.get("lon") or (ev.get("point", {}).get("lon"))
            if lat_clicked is None or lon_clicked is None:
                # інші формати події — спробуємо розпакувати
                pts = ev.get("points")
                if pts and isinstance(pts, list):
                    p0 = pts[0]
                    lat_clicked = p0.get("lat") or p0.get("y")
                    lon_clicked = p0.get("lon") or p0.get("x")
            if lat_clicked is not None and lon_clicked is not None:
                click_alt = st.number_input("Висота кліку (м)", value=alt_grid, key=f"click_alt_{len(st.session_state.click_points)}")
                res = calc_point(lat_clicked, lon_clicked, click_alt, decimal_year(datetime.utcnow()))
                if "error" not in res:
                    st.session_state.click_points.append(res)
                    st.experimental_rerun()
    else:
        # fallback: просто рендеримо графік і даємо полям для ручного введення координат користувачем
        st.plotly_chart(fig, use_container_width=True, key="map_final_no_events")
        st.info("Автоматичне захоплення кліків не встановлено. Можна ввести координати вручну або встановити пакет `streamlit-plotly-events`.")
        with st.form("manual_click_form"):
            mlat = st.number_input("Ввести широту (lat)", format="%.6f", key="manual_lat")
            mlon = st.number_input("Ввести довготу (lon)", format="%.6f", key="manual_lon")
            malt = st.number_input("Висота (м)", value=alt_grid, step=1.0, key="manual_alt")
            add_btn = st.form_submit_button("Додати точку")
            if add_btn:
                res = calc_point(mlat, mlon, malt, decimal_year(datetime.utcnow()))
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.session_state.click_points.append(res)
                    st.experimental_rerun()

    # Показ таблиці кліків
    if st.session_state.click_points:
        df_click = pd.DataFrame(st.session_state.click_points)
        st.dataframe(df_click[["lat", "lon", "alt", "mgrs", "decl", "total"]], use_container_width=True)
        if st.button("Очистити кліки"):
            st.session_state.click_points = []
            st.experimental_rerun()

# === ПАКЕТ (CSV масив точок) ===
with tabs[2]:
    file = st.file_uploader("CSV (стовпці: lat, lon, [alt])", ["csv"], key="pkg_file")
    if file:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"Не вдалось прочитати CSV: {e}")
            st.stop()
        required = ["lat", "lon"]
        if not all(col in df.columns for col in required):
            st.error("CSV повинен містити колонки: lat, lon")
            st.stop()
        if "alt" not in df.columns:
            df["alt"] = 0.0
        else:
            df["alt"] = pd.to_numeric(df["alt"], errors="coerce").fillna(0.0)
        points = [(float(row.lat), float(row.lon), float(row.alt)) for _, row in df.iterrows()]
        with st.spinner("Обчислення для пакету..."):
            results = [calc_point(lat, lon, alt_m, decimal_year(datetime.utcnow())) for lat, lon, alt_m in points]
        df_out = pd.DataFrame(results)
        st.dataframe(df_out, use_container_width=True)
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("Завантажити результат (CSV)", csv_bytes, "geomag_results.csv", "text/csv")

# ---------- Кінець скрипта ----------
