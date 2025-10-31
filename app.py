"""
Geomagnetic Pro 2025 v5.1 — Виправлено кеш, width='stretch', стабільний запуск
"""
from __future__ import annotations

import os
import io
import zipfile
import xml.etree.ElementTree as ET
from datetime import date
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional
import json
import requests
import numpy as np
import pandas as pd
import mgrs
import plotly.graph_objects as go
from pyproj import Proj
from pygeomag import GeoMag
import streamlit as st

# =============================================
# КОНФІГУРАЦІЯ
# =============================================
st.set_page_config(
    page_title="Geomagnetic Pro 2025",
    page_icon="compass",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Папка кешу
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Константи
DATUM = "WGS84"
MGRS_CONV = mgrs.MGRS()
COF_PATH = "wmm/WMM_2025.COF"
COF_URL = "https://www.ncei.noaa.gov/data/world-magnetic-model-2025/full/WMM_COEFS-2025.zip"
RTDM_API = "https://geomag.usgs.gov/ws/edge/"
DEM_API = "https://api.opentopodata.org/v1/srtm30m"

# =============================================
# ОФЛАЙН WMM
# =============================================
@st.cache_resource(show_spinner="Завантаження WMM2025 (офлайн кеш)...")
def load_wmm() -> GeoMag:
    if os.path.exists(COF_PATH) and os.path.getsize(COF_PATH) > 1000:
        return GeoMag(coefficients_file=COF_PATH)
    os.makedirs("wmm", exist_ok=True)
    try:
        r = requests.get(COF_URL, timeout=30)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            cof_file = next(n for n in z.namelist() if n.lower().endswith('.cof'))
            with z.open(cof_file) as src, open(COF_PATH, "wb") as dst:
                dst.write(src.read())
        return GeoMag(coefficients_file=COF_PATH)
    except Exception as e:
        if os.path.exists(COF_PATH):
            st.warning("Використовується локальний WMM2025 (офлайн)")
            return GeoMag(coefficients_file=COF_PATH)
        st.error(f"Помилка WMM: {e}")
        st.info("Завантажте вручну: [NOAA WMM2025](https://www.ncei.noaa.gov)")
        st.stop()

wmm_model = load_wmm()

# =============================================
# ОФЛАЙН DEM
# =============================================
DEM_CACHE = os.path.join(CACHE_DIR, "dem_cache.csv")

def load_dem_cache() -> pd.DataFrame:
    if os.path.exists(DEM_CACHE):
        try:
            df = pd.read_csv(DEM_CACHE)
            if {"lat", "lon", "elev"}.issubset(df.columns):
                return df.set_index(["lat", "lon"])["elev"]
        except Exception as e:
            st.warning(f"Помилка читання кешу DEM: {e}. Перестворюється.")
    return pd.DataFrame(columns=["elev"]).astype(float)

_dem_series = load_dem_cache()

@st.cache_data(ttl=86400)
def get_elevation(lat: float, lon: float) -> float:
    key = (round(lat, 6), round(lon, 6))
    if key in _dem_series.index:
        return float(_dem_series.loc[key])
    try:
        url = f"{DEM_API}?locations={lat},{lon}"
        r = requests.get(url, timeout=10).json()
        elev = r["results"][0]["elevation"]
        if elev is not None:
            new_row = pd.DataFrame([{"lat": key[0], "lon": key[1], "elev": elev}])
            # Виправлення: уникаємо concat з порожнім DataFrame
            if os.path.exists(DEM_CACHE) and os.path.getsize(DEM_CACHE) > 0:
                existing = pd.read_csv(DEM_CACHE)
                if not existing.empty:
                    pd.concat([existing, new_row], ignore_index=True).to_csv(DEM_CACHE, index=False)
                else:
                    new_row.to_csv(DEM_CACHE, index=False)
            else:
                new_row.to_csv(DEM_CACHE, index=False)
            return round(elev, 1)
    except Exception as e:
        st.debug(f"DEM error: {e}")
    return 0.0

# =============================================
# ОФЛАЙН RTDM
# =============================================
RTDM_CACHE = os.path.join(CACHE_DIR, "rtdm_cache.csv")

def load_rtdm_cache() -> pd.DataFrame:
    if os.path.exists(RTDM_CACHE):
        try:
            df = pd.read_csv(RTDM_CACHE)
            if {"lat", "lon", "decl_rt", "total_rt", "storm"}.issubset(df.columns):
                return df.set_index(["lat", "lon"])
        except:
            st.warning("Пошкоджений кеш RTDM. Перестворюється.")
    return pd.DataFrame(columns=["decl_rt", "total_rt", "storm"])

_rtdm_df = load_rtdm_cache()

@st.cache_data(ttl=3600)
def get_rtdm(lat: float, lon: float, alt: float = 0) -> Dict:
    key = (round(lat, 4), round(lon, 4))
    if key in _rtdm_df.index:
        row = _rtdm_df.loc[key]
        return {"decl_rt": row["decl_rt"], "total_rt": row["total_rt"], "storm": row["storm"]}
    try:
        params = {"latitude": lat, "longitude": lon, "altitude": alt / 1000, "format": "json"}
        r = requests.get(RTDM_API, params=params, timeout=10)
        if r.status_code == 200:
            d = r.json()
            decl_rt = d.get("declination", 0) - d.get("quiet_declination", 0)
            total_rt = d.get("total_intensity", 0) - d.get("quiet_intensity", 0)
            storm = d.get("storm_level", "quiet")
            new_row = pd.DataFrame([{"lat": key[0], "lon": key[1], "decl_rt": decl_rt, "total_rt": total_rt, "storm": storm}])
            # Виправлення: безпечний concat
            if os.path.exists(RTDM_CACHE) and os.path.getsize(RTDM_CACHE) > 0:
                existing = pd.read_csv(RTDM_CACHE)
                if not existing.empty:
                    pd.concat([existing, new_row], ignore_index=True).to_csv(RTDM_CACHE, index=False)
                else:
                    new_row.to_csv(RTDM_CACHE, index=False)
            else:
                new_row.to_csv(RTDM_CACHE, index=False)
            return {"decl_rt": decl_rt, "total_rt": total_rt, "storm": storm}
    except:
        pass
    return {"decl_rt": 0.0, "total_rt": 0.0, "storm": "offline"}


# =============================================
# КОНВЕРТОРИ
# =============================================
@st.cache_data
def latlon_to_mgrs(lat: float, lon: float, precision: int = 5) -> str:
    return MGRS_CONV.toMGRS(lat, lon, precision) if -90 <= lat <= 90 and -180 <= lon <= 180 else "Invalid"

@st.cache_data
def mgrs_to_latlon(mgrs_str: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        mgrs_str = mgrs_str.strip().upper()
        digits = ''.join(c for c in mgrs_str if c.isdigit())
        if len(digits) % 2 != 0 or len(digits) > 10 or len(digits) // 2 not in range(1, 6):
            return None, None
        lat, lon = MGRS_CONV.fromMGRS(mgrs_str)
        return round(lat, 8), round(lon, 8)
    except:
        return None, None

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
def decimal_year(d: date) -> float:
    start = date(d.year, 1, 1)
    return d.year + (d - start).days / 365.25

@st.cache_data
def calc_point(lat: float, lon: float, alt: float, year: float) -> Dict:
    final_alt = alt if alt > 0 else get_elevation(lat, lon)
    wmm = wmm_model.calculate(glat=lat, glon=lon, alt=final_alt/1000, time=year)
    rt = get_rtdm(lat, lon, final_alt)
    decl = wmm.d + rt["decl_rt"]
    total = wmm.f + rt["total_rt"]
    mgrs_str = latlon_to_mgrs(lat, lon)
    utm_zone, utm_e, utm_n = latlon_to_utm(lat, lon)
    return {
        "lat": round(lat, 8), "lon": round(lon, 8), "alt": round(final_alt, 1),
        "decl": round(decl, 4), "total": round(total, 2),
        "mgrs": mgrs_str, "utm_zone": utm_zone, "utm_e": utm_e, "utm_n": utm_n
    }

def calc_batch(points: List[Tuple[float, float, float]], year: float) -> List[Dict]:
    with ThreadPoolExecutor() as executor:
        return list(executor.map(lambda p: calc_point(p[0], p[1], p[2], year), points))

# =============================================
# ФОРМАТИ ЕКСПОРТУ
# =============================================
def export_kml(points: List[Dict]) -> bytes:
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = ET.SubElement(kml, "Document")
    for pt in points:
        pm = ET.SubElement(doc, "Placemark")
        ET.SubElement(pm, "name").text = pt["mgrs"]
        ET.SubElement(pm, "description").text = f"Decl: {pt['decl']}° | Int: {pt['total']} nT | Alt: {pt['alt']} m"
        point = ET.SubElement(pm, "Point")
        ET.SubElement(point, "coordinates").text = f"{pt['lon']},{pt['lat']},{pt['alt']}"
    return ET.tostring(kml, encoding="utf-8", xml_declaration=True)

def export_gpx(points: List[Dict]) -> bytes:
    gpx = ET.Element("gpx", version="1.1", creator="Geomagnetic Pro", xmlns="http://www.topografix.com/GPX/1/1")
    for pt in points:
        wpt = ET.SubElement(gpx, "wpt", lat=str(pt["lat"]), lon=str(pt["lon"]))
        ET.SubElement(wpt, "ele").text = str(pt["alt"])
        ET.SubElement(wpt, "name").text = pt["mgrs"]
    return ET.tostring(gpx, encoding="utf-8", xml_declaration=True)

def export_geojson(points: List[Dict]) -> bytes:
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [pt["lon"], pt["lat"], pt["alt"]]},
            "properties": {
                "mgrs": pt["mgrs"], "decl": pt["decl"], "total": pt["total"], "alt": pt["alt"]
            }
        } for pt in points
    ]
    return json.dumps({"type": "FeatureCollection", "features": features}, indent=2).encode()

def import_geojson(file) -> List[Tuple[float, float, float]]:
    data = json.load(file)
    points = []
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") == "Point":
            coords = geom.get("coordinates", [])
            lon, lat = coords[0], coords[1]
            alt = coords[2] if len(coords) > 2 else 0.0
            points.append((lat, lon, alt))
    return points

# =============================================
# ІНТЕРФЕЙС
# =============================================
st.title("Geomagnetic Pro 2025")
st.markdown("**Офлайн-режим | GeoJSON | GPX | KML | WMM2025 + RTDM**")

if "history_points" not in st.session_state:
    st.session_state.history_points = []

tab_conv, tab_calc, tab_map, tab_hist, tab_pkg = st.tabs([
    "Конвертер", "Калькулятор", "Карта", "Історія", "Пакет"
])

# === КОНВЕРТЕР ===
with tab_conv:
    st.subheader("Конвертер координат")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**MGRS to Lat/Lon**")
        mgrs_in = st.text_input("MGRS", key="mgrs_conv")
        if st.button("Конвертувати", key="c1"):
            lat, lon = mgrs_to_latlon(mgrs_in)
            if lat: st.success(f"Lat: {lat}° | Lon: {lon}°")
            else: st.error("Невірний MGRS")
    with col2:
        st.markdown("**Lat/Lon to MGRS**")
        lat_i = st.number_input("Широта", format="%.8f", key="lat_i")
        lon_i = st.number_input("Довгота", format="%.8f", key="lon_i")
        prec = st.select_slider("Точність", [1,2,3,4,5], 5, format_func=lambda x: f"{10**(5-x)} м")
        if st.button("Конвертувати", key="c2"):
            st.success(f"MGRS: `{latlon_to_mgrs(lat_i, lon_i, prec)}`")

    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**UTM to Lat/Lon**")
        z = st.number_input("Зона", 1, 60, 36, key="z1")
        e = st.number_input("E", format="%.3f", key="e1")
        n = st.number_input("N", format="%.3f", key="n1")
        s = st.checkbox("Південь", key="s1")
        if st.button("Конвертувати", key="c3"):
            lat, lon = utm_to_latlon(z, e, n, s)
            st.success(f"Lat: {lat}° | Lon: {lon}°")
    with col4:
        st.markdown("**Lat/Lon to UTM**")
        lat_u = st.number_input("Широта", format="%.8f", key="lat_u")
        lon_u = st.number_input("Довгота", format="%.8f", key="lon_u")
        if st.button("Конвертувати", key="c4"):
            zone, ee, nn = latlon_to_utm(lat_u, lon_u)
            st.success(f"Зона: {zone} | E: {ee} м | N: {nn} м")

# === КАЛЬКУЛЯТОР ===
with tab_calc:
    st.subheader("Калькулятор")
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Широта", format="%.8f", key="calc_lat")
        lon = st.number_input("Довгота", format="%.8f", key="calc_lon")
    with col2:
        use_dem = st.checkbox("Автовисота", True)
        if use_dem:
            with st.spinner("Визначення..."):
                elev = get_elevation(lat, lon)
            alt = elev
            st.info(f"Висота: {elev} м")
        else:
            alt = st.number_input("Висота (м)", value=0.0)
    if st.button("Обчислити", type="primary"):
        res = calc_point(lat, lon, alt, decimal_year(date.today()))
        cols = st.columns(3)
        with cols[0]: st.metric("MGRS", res["mgrs"])
        with cols[1]: st.metric("UTM", f"{res['utm_zone']} {res['utm_e']}E")
        with cols[2]: st.metric("Деклінація", f"{res['decl']}°")
        st.json(res)

# === КАРТА ===
with tab_map:
    st.subheader("Теплова карта по точкам")
    col1, col2 = st.columns(2)
    with col1:
        grid_size = st.slider("Сітка (км)", 5, 50, 10, 5)
        grid_step = st.slider("Крок (км)", 1, 10, 10, 1)
    with col2:
        grid_alt = st.number_input("Висота сітки (м)", value=2000.0)

    km_per_deg = 111.32
    size_deg = grid_size / km_per_deg
    step_deg = grid_step / km_per_deg

    def add_point(lat: float, lon: float, source: str, alt: float = 0.0):
        res = calc_point(lat, lon, alt, decimal_year(date.today()))
        res["source"] = source
        st.session_state.history_points.append(res)

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**MGRS**")
        mgrs_in = st.text_input("MGRS", key="m_map")
        if st.button("Додати", key="a1"):
            lat, lon = mgrs_to_latlon(mgrs_in)
            if lat: add_point(lat, lon, "MGRS"); st.rerun()
            else: st.error("Невірний MGRS")
    with col_b:
        st.markdown("**UTM**")
        z = st.number_input("Зона", 1, 60, 36, key="z_map")
        e = st.number_input("E", format="%.3f", key="e_map")
        n = st.number_input("N", format="%.3f", key="n_map")
        s = st.checkbox("Південь", key="s_map")
        if st.button("Додати", key="a2"):
            lat, lon = utm_to_latlon(z, e, n, s)
            add_point(lat, lon, "UTM"); st.rerun()

    if st.session_state.get("plotly_events"):
        e = st.session_state.plotly_events[0]
        if e["type"] == "click":
            lat, lon = e["points"][0]["lat"], e["points"][0]["lon"]
            add_point(lat, lon, "Клік"); st.rerun()

    # Сітка
    grid_points = []
    if st.session_state.history_points:
        for pt in st.session_state.history_points:
            lats = np.arange(pt["lat"] - size_deg/2, pt["lat"] + size_deg/2 + step_deg, step_deg)
            lons = np.arange(pt["lon"] - size_deg/2, pt["lon"] + size_deg/2 + step_deg, step_deg)
            for la in lats:
                for lo in lons:
                    if -90 <= la <= 90 and -180 <= lo <= 180:
                        grid_points.append((la, lo, grid_alt))

    df_grid = pd.DataFrame(calc_batch(grid_points, decimal_year(date.today()))) if grid_points else pd.DataFrame()

    # Карта
    fig = go.Figure()
    if not df_grid.empty:
        fig.add_trace(go.Densitymapbox(
            lat=df_grid["lat"], lon=df_grid["lon"], z=df_grid["decl"],
            radius=12, colorscale="RdBu", zmid=0, opacity=0.7,
            colorbar=dict(title="Деклінація (°)")
        ))

    for pt in st.session_state.history_points:
        color = {"Клік": "red", "MGRS": "dodgerblue", "UTM": "orange", "GeoJSON": "green", "GPX": "purple"}.get(pt["source"], "white")
        fig.add_trace(go.Scattermapbox(
            lat=[pt["lat"]], lon=[pt["lon"]], mode="markers",
            marker=dict(color=color, size=12),
            text=f"{pt['mgrs']}<br>{pt['decl']}°",
            hoverinfo="text"
        ))

    center = st.session_state.history_points[-1] if st.session_state.history_points else {"lat": 61.0, "lon": 100.0}
    fig.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox=dict(center=dict(lat=center["lat"], lon=center["lon"]), zoom=8),
        height=600, margin=dict(l=0,r=0,b=0,t=0)
    )
    st.plotly_chart(fig, width='stretch', key="map")

# === ІСТОРІЯ ===
with tab_hist:
    st.subheader("Історія точок")

    uploaded = st.file_uploader("Імпорт (GeoJSON, GPX)", ["geojson", "json", "gpx"], key="import_file")
    if uploaded:
        try:
            if uploaded.name.endswith((".geojson", ".json")):
                points = import_geojson(uploaded)
                source = "GeoJSON"
            else:
                tree = ET.parse(uploaded)
                ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
                points = [(float(w.attrib["lat"]), float(w.attrib["lon"]),
                           float(w.find("gpx:ele", ns).text) if w.find("gpx:ele", ns) is not None else 0.0)
                          for w in tree.findall(".//gpx:wpt", ns)]
                source = "GPX"
            for lat, lon, alt in points:
                add_point(lat, lon, source, alt)
            st.success(f"Імпортовано {len(points)} точок")
            st.rerun()
        except Exception as e:
            st.error(f"Помилка імпорту: {e}")

    if st.session_state.history_points:
        df = pd.DataFrame(st.session_state.history_points)
        df_disp = df[["source", "lat", "lon", "alt", "mgrs", "decl", "total"]].copy()
        df_disp.columns = ["Джерело", "Широта", "Довгота", "Висота", "MGRS", "Деклінація", "Інтенсивність"]
        st.dataframe(df_disp, width='stretch')

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.download_button("CSV", df.to_csv(index=False).encode(), f"geomag_{date.today()}.csv", "text/csv")
        with col2:
            st.download_button("KML", export_kml(st.session_state.history_points), f"geomag_{date.today()}.kml", "application/vnd.google-earth.kml+xml")
        with col3:
            st.download_button("GPX", export_gpx(st.session_state.history_points), f"geomag_{date.today()}.gpx", "application/gpx+xml")
        with col4:
            st.download_button("GeoJSON", export_geojson(st.session_state.history_points), f"geomag_{date.today()}.geojson", "application/json")

        if st.button("Очистити історію", type="secondary"):
            st.session_state.history_points = []
            st.rerun()
    else:
        st.info("Додайте точки або імпортуйте файл.")

# === ПАКЕТ ===
with tab_pkg:
    st.subheader("Пакетне обчислення")
    file = st.file_uploader("CSV (lat,lon,alt)", ["csv"], key="pkg_csv")
    if file:
        df = pd.read_csv(file)
        df["alt"] = df.get("alt", 0).fillna(0)
        points = [(r["lat"], r["lon"], r["alt"]) for _, r in df.iterrows()]
        with st.spinner("Обчислення..."):
            results = calc_batch(points, decimal_year(date.today()))
        df_out = pd.DataFrame(results)
        st.dataframe(df_out, width='stretch')
        st.download_button("Експорт CSV", df_out.to_csv(index=False).encode(), "results.csv", "text/csv")
        