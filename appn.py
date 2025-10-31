"""
Geomagnetic Pro 2025 v6.2
Повністю оптимізований | CoordinateConverter | Офлайн |
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
import plotly.graph_objects as go
from pygeomag import GeoMag
import streamlit as st

# Імпорт модуля
from modules.coordinates import CoordinateConverter

# =============================================
# КОНФІГУРАЦІЯ
# =============================================
st.set_page_config(
    page_title="Geomagnetic Pro 2025",
    page_icon="compass",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Папки
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("wmm", exist_ok=True)

# Константи
COF_PATH = "wmm/WMM_2025.COF"
COF_URL = "https://www.ncei.noaa.gov/data/world-magnetic-model-2025/full/WMM_COEFS-2025.zip"
RTDM_API = "https://geomag.usgs.gov/ws/edge/"
DEM_API = "https://api.opentopodata.org/v1/srtm30m"

# Ініціалізація
converter = CoordinateConverter()

# =============================================
# ОФЛАЙН WMM
# =============================================
@st.cache_resource(show_spinner="Завантаження WMM2025...")
def load_wmm() -> GeoMag:
    if os.path.exists(COF_PATH) and os.path.getsize(COF_PATH) > 1000:
        return GeoMag(coefficients_file=COF_PATH)
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
            st.warning("Офлайн WMM2025")
            return GeoMag(coefficients_file=COF_PATH)
        st.error(f"WMM помилка: {e}")
        st.info("Завантажте вручну: [NOAA WMM2025](https://www.ncei.noaa.gov)")
        st.stop()

wmm_model = load_wmm()

# =============================================
# ОФЛАЙН DEM (безпечний кеш)
# =============================================
DEM_CACHE = os.path.join(CACHE_DIR, "dem_cache.csv")
_dem_series = pd.Series(dtype=float)

if os.path.exists(DEM_CACHE):
    try:
        df = pd.read_csv(DEM_CACHE)
        if {"lat", "lon", "elev"}.issubset(df.columns):
            _dem_series = df.set_index(["lat", "lon"])["elev"]
    except Exception as e:
        st.warning(f"Кеш DEM пошкоджено: {e}")

@st.cache_data(ttl=86400)
def get_elevation(lat: float, lon: float) -> float:
    if not converter.validate_coordinates(lat, lon):
        return 0.0
    key = (round(lat, 6), round(lon, 6))
    if key in _dem_series.index:
        return float(_dem_series.loc[key])
    try:
        r = requests.get(f"{DEM_API}?locations={lat},{lon}", timeout=10).json()
        elev = r["results"][0]["elevation"]
        if elev is not None:
            new_row = pd.DataFrame([{"lat": key[0], "lon": key[1], "elev": elev}])
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
        st.debug(f"DEM: {e}")
    return 0.0

# =============================================
# ОФЛАЙН RTDM
# =============================================
RTDM_CACHE = os.path.join(CACHE_DIR, "rtdm_cache.csv")
_rtdm_df = pd.DataFrame(columns=["decl_rt", "total_rt", "storm"])

if os.path.exists(RTDM_CACHE):
    try:
        df = pd.read_csv(RTDM_CACHE)
        if {"lat", "lon", "decl_rt", "total_rt", "storm"}.issubset(df.columns):
            _rtdm_df = df.set_index(["lat", "lon"])
    except:
        st.warning("Кеш RTDM пошкоджено")

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
# ОБЧИСЛЕННЯ
# =============================================
def decimal_year(d: date) -> float:
    start = date(d.year, 1, 1)
    return d.year + (d - start).days / 365.25

@st.cache_data
def calc_point(lat: float, lon: float, alt: float, year: float) -> Dict:
    if not converter.validate_coordinates(lat, lon, alt):
        return {}
    final_alt = alt if alt > 0 else get_elevation(lat, lon)
    wmm = wmm_model.calculate(glat=lat, glon=lon, alt=final_alt/1000, time=year)
    rt = get_rtdm(lat, lon, final_alt)
    decl = wmm.d + rt["decl_rt"]
    total = wmm.f + rt["total_rt"]
    mgrs_str = converter.latlon_to_mgrs(lat, lon)
    utm_zone, utm_e, utm_n = converter.latlon_to_utm(lat, lon)
    return {
        "lat": round(lat, 8), "lon": round(lon, 8), "alt": round(final_alt, 1),
        "decl": round(decl, 4), "total": round(total, 2),
        "mgrs": mgrs_str, "utm_zone": utm_zone, "utm_e": utm_e, "utm_n": utm_n
    }

def calc_batch(points: List[Tuple[float, float, float]], year: float) -> List[Dict]:
    with ThreadPoolExecutor() as executor:
        return [p for p in executor.map(lambda pt: calc_point(*pt, year), points) if p]

# =============================================
# ФОРМАТИ ЕКСПОРТУ
# =============================================
def export_kml(points: List[Dict]) -> bytes:
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = ET.SubElement(kml, "Document")
    for pt in points:
        pm = ET.SubElement(doc, "Placemark")
        ET.SubElement(pm, "name").text = pt.get("mgrs", "N/A")
        ET.SubElement(pm, "description").text = f"Decl: {pt['decl']}° | Int: {pt['total']} nT | Alt: {pt['alt']} m"
        point = ET.SubElement(pm, "Point")
        ET.SubElement(point, "coordinates").text = f"{pt['lon']},{pt['lat']},{pt['alt']}"
    return ET.tostring(kml, encoding="utf-8", xml_declaration=True)

def export_gpx(points: List[Dict]) -> bytes:
    gpx = ET.Element("gpx", version="1.1", creator="Geomagnetic Pro", xmlns="http://www.topografix.com/GPX/1/1")
    for pt in points:
        wpt = ET.SubElement(gpx, "wpt", lat=str(pt["lat"]), lon=str(pt["lon"]))
        ET.SubElement(wpt, "ele").text = str(pt["alt"])
        ET.SubElement(wpt, "name").text = pt.get("mgrs", "N/A")
    return ET.tostring(gpx, encoding="utf-8", xml_declaration=True)

def export_geojson(points: List[Dict]) -> bytes:
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [pt["lon"], pt["lat"], pt["alt"]]},
            "properties": {"mgrs": pt["mgrs"], "decl": pt["decl"], "total": pt["total"], "alt": pt["alt"]}
        } for pt in points
    ]
    return json.dumps({"type": "FeatureCollection", "features": features}, indent=2).encode()

def import_geojson(file) -> List[Tuple[float, float, float]]:
    try:
        data = json.load(file)
        points = []
        for feat in data.get("features", []):
            geom = feat.get("geometry", {})
            if geom.get("type") == "Point":
                coords = geom.get("coordinates", [])
                if len(coords) >= 2:
                    lon, lat = coords[0], coords[1]
                    alt = coords[2] if len(coords) > 2 else 0.0
                    if converter.validate_coordinates(lat, lon, alt):
                        points.append((lat, lon, alt))
        return points
    except:
        return []

# =============================================
# ІНТЕРФЕЙС
# =============================================
st.title("Geomagnetic Pro 2025")
st.markdown("**Офлайн | GeoJSON | GPX | KML | MGRS/UTM | WMM2025 + RTDM**")

if "history_points" not in st.session_state:
    st.session_state.history_points = []

tabs = st.tabs(["Конвертер", "Калькулятор", "Карта", "Історія", "Пакет"])

# === КОНВЕРТЕР ===
with tabs[0]:
    st.subheader("Конвертер координат")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**MGRS → Lat/Lon**")
        mgrs_in = st.text_input("MGRS", "37UDB1283987243", key="mgrs_in")
        if st.button("Конвертувати", key="conv1"):
            lat, lon = converter.convert_mgrs_to_latlon(mgrs_in)
            if lat:
                st.success(f"**Lat:** {lat}° | **Lon:** {lon}°")
                st.json(converter.get_mgrs_details(mgrs_in))
            else:
                st.error("Невірний MGRS")
    with c2:
        st.markdown("**Lat/Lon → MGRS**")
        lat_i = st.number_input("Широта", format="%.8f", key="lat_in")
        lon_i = st.number_input("Довгота", format="%.8f", key="lon_in")
        prec = st.select_slider("Точність", [1,2,3,4,5], 5)
        if st.button("Конвертувати", key="conv2"):
            mgrs_out = converter.latlon_to_mgrs(lat_i, lon_i, prec)
            st.success(f"MGRS: `{mgrs_out}`")

    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**UTM → Lat/Lon**")
        z = st.number_input("Зона", 1, 60, 36, key="utm_z")
        e = st.number_input("E", format="%.3f", key="utm_e")
        n = st.number_input("N", format="%.3f", key="utm_n")
        s = st.checkbox("Південь", key="utm_s")
        if st.button("Конвертувати", key="conv3"):
            lat, lon = converter.utm_to_latlon(z, e, n, s)
            st.success(f"Lat: {lat}° | Lon: {lon}°")
    with c4:
        st.markdown("**Lat/Lon → UTM**")
        lat_u = st.number_input("Широта", format="%.8f", key="lat_u")
        lon_u = st.number_input("Довгота", format="%.8f", key="lon_u")
        if st.button("Конвертувати", key="conv4"):
            zone, ee, nn = converter.latlon_to_utm(lat_u, lon_u)
            st.success(f"Зона: {zone} | E: {ee} м | N: {nn} м")

# === КАЛЬКУЛЯТОР ===
with tabs[1]:
    st.subheader("Калькулятор")
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Широта", format="%.8f", key="calc_lat")
        lon = st.number_input("Довгота", format="%.8f", key="calc_lon")
    with col2:
        use_dem = st.checkbox("Автовисота", True)
        if use_dem:
            with st.spinner("Визначення висоти..."):
                elev = get_elevation(lat, lon)
            alt = elev
            st.info(f"Висота: {elev} м")
        else:
            alt = st.number_input("Висота (м)", value=0.0)
    if st.button("Обчислити", type="primary"):
        res = calc_point(lat, lon, alt, decimal_year(date.today()))
        if res:
            cols = st.columns(3)
            with cols[0]: st.metric("MGRS", res["mgrs"])
            with cols[1]: st.metric("UTM", f"{res['utm_zone']} {res['utm_e']}E")
            with cols[2]: st.metric("Деклінація", f"{res['decl']}°")
            st.json(res)
        else:
            st.error("Невірні координати")

# === КАРТА ===
with tabs[2]:
    st.subheader("Теплова карта по точкам")
    col1, col2 = st.columns(2)
    with col1:
        grid_size = st.slider("Сітка (км)", 5, 50, 10, 5)
        grid_step = st.slider("Крок (км)", 1, 10, 5, 1)
    with col2:
        grid_alt = st.number_input("Висота сітки (м)", value=2000.0)

    km_per_deg = 111.32
    size_deg = grid_size / km_per_deg
    step_deg = grid_step / km_per_deg

    def add_point(lat: float, lon: float, source: str, alt: float = 0.0):
        if converter.validate_coordinates(lat, lon, alt):
            res = calc_point(lat, lon, alt, decimal_year(date.today()))
            if res:
                res["source"] = source
                st.session_state.history_points.append(res)

    st.divider()
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**MGRS**")
        mgrs_map = st.text_input("MGRS", key="mgrs_map")
        if st.button("Додати", key="add_mgrs"):
            lat, lon = converter.convert_mgrs_to_latlon(mgrs_map)
            if lat: add_point(lat, lon, "MGRS"); st.rerun()
            else: st.error("Невірний MGRS")
    with cb:
        st.markdown("**UTM**")
        z = st.number_input("Зона", 1, 60, 36, key="utm_z_map")
        e = st.number_input("E", format="%.3f", key="utm_e_map")
        n = st.number_input("N", format="%.3f", key="utm_n_map")
        s = st.checkbox("Південь", key="utm_s_map")
        if st.button("Додати", key="add_utm"):
            lat, lon = converter.utm_to_latlon(z, e, n, s)
            add_point(lat, lon, "UTM"); st.rerun()

    if st.session_state.get("plotly_events"):
        e = st.session_state.plotly_events[0]
        if e["type"] == "click":
            lat, lon = e["points"][0]["lat"], e["points"][0]["lon"]
            add_point(lat, lon, "Клік"); st.rerun()

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

    center = st.session_state.history_points[-1] if st.session_state.history_points else {"lat": 50.0, "lon": 30.0}
    fig.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox=dict(center=dict(lat=center["lat"], lon=center["lon"]), zoom=8),
        height=600, margin=dict(l=0,r=0,b=0,t=0)
    )
    st.plotly_chart(fig, width='stretch', key="map_plot")

# === ІСТОРІЯ ===
with tabs[3]:
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
with tabs[4]:
    st.subheader("Пакетне обчислення")
    file = st.file_uploader("CSV (lat,lon,alt)", ["csv"], key="pkg_csv")
    if file:
        try:
            df = pd.read_csv(file)
            df["alt"] = df.get("alt", 0).fillna(0)
            points = [(r["lat"], r["lon"], r["alt"]) for _, r in df.iterrows()]
            with st.spinner("Обчислення..."):
                results = calc_batch(points, decimal_year(date.today()))
            df_out = pd.DataFrame(results)
            st.dataframe(df_out, width='stretch')
            st.download_button("Експорт CSV", df_out.to_csv(index=False).encode(), "results.csv", "text/csv")
        except Exception as e:
            st.error(f"Помилка читання CSV: {e}")
