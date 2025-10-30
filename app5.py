"""
Geomagnetic Pro 2025 — Повний, оптимізований, паралельний
WMM2025 | MGRS | UTM | Імпорт/Експорт | Карта | QGIS
"""
from __future__ import annotations

import os
import io
import json
import zipfile
import tempfile
from datetime import date
from typing import List, Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import pandas as pd
import numpy as np
import mgrs
import plotly.express as px
import plotly.graph_objects as go
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

# =============================================
# ЗАВАНТАЖЕННЯ WMM_2025.COF
# =============================================
@st.cache_resource(show_spinner="Завантаження WMM_2025.COF...")
def download_cof() -> str:
    if os.path.exists(COF_PATH):
        return COF_PATH
    os.makedirs("wmm", exist_ok=True)
    import requests
    try:
        r = requests.get(COF_URL, timeout=30)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open("WMM_2025.COF") as src, open(COF_PATH, "wb") as dst:
                dst.write(src.read())
        return COF_PATH
    except Exception as e:
        st.error(f"Помилка завантаження COF: {e}")
        st.info("Завантажте вручну: [NOAA WMM2025](https://www.ncei.noaa.gov/products/world-magnetic-model)")
        st.stop()

COF_PATH = download_cof()

# =============================================
# МОДЕЛЬ
# =============================================
@st.cache_resource
def get_model() -> GeoMag:
    return GeoMag(coefficients_file=COF_PATH)

def decimal_year(d: date) -> float:
    return d.year + (d - date(d.year, 1, 1)).days / 365.25

# =============================================
# КОНВЕРТЕРИ
# =============================================
def mgrs_to_latlon(mgrs_str: str) -> Tuple[float | None, float | None]:
    try:
        lat, lon = MGRS.toLatLon(mgrs_str.upper().replace(" ", ""))
        return round(lat, 6), round(lon, 6)
    except:
        return None, None

def utm_to_latlon(zone: str, easting: float, northing: float) -> Tuple[float | None, float | None]:
    try:
        zn = int(zone[:-1])
        hemi = zone[-1]
        proj = f"+proj=utm +zone={zn} +datum=WGS84 +units=m +no_defs" + (" +south" if hemi == 'S' else "")
        p = Proj(proj)
        lon, lat = p(easting, northing, inverse=True)
        return round(lat, 6), round(lon, 6)
    except:
        return None, None

def to_wgs84(x: float, y: float, crs_from: CRS) -> Tuple[float, float]:
    if crs_from.to_epsg() == 4326:
        return y, x
    t = Transformer.from_crs(crs_from, CRS.from_epsg(4326), always_xy=True)
    lon, lat = t.transform(x, y)
    return lat, lon

def to_utm_mgrs_batch(lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    zones = np.empty(len(lats), dtype='U4')
    eastings = np.empty(len(lats))
    northings = np.empty(len(lats))
    mgrs_list = []
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        zone_num = int((lon + 180) / 6) + 1
        hemi = 'S' if lat < 0 else 'N'
        zones[i] = f"{zone_num}{hemi}"
        proj = f"+proj=utm +zone={zone_num} +datum=WGS84 +units=m +no_defs" + (" +south" if hemi == 'S' else "")
        p = Proj(proj)
        e, n = p(lon, lat)
        eastings[i], northings[i] = round(e, 1), round(n, 1)
        mgrs_list.append(MGRS.toMGRS(lat, lon, 5))
    return zones, eastings, northings, mgrs_list

# =============================================
# ГЕОМАГНІТНИЙ РОЗРАХУНОК (кешований)
# =============================================
@st.cache_data
def _calc_single(lat: float, lon: float, alt: float, year: float) -> Dict:
    r = get_model().calculate(glat=lat, glon=lon, alt=alt/1000, time=year)
    return {
        "decl": round(r.d, 4), "incl": round(r.i, 4), "total": round(r.f, 2),
        "horiz": round(r.h, 2), "X": round(r.x, 2), "Y": round(r.y, 2), "Z": round(r.z, 2),
        "lat": round(lat, 6), "lon": round(lon, 6)
    }

def calc_geomag_batch(wgs_points: List[Tuple[float, float, float]], year: float) -> List[Dict]:
    def worker(p):
        lat, lon, alt = p
        base = _calc_single(lat, lon, alt, year)
        return base

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(worker, wgs_points))

    lats = np.array([r["lat"] for r in results])
    lons = np.array([r["lon"] for r in results])
    zones, es, ns, mgrs_list = to_utm_mgrs_batch(lats, lons)
    for r, z, e, n, m in zip(results, zones, es, ns, mgrs_list):
        r.update({"utm_zone": z, "utm_e": e, "utm_n": n, "mgrs": m})
    return results

# =============================================
# ПАРСЕРИ
# =============================================
def parse_csv(file) -> List[Dict]:
    df = pd.read_csv(file, na_filter=False)
    crs = CRS.from_epsg(int(df["epsg"].iloc[0]) if "epsg" in df.columns and pd.notna(df["epsg"].iloc[0]) else 4326)
    return [
        {"name": row.get("name", f"P{i+1}"), "x": row["lon"], "y": row["lat"], "alt": row.get("alt", 0), "crs": crs}
        for i, row in df.iterrows() if pd.notna(row.get("lon")) and pd.notna(row.get("lat"))
    ]

def parse_geojson(file) -> List[Dict]:
    data = json.load(file)
    crs = CRS.from_epsg(4326)
    if data.get("crs"):
        name = data["crs"]["properties"].get("name", "")
        if "EPSG" in name:
            crs = CRS.from_epsg(int(name.split(":")[-1]))
    return [
        {"name": f["properties"].get("name", "P"), "x": f["geometry"]["coordinates"][0],
         "y": f["geometry"]["coordinates"][1], "alt": f["properties"].get("alt", 0), "crs": crs}
        for f in data.get("features", [])
    ]

def parse_shapefile(zip_file) -> List[Dict]:
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_file) as z:
            z.extractall(tmp)
        gdf = gpd.read_file(tmp)
        return [
            {"name": row.get("name", f"P{i+1}"), "x": row.geometry.x, "y": row.geometry.y,
             "alt": row.get("alt", 0), "crs": gdf.crs or CRS.from_epsg(4326)}
            for i, row in gdf.iterrows()
        ]

def parse_kml_kmz(file_bytes: bytes) -> List[Dict]:
    if file_bytes.startswith(b"PK"):
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            kml_file = [f for f in z.namelist() if f.endswith(".kml")][0]
            content = z.read(kml_file)
    else:
        content = file_bytes
    kml_doc = kml.KML()
    kml_doc.from_string(content)

    def extract(feature):
        if hasattr(feature, "geometry") and feature.geometry and feature.geometry.geom_type == "Point":
            lon, lat = feature.geometry.x, feature.geometry.y
            alt = getattr(feature.geometry, "z", 0) or 0
            return {"name": feature.name or "Point", "x": lon, "y": lat, "alt": alt, "crs": CRS.from_epsg(4326)}
        return None

    points = []
    for f in kml_doc.features():
        if hasattr(f, "features"):
            for sub in f.features():
                p = extract(sub)
                if p:
                    points.append(p)
        else:
            p = extract(f)
            if p:
                points.append(p)
    return points

def parse_gpx(file_bytes: bytes) -> List[Dict]:
    gpx = gpxpy.parse(file_bytes.decode("utf-8", errors="ignore"))
    points = []
    for wp in gpx.waypoints:
        points.append({"name": wp.name or "WP", "x": wp.longitude, "y": wp.latitude, "alt": wp.elevation or 0, "crs": CRS.from_epsg(4326)})
    return points

def parse_geotiff(file) -> List[Dict]:
    with rasterio.open(file) as src:
        crs = src.crs or CRS.from_epsg(4326)
        transform = src.transform
        band = src.read(1)
        height, width = band.shape
        step = max(1, height // 50)
        points = []
        for row in range(0, height, step):
            for col in range(0, width, step):
                if not np.isnan(band[row, col]):
                    x, y = transform * (col, row)
                    lat, lon = to_wgs84(x, y, crs)
                    points.append({"name": f"Pixel_{row}_{col}", "x": lon, "y": lat, "alt": 0, "crs": CRS.from_epsg(4326)})
        return points

# =============================================
# ЕКСПОРТ
# =============================================
def export_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def export_geojson(df: pd.DataFrame) -> str:
    features = [
        geojson.Feature(geometry=geojson.Point((r.lon, r.lat)), properties=r.drop(["lat", "lon"]).to_dict())
        for _, r in df.iterrows()
    ]
    return json.dumps(geojson.FeatureCollection(features))

def export_shapefile(df: pd.DataFrame) -> bytes:
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/results"
        gdf.to_file(path, driver="ESRI Shapefile")
        with zipfile.ZipFile(buf, "w") as z:
            for ext in ["shp", "shx", "dbf", "prj", "cpg"]:
                f = f"{path}.{ext}"
                if os.path.exists(f):
                    z.write(f, f"results.{ext}")
    buf.seek(0)
    return buf.read()

def export_kml(df: pd.DataFrame) -> bytes:
    kml_doc = simplekml.Kml()
    for _, r in df.iterrows():
        p = kml_doc.newpoint(name=r["name"], coords=[(r["lon"], r["lat"])])
        p.description = f"Decl: {r['decl']}° | MGRS: {r['mgrs']}"
    buf = io.BytesIO()
    buf.write(kml_doc.kml().encode("utf-8"))
    buf.seek(0)
    return buf.read()

def export_gpx(df: pd.DataFrame) -> str:
    gpx = gpxpy.gpx.GPX()
    for _, r in df.iterrows():
        wp = gpxpy.gpx.GPXWaypoint()
        wp.latitude = r["lat"]
        wp.longitude = r["lon"]
        wp.elevation = 0
        wp.name = r["name"]
        wp.description = f"Decl: {r['decl']}°"
        gpx.waypoints.append(wp)
    return gpx.to_xml()

def export_geotiff(df: pd.DataFrame, field: str = "decl") -> bytes:
    if len(df) < 3:
        return b""
    lons, lats, values = df["lon"].values, df["lat"].values, df[field].values
    res = max(0.001, (lons.max() - lons.min()) / 1000)
    lon_grid = np.linspace(lons.min(), lons.max(), 1000)
    lat_grid = np.linspace(lats.min(), lats.max(), 1000)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
    points = np.column_stack((lons, lats))
    interp = NearestNDInterpolator(points, values)
    grid = interp(lon_grid, lat_grid)
    transform = from_origin(lons.min(), lats.max(), res, res)
    with rasterio.MemoryFile() as memfile:
        with memfile.open(driver="GTiff", height=1000, width=1000, count=1, dtype='float32', crs='EPSG:4326', transform=transform) as dst:
            dst.write(grid.astype('float32'), 1)
        return memfile.read()

# =============================================
# ІНТЕРФЕЙС
# =============================================
st.title("🧭 Geomagnetic Pro 2025")
st.markdown("**WMM2025 | Паралельні обчислення | MGRS | UTM | Імпорт/Експорт**")

tabs = st.tabs(["Калькулятор", "Пакет", "Карта + Пошук", "QGIS"])

# === КАЛЬКУЛЯТОР ===
with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Широта", value=55.7558, format="%.6f", key="calc_lat")
        lon = st.number_input("Довгота", value=37.6173, format="%.6f", key="calc_lon")
    with col2:
        alt = st.number_input("Висота (м)", value=0.0, key="calc_alt")
        calc_date = st.date_input("Дата", value=date(2025, 11, 5), key="calc_date")
    if st.button("Обчислити", type="primary", key="calc_btn"):
        res = _calc_single(lat, lon, alt, decimal_year(calc_date))
        utm, e, n, mgrs_str = to_utm_mgrs_batch(np.array([lat]), np.array([lon]))[0][0], to_utm_mgrs_batch(np.array([lat]), np.array([lon]))[1][0], to_utm_mgrs_batch(np.array([lat]), np.array([lon]))[2][0], to_utm_mgrs_batch(np.array([lat]), np.array([lon]))[3][0]
        res.update({"name": "Тест", "mgrs": mgrs_str, "utm_zone": utm, "utm_e": e, "utm_n": n})
        st.metric("Деклінація", f"{res['decl']}°")
        st.metric("MGRS", res["mgrs"])
        st.json(res)

# === ПАКЕТ (ПАРАЛЕЛЬНИЙ) ===
with tabs[1]:
    uploaded = st.file_uploader(
        "CSV | GeoJSON | Shapefile | KML/KMZ | GPX | GeoTIFF",
        type=["csv", "json", "geojson", "zip", "kml", "kmz", "gpx", "tif", "tiff"],
        key="pkg_upload"
    )
    if uploaded:
        ext = uploaded.name.split(".")[-1].lower()
        try:
            if ext == "csv":
                points = parse_csv(uploaded)
            elif ext in ["json", "geojson"]:
                points = parse_geojson(uploaded)
            elif ext == "zip":
                points = parse_shapefile(uploaded)
            elif ext in ["kml", "kmz"]:
                points = parse_kml_kmz(uploaded.read())
            elif ext == "gpx":
                points = parse_gpx(uploaded.read())
            elif ext in ["tif", "tiff"]:
                points = parse_geotiff(uploaded)
            else:
                st.error("Формат не підтримується")
                st.stop()

            year = decimal_year(st.date_input("Дата", date.today(), key="pkg_date"))
            wgs_points = [(to_wgs84(p["x"], p["y"], p["crs"]) + (p.get("alt", 0),)) for p in points]
            with st.spinner(f"Паралельне обчислення {len(wgs_points)} точок..."):
                results = calc_geomag_batch(wgs_points, year)
                for r, p in zip(results, points):
                    r["name"] = p.get("name", f"P{results.index(r)+1}")
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1: st.download_button("CSV", export_csv(df), "geomag.csv", "text/csv")
            with col2: st.download_button("GeoJSON", export_geojson(df), "geomag.geojson", "application/json")
            with col3: st.download_button("Shapefile", export_shapefile(df), "geomag.zip", "application/zip")
            with col4: st.download_button("KML", export_kml(df), "geomag.kml", "application/vnd.google-earth.kml+xml")
            with col5: st.download_button("GPX", export_gpx(df), "geomag.gpx", "application/gpx+xml")
            with col6:
                field = st.selectbox("Поле", ["decl", "incl", "total"], key="tif_field")
                tif = export_geotiff(df, field)
                if tif:
                    st.download_button("GeoTIFF", tif, f"geomag_{field}.tif", "image/tiff")
        except Exception as e:
            st.error(f"Помилка: {e}")

# === КАРТА + ПОШУК (ПАРАЛЕЛЬНИЙ) ===
with tabs[2]:
    if "map_points" not in st.session_state:
        st.session_state.map_points = []

    st.subheader("🎯 Введіть координати lat/lon")
    col_lat, col_lon, col_add = st.columns([1, 1, 1])
    with col_lat:
        lat_input = st.number_input("Широта", value=50.450000, format="%.6f", key="map_lat_input")
    with col_lon:
        lon_input = st.number_input("Довгота", value=30.524000, format="%.6f", key="map_lon_input")
    with col_add:
        if st.button("➕ Додати", key="map_add_latlon_btn"):
            st.session_state.map_points.append((lat_input, lon_input))
            st.success(f"Додано: {lat_input:.6f}, {lon_input:.6f}")
            st.rerun()

    st.markdown("---")
    st.subheader("🔢 MGRS → lat/lon")
    mgrs_input = st.text_input("MGRS", key="map_mgrs_input")
    if st.button("➕ Конвертувати MGRS", key="map_convert_mgrs_btn"):
        lat, lon = mgrs_to_latlon(mgrs_input)
        if lat:
            st.session_state.map_points.append((lat, lon))
            st.success(f"MGRS → {lat:.6f}, {lon:.6f}")
            st.rerun()
        else:
            st.error("Невірний MGRS")

    st.markdown("---")
    st.subheader("📏 UTM → lat/lon")
    c1, c2, c3, c4 = st.columns(4)
    with c1: utm_zone = st.text_input("Зона", "35U", key="map_utm_zone")
    with c2: utm_e = st.number_input("Easting", value=524136.0, key="map_utm_e")
    with c3: utm_n = st.number_input("Northing", value=5584136.0, key="map_utm_n")
    with c4:
        if st.button("➕ Конвертувати UTM", key="map_convert_utm_btn"):
            lat, lon = utm_to_latlon(utm_zone, utm_e, utm_n)
            if lat:
                st.session_state.map_points.append((lat, lon))
                st.success(f"UTM → {lat:.6f}, {lon:.6f}")
                st.rerun()
            else:
                st.error("Невірний UTM")

    st.markdown("---")
    st.subheader("🗺️ Карта")
    if st.session_state.map_points:
        year = decimal_year(date.today())
        wgs_points = [(lat, lon, 0) for lat, lon in st.session_state.map_points]
        with st.spinner("Паралельне обчислення..."):
            results = calc_geomag_batch(wgs_points, year)
        df_map = pd.DataFrame(results)
        fig = px.scatter_mapbox(df_map, lat="lat", lon="lon", color="decl", hover_data=["mgrs"], mapbox_style="open-street-map")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True, key="geomag_map")

        if st.button("🗑️ Очистити", key="clear_map_btn"):
            st.session_state.map_points = []
            st.rerun()
    else:
        st.info("Додайте точки")

# === QGIS ===
with tabs[3]:
    st.markdown("### Експорт для QGIS")
    st.code('''layer = QgsVectorLayer("geomag.geojson", "Geomag", "ogr")
QgsProject.instance().addMapLayer(layer)''', language="python")
