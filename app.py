"""
Geomagnetic Pro 2025
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
from shapely.geometry import Point
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
def download_wmm_cof() -> None:
    if not os.path.exists(COF_PATH):
        os.makedirs("wmm", exist_ok=True)
        with st.spinner("Завантаження WMM_2025.COF..."):
            try:
                import requests
                zip_data = requests.get(COF_URL, timeout=30).content
                with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                    with z.open("WMM_2025.COF") as src, open(COF_PATH, "wb") as dst:
                        dst.write(src.read())
                st.success("WMM_2025.COF завантажено!")
            except Exception as e:
                st.error(f"Не вдалося завантажити COF: {e}")


download_wmm_cof()


# =============================================
# МОДЕЛЬ
# =============================================
@st.cache_resource
def get_geomag_model() -> GeoMag:
    return GeoMag(coefficients_file=COF_PATH)


def decimal_year(d: date) -> float:
    start = date(d.year, 1, 1)
    return d.year + (d - start).days / 365.25


# =============================================
# КОНВЕРТЕРИ
# =============================================
def mgrs_to_latlon(mgrs_str: str) -> Tuple[float | None, float | None]:
    try:
        lat, lon = MGRS.toLatLon(mgrs_str.upper().replace(" ", ""))
        return round(lat, 6), round(lon, 6)
    except Exception:
        return None, None


def utm_to_latlon(zone: str, easting: float, northing: float) -> Tuple[float | None, float | None]:
    try:
        zone_num = int(zone[:-1])
        hemi = zone[-1]
        proj_str = f"+proj=utm +zone={zone_num} +datum=WGS84 +units=m +no_defs"
        if hemi == "S":
            proj_str += " +south"
        p = Proj(proj_str)
        lon, lat = p(easting, northing, inverse=True)
        return round(lat, 6), round(lon, 6)
    except Exception:
        return None, None


def to_utm_mgrs(lat: float, lon: float) -> Tuple[str, float, float, str]:
    zone = int((lon + 180) / 6) + 1
    hemi = "S" if lat < 0 else "N"
    proj_str = f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs"
    if hemi == "S":
        proj_str += " +south"
    p = Proj(proj_str)
    e, n = p(lon, lat)
    return f"{zone}{hemi}", round(e, 1), round(n, 1), MGRS.toMGRS(lat, lon, 5)


def to_wgs84(x: float, y: float, crs_from: CRS) -> Tuple[float, float]:
    if crs_from.to_epsg() == 4326:
        return y, x
    t = Transformer.from_crs(crs_from, CRS.from_epsg(4326), always_xy=True)
    lon, lat = t.transform(x, y)
    return lat, lon


# =============================================
# ГЕОМАГНІТНИЙ РОЗРАХУНОК
# =============================================
@st.cache_data
def calc_geomag_point(lat: float, lon: float, alt: float, year: float) -> Dict[str, Any]:
    model = get_geomag_model()
    r = model.calculate(glat=lat, glon=lon, alt=alt / 1000, time=year)
    utm_zone, utm_e, utm_n, mgrs_str = to_utm_mgrs(lat, lon)
    return {
        "name": "",
        "decl": round(r.d, 4),
        "incl": round(r.i, 4),
        "total": round(r.f, 2),
        "horiz": round(r.h, 2),
        "X": round(r.x, 2),
        "Y": round(r.y, 2),
        "Z": round(r.z, 2),
        "mgrs": mgrs_str,
        "utm_zone": utm_zone,
        "utm_e": utm_e,
        "utm_n": utm_n,
        "lat": round(lat, 6),
        "lon": round(lon, 6),
    }


# =============================================
# ПАРСЕРИ
# =============================================
def parse_csv(file) -> List[Dict]:
    df = pd.read_csv(file)
    crs = CRS.from_epsg(int(df.get("epsg", [4326]).iloc[0]))
    return [
        {
            "name": row.get("name", f"P{i+1}"),
            "x": row.get("lon", row.get("x")),
            "y": row.get("lat", row.get("y")),
            "alt": row.get("alt", 0),
            "crs": crs,
        }
        for i, row in df.iterrows()
        if pd.notna(row.get("lon")) or pd.notna(row.get("lat"))
    ]


def parse_geojson(file) -> List[Dict]:
    data = json.load(file)
    crs = CRS.from_epsg(4326)
    if data.get("crs"):
        name = data["crs"]["properties"].get("name", "")
        if "EPSG" in name:
            crs = CRS.from_epsg(int(name.split(":")[-1]))
    points = []
    for f in data.get("features", []):
        coords = f["geometry"]["coordinates"]
        props = f["properties"]
        points.append(
            {
                "name": props.get("name", "Point"),
                "x": coords[0],
                "y": coords[1],
                "alt": props.get("alt", 0),
                "crs": crs,
            }
        )
    return points


def parse_shapefile(zip_file) -> List[Dict]:
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_file) as z:
            z.extractall(tmp)
        gdf = gpd.read_file(tmp)
        return [
            {
                "name": row.get("name", f"P{i+1}"),
                "x": row.geometry.x,
                "y": row.geometry.y,
                "alt": row.get("alt", 0),
                "crs": gdf.crs or CRS.from_epsg(4326),
            }
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
        points.append(
            {
                "name": wp.name or "WP",
                "x": wp.longitude,
                "y": wp.latitude,
                "alt": wp.elevation or 0,
                "crs": CRS.from_epsg(4326),
            }
        )
    return points


def parse_geotiff(file) -> List[Dict]:
    with rasterio.open(file) as src:
        crs = src.crs or CRS.from_epsg(4326)
        transform = src.transform
        band = src.read(1)
        height, width = band.shape
        points = []
        step = max(1, height // 50)
        for row in range(0, height, step):
            for col in range(0, width, step):
                if not np.isnan(band[row, col]):
                    x, y = transform * (col, row)
                    lat, lon = to_wgs84(x, y, crs)
                    points.append(
                        {
                            "name": f"Pixel_{row}_{col}",
                            "x": lon,
                            "y": lat,
                            "alt": 0,
                            "crs": CRS.from_epsg(4326),
                        }
                    )
        return points


# =============================================
# ЕКСПОРТ
# =============================================
def export_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def export_geojson(df: pd.DataFrame) -> str:
    features = [
        geojson.Feature(
            geometry=geojson.Point((r.lon, r.lat)),
            properties=r.drop(["lat", "lon"]).to_dict(),
        )
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
    lons, lats = df["lon"].values, df["lat"].values
    values = df[field].values
    res = 0.001
    lon_grid = np.arange(lons.min(), lons.max() + res, res)
    lat_grid = np.arange(lats.min(), lats.max() + res, res)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
    points = np.column_stack((lons, lats))
    interp = NearestNDInterpolator(points, values)
    grid = interp(lon_grid, lat_grid)
    transform = from_origin(lons.min(), lats.max(), res, res)
    buf = io.BytesIO()
    with rasterio.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=grid.shape[0],
            width=grid.shape[1],
            count=1,
            dtype=grid.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(grid, 1)
        return memfile.read()
    return b""


# =============================================
# ІНТЕРФЕЙС
# =============================================
st.title("🧭 Geomagnetic Pro 2025")
st.markdown("**WMM2025 | MGRS | UTM | Імпорт/Експорт | Карта | QGIS**")

tabs = st.tabs(["Калькулятор", "Пакет", "Карта", "QGIS"])

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
        res = calc_geomag_point(lat, lon, alt, decimal_year(calc_date))
        res["name"] = "Тест"
        st.metric("Деклінація", f"{res['decl']}°")
        st.metric("MGRS", res["mgrs"])
        st.json(res)


# === ПАКЕТ ===
with tabs[1]:
    uploaded = st.file_uploader(
        "CSV | GeoJSON | Shapefile | KML/KMZ | GPX | GeoTIFF",
        type=["csv", "json", "geojson", "zip", "kml", "kmz", "gpx", "tif", "tiff"],
        key="pkg_upload",
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
            results = []
            progress = st.progress(0)
            for i, p in enumerate(points):
                lat_wgs, lon_wgs = to_wgs84(p["x"], p["y"], p["crs"])
                mag = calc_geomag_point(lat_wgs, lon_wgs, p.get("alt", 0), year)
                mag["name"] = p.get("name", f"P{i+1}")
                results.append(mag)
                progress.progress((i + 1) / len(points))
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.download_button("CSV", export_csv(df), "geomag.csv", "text/csv")
            with col2:
                st.download_button("GeoJSON", export_geojson(df), "geomag.geojson", "application/json")
            with col3:
                st.download_button("Shapefile", export_shapefile(df), "geomag.zip", "application/zip")
            with col4:
                st.download_button("KML", export_kml(df), "geomag.kml", "application/vnd.google-earth.kml+xml")
            with col5:
                st.download_button("GPX", export_gpx(df), "geomag.gpx", "application/gpx+xml")
            with col6:
                field = st.selectbox("Поле", ["decl", "incl", "total"], key="tif_field")
                tif = export_geotiff(df, field)
                if tif:
                    st.download_button("GeoTIFF", tif, f"geomag_{field}.tif", "image/tiff")
        except Exception as e:
            st.error(f"Помилка: {e}")


# === КАРТА ===
with tabs[2]:
    if "map_points" not in st.session_state:
        st.session_state.map_points = []

    st.subheader("Введіть координати")

    # lat/lon
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        lat_in = st.number_input("Широта", value=50.450000, format="%.6f", key="map_lat")
    with col2:
        lon_in = st.number_input("Довгота", value=30.524000, format="%.6f", key="map_lon")
    with col3:
        if st.button("Додати", key="map_add_ll"):
            st.session_state.map_points.append((lat_in, lon_in))
            st.rerun()

    # MGRS
    st.markdown("---")
    mgrs_in = st.text_input("MGRS", key="map_mgrs")
    if st.button("Конвертувати MGRS", key="map_conv_mgrs"):
        lat, lon = mgrs_to_latlon(mgrs_in)
        if lat:
            st.session_state.map_points.append((lat, lon))
            st.rerun()
        else:
            st.error("Невірний MGRS")

    # UTM
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        zone = st.text_input("Зона", "35U", key="map_utm_zone")
    with c2:
        e = st.number_input("Easting", value=524136.0, key="map_utm_e")
    with c3:
        n = st.number_input("Northing", value=5584136.0, key="map_utm_n")
    with c4:
        if st.button("Конвертувати UTM", key="map_conv_utm"):
            lat, lon = utm_to_latlon(zone, e, n)
            if lat:
                st.session_state.map_points.append((lat, lon))
                st.rerun()
            else:
                st.error("Невірний UTM")

    # Карта
    st.markdown("---")
    if st.session_state.map_points:
        df_map = pd.DataFrame(st.session_state.map_points, columns=["lat", "lon"])
        year = decimal_year(date.today())
        df_map["decl"] = [calc_geomag_point(lat, lon, 0, year)["decl"] for lat, lon in st.session_state.map_points]
        df_map["mgrs"] = [calc_geomag_point(lat, lon, 0, year)["mgrs"] for lat, lon in st.session_state.map_points]

        fig = px.scatter_mapbox(df_map, lat="lat", lon="lon", color="decl", hover_data=["mgrs"], mapbox_style="open-street-map")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Очистити", key="map_clear"):
            st.session_state.map_points = []
            st.rerun()
    else:
        st.info("Додайте точки")


# === QGIS ===
with tabs[3]:
    st.code(
        '''layer = QgsVectorLayer("geomag.geojson", "Geomag", "ogr")
QgsProject.instance().addMapLayer(layer)''',
        language="python",
    )