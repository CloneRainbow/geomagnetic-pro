import streamlit as st
import pandas as pd
import numpy as np
from pygeomag import GeoMag
from datetime import date
import plotly.graph_objects as go
import mgrs
from pyproj import CRS, Transformer, Proj
from shapely.geometry import Point
import geopandas as gpd
import gpxpy
import gpxpy.gpx
import simplekml
from fastkml import kml
import zipfile
import io
import tempfile
import os
import json
import geojson
import requests
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import NearestNDInterpolator
from typing import List, Dict

# =============================================
# КОНФІГУРАЦІЯ
# =============================================
st.set_page_config(page_title="Geomagnetic Pro 2025", page_icon="🧭", layout="wide")
m = mgrs.MGRS()

# Автозавантаження WMM_2025.COF
COF_URL = "https://www.ncei.noaa.gov/data/world-magnetic-model-2025/full/WMM_COEFS-2025.zip"
COF_PATH = "wmm/WMM_2025.COF"

if not os.path.exists(COF_PATH):
    os.makedirs("wmm", exist_ok=True)
    with st.spinner("Завантаження WMM_2025.COF..."):
        try:
            zip_data = requests.get(COF_URL).content
            with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                with z.open("WMM_2025.COF") as src, open(COF_PATH, "wb") as dst:
                    dst.write(src.read())
            st.success("WMM_2025.COF завантажено!")
        except Exception as e:
            st.error(f"Не вдалося завантажити COF: {e}")

@st.cache_resource
def get_model():
    return GeoMag(coefficients_file=COF_PATH)

def decimal_year(d: date) -> float:
    start = date(d.year, 1, 1)
    return d.year + (d - start).days / 365.25

# =============================================
# ПЕРЕТВОРЕННЯ
# =============================================
def to_wgs84(x: float, y: float, crs_from: CRS) -> tuple:
    if crs_from.to_epsg() == 4326:
        return y, x
    t = Transformer.from_crs(crs_from, CRS.from_epsg(4326), always_xy=True)
    lon, lat = t.transform(x, y)
    return lat, lon

def to_utm_mgrs(lat: float, lon: float) -> tuple:
    zone = int((lon + 180) / 6) + 1
    hemi = 'S' if lat < 0 else 'N'
    proj_str = f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs"
    if hemi == 'S': proj_str += " +south"
    p = Proj(proj_str)
    e, n = p(lon, lat)
    return f"{zone}{hemi}", round(e, 1), round(n, 1), m.toMGRS(lat, lon, 5)

# =============================================
# ГЕОМАГНІТНИЙ РОЗРАХУНОК
# =============================================
@st.cache_data
def calc_point(lat: float, lon: float, alt: float, year: float) -> Dict:
    model = get_model()
    r = model.calculate(glat=lat, glon=lon, alt=alt/1000, time=year)
    utm, e, n, mgrs_str = to_utm_mgrs(lat, lon)
    return {
        "name": "", "decl": round(r.d, 4), "incl": round(r.i, 4), "total": round(r.f, 2),
        "horiz": round(r.h, 2), "X": round(r.x, 2), "Y": round(r.y, 2), "Z": round(r.z, 2),
        "mgrs": mgrs_str, "utm_zone": utm, "utm_e": e, "utm_n": n,
        "lat": round(lat, 6), "lon": round(lon, 6)
    }

# =============================================
# ІМПОРТ (усі формати)
# =============================================
def parse_csv(file) -> List[Dict]:
    df = pd.read_csv(file)
    crs = CRS.from_epsg(int(df.get('epsg', [4326]).iloc[0]))
    return [{"name": row.get('name', f"P{i}"), "x": row['lon'], "y": row['lat'], "alt": row.get('alt', 0), "crs": crs}
            for i, row in df.iterrows()]

def parse_geojson(file) -> List[Dict]:
    data = json.load(file)
    crs = CRS.from_epsg(4326)
    if data.get("crs"):
        name = data["crs"]["properties"].get("name", "")
        if "EPSG" in name: crs = CRS.from_epsg(int(name.split(":")[-1]))
    return [{"name": f["properties"].get("name", "P"), "x": f["geometry"]["coordinates"][0],
             "y": f["geometry"]["coordinates"][1], "alt": f["properties"].get("alt", 0), "crs": crs}
            for f in data.get("features", [])]

def parse_shapefile(zip_file) -> List[Dict]:
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_file) as z:
            z.extractall(tmp)
        gdf = gpd.read_file(tmp)
        return [{"name": row.get("name", f"P{i}"), "x": row.geometry.x, "y": row.geometry.y,
                 "alt": row.get("alt", 0), "crs": gdf.crs or CRS.from_epsg(4326)}
                for i, row in gdf.iterrows()]

def parse_kml_kmz(file_bytes) -> List[Dict]:
    if file_bytes.startswith(b'PK'):
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            kml_file = [f for f in z.namelist() if f.endswith('.kml')][0]
            content = z.read(kml_file)
    else:
        content = file_bytes
    k = kml.KML()
    k.from_string(content)
    points = []
    def extract(feature):
        if hasattr(feature, 'geometry') and feature.geometry and feature.geometry.geom_type == 'Point':
            lon, lat = feature.geometry.x, feature.geometry.y
            alt = getattr(feature.geometry, 'z', 0) or 0
            points.append({"name": feature.name or "Point", "x": lon, "y": lat, "alt": alt, "crs": CRS.from_epsg(4326)})
        if hasattr(feature, 'features'):
            for f in feature.features(): extract(f)
    for f in k.features(): extract(f)
    return points

def parse_gpx(file_bytes) -> List[Dict]:
    gpx = gpxpy.parse(file_bytes.decode('utf-8', errors='ignore'))
    points = []
    for wp in gpx.waypoints:
        points.append({"name": wp.name or "WP", "x": wp.longitude, "y": wp.latitude, "alt": wp.elevation or 0, "crs": CRS.from_epsg(4326)})
    for track in gpx.tracks:
        for seg in track.segments:
            for pt in seg.points:
                points.append({"name": f"Track: {track.name or 'Unknown'}", "x": pt.longitude, "y": pt.latitude, "alt": pt.elevation or 0, "crs": CRS.from_epsg(4326)})
    for route in gpx.routes:
        for pt in route.points:
            points.append({"name": f"Route: {route.name or 'Unknown'}", "x": pt.longitude, "y": pt.latitude, "alt": pt.elevation or 0, "crs": CRS.from_epsg(4326)})
    return points

def parse_geotiff(file) -> List[Dict]:
    with rasterio.open(file) as src:
        crs = src.crs or CRS.from_epsg(4326)
        transform = src.transform
        band = src.read(1)
        height, width = band.shape
        points = []
        for row in range(0, height, max(1, height // 1000)):  # Підвибірка
            for col in range(0, width, max(1, width // 1000)):
                if not np.ma.is_masked(band[row, col]) and not np.isnan(band[row, col]):
                    x, y = transform * (col, row)
                    lat, lon = to_wgs84(x, y, crs)
                    points.append({"name": f"Pixel_{row}_{col}", "x": lon, "y": lat, "alt": 0, "crs": CRS.from_epsg(4326)})
        return points

# =============================================
# ЕКСПОРТ (усі формати)
# =============================================
def export_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

def export_geojson(df: pd.DataFrame) -> str:
    features = [geojson.Feature(geometry=geojson.Point((r.lon, r.lat)), properties=r.drop(['lat','lon']).to_dict())
                for _, r in df.iterrows()]
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
        p.description = f"Decl: {r['decl']}° | Incl: {r['incl']}° | Total: {r['total']} nT\nMGRS: {r['mgrs']}"
    buf = io.BytesIO()
    buf.write(kml_doc.kml().encode('utf-8'))
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
        wp.description = f"Decl: {r['decl']}° | MGRS: {r['mgrs']}"
        gpx.waypoints.append(wp)
    return gpx.to_xml()

def export_geotiff(df: pd.DataFrame, field: str = "decl") -> bytes:
    if len(df) < 3:
        return b""
    lons = df["lon"].values
    lats = df["lat"].values
    values = df[field].values
    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()
    res = 0.001
    lon_grid = np.arange(lon_min, lon_max + res, res)
    lat_grid = np.arange(lat_min, lat_max + res, res)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
    points = np.column_stack((lons, lats))
    interp = NearestNDInterpolator(points, values)
    grid = interp(lon_grid, lat_grid)
    transform = from_origin(lon_min, lat_max, res, res)
    buf = io.BytesIO()
    with rasterio.MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff', height=grid.shape[0], width=grid.shape[1], count=1,
            dtype=grid.dtype, crs='EPSG:4326', transform=transform
        ) as dataset:
            dataset.write(grid, 1)
        buf = memfile.read()
    return buf

# =============================================
# ІНТЕРФЕЙС
# =============================================
st.title("🧭 Geomagnetic Pro 2025")
st.markdown("**Повний імпорт + експорт: CSV | GeoJSON | Shapefile | KML/KMZ | GPX | GeoTIFF**")

tabs = st.tabs(["Калькулятор", "Пакет", "Карта", "QGIS"])

with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Широта", value=55.7558, format="%.6f")
        lon = st.number_input("Довгота", value=37.6173, format="%.6f")
    with col2:
        alt = st.number_input("Висота (м)", value=0.0)
        calc_date = st.date_input("Дата", value=date(2025, 11, 5))
    if st.button("Обчислити", type="primary"):
        res = calc_point(lat, lon, alt, decimal_year(calc_date))
        res["name"] = "Тестова точка"
        st.metric("Деклінація", f"{res['decl']}°")
        st.metric("MGRS", res['mgrs'])
        st.json(res)

with tabs[1]:
    uploaded = st.file_uploader(
        "CSV | GeoJSON | Shapefile | KML/KMZ | GPX | GeoTIFF",
        type=["csv", "json", "geojson", "zip", "kml", "kmz", "gpx", "tif", "tiff"]
    )
    if uploaded:
        ext = uploaded.name.split(".")[-1].lower()
        try:
            if ext == "csv": points = parse_csv(uploaded)
            elif ext in ["json", "geojson"]: points = parse_geojson(uploaded)
            elif ext == "zip": points = parse_shapefile(uploaded)
            elif ext in ["kml", "kmz"]: points = parse_kml_kmz(uploaded.read())
            elif ext == "gpx": points = parse_gpx(uploaded.read())
            elif ext in ["tif", "tiff"]: points = parse_geotiff(uploaded)
            else: st.error("Формат не підтримується"); st.stop()

            year = decimal_year(st.date_input("Дата", date.today()))
            results = []
            progress = st.progress(0)
            for i, p in enumerate(points):
                lat_wgs, lon_wgs = to_wgs84(p["x"], p["y"], p["crs"])
                mag = calc_point(lat_wgs, lon_wgs, p.get("alt", 0), year)
                mag["name"] = p.get("name", f"P{i+1}")
                results.append(mag)
                progress.progress((i + 1) / len(points))
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            # ЕКСПОРТ УСІХ ФОРМАТІВ
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1: st.download_button("CSV", export_csv(df), "geomag.csv", "text/csv")
            with col2: st.download_button("GeoJSON", export_geojson(df), "geomag.geojson", "application/json")
            with col3: st.download_button("Shapefile", export_shapefile(df), "geomag.zip", "application/zip")
            with col4: st.download_button("KML", export_kml(df), "geomag.kml", "application/vnd.google-earth.kml+xml")
            with col5: st.download_button("GPX", export_gpx(df), "geomag.gpx", "application/gpx+xml")
            with col6:
                field = st.selectbox("Поле для GeoTIFF", ["decl", "incl", "total"], key="tif_field")
                tif_data = export_geotiff(df, field)
                if tif_data:
                    st.download_button("GeoTIFF", tif_data, f"geomag_{field}.tif", "image/tiff")
        except Exception as e:
            st.error(f"Помилка: {e}")
with tabs[2]:
    # === ІНІЦІАЛІЗАЦІЯ СЕСІЇ ===
    if 'map_points' not in st.session_state:
        st.session_state.map_points = []

    # === СТВОРЕННЯ КАРТИ З БАЗОВИМ ШАРОМ ===
    fig = go.Figure()

    # Базовий шар карти (обов’язково!)
    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=[0], lat=[0],
        marker={'size': 0, 'color': []},
        showlegend=False
    ))

    # Додавання точок користувача
    if st.session_state.map_points:
        lats, lons = zip(*st.session_state.map_points)
        fig.add_trace(go.Scattermapbox(
            lat=lats, lon=lons,
            mode="markers+lines",
            marker=dict(size=12, color="red"),
            line=dict(width=2, color="red"),
            name="Точки"
        ))

    # Налаштування карти
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=2,
        mapbox_center={"lat": 50, "lon": 30},  # Центр (Україна)
        height=600,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    # === ВІДОБРАЖЕННЯ КАРТИ ===
    st.plotly_chart(fig, use_container_width=True, key="interactive_map")

    # === БЕЗПЕЧНА ОБРОБКА КЛІКУ ===
    if hasattr(st.session_state, 'interactive_map') and st.session_state.interactive_map:
        click_data = st.session_state.interactive_map.get("click")
        if click_data:
            lat, lon = click_data["lat"], click_data["lon"]
            st.session_state.map_points.append((lat, lon))
            st.success(f"Додано точку: {lat:.6f}, {lon:.6f}")

    # === КНОПКА ОЧИЩЕННЯ ===
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Очистити"):
            st.session_state.map_points = []
            st.rerun()

with tabs[3]:
    st.code('''layer = QgsVectorLayer("geomag.kml", "Geomag", "ogr")
QgsProject.instance().addMapLayer(layer)''')
