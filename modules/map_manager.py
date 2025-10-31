# =============================================
# Управління картою Leaflet
# =============================================
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
import json
from datetime import datetime


from typing import List, Dict, Optional
from .session_manager import SessionManager
from .data_export import DataExporter
from .coordinates import CoordinateConverter
from .calculator import Calculator  # для динамічного імпорту

class MapManager:
    def __init__(self, session: SessionManager, exporter: DataExporter):
        self.session = session
        self.exporter = exporter
        self.config = session.config
    def create_leaflet_map(self, center_lat: float, center_lon: float, zoom: int, 
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
    
    def render_map_tab(self, session, calculator):
        """Відображення вкладки карти"""
        st.subheader("🗺️ Точна карта з Leaflet")
        
        col_map1, col_map2 = st.columns([3, 1])
        
        with col_map2:
            self._render_map_settings(session, calculator)
        
        with col_map1:
            self._render_map_display(session, calculator)
    
    def _render_map_settings(self, session, calculator):
        """Відображення налаштувань карти"""
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
                    if calculator.coord_converter.validate_mgrs(mgrs_search_input.strip()):
                        lat, lon = calculator.coord_converter.convert_mgrs_to_latlon(mgrs_search_input.strip())
                        if lat is not None and lon is not None:
                            # Створюємо точку для пошуку
                            search_result = {
                                "lat": lat,
                                "lon": lon,
                                "mgrs": mgrs_search_input.strip(),
                                "precision": calculator.coord_converter.get_mgrs_precision(mgrs_search_input.strip()),
                                "type": "search"
                            }
                            session.set_mgrs_search(search_result)
                            # Оновлюємо центр карти
                            session.set_last_click(None)
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
                session.set_mgrs_search(None)
                st.rerun()
        
        if session.get_mgrs_search():
            st.info(f"**Пошук:** {session.get_mgrs_search()['mgrs']}")
        
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
            result = calculator.calc_point(click_lat, click_lon, click_alt, calculator.wmm_loader.decimal_year(st.session_state.get('today')))
            if "error" not in result:
                session.set_last_click(result)
                session.add_to_history(result)
                st.success(f"Додано точку: {click_lat:.4f}°, {click_lon:.4f}°")
                st.rerun()
    
    def _render_map_display(self, session, calculator):
        """Відображення карти"""
        # Визначення центру карти
        if session.get_mgrs_search():
            # Якщо є пошук - центруємо на ньому
            map_center_lat = session.get_mgrs_search()['lat']
            map_center_lon = session.get_mgrs_search()['lon']
            map_zoom = 14  # Збільшений масштаб для пошуку
        elif session.get_last_click():
            # Якщо є остання точка - центруємо на ній
            map_center_lat = session.get_last_click()['lat']
            map_center_lon = session.get_last_click()['lon']
            map_zoom = 12
        else:
            # Інакше використовуємо налаштування користувача
            map_center_lat = st.session_state.get('center_lat', 50.4501)
            map_center_lon = st.session_state.get('center_lon', 30.5234)
            map_zoom = st.session_state.get('zoom_level', 10)
        
        # Підготовка точок для карти
        all_points = session.get_selected_points()
        
        # Створення Leaflet карти
        map_html = self.create_leaflet_map(
            center_lat=map_center_lat,
            center_lon=map_center_lon,
            zoom=map_zoom,
            points=all_points,
            search_point=session.get_mgrs_search()
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
        if session.get_last_click():
            self._render_last_click_info(session.get_last_click())
        
        # Інформація про точку пошуку MGRS
        if session.get_mgrs_search():
            self._render_search_info(session, calculator)
    
    def _render_last_click_info(self, last_click):
        """Відображення інформації про останній вибір"""
        lc = last_click
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
    
    def _render_search_info(self, session, calculator):
        """Відображення інформації про пошук"""
        search = session.get_mgrs_search()
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
                search_result = calculator.calc_point(search['lat'], search['lon'], 0, calculator.wmm_loader.decimal_year(st.session_state.get('today')))
                if "error" not in search_result:
                    st.write(f"**⚡ Геомагнітні параметри:**")
                    st.write(f"Деклінація: {search_result['decl']}°")
                    st.write(f"Інтенсивність: {search_result['total']:,.1f} nT")
                    st.write(f"UTM: {search_result['utm_zone']}")
                
                if st.button("➕ Додати до історії", key="add_search_to_history"):
                    if "error" not in search_result:
                        session.set_last_click(search_result)
                        session.add_to_history(search_result)
                        st.success("Точку додано до історії!")
                        st.rerun()
