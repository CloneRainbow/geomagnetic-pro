"""
calculator.py — Геомагнітні розрахунки (WMM2025 + RTDM + MGRS/UTM)
"""
from __future__ import annotations

import streamlit as st
import requests
import numpy as np
from datetime import datetime
from typing import Dict
from .wmm_loader import WMMLoader
from .coordinates import CoordinateConverter
from .session_manager import SessionManager

class Calculator:
    def __init__(self, session: SessionManager):
        self.session = session
        self.config = session.config
        self.wmm_loader = WMMLoader(self.config.COF_URL, self.config.COF_PATH)
        self.coord_converter = CoordinateConverter()
        self.rtdm_api = self.config.RTDM_API
    
        
        # Завантаження WMM
        self.wmm_loader.download_wmm()
    
    @st.cache_resource
    def _get_wmm_model(self) -> Any:
        """Кешована WMM модель"""
        return self.wmm_loader.get_model()
    
    @st.cache_data(ttl=3600)
    def get_rtdm(self, lat: float, lon: float, alt: float = 0) -> Dict:
        """Отримання реальних корекцій"""
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return {"decl_rt": 0.0, "total_rt": 0.0, "storm": "invalid_coords", "available": False}
        
        try:
            params = {"latitude": lat, "longitude": lon, "altitude": alt / 1000, "format": "json"}
            r = requests.get(self.rtdm_api, params=params, timeout=5)
            if r.status_code == 200:
                d = r.json()
                return {
                    "decl_rt": d.get("declination", 0) - d.get("quiet_declination", 0),
                    "total_rt": d.get("total_intensity", 0) - d.get("quiet_intensity", 0),
                    "storm": d.get("storm_level", "quiet"),
                    "available": True
                }
        except Exception as e:
            st.warning(f"RTDM недоступний: {e}")
        
        return {"decl_rt": 0.0, "total_rt": 0.0, "storm": "offline", "available": False}
    
    @st.cache_data
    def calc_point(self, lat: float, lon: float, alt: float, year: float) -> Dict:
        """Розрахунок геомагнітних параметрів для точки"""
        if not self.coord_converter.validate_coordinates(lat, lon, alt):
            return {"error": "Invalid coordinates"}
        
        try:
            model = self._get_wmm_model()
            wmm = model.calculate(glat=lat, glon=lon, alt=alt/1000, time=year)
            rt = self.get_rtdm(lat, lon, alt)
            
            decl = wmm.d + rt["decl_rt"]
            total = wmm.f + rt["total_rt"]
            mgrs_str = self.coord_converter.safe_mgrs_conversion(lat, lon)
            mgrs_details = self.coord_converter.get_mgrs_details(mgrs_str)
            utm_zone, utm_e, utm_n = self.coord_converter.calculate_utm(lat, lon)
            
            result = {
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
            
            # Автозбереження
            self.session.add_to_history(result)
            self.session.set_last_click(result)
            
            return result
            
        except Exception as e:
            return {"error": f"Calculation error: {str(e)}"}
    
    def display_service_status(self):
        """Відображення статусу сервісів"""
        test = self.get_rtdm(50.45, 30.52, 0)
        if not test["available"]:
            st.info("""
            **ℹ️ RTDM тимчасово недоступний**  
            Використовується тільки WMM2025 (точність ±0.5°)
            """)
    
    def render_calculator_tab(self):
        """Відображення вкладки калькулятора"""
        st.subheader("Калькулятор геомагнітних параметрів")
        
        col1, col2 = st.columns(2)
        
        with col1:
            coord_type = st.radio(
                "Тип координат", 
                ["Географічні", "MGRS"], 
                horizontal=True, 
                key="calc_coord_type"
            )
            
            lat, lon = 50.4501, 30.5234  # дефолт
            
            if coord_type == "Географічні":
                lat = st.number_input(
                    "Широта (°)", 
                    min_value=-90.0, max_value=90.0, 
                    value=50.4501, format="%.6f", 
                    key="calc_lat_geo"
                )
                lon = st.number_input(
                    "Довгота (°)", 
                    min_value=-180.0, max_value=180.0, 
                    value=30.5234, format="%.6f", 
                    key="calc_lon_geo"
                )
            else:
                mgrs_input = st.text_input(
                    "MGRS координати", 
                    value="36TWN1234567890", 
                    key="calc_mgrs_input"
                ).strip().upper()
                
                if mgrs_input:
                    lat_conv, lon_conv = self.coord_converter.convert_mgrs_to_latlon(mgrs_input)
                    if lat_conv is not None:
                        lat, lon = lat_conv, lon_conv
                        st.success(f"Конвертовано: {lat:.6f}°, {lon:.6f}°")
                    else:
                        st.error("Невірний MGRS формат")
                        return
        
        with col2:
            alt = st.number_input(
                "Висота (м)", 
                min_value=-10000.0, max_value=100000.0, 
                value=0.0, step=100.0, 
                key="calc_alt"
            )
            calc_date = st.date_input(
                "Дата розрахунку", 
                value=self.session.today, 
                key="calc_date"
            )

        if st.button("Обчислити", type="primary", key="calc_btn"):
            with st.spinner("Обчислення..."):
                year = self.wmm_loader.decimal_year(calc_date)
                res = self.calc_point(lat, lon, alt, year)
                
                if "error" in res:
                    st.error(res["error"])
                else:
                    self._display_results(res)
    
    def _display_results(self, res: Dict):
        """Відображення результатів"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("MGRS", res["mgrs"])
            if res["mgrs_details"]:
                with st.expander("Деталі MGRS"):
                    d = res["mgrs_details"]
                    st.write(f"**Зона:** {d.get('zone', '-')}")
                    st.write(f"**Банда:** {d.get('band', '-')}")
                    st.write(f"**Квадрат:** {d.get('square', '-')}")
        
        with col2:
            st.metric("UTM Зона", res["utm_zone"])
            st.metric("Easting", f"{res['utm_e']:,.1f} m" if not np.isnan(res['utm_e']) else "—")
            st.metric("Northing", f"{res['utm_n']:,.1f} m" if not np.isnan(res['utm_n']) else "—")
        
        with col3:
            st.metric("Деклінація", f"{res['decl']}°")
            st.metric("Інтенсивність", f"{res['total']:,.0f} nT")
            storm_emoji = "Тихий" if res["storm"] == "quiet" else "Помірний" if res["storm"] == "moderate" else "Офлайн"
            st.metric("Статус", storm_emoji)
            st.metric("Модель", res["model"])
        
        with st.expander("Повні дані"):
            st.json({k: v for k, v in res.items() if k != "mgrs_details"})
