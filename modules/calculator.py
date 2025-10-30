# =============================================
# Геомагнітні розрахунки
# =============================================

import streamlit as st
import requests
import numpy as np
from datetime import datetime
from typing import Dict
from .wmm_loader import WMMLoader
from .coordinates import CoordinateConverter

class Calculator:
    """Клас для геомагнітних розрахунків"""
    
    def __init__(self):
        self.config = st.session_state.get('config')
        self.wmm_loader = WMMLoader(
            self.config.COF_URL if self.config else "https://www.ncei.noaa.gov/data/world-magnetic-model-2025/full/WMM_COEFS-2025.zip",
            self.config.COF_PATH if self.config else "wmm/WMM_2025.COF"
        )
        self.coord_converter = CoordinateConverter()
        
        # Завантаження WMM моделі
        self.wmm_loader.download_wmm()
    
    @st.cache_data(ttl=3600)
    def get_rtdm(self, lat: float, lon: float, alt: float = 0) -> Dict:
        """Отримання реальних корекцій"""
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return {"decl_rt": 0.0, "total_rt": 0.0, "storm": "invalid_coords", "available": False}
        
        try:
            params = {
                "latitude": lat, 
                "longitude": lon, 
                "altitude": alt / 1000, 
                "format": "json"
            }
            
            r = requests.get(
                self.config.RTDM_API if self.config else "https://geomag.usgs.gov/ws/edge/", 
                params=params, 
                timeout=5
            )
            
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
    
    @st.cache_data
    def calc_point(self, lat: float, lon: float, alt: float, year: float) -> Dict:
        """Розрахунок геомагнітних параметрів для точки"""
        if not self.coord_converter.validate_coordinates(lat, lon, alt):
            return {"error": "Invalid coordinates"}
        
        try:
            model = self.wmm_loader.get_model()
            wmm = model.calculate(glat=lat, glon=lon, alt=alt/1000, time=year)
            rt = self.get_rtdm(lat, lon, alt)
            
            decl = wmm.d + rt["decl_rt"]
            total = wmm.f + rt["total_rt"]
            mgrs_str = self.coord_converter.safe_mgrs_conversion(lat, lon)
            mgrs_details = self.coord_converter.get_mgrs_details(mgrs_str)
            utm_zone, utm_e, utm_n = self.coord_converter.calculate_utm(lat, lon)
            
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
    
    def display_service_status(self):
        """Відображення статусу сервісів"""
        rtdm_status = self.get_rtdm(50.45, 30.52, 0)["available"]
        if not rtdm_status:
            st.info("""
            **ℹ️ Інформація про сервіс:** 
            Real-time магнітні корекції (RTDM) тимчасово недоступні. 
            Використовуються точні дані WMM 2025 моделі.
            """)
    
    def render_calculator_tab(self, session):
        """Відображення вкладки калькулятора"""
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
                lat, lon = 50.4501, 30.5234
                
                if mgrs_input and mgrs_input.strip():
                    lat_conv, lon_conv = self.coord_converter.convert_mgrs_to_latlon(mgrs_input.strip())
                    if lat_conv is not None and lon_conv is not None:
                        lat, lon = lat_conv, lon_conv
                        st.success(f"Конвертовано: {lat:.6f}°, {lon:.6f}°")
        
        with col2:
            alt = st.number_input("Висота (м)", min_value=-10000.0, max_value=100000.0, value=0.0, step=100.0, key="calc_alt")
            calc_date = st.date_input("Дата розрахунку", value=st.session_state.get('today'), key="calc_date")

        if st.button("🔄 Обчислити", type="primary", key="calc_btn"):
            if coord_type == "MGRS" and (not mgrs_input or not mgrs_input.strip()):
                st.error("Будь ласка, введіть MGRS координати")
            else:
                with st.spinner("Виконуються розрахунки..."):
                    res = self.calc_point(lat, lon, alt, self.wmm_loader.decimal_year(calc_date))
                
                if "error" in res:
                    st.error(f"Помилка розрахунку: {res['error']}")
                else:
                    self._display_results(res, session)
    
    def _display_results(self, res, session):
        """Відображення результатів розрахунку"""
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
        session.add_to_history(res)
        session.set_last_click(res)
        
        with st.expander("📊 Детальна інформація"):
            st.json({k: v for k, v in res.items() if k != 'mgrs_details'})
