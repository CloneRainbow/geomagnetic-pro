# =============================================
# Робота з системами координат
# =============================================

import numpy as np
import mgrs
from pyproj import CRS, Transformer, Proj
from typing import Tuple, Optional, Dict

class CoordinateConverter:
    """Клас для конвертації між системами координат"""
    
    def __init__(self):
        self.mgrs_conv = mgrs.MGRS()
    
    def safe_mgrs_conversion(self, lat: float, lon: float) -> str:
        """Безпечна конвертація в MGRS"""
        try:
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return self.mgrs_conv.toMGRS(lat, lon, 5)
            else:
                return "Invalid coordinates"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def convert_mgrs_to_latlon(self, mgrs_str: str) -> Tuple[Optional[float], Optional[float]]:
        """Конвертація MGRS в географічні координати"""
        try:
            if not mgrs_str or len(mgrs_str) < 5:
                return None, None
            lat, lon = self.mgrs_conv.toLatLon(mgrs_str)
            return float(lat), float(lon)
        except Exception as e:
            import streamlit as st
            st.error(f"Помилка конвертації MGRS '{mgrs_str}': {e}")
            return None, None
    
    def validate_mgrs(self, mgrs_str: str) -> bool:
        """Валідація MGRS координат"""
        try:
            if not mgrs_str or len(mgrs_str) < 5:
                return False
            parts = mgrs_str.strip()
            if len(parts) < 5:
                return False
            lat, lon = self.mgrs_conv.toLatLon(parts)
            return lat is not None and lon is not None
        except Exception:
            return False
    
    def get_mgrs_precision(self, mgrs_str: str) -> str:
        """Визначення точності MGRS координат"""
        length = len(mgrs_str)
        if length <= 5:
            return "Низька (100 км)"
        elif length <= 7:
            return "Середня (10 км)"
        elif length <= 9:
            return "Висока (1 км)"
        elif length <= 11:
            return "Дуже висока (100 м)"
        else:
            return "Найвища (10 м)"
    
    def get_mgrs_details(self, mgrs_str: str) -> Dict:
        """Детальна інформація MGRS"""
        try:
            if not mgrs_str or "Error" in mgrs_str or "Invalid" in mgrs_str:
                return {"full": mgrs_str}
            
            if len(mgrs_str) < 5:
                return {"full": mgrs_str}
                
            zone = mgrs_str[:2] if mgrs_str[:2].isdigit() else ""
            band = mgrs_str[2] if len(mgrs_str) > 2 else ""
            square = mgrs_str[3:5] if len(mgrs_str) > 4 else ""
            
            return {
                "zone": zone,
                "band": band,
                "square": square,
                "full": mgrs_str,
                "precision": self.get_mgrs_precision(mgrs_str)
            }
        except Exception:
            return {"full": mgrs_str}
    
    def calculate_utm(self, lat: float, lon: float) -> Tuple[str, float, float]:
        """Обчислення UTM координат"""
        try:
            zone = int((lon + 180) / 6) + 1
            hemi = 'S' if lat < 0 else 'N'
            proj_string = f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs"
            if hemi == 'S':
                proj_string += " +south"
            
            transformer = Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)
            e, n = transformer.transform(lon, lat)
            utm_zone = f"{zone}{hemi}"
            return utm_zone, round(e, 1), round(n, 1)
            
        except Exception:
            return "Error", np.nan, np.nan
    
    def validate_coordinates(self, lat: float, lon: float, alt: float = 0) -> bool:
        """Валідація координат"""
        import streamlit as st
        
        if not (-90 <= lat <= 90):
            st.error(f"Некоректна широта: {lat}. Має бути в діапазоні -90 до 90.")
            return False
        if not (-180 <= lon <= 180):
            st.error(f"Некоректна довгота: {lon}. Має бути в діапазоні -180 до 180.")
            return False
        if not (-10000 <= alt <= 100000):
            st.error(f"Некоректна висота: {alt}. Має бути в діапазоні -10,000 до 100,000 м.")
            return False
        return True
