"""
Координатний конвертер

"""
from __future__ import annotations

import numpy as np
import mgrs
from pyproj import Transformer
from typing import Tuple, Optional, Dict, Any


class CoordinateConverter:
    def __init__(self) -> None:
        self.mgrs_conv = mgrs.MGRS()

    def convert_mgrs_to_latlon(self, mgrs_str: str) -> Tuple[Optional[float], Optional[float]]:
        mgrs_str = mgrs_str.strip().upper().replace(" ", "")
        if len(mgrs_str) < 5:
            return None, None
        try:
            lat, lon = self.mgrs_conv.fromMGRS(mgrs_str)
            return round(float(lat), 8), round(float(lon), 8)
        except Exception:
            return None, None

    def latlon_to_mgrs(self, lat: float, lon: float, precision: int = 5) -> str:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return "Invalid"
        try:
            return self.mgrs_conv.toMGRS(lat, lon, precision)
        except Exception:
            return "Error"

    def validate_mgrs(self, mgrs_str: str) -> bool:
        mgrs_str = mgrs_str.strip().upper().replace(" ", "")
        if len(mgrs_str) < 5:
            return False
        try:
            self.mgrs_conv.fromMGRS(mgrs_str)
            return True
        except Exception:
            return False

    def get_mgrs_details(self, mgrs_str: str) -> Dict[str, Any]:
        clean = mgrs_str.strip().upper().replace(" ", "")
        if len(clean) < 5 or not self.validate_mgrs(clean):
            return {"full": clean, "valid": False}
        try:
            zone_num = clean[:2]
            zone_letter = clean[2]
            band = clean[3]
            square = clean[4:6]
            digits = ''.join(c for c in clean[6:] if c.isdigit())
            precision = len(digits) // 2 if digits else 0
            precision_map = {0: "100 км", 1: "10 км", 2: "1 км", 3: "100 м", 4: "10 м", 5: "1 м"}
            prec_text = precision_map.get(precision, "Невідомо")
            return {
                "full": clean,
                "zone": f"{zone_num}{zone_letter}",
                "band": band,
                "square": square,
                "eastings": clean[6:6+len(digits)//2],
                "northings": clean[6+len(digits)//2:6+len(digits)],
                "precision": prec_text,
                "valid": True
            }
        except Exception:
            return {"full": clean, "valid": False}

    def latlon_to_utm(self, lat: float, lon: float) -> Tuple[str, float, float]:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return "Invalid", np.nan, np.nan
        try:
            zone = int((lon + 180) / 6) + 1
            hemi = 'S' if lat < 0 else 'N'
            proj_str = f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs"
            if hemi == 'S':
                proj_str += " +south"
            transformer = Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)
            e, n = transformer.transform(lon, lat)
            return f"{zone}{hemi}", round(e, 3), round(n, 3)
        except Exception:
            return "Error", np.nan, np.nan

    def utm_to_latlon(self, zone: int, east: float, north: float, south: bool = False) -> Tuple[float, float]:
        try:
            proj_str = f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs"
            if south:
                proj_str += " +south"
            transformer = Transformer.from_crs(proj_str, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(east, north)
            return round(lat, 8), round(lon, 8)
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def validate_coordinates(lat: float, lon: float, alt: float = 0.0) -> bool:
        return (-90 <= lat <= 90 and -180 <= lon <= 180 and -10000 <= alt <= 100000)
