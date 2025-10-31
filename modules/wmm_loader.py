# =============================================
# Завантаження та управління WMM моделлю
# =============================================

import os
import io
import zipfile
import requests
import streamlit as st
from datetime import date
from pygeomag import GeoMag

class WMMLoader:
    """Клас для завантаження та управління WMM моделлю"""
    
    def __init__(self, cof_url, cof_path):
        self.cof_url = cof_url
        self.cof_path = cof_path
        self._model = None
    
    def safe_extract(self, zf: zipfile.ZipFile, path: str) -> None:
        """Безпечне розпакування ZIP архіву"""
        abs_target = os.path.abspath(path)
        for member in zf.namelist():
            member_path = os.path.abspath(os.path.join(path, member))
            if not member_path.startswith(abs_target + os.sep) and member_path != abs_target:
                raise Exception("Unsafe zip archive (zip-slip attempt)")
        zf.extractall(path)
    
    @st.cache_resource(show_spinner="Завантаження WMM2025...")
    def download_wmm(_self) -> str:
        """Завантаження WMM моделі"""
        os.makedirs("wmm", exist_ok=True)
        if os.path.exists(_self.cof_path):
            return _self.cof_path
        
        try:
            resp = requests.get(_self.cof_url, timeout=30)
            resp.raise_for_status()
            zip_data = resp.content
            
            with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                # Пошук COF файлу в архіві
                names = [n for n in z.namelist() if n.endswith(".COF") or n.endswith(".cof")]
                if not names:
                    raise Exception("COF файл не знайдено в архіві")
                
                # Читання конкретного файлу
                with z.open(names[0]) as src, open(_self.cof_path, "wb") as dst:
                    dst.write(src.read())
            
            return _self.cof_path
            
        except Exception as e:
            st.error(f"Помилка завантаження WMM: {e}")
            st.stop()
    
    @st.cache_resource
    def get_model(_self) -> GeoMag:
        """Отримання WMM моделі"""
        # Перевірка наявності файлу
        if not os.path.exists(_self.cof_path) or os.path.getsize(_self.cof_path) == 0:
            raise FileNotFoundError("Файл коефіцієнтів WMM відсутній або порожній")
        
        return GeoMag(coefficients_file=_self.cof_path)
        # modules/wmm_loader.py
    def decimal_year(self, d: date) -> float:
       year = d.year
       start = date(year, 1, 1)
       end = date(year + 1, 1, 1)
       return year + (d - start).days / (end - start).days
        
    


