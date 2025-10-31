# modules/wmm_loader.py
"""
wmm_loader.py — Безпечне завантаження WMM2025
"""
from __future__ import annotations

import os
import io
import zipfile
import requests
import streamlit as st
from datetime import date
from pygeomag import GeoMag
from typing import List


class WMMLoader:
    """Клас для завантаження та управління WMM моделлю"""
    
    def __init__(self, cof_url: str, cof_path: str):
        self.cof_url = cof_url
        self.cof_path = cof_path
        self._model: GeoMag | None = None

    def _safe_extract(self, zf: zipfile.ZipFile, target_dir: str) -> None:
        """Безпечне розпакування (запобігає zip-slip)"""
        abs_target = os.path.abspath(target_dir)
        for member in zf.namelist():
            member_path = os.path.abspath(os.path.join(target_dir, member))
            if not (member_path.startswith(abs_target + os.sep) or member_path == abs_target):
                raise ValueError("Небезпечний ZIP архів (zip-slip)")
        zf.extractall(target_dir)

    @st.cache_resource(show_spinner="Завантаження WMM2025...")
    def download_wmm(_self) -> str:
        """Завантаження WMM моделі"""
        os.makedirs(os.path.dirname(_self.cof_path), exist_ok=True)
        
        if os.path.exists(_self.cof_path) and os.path.getsize(_self.cof_path) > 1000:
            return _self.cof_path
        
        try:
            resp = requests.get(_self.cof_url, timeout=30)
            resp.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                # Пошук .COF файлу
                cof_files: List[str] = [n for n in z.namelist() if n.lower().endswith(".cof")]
                if not cof_files:
                    raise FileNotFoundError("COF файл не знайдено в архіві")
                
                # Безпечне розпакування
                temp_dir = os.path.dirname(_self.cof_path)
                _self._safe_extract(z, temp_dir)
                
                # Переміщення знайденого файлу
                extracted_path = os.path.join(temp_dir, cof_files[0])
                if os.path.exists(extracted_path):
                    os.replace(extracted_path, _self.cof_path)
            
            if not os.path.exists(_self.cof_path):
                raise FileNotFoundError("Не вдалося зберегти COF файл")
                
            return _self.cof_path
            
        except Exception as e:
            st.error(f"Помилка завантаження WMM: {e}")
            st.stop()

    @st.cache_resource(show_spinner="Ініціалізація моделі WMM...")
    def get_model(_self) -> GeoMag:
        """Отримання WMM моделі"""
        _self.download_wmm()  # Гарантія наявності файлу
        
        if not os.path.exists(_self.cof_path):
            raise FileNotFoundError("Файл WMM не знайдено")
        if os.path.getsize(_self.cof_path) == 0:
            raise ValueError("Файл WMM порожній")
        
        return GeoMag(coefficients_file=_self.cof_path)

    def decimal_year(self, d: date) -> float:
        """Перетворення дати в десятковий рік"""
        year = d.year
        start = date(year, 1, 1)
        end = date(year + 1, 1, 1)
        return year + (d - start).days / (end - start).days
