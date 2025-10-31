# modules/session_manager.py
"""
session_manager.py — Управління станом сесії
Історія, кліки, MGRS пошук, експорт
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional


class SessionManager:
    """Клас для управління станом сесії"""
    
    def __init__(self, config):
        self.config = config
        self.today = date.today()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Ініціалізація стану сесії"""
        defaults = {
            "selected_points": [],
            "last_click": None,
            "mgrs_search": None,
            "today": datetime.now().date(),
            "grid_cache": {},
            "click_points": []
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def add_to_history(self, point: Dict):
        """Додавання точки до історії (max 10)"""
        if len(st.session_state.selected_points) >= 10:
            st.session_state.selected_points.pop(0)
        st.session_state.selected_points.append(point)
    
    def get_selected_points(self) -> List[Dict]:
        """Отримання списку вибраних точок"""
        return st.session_state.selected_points.copy()
    
    def set_last_click(self, point: Optional[Dict]):
        """Встановлення останньої обраної точки"""
        st.session_state.last_click = point
    
    def get_last_click(self) -> Optional[Dict]:
        """Отримання останньої обраної точки"""
        return st.session_state.last_click
    
    def set_mgrs_search(self, search_point: Optional[Dict]):
        """Встановлення точки пошуку MGRS"""
        st.session_state.mgrs_search = search_point
    
    def get_mgrs_search(self) -> Optional[Dict]:
        """Отримання точки пошуку MGRS"""
        return st.session_state.mgrs_search
    
    def add_click_point(self, point: Dict):
        """Додає точку з карти"""
        st.session_state.click_points.append(point)
        self.set_last_click(point)
    
    def get_click_switch(self) -> List[Dict]:
        """Отримання кліків з карти"""
        return st.session_state.click_points.copy()
    
    def clear_clicks(self):
        """Очищення кліків з карти"""
        st.session_state.click_points = []
        self.set_last_click(None)
    
    def clear_history(self):
        """Очищення історії"""
        st.session_state.selected_points = []
        self.set_last_click(None)
    
    def render_history_tab(self, data_exporter):
        """Відображення вкладки історії"""
        st.subheader("Історія обраних точок")
        
        if not st.session_state.selected_points:
            st.info("ℹ️ Історія порожня. Додайте точки через калькулятор або карту.")
            return
        
        history_df = pd.DataFrame(st.session_state.selected_points)
        display_cols = ['lat', 'lon', 'alt', 'mgrs', 'decl', 'total', 'storm', 'timestamp']
        available_cols = [col for col in display_cols if col in history_df.columns]
        
        st.dataframe(
            history_df[available_cols],
            use_container_width=True,
            height=300,
            hide_index=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Очистити історію", type="secondary", key="clear_hist"):
                self.clear_history()
                st.success("Історія очищена")
                st.rerun()
        with col2:
            if st.button("Експортувати історію", type="primary", key="export_hist"):
                data_exporter.export_results(history_df, "history")
                st.success("Експорт запущено")
