# =============================================
# Управління сесією та станом додатку
# =============================================

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

class SessionManager:
    """Клас для управління станом сесії"""
    
    def __init__(self):
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Ініціалізація стану сесії"""
        if "selected_points" not in st.session_state:
            st.session_state.selected_points = []
        if "last_click" not in st.session_state:
            st.session_state.last_click = None
        if "mgrs_search" not in st.session_state:
            st.session_state.mgrs_search = None
        if "today" not in st.session_state:
            st.session_state.today = datetime.now().date()
    
    def add_to_history(self, point: Dict):
        """Додавання точки до історії"""
        if len(st.session_state.selected_points) >= 10:
            st.session_state.selected_points.pop(0)
        st.session_state.selected_points.append(point)
    
    def get_selected_points(self) -> List[Dict]:
        """Отримання списку вибраних точок"""
        return st.session_state.selected_points
    
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
    
    def clear_history(self):
        """Очищення історії"""
        st.session_state.selected_points = []
        st.session_state.last_click = None
    
    def render_history_tab(self, data_exporter):
        """Відображення вкладки історії"""
        st.subheader("📚 Історія обраних точок")
        
        if not st.session_state.selected_points:
            st.info("ℹ️ Ще немає обраних точок. Використовуйте карту або калькулятор для додавання точок.")
        else:
            history_df = pd.DataFrame(st.session_state.selected_points)
            display_cols = ['lat', 'lon', 'alt', 'mgrs', 'decl', 'total', 'storm', 'timestamp']
            available_cols = [col for col in display_cols if col in history_df.columns]
            
            st.dataframe(history_df[available_cols], use_container_width=True, height=300)
            
            col_hist1, col_hist2 = st.columns(2)
            with col_hist1:
                if st.button("🗑️ Очистити історію", type="secondary"):
                    self.clear_history()
                    st.rerun()
            with col_hist2:
                if st.button("💾 Експортувати історію", type="primary"):
                    data_exporter.export_results(history_df, "history")
