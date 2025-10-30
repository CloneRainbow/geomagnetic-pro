# =============================================
# Конфігурація додатку
# =============================================

import streamlit as st

class Config:
    """Клас для управління конфігурацією додатку"""
    
    def __init__(self):
        self.app_name = "Geomagnetic Pro 2025"
        self.app_icon = "🧭"
        self.layout = "wide"
        
        # Константи
        self.COF_URL = "https://www.ncei.noaa.gov/data/world-magnetic-model-2025/full/WMM_COEFS-2025.zip"
        self.COF_PATH = "wmm/WMM_2025.COF"
        self.RTDM_API = "https://geomag.usgs.gov/ws/edge/"
    
    def setup_page(self):
        """Налаштування сторінки Streamlit"""
        st.set_page_config(
            page_title=self.app_name,
            page_icon=self.app_icon,
            layout=self.layout
        )
    
    def render_footer(self):
        """Відображення футера"""
        st.markdown("---")
        st.markdown(
            f"""
            <div style='text-align: center'>
                <strong>{self.app_name}</strong> • WMM 2025 • Leaflet Map<br>
                <small>Розроблено для геомагнітних досліджень та навігації</small>
            </div>
            """,
            unsafe_allow_html=True
      )
