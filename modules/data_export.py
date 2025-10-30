
# =============================================
# Експорт даних у різні формати
# =============================================

import streamlit as st
import pandas as pd
import io
from datetime import datetime

class DataExporter:
    """Клас для експорту даних у різні формати"""
    
    def __init__(self):
        pass
    
    def export_results(self, df, export_type="batch"):
        """Експорт результатів"""
        if export_type == "batch":
            csv_data = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                "⬇️ Завантажити CSV", 
                csv_data,
                f"geomag_batch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", 
                "text/csv",
                key="download_batch"
            )
        elif export_type == "history":
            csv_data = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                "⬇️ Завантажити історію", 
                csv_data,
                f"geomag_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", 
                "text/csv",
                key="download_history"
            )
