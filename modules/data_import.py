# =============================================
# Імпорт даних з різних форматів
# =============================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

class DataImporter:
    """Клас для імпорту даних з різних форматів"""
    
    def __init__(self):
        pass
    
    def render_import_tab(self, session, calculator, data_exporter):
        """Відображення вкладки імпорту даних"""
        st.subheader("📦 Пакетна обробка даних")
        
        file = st.file_uploader("Завантажте CSV файл", type=["csv"], key="batch_csv")
        
        if file:
            try:
                df = pd.read_csv(file)
                st.success(f"Файл успішно завантажено: {len(df)} рядків")
                
                if not all(c in df.columns for c in ["lat", "lon"]):
                    st.error("CSV файл повинен містити колонки 'lat' та 'lon'")
                    st.stop()
                
                df["alt"] = df.get("alt", 0).fillna(0)
                
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("Кількість точок", len(df))
                with col_stats2:
                    valid_lats = df[(df['lat'] >= -90) & (df['lat'] <= 90)]
                    st.metric("Валідні широти", len(valid_lats))
                with col_stats3:
                    valid_lons = df[(df['lon'] >= -180) & (df['lon'] <= 180)]
                    st.metric("Валідні довготи", len(valid_lons))
                
                col_batch1, col_batch2 = st.columns(2)
                with col_batch1:
                    batch_date = st.date_input("Дата для розрахунків", value=date.today(), key="batch_date")
                with col_batch2:
                    include_mgrs_details = st.checkbox("Включити деталі MGRS", value=True, key="include_mgrs")
                
                pts = [(r["lat"], r["lon"], r["alt"]) for _, r in df.iterrows()]
                
                if st.button("🚀 Запустити пакетну обробку", type="primary", key="batch_btn"):
                    self._process_batch_data(pts, batch_date, include_mgrs_details, calculator, data_exporter)
                    
            except Exception as e:
                st.error(f"Помилка читання CSV файлу: {e}")
    
    def _process_batch_data(self, points, batch_date, include_mgrs_details, calculator, data_exporter):
        """Обробка пакетних даних"""
        with st.spinner(f"Обчислення для {len(points)} точок..."):
            results = []
            progress_bar = st.progress(0)
            
            for i, point in enumerate(points):
                lat, lon, alt = point
                if calculator.coord_converter.validate_coordinates(lat, lon, alt):
                    res = calculator.calc_point(lat, lon, alt, calculator.wmm_loader.decimal_year(batch_date))
                    if "error" not in res and include_mgrs_details:
                        res.update(res.get("mgrs_details", {}))
                    results.append(res)
                else:
                    results.append({"error": "Invalid coordinates", "lat": lat, "lon": lon, "alt": alt})
                
                progress_bar.progress((i + 1) / len(points))
            
            df_out = pd.DataFrame(results)
            
            st.subheader("📊 Результати обробки")
            st.dataframe(df_out, use_container_width=True, height=400)
            
            successful = len([r for r in results if "error" not in r])
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Успішні розрахунки", successful)
            with col_stat2:
                st.metric("Помилки", len(results) - successful)
            
            # Експорт результатів
            data_exporter.export_results(df_out, "batch")
