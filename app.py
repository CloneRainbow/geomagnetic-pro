# =============================================
# 🧭 Geomagnetic Pro 2025
# =============================================

import streamlit as st
from modules.config import Config
from modules.session_manager import SessionManager
from modules.calculator import Calculator
from modules.map_manager import MapManager
from modules.data_export import DataExporter

# === Ініціалізація ===
config = Config()
session = SessionManager(config)
exporter = DataExporter()

calculator = Calculator(session)
map_manager = MapManager(session, exporter)
    
    # Заголовок додатку
    st.title("🧭 Geomagnetic Pro 2025")
    st.markdown("Карта | MGRS пошук | Геомагнітні параметри")
    
    # Інформація про статус сервісів
    calculator.display_service_status()
    
    # Створення вкладок
    tabs = st.tabs(["Калькулятор", "Карта", "Пакет", "Історія вибору"])
    
    # Вкладка Калькулятор
    with tabs[0]:
        calculator.render_calculator_tab(session)
    
    # Вкладка Карта
    with tabs[1]:
        map_manager.render_map_tab(session, calculator)
    
    # Вкладка Пакетна обробка
    with tabs[2]:
        data_importer.render_import_tab(session, calculator, data_exporter)
    
    # Вкладка Історія
    with tabs[3]:
        session.render_history_tab(data_exporter)
    
    # Футер
    config.render_footer()

if __name__ == "__main__":
    main()
