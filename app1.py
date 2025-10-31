"""
Geomagnetic Pro 2025 — Головний додаток
"""
from __future__ import annotations

import streamlit as st
from modules.config import Config
from modules.session_manager import SessionManager
from modules.calculator import Calculator
from modules.map_manager import MapManager
from modules.data_import import DataImporter
from modules.data_export import DataExporter


def main() -> None:
    # === Ініціалізація ===
    config = Config()
    session = SessionManager(config)
    exporter = DataExporter()
    importer = DataImporter()  # ← Додано

    calculator = Calculator(session)
    map_manager = MapManager(session, exporter)

    # === Налаштування сторінки ===
    st.set_page_config(
        page_title="Geomagnetic Pro 2025",
        page_icon="compass",
        layout="wide"
    )

    # === Заголовок ===
    st.title("Geomagnetic Pro 2025")
    st.markdown("**WMM2025 + RTDM | Теплова карта | MGRS/UTM | Висота | Історія**")

    # === Статус сервісів ===
    calculator.display_service_status()

    # === Вкладки ===
    tab_calc, tab_map, tab_pkg, tab_hist = st.tabs([
        "Калькулятор", "Карта", "Пакет", "Історія"
    ])

    # === Калькулятор ===
    with tab_calc:
        calculator.render_calculator_tab()

    # === Карта ===
    with tab_map:
        map_manager.render_map_tab()

    # === Пакетна обробка ===
    with tab_pkg:
        importer.render_import_tab(session, calculator, exporter)

    # === Історія ===
    with tab_hist:
        session.render_history_tab(exporter)

    # === Футер ===
    st.markdown("---")
    st.caption(f"© 2025 Geomagnetic Pro | WMM2025 | RTDM | v1.0")


if __name__ == "__main__":
    main()
