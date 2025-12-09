# Changelog

## 2025-12-07
- Replaced the legacy CLI/scripting workflow with a layered architecture (`core/`, `services/`, `app.py`) that powers the Streamlit UI.
- Removed slide-packaging, notebooklm artifacts, batch scripts, and CLI helpers that were unrelated to the core backtest/signal experience.
- Added sample datasets (`data/sample/`) plus unified configuration files for watchlists and strategy parameters.
- Delivered a Streamlit dashboard featuring signal summaries, backtest metrics, model registry controls, and feature previews, along with deployment assets (`README`, `.env.example`, `.streamlit/config.toml`).
