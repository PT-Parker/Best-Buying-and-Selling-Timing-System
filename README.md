# Best Buying and Selling Timing System

Streamlit application for analysing Taiwan stock strategies end-to-end: ingest sample or live Yahoo Finance data, engineer features, run the indicator-based backtest, surface buy/sell signals, and manage model versions – all without touching the CLI.

## Key Features
- **Single Streamlit entry point** (`streamlit run app.py`) with sidebar controls for watchlist、日期區間與指標參數，後端始終透過 yfinance 取得即時 OHLCV。
- **Signal dashboard** summarising the latest buy/sell recommendations, confidence, rationale, and QC anomalies with CSV export.
- **Backtest view** showing metrics (CAGR, MDD, win rate), equity curve, and trade ledger per symbol with download buttons.
- **Model registry & training** preview of the latest XGBoost classifier metrics plus an Advanced-mode action to retrain using the current dataset.
- **Feature preview** to inspect engineered indicators feeding the ML/strategy layers.
- **Data abstraction** through `services/data_source.py` that supports sample CSVs (for Streamlit Cloud), yfinance downloads, or future extensions.

## Project Layout
```
app.py                    # Streamlit UI
core/                     # Pure analytics primitives (backtest, features, labeling, inference)
services/                 # Data loading, backtest orchestration, signals, model registry
config/                   # Watchlist + strategy defaults
data/sample/              # Self-contained demo dataset for cloud deployments
metrics/                  # JSONL registries (model history, etc.)
tests/                    # Unit tests covering core/services layers
```

## Getting Started
1. **Environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # on Windows
   pip install -r requirements.txt
   ```
2. **Configuration**
   - Copy `.env.example` to `.env` and adjust as needed（主要是 watchlist / registry 路徑設定）。
   - Update `config/watchlist.yaml` 以及 `config/strategy.yaml`；`strategy.yaml` 內現在包含 `labeling` 區塊，可調整 `horizon_days`、`take_profit_pct`、`stop_loss_pct` 來同時產生作多/作空的標記。
3. **Run locally**
   ```bash
   streamlit run app.py
   ```
   App 會直接從 yfinance 下載資料，請確保環境具備網路與 `yfinance` 套件。

### 模型訓練時間窗
- 在 Advanced 模式按下「以目前資料重新訓練模型」時，系統會自動：
  1. 以「今天往前 30 天」的日期為訓練結束日，往前再取一年做訓練樣本（建立訊號與標籤）。
  2. 以訓練結束日隔天到今天為驗證區間，對照模型預測與實際標籤，顯示驗證 Precision / Recall / F1。
- 兩段資料都透過 yfinance 取得，無需手動切換日期或資料來源。

## Deployment Notes
- **Streamlit Cloud**: keep `DATA_MODE=sample` and commit the latest `data/sample/*.csv` so the app runs without external data sources.
- **Local full mode**: set `DATA_MODE=auto` (default) and ensure outbound network access for yfinance. Set `STREAMLIT_OFFLINE=1` if you need to force sample data even when online.
- **Model registry**: artifacts are appended to `metrics/model_registry.jsonl`. Provide persistent storage or replace with a remote bucket if deploying multi-user.

## Testing
```
pytest
```
The suite covers backtest mechanics, parameter sweeps, labeling logic, and signal summarisation.

## Roadmap / Next Steps
- Integrate optional realtime twstock feed inside `services/data_source`.
- Add authentication or sharing controls before exposing on public Streamlit Cloud workspaces.
