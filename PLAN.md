# PLAN

## 現況摘要
### 專案結構與主要入口
- `cli.py` 是唯一的 CLI 入口，聚合了回測 (`backtest/run_backtest.py`)、訊號輸出、繪圖、Slides/打包、批次參數掃描 (`scripts/batch_sweep.py`)、以及 pipeline（data ingest / train）等大量功能，沒有圖形化介面或 Streamlit 入口。
- 策略核心集中在 `backtest/run_backtest.py`：從 yfinance 下載價量資料，計算 EMA/RSI/Bollinger 訊號並回測，輸出 equity/trade CSV。`cli.py`、`scripts/batch_sweep.py`、`scripts/pipeline/historical_evaluator.py` 都直接呼叫 `generate_signals/backtest`。
- `scripts/pipeline/` 內含一組資料科學模組：`data_ingest.py`、`feature_builder.py`、`labeling.py`、`train_model.py`、`live_features.py`、`historical_evaluator.py`、`reporting.py`、`metadata.py` 等，對應 ingest → 特徵 → 標籤 → 訓練 → 評估 → 紀錄；僅透過 CLI 暗示用法，尚未與 UI 整合。
- 即時監控(`scripts/realtime_monitor.py`)獨立運作，直接讀 `config/watchlist.yaml`，從 twstock 抓取分 K，執行 EMA/RSI 規則與 (可選的) XGBoost 模型，再透過多種通知模式 (LINE/Discord/GAS/N8N) 寫入 `logs/`、`metrics/`、`state/`。
- 其他腳本：`scripts/test_historical_signal.py`（單日回放 + GAS 通知）、`scripts/find_volatile_stocks.py`（批次波動率）、多個 PowerShell 腳本（webhook/monitor helper）、`scripts/build_slides.py` + `tools/generate_pptx.py`（產簡報）、`scripts/package_deliverables.py`（打包成果）。
- 專案資料夾：`backtest_out/` (成果)、`metrics/` (JSONL)、`logs/`、`state/`、`models/`、`data/`、`rda股票數據/`、`deliverables/`（含 notebooklm_source_dir 的大量遞迴副本）、`google sheets api/`（私密金鑰）、`specs/`（舊功能說明）。

### 資料流 / 模組依賴
- 歷史回測流程：`cli._download_data()` → `backtest.generate_signals()` → `backtest.backtest()` → `backtest_out/` PNG/CSV → `scripts/build_slides.py`/`package_deliverables.py`。Batch sweep 與 historical evaluator 也重用 `generate_signals`，但各自再實作輸出格式。
- 數據/特徵/模型：`scripts/pipeline/data_ingest.py` 合併 RDA/CSV → `feature_builder.build_features()` 產特徵 → `labeling.assign_labels()` 打標 → `train_model.py` 讀 feature parquet + Labeling + XGBoost 訓練 + 記錄到 `metrics/model_registry.jsonl`。`live_features.py`/`ModelPredictor` 則是線上版 pipeline。
- 即時訊號：`scripts/realtime_monitor.Engine` -> twstock data -> features -> rule decision + optional `ModelPredictor` → `logs/alerts.csv`, `metrics/live_trades.csv`, GAS webhook。
- 配置：`config/rules.yaml`（指標/fees）、`config/watchlist.yaml`（觀察清單 + 策略 + 通知 + 靜態 secrets）、`config/monitor.yaml`、`config/data_ingest.yaml.example`、`config/train.yaml.example`。目前缺統一的 config loader/service。
- 判斷/可視化：只有 PNG、CSV、PPTX、及 CLI 印出的摘要；無任何 UI 來串資料→特徵→模型→訊號→回測的流程。

### 核心必留功能
1. **指標式訊號與回測引擎**（`backtest/run_backtest.py` + `cli.py` 使用）：是 80/20 決策價值的來源，Streamlit 需以此展示歷史績效、交易軌跡與報表。
2. **資料/特徵/模型管線模組**（`scripts/pipeline`）：提供 ingest、特徵工程、標籤、訓練、模型登錄；未來需要以 Service 層呼叫並在 UI 中選模型或載入指標。
3. **即時/今日訊號邏輯**：`cli.compute_signal` + `realtime_monitor.Engine` 代表如何將策略應用在最新資料，是 Streamlit「今日訊號摘要」的基礎（可改以 yfinance / 示例資料供雲端展示）。
4. **配置與成果儲存**：`config/*.yaml`、`metrics/*.jsonl`、`backtest_out/`、`signals_today.csv` 等需改裝為 config/service+儲存位置，供 UI 載入與下載。
5. **模型與指標共用函式**：`scripts/pipeline/live_features.py`, `historical_evaluator.py`, `metadata.py` 等已實作的邏輯應保留並模組化，避免重寫。

### 可合併 / 簡化功能
- 多處重複的訊號計算（`backtest.generate_signals`, `cli.compute_signal`, `scripts/test_historical_signal.compute_signal`）應收斂為 `features.py/signals.py`，供回測、今日訊號、即時監控共用。
- CLI 中的 backtest/backtest-all/backtest-one/backtest_all_simple/batch sweep，可以重構為一個 `services/backtest.py`（同步支援單檔、批次、參數組合）；UI 以配置化叫用，而非 subprocess。
- Pipeline 模組彼此分散，需要以 `core/features.py`, `core/labeling.py`, `core/inference.py` 等新分層對應；`data_ingest` / `historical_evaluator` / `reporting` 可以整併進共享 data_source/backtest/service。
- 即時監控 (twstock) 與今日訊號 (yfinance) 多數程式碼是一樣的條件判斷，可改為 `services/signals.py` + `services/data_source.py`（回傳最新 Bar Data）。
- 成果輸出 (CSV/PNG/PPTX/ZIP) 應聚焦在必要的 CSV/圖表；Slides/Zip 可從 UI 下載 CSV 的 approach 取代。

### 應移除 / 視為死碼的功能
- **Deliverables/Slides 工具**：`scripts/build_slides.py`, `scripts/package_deliverables.py`, `tools/generate_pptx.py`, `tools/slides.json`, 以及整個 `deliverables/`（尤其是 `notebooklm_source_dir` 的遞迴副本），和目標「Streamlit 單一入口」無關。
- **Powershell / Webhook helper**：`scripts/monitor_live.ps1`, `test_n8n_webhook.ps1`, `switch_notify_to_webhook.ps1`, `monitor_smoketest.ps1`, `set_webhook_env_local.ps1` 等僅支援舊 CLI 佈署。
- **n8n / Google Sheets API 殘留**：`google sheets api/`（含金鑰）、`n8n_api_key.txt`, `specs/00x` 中的舊工作流程對目前 Streamlit 方向不再適用，需從程式碼路徑移除。
- **冗餘腳本**：`scripts/test_historical_signal.py`, `scripts/find_volatile_stocks.py`, `cli` 中與 slides/package 有關的命令、`speckit*.py`、`deliverables/notebooklm_*` 的拷貝檔案皆屬 demo/歷史需求。
- **多層 nesting artifacts**：`deliverables/notebooklm_source_dir` 內含成百上千的重複檔案，僅用於 NotebookLM，會阻礙後續維護與部署。
- **未實作的 pipeline subcommand**：`cli pipeline monitor-live` 直接回傳「not implemented」，應一併刪除或改為新 UI 內建功能。

以上盤點為後續「重構計畫」的基礎，下一步會依此撰寫具體刪改清單與新架構提案。

## 重構計畫
### 預計刪除 / 停用項目（含理由）
- `scripts/build_slides.py`, `scripts/package_deliverables.py`, `tools/generate_pptx.py`, `tools/slides.json`、`deliverables/**`：僅供簡報/打包展示，與 Streamlit 操作流程無關，且大幅增加部署體積。
- `deliverables/notebooklm_source_dir/**` 與遞迴副本：NotebookLM 專用的素材備份，無法在雲端部署時使用；刪除以減少 repo 體積與噪音。
- `scripts/test_historical_signal.py`, `scripts/find_volatile_stocks.py`, 所有 `*.ps1`（monitor_live、switch_notify、test_n8n 等）：CLI 與 webhook 測試工具，只支援舊自動化流程，會被 Streamlit 的 `services/signals.py` 取代。
- `google sheets api/`、`n8n_api_key.txt` 等私密金鑰或舊整合檔案：未來改由環境變數 + config 值控制，不再捆綁憑證。
- `scripts/realtime_monitor.py`、`cli.py` 中非回測/訊號的子命令（slides、package、backtest-all-simple 等）：將邏輯抽成 service 後，舊 CLI 不再外露；Streamlit 是唯一正式入口。
- 過期示範檔（`speckit*.py`, `PROJECT_one/**`, `slides/**` 中手工產出 PPTX）：不提供決策價值，刪除以專注於核心策略。

### 合併 / 模組化策略
- **訊號計算統一**：把 `backtest.run_backtest.generate_signals`、`cli.compute_signal`、`scripts/test_historical_signal.compute_signal`、`realtime_monitor.Engine` 指標邏輯合併為 `core/signals.py`（統一回傳帶信心指標的 DataFrame / dict），以便回測、即時訊號、今日摘要共用。
- **資料來源統一**：整理 `cli._download_data`, `scripts/batch_sweep._download_ohlc`, `scripts/pipeline/data_ingest` 內的 I/O，到 `services/data_source.py`（支援 yfinance、twstock、CSV/RDA、sample data），回傳標準 schema。
- **回測/策略整合**：將 `backtest/backtest.py`（新命名）提供純演算法，`services/backtest.py` 負責 orchestrate（載入資料 → 計算特徵/訊號 → 執行 backtest → 回傳 trades + metrics + equity）。Batch sweep 需要的參數組合會實作為此 service 的 helper。
- **特徵/標籤/模型**：把 `scripts/pipeline/feature_builder.py`, `labeling.py`, `train_model.py`, `live_features.py`, `model_registry.py` 轉進 `core/` 與 `services/`：
  - `core/features.py`：包含 indicator features + ML 特徵建構。
  - `core/labeling.py`：保留 LabelConfig 與 assign_labels。
  - `core/inference.py`：抽出模型載入、預測 (含 XGBoost Booster) 與版本管理。
  - `services/model.py`（或在 `core/inference` 中提供 helper）負責寫入/讀取 registry、存取 `metrics/model_registry.jsonl`。
- **Config 統一**：建立 `config/strategy.yaml`（或沿用 `watchlist.yaml`），加上模式/門檻/停利停損/冷卻，並提供 `config/app.yaml` 控制 UI 顯示。原 `rules.yaml` 的 fees + indicator 參數將移入該檔；`watchlist.yaml` 僅保留清單。
- **產出/樣本資料**：建立 `data/sample/`（例如 `sample_prices.csv`, `sample_signals.csv`, `sample_metrics.json`）供 Streamlit Cloud 使用；同時維持 `backtest_out/` 供在地模式寫入。
- **Change log**：新增 `CHANGELOG.md` 記錄刪除/搬移，並在 PLAN.md 的「重構計畫」條列與實作保持同步。

### 新目錄 / 分層設計
```
app.py                      # Streamlit 主入口，包含 sidebar + main tabs
core/
  __init__.py
  backtest.py               # 原 backtest/run_backtest 的純邏輯 + 指標與 equity 計算
  features.py               # 由 pipeline/feature_builder + signal indicators 重構
  labeling.py               # 由 pipeline/labeling 搬移
  inference.py              # 模型載入、推論、訓練 API（封裝 XGBoost + registry）
services/
  __init__.py
  data_source.py            # 統一資料取得（yfinance, sample CSV, RDA, twstock stub）
  backtest.py               # orchestration（接收參數、呼叫 core、輸出 charts/trades）
  signals.py                # 今日/即時訊號，呼叫 data_source + core.signals
  registry.py               # 讀寫 metrics/model_registry.jsonl, 回傳版本表
  cache.py (可選)           # 共用快取/路徑 helper
config/
  strategy.yaml             # 指標 + fees + thresholds
  watchlist.yaml            # 股票清單/metadata
  app.yaml (可選)           # UI 預設參數
data/sample/
  prices.csv                # 示範資料
  watchlist.yaml            # 雲端模式 fallback
assets/
  logo.png 或示例圖片       # Streamlit UI 用
```

### Streamlit 操作流程設計
1. **Sidebar**  
   - Watchlist 選擇：讀取 `config/watchlist.yaml`（無檔時 fall back 到 `data/sample/watchlist.yaml`）。  
   - 時間區間選擇：預設近期 1Y，支援 `Basic`（固定）與 `Advanced`（自訂 start/end）。  
   - 策略參數：EMA fast/slow、RSI 門檻、Bollinger 參數、停利/停損、cooldown、T days（對 labeling/backtest）等 slider。  
   - 模式切換：`Basic` 使用 sample data +預測; `Advanced` 可選擇資料來源（yfinance / 檔案上傳 / sample）與模型版本、是否立即再訓練。  
   - 其他控制：`Show debug details` checkbox、`Refresh data` button。

2. **Main 頁面結構**  
   - **今日訊號摘要卡**：`services/signals.get_signal_overview()` 回傳 summary（每檔 action/confidence/score/理由），以 `st.dataframe` + `st.metric` 呈現；提供 loading/empty/error state（spinner、`st.info`、`st.error`）。  
   - **回測結果區**：Tabs `["Performance", "Trades", "Equity"]`，使用 `services/backtest.run_backtest()` output (metrics dict, trades df, equity df)。  
     - Performance: `st.metric` for CAGR/MDD/WinRate，`st.bar_chart` for returns by year。  
     - Trades: table + download button (CSV via `st.download_button`)。  
     - Equity: `st.line_chart` with `equity` series。  
     - 若 `backtest_out/` 有現成 CSV/PNG，`Advanced` 模式可載入/比較歷史 run。  
   - **模型版本 / 績效**：讀 `services/registry.list_models()` → 顯示最新模型 metrics、feature importance (bar chart)、版本日誌; `Advanced` 模式可觸發 `core.inference.train_model()`（用 `core.features`+`labeling`）並寫入 `models/` + `metrics/model_registry.jsonl`。  
   - **資料 & 特徵檢視**：顯示 `services.data_source.preview()` + `core.features.build_features()` 的樣本10列，協助確認特徵工程結果。  
   - **輸出整合**：提供「下載最新訊號 / 交易紀錄 / 特徵集」按鈕，如果雲端無法寫檔，就以 `st.download_button` 直接輸出 CSV。  
   - **Debug 面板**（僅在 `Show debug details`）：顯示原始 JSON、API latency等，以 `st.expander` 包裹。  
   - 所有區塊都需處理 `empty`（資料不足）+ `error`（exception）狀態，並顯示 `st.warning` 或 `st.error`。

3. **資料 → 特徵 → 模型 → 回測 → 訊號 → 輸出整合流程**  
   - **資料取得**：`services.data_source.fetch_prices(symbols, start, end, mode)` 先檢查 `local mode` vs `sample mode` vs `live (yfinance)` vs `twstock minute` stub；支援 caching，若 Streamlit Cloud 無法讀真實 RDA，就自動切 sample。  
   - **特徵工程**：`core.features.build_features()` 接 DataFrame + `FeatureConfig`，在 Streamlit 中 `Advanced` mode show options (choose feature list)；結果 feed into labeling/backtest/inference。  
   - **模型訓練與版本管理**：`core.inference.train(dataset, params)` 直接回傳 metrics + Booster + feature importance；`services.registry.save_model()` 寫 JSONL + `.json` model，UI 顯示 `Model version` + metrics + `Download model` button。  
   - **回測**：`services.backtest.run()` 會 (a) 透過 `core.signals.generate_signals()` 取得 `signal_df`，(b) `core.backtest.execute()` 產 trades & metrics, (c) optional labeling/backtest to compare with ML scores; outputs stored to `backtest_out/` (local) or `tmp` (cloud).  
   - **今日/即時訊號**：`services.signals.live_summary()` 會摘取 `latest prices` (yfinance day close or sample), apply `core.signals`, optionally combine `core.inference.score()` (if model available)，生成 summary/notes, plus detection of stale data/anomaly (QC) -> `st.warning`.  
   - **輸出整合**：UI 內建 `Download CSV`/`Download JSON` actions (no CLI needed)。 `Advanced` mode 也可以 push result to `metrics/` by writing JSONL through service; for Streamlit Cloud, fallback to `st.session_state` downloads。

4. **部署 / 環境配置**  
   - `requirements.txt` 只保留 Streamlit + 必要 data/ML 套件 (pandas, numpy, yfinance, twstock, pyarrow, xgboost, scikit-learn, matplotlib, PyYAML)。  
   - `.env.example`：新增 `DATA_MODE`, `WATCHLIST_PATH`, `MODEL_REGISTRY_PATH`, `ENABLE_REALTIME`, `LINE_NOTIFY_TOKEN` (可留空)，Streamlit Cloud 讀 `.env`/secrets。  
   - `.streamlit/config.toml`：設定 page title、theme、server.headless。  
   - `README.md`：撰寫「本機模式」「Streamlit Cloud 模式」操作、主要功能說明、資料流圖。  
   - `data/sample/`：至少包含 `sample_prices.csv`, `sample_trades.csv`, `sample_model.json`, `sample_signals.csv` 以便雲端 demo。  
   - `CHANGELOG.md`：紀錄刪除/搬遷/新增架構，並鏈接到 PLAN.md 條列。

此計畫完成後，再進入 Step 3 進行實際精簡與重構。
