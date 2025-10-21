# 功能規格：統一化 CLI（回測與訊號輸出）

**功能分支**：`feature/unified-cli`  
**建立日期**：2025-10-18  
**狀態**：草稿  
**輸入來源**：使用者需求：「CLI」與現有 backtest 腳本與 `config/rules.yaml`。

## 使用者情境與測試（必填）

### 使用者故事 1－單檔快速回測（優先序：P1）

身為投資人，我想用單一命令對一檔股票進行回測，並輸出損益曲線圖與交易點 CSV，讓我能快速評估策略表現。

**優先原因**：最小可用價值，對現有 `backtest/run_backtest.py` 直接封裝即可落地。

**獨立測試方式**：執行 `python cli.py backtest one --symbol 2330.TW --start 2023-01-01 --end 2025-09-30`，驗證輸出檔在 `backtest_out/` 生成，且年化報酬與 MDD 有印出。

**驗收情境**：
1. Given 安裝依賴就緒，When 執行 `backtest one`，Then 產生 `<SYMBOL>_equity.png` 與 `<SYMBOL>_trades.csv` 並顯示指標。
2. Given 指定不存在代碼，When 執行，Then 以非零退出碼回報錯誤並顯示可讀訊息。

---

### 使用者故事 2－依設定批次回測（優先序：P2）

身為操盤手，我要用同一命令讀取 `config/rules.yaml` 的股票清單、區間與費用參數，批次執行回測，讓流程自動化且可重複。

**優先原因**：把既有設定檔落地到實務批次，節省重覆操作時間。

**獨立測試方式**：執行 `python cli.py backtest all --config config/rules.yaml`，驗證對每個 `symbols` 皆生成輸出，且參數與設定一致。

**驗收情境**：
1. Given `rules.yaml` 合法，When 執行 `backtest all`，Then 為每個 symbol 生成對應輸出檔並彙總列印摘要表。
2. Given `rules.yaml` 缺欄位，When 執行，Then 顯示缺失鍵與預設處理或中止訊息。

---

### 使用者故事 3－訊號檢視與導出（優先序：P3）

身為分析師，我想快速列印某檔標的的交易訊號（含信心等級）到終端機或另存 CSV，便於檢閱與分享。

**優先原因**：輔助決策與除錯，非 P1 必要但價值高。

**獨立測試方式**：`python cli.py signals --symbol 2330.TW --start 2024-01-01 --end 2024-12-31 --out backtest_out/2330_signals.csv`，驗證檔案存在且欄位齊全。

**驗收情境**：
1. Given 期間內有訊號，When 執行 `signals`，Then 終端打印前 N 筆並可選 `--out` 另存。
2. Given 無訊號，When 執行，Then 顯示「無訊號」並正常退出（退出碼 0）。

---

### 邊界情境

- 目標資料源（yfinance）短暫故障或回傳空資料：應提示重試或變更期間，退出碼非零。
- 多重索引欄位（yfinance 回傳 MultiIndex）需穩定展平，避免欄名衝突造成後續運算錯誤。
- Windows/Unix 皆可執行；輸出路徑與文字編碼（CSV 採 `utf-8-sig`）一致。
- 目錄不存在時自動建立（如 `backtest_out/`）。
- 參數驗證：日期格式錯誤、手續費/滑價為負、symbol 空字串等。

## 需求（必填）

### 功能性需求

- FR-001：提供 `cli.py`，包含子命令：`backtest one`、`backtest all`、`signals`、`plot`（選配）。
- FR-002：`backtest one` 需接受 `--symbol --start --end --commission --slippage`，並重用現有回測邏輯。
- FR-003：`backtest all` 需讀取 `--config`（預設 `config/rules.yaml`），解析 symbols、期間與費用，逐檔執行並彙總報表。
- FR-004：`signals` 需輸出欄位至少含：`Date, signal, confidence, Close`，支援 `--out`。
- FR-005：所有子命令需具備明確錯誤訊息與退出碼；正常完成退出碼 0。
- FR-006：產出檔寫入 `backtest_out/` 目錄；若不存在需自動建立。
- FR-007：日誌/輸出為人類可讀；支援 `--quiet`/`--verbose`（選配）。
- FR-008：與現有 `backtest/run_backtest.py` 與 `config/rules.yaml` 相容，避免破壞既有流程。

### 關鍵實體

- 回測任務：輸入（symbol、期間、費用）、輸出（equity.png、trades.csv、摘要指標）。
- 批次設定：`rules.yaml` 的 symbols、backtest_period、fees、indicators（留待後續策略擴充使用）。
- 訊號紀錄：時間索引、訊號類型（BUY/SELL/HOLD）、信心、價格。

## 成功準則（必填）

### 可量測成果

- SC-001：`backtest one` 在 10 秒內完成單檔下載與回測（網路可用前提）。
- SC-002：`backtest all` 能對 `rules.yaml` 中所有 symbols 生成對應輸出檔且無錯誤退出。
- SC-003：`signals` 產出的 CSV 與終端列印欄位一致、無缺失欄名。
- SC-004：所有子命令提供 `--help` 並涵蓋可用選項與範例。

