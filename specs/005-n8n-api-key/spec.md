# Feature Specification: n8n API Key 檔案式載入與優先序（Runner/Tasks）

**Feature Branch**: `005-n8n-api-key`  
**Created**: 2025-10-20  
**Status**: Draft  
**Input**: User description: "我把n8n api key放進C:UsersParkerBest Buying and Selling Systemn8n_api_key.txt"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 自動讀取 API Key（P1）

身為使用者，我希望 Runner/任務能自動從「環境變數或本地檔案」取得 n8n API Key，避免每次開新視窗都要重新 export/setx。

**Why this priority**: 降低操作摩擦，提升自動化腳本成功率。

**Independent Test**: 僅提供本地檔案的 API Key，未設定環境變數，仍可成功呼叫 `GET /rest/workflows`。

**Acceptance Scenarios**:
1. Given 未設定 `N8N_API_KEY` 環境變數且存在預設金鑰檔，When 執行 `speckit run n8n_check_endpoints`，Then 能以檔案中的 API Key 完成 API 檢查。
2. Given 同時存在環境變數與檔案，When 執行任務，Then 以「環境變數」優先，檔案作為後備不得覆寫。

---

### User Story 2 - 可設定金鑰來源位置（P2）

身為維運，我希望可以透過環境變數指定金鑰檔案路徑，以適配不同目錄結構或機密檔配置方式。

**Why this priority**: 增強彈性，與各種部署方式兼容。

**Independent Test**: 僅設定 `N8N_API_KEY_FILE` 指向自定義路徑，未設定 `N8N_API_KEY`，仍可成功呼叫 API。

**Acceptance Scenarios**:
1. Given 設定 `N8N_API_KEY_FILE`，When 執行任務，Then 從該檔讀取金鑰字串並使用。
2. Given 檔案不存在或內容為空，When 執行任務，Then 輸出清楚錯誤並以非 0 代碼結束（或跳回其它後備來源）。

---

### User Story 3 - 安全提示與掩碼（P3）

身為使用者，我希望日誌或報告中不會直接印出完整 API Key，僅顯示前後各幾個字元與長度，以降低外洩風險。

**Why this priority**: 基本安全衛生，避免敏感資訊洩漏。

**Independent Test**: 在報告檔與終端輸出中，API Key 僅以掩碼形式顯示。

**Acceptance Scenarios**:
1. Given 任務成功讀到金鑰，When 輸出診斷訊息，Then 顯示掩碼（如 `sk_****abcd (len=... )`），而非明文。

---

### Edge Cases

- 環境變數、使用者層級變數與檔案同時存在：需明確優先序與行為一致。
- 檔案中出現前後空白或換行：需自動 trim。
- 檔案權限過寬（可選）：可提醒風險但不強制阻擋。
- Windows 與 Posix 的路徑差異：支援相對/絕對路徑；Windows 建議使用 `C:\...\n8n_api_key.txt` 格式。

## Requirements *(mandatory)*

### Functional Requirements

- FR-001：Runner/任務在取得 API Key 時，應依下列優先序取值：
  1) 進程環境變數 `N8N_API_KEY`
  2) 使用者層級環境變數 `N8N_API_KEY`
  3) 檔案來源：`N8N_API_KEY_FILE` 指向的路徑
  4) 預設檔案位置（例如：`./n8n_api_key.txt` 或 `C:\Users\<User>\...\n8n_api_key.txt`）[NEEDS CLARIFICATION: 預設檔放哪個路徑最合適？]
- FR-002：成功讀取後，應將字串 trim 去除前後空白與換行。
- FR-003：若無任何來源可用，需輸出清楚錯誤訊息與建議（設定環境變數或放置檔案）。
- FR-004：在 `n8n_check_endpoints`、`n8n_import_workflow`、`n8n_activate_workflow`、`n8n_update_workflow_params`、`n8n_run_via_api` 等任務中，統一套用此金鑰取得邏輯，不得各自為政。
- FR-005：診斷輸出與報告中，API Key 僅以掩碼呈現，避免印出明文。

### Key Entities *(include if feature involves data)*

- 金鑰來源（Key Source）：環境變數、使用者層級環境、指定檔案、預設檔案。
- 載入器（Key Loader）：依優先序解析來源並回傳字串（含掩碼診斷）。

## Assumptions

- 使用者可接受在專案根目錄放置 `n8n_api_key.txt` 作為預設（若不希望，將以變數 `N8N_API_KEY_FILE` 客製）。
- 任務腳本可在 Windows/Posix 中安全地讀檔；若路徑不可用，會回退至其它來源。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- SC-001：在未設任何環境變數、僅提供檔案的情況下，API 類任務成功率 ≥ 95%。
- SC-002：當同時存在變數與檔案時，使用「環境變數優先」的行為一致性達到 100%。
- SC-003：報告與日誌中無明文 API Key（抽查 100% 不外洩）。

