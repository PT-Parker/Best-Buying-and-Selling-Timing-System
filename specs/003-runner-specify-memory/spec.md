# Feature Specification: 專案內極簡 Runner（解析 tasks-cli.yaml 並執行對應平台腳本）

**Feature Branch**: `003-runner-specify-memory`  
**Created**: 2025-10-20  
**Status**: Draft  
**Input**: User description: "在專案裡加一個極簡 Runner，它會讀 .specify/memory/tasks-cli.yaml，根據你電腦是 Windows/Posix 自動執行對應的 run.windows / run.posix 腳本。加完就能在專案根目錄用 speckit run <task>。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 從任務名稱直接執行（P1）

身為專案使用者，我希望在專案根目錄執行 `speckit run <task>`，Runner 會自動讀取 `.specify/memory/tasks-cli.yaml`，並根據我電腦是 Windows 或 Posix 自動執行對應的 `run.windows` 或 `run.posix` 腳本，讓常見任務一鍵執行。

**Why this priority**: 降低跨平台操作與指令記憶成本，快速啟動專案既有任務流程。

**Independent Test**: 僅安裝 Runner，放入一個簡單任務（例如 echo），即可在兩大平台上用相同命令成功執行。

**Acceptance Scenarios**:

1. Given `tasks-cli.yaml` 中存在 `<task>` 的 `run.windows` 與/或 `run.posix`，When 使用者執行 `speckit run <task>`，Then Runner 根據平台選擇並執行對應腳本且以原生退出碼結束。
2. Given 找不到 `<task>`，When 執行 `speckit run <task>`，Then 提示可用任務清單並以非 0 退出碼結束。
3. Given `<task>` 僅定義其中一個平台腳本，When 在該平台上執行，Then 正常執行；在另一平台執行，Then 以友善訊息告知該任務未提供對應平台腳本並以非 0 結束。

---

### User Story 2 - 列出任務與說明（P2）

身為使用者，我希望可以使用 `speckit list` 查看所有任務與描述，方便快速探索。

**Why this priority**: 提升可發現性與可用性，降低學習成本。

**Independent Test**: 不執行任何任務，也可列出任務清單與描述。

**Acceptance Scenarios**:

1. Given 存在 `tasks-cli.yaml`，When 執行 `speckit list`，Then 以清單顯示任務名稱與描述（若描述缺省則留空）。

---

### User Story 3 - 傳遞參數到任務（P3）

身為進階使用者，我希望 `speckit run <task> -- <args...>` 能將 `--` 後的參數原封不動傳給對應平台腳本，以在必要時覆蓋預設或帶入額外選項。

**Why this priority**: 擴充彈性，不影響初學者體驗。

**Independent Test**: 建立一個回聲任務，驗證 `-- foo bar` 能被腳本接收。

**Acceptance Scenarios**:

1. Given `<task>` 存在且在 Windows/Posix 執行，When 執行 `speckit run <task> -- a b c`，Then 對應平台腳本能接收到 `a b c`。

---

### Edge Cases

- 缺少 `tasks-cli.yaml`：需提示缺少檔案並非 0 結束。
- YAML 結構不正確：提示可讀錯誤並非 0 結束。
- 任務不存在：提示與 `speckit list` 引導。
- 任務無對應平台的 `run.*`：提示任務不支援當前平台。
- 腳本執行失敗（退出碼非 0）：Runner 應以相同退出碼結束並顯示標準錯誤輸出。
- 輸出與檔案生成：Runner 不主動檢查輸出檔案存在性（交由任務腳本負責）。
- 編碼/換行差異：Runner 僅負責轉交命令；平台細節由腳本自行處理。

## Requirements *(mandatory)*

### Functional Requirements

- FR-001：提供 `speckit` 極簡命令介面，支援子命令 `run` 與 `list`，於專案根目錄執行。
- FR-002：`speckit run <task> [-- <args...>]` 會讀取 `.specify/memory/tasks-cli.yaml` 並根據目前平台（Windows/Posix）執行對應 `run.windows` 或 `run.posix`。
- FR-003：當任務不存在、YAML 缺失或無對應平台腳本時，必須輸出清楚錯誤訊息並以非 0 退出碼結束。
- FR-004：Runner 執行子程序時，應傳遞標準輸入/輸出/錯誤，並在完成後以子程序退出碼作為自身退出碼。
- FR-005：`speckit list` 讀取 YAML 並列出所有任務名稱與描述；若描述不存在則留空。
- FR-006：Runner 不修改任務腳本內容；參數透過 `--` 後原樣傳遞給對應平台腳本。[NEEDS CLARIFICATION: 若 YAML 已內含多行命令，是否需以殼層啟動（如 cmd/bash）強制整段執行？]
- FR-007：相依檢查：若當前平台缺少可執行殼層（例如 Posix 無 bash、Windows 無 powershell/cmd），需提示用戶安裝或改用另一平台腳本。

### Key Entities *(include if feature involves data)*

- 任務（Task）：以名稱為鍵，包含 `description`、`run.windows` 與/或 `run.posix`、`outputs` 等屬性。
- 執行請求（Run Invocation）：由 `speckit run <task> -- <args...>` 構成，包含解析後的平台與傳遞參數。

## Assumptions

- `tasks-cli.yaml` 的結構與目前專案一致（tasks: <name>: description/run/...）。
- 使用者於對應平台具備可用的殼層（Windows: cmd/powershell；Posix: bash/sh）。
- Runner 僅在專案根目錄執行；相對路徑以專案根目錄為準。
- 不處理輸出檔案校驗與產物歸檔（交由各任務腳本實作）。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- SC-001：在 Windows 與 Posix 上，`speckit run backtest_all` 能於首次嘗試成功啟動對應任務（前提：任務本身可執行）。
- SC-002：`speckit list` 能在 1 秒內列出所有任務與描述（以本機測試為準）。
- SC-003：當任務不存在或 YAML 缺失時，錯誤訊息明確且退出碼非 0；90% 使用者能從訊息中找出下一步（如使用 `speckit list`）。
- SC-004：參數傳遞驗證：`speckit run <task> -- foo bar` 能在兩平台將 `foo bar` 傳至子腳本（以示範任務測試）。

