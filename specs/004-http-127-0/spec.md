# Feature Specification: n8n 端點與 URL 指南（UI / API / Webhook 驗證）

**Feature Branch**: `004-http-127-0`  
**Created**: 2025-10-20  
**Status**: Draft  
**Input**: User description: "http://127.0.0.1:5678/home/workflows 這個是嗎?"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 確認本機 n8n UI 是否可用 (Priority: P1)

身為使用者，我希望快速確認本機或指定主機上的 n8n UI 是否可開啟（如 `http://127.0.0.1:5678/home/workflows`），以確保服務運行、可登入與操作工作流。

**Why this priority**: 若 UI 無法存取，之後的 API／Webhook 測試也無從進行。

**Independent Test**: 以瀏覽器開啟 `BASE_URL/home/workflows`；可看到工作流清單或登入頁即視為可用。

**Acceptance Scenarios**:
1. Given n8n 正在執行，When 使用者開啟 `BASE_URL/home/workflows`，Then 顯示 UI 頁面（登入或工作流清單）。
2. Given 服務未啟動或網址錯誤，When 存取該 URL，Then 提示無法連線（或 404/502），並指引使用者檢查服務／連線設定。

---

### User Story 2 - 取得正確的 Webhook URL 並驗證 (Priority: P1)

身為整合者，我希望明確知道可對外呼叫的 Webhook URL（例如 `BASE_URL/webhook/tw-live-signal`），並能以一筆測試資料驗證是否能被工作流接收與處理。

**Why this priority**: Webhook 是外部系統觸發流程的關鍵入口。

**Independent Test**: 用 `curl` 或任一 HTTP 工具對 Webhook URL POST 一筆 JSON，確認工作流有記錄（例如寫入 Google Sheets 或日誌）。

**Acceptance Scenarios**:
1. Given 已知 `BASE_URL` 與路徑，When 對 `BASE_URL/webhook/<path>` 送出 JSON，Then 工作流節點被觸發且返回 200/OK。
2. Given 路徑錯誤或未啟用，When 送出請求，Then 返回 404/錯誤並提供排查建議（確認路徑、啟用狀態、權限）。

---

### User Story 3 - 以 API Key 測試 REST API 可用性 (Priority: P2)

身為維運人員，我希望確認 `BASE_URL/rest` 的 API 可用，並用 API Key 調用 `GET /rest/workflows` 或 `GET /rest/workflows/{id}`，以確定自動化腳本可正常工作。

**Why this priority**: 自動化（匯入、啟用、更新節點參數）依賴 REST API 正常運作。

**Independent Test**: 以 `X-N8N-API-KEY` 呼叫 `GET /rest/workflows`，收到 200 與 JSON 列表。

**Acceptance Scenarios**:
1. Given 正確 API Key，When 呼叫 `GET /rest/workflows`，Then 返回 200 與工作流清單。
2. Given 缺少或錯誤的 API Key，When 呼叫 API，Then 返回權限錯誤並指引使用者設定金鑰或權限。

---

### Edge Cases

- `BASE_URL` 與實際服務位址不一致（127.0.0.1 vs 主機 IP/網域）：需提示使用適合對外存取的 URL。
- HTTP 與 HTTPS：若前端使用反向代理或憑證，需明確指出 `https://` 與正確連接埠。
- 服務未啟動、埠被占用或防火牆阻擋：提供檢查清單（服務狀態、連接埠、網路規則）。
- API Key 缺失或權限不足：清楚錯誤訊息與產生／設定方式。
- Webhook 路徑與工作流未啟用：提示啟用工作流與路徑大小寫匹配。

## Requirements *(mandatory)*

### Functional Requirements

- FR-001：提供一份清楚的 URL 指南，說明 UI、REST API、Webhook 的標準路徑與組合方式（以 `BASE_URL` 為核心）。
- FR-002：提供「快速檢查步驟」，協助使用者驗證 `BASE_URL/home/workflows` 是否可開啟（UI 可用性）。
- FR-003：提供 Webhook 測試指引與範例負載，協助驗證 `BASE_URL/webhook/<path>` 是否可被工作流接收與處理。
- FR-004：提供 REST API 測試指引（含 `X-N8N-API-KEY` 標頭），協助驗證 `BASE_URL/rest/workflows` 可用。
- FR-005：當驗證失敗時，提供具體排查建議（服務狀態、URL/路徑、啟用狀態、HTTPS/Port、API Key）。
- FR-006：以環境變數或設定檔記錄 `BASE_URL`，降低不同環境（本機、LAN、網域、雲端）切換成本。[NEEDS CLARIFICATION: `BASE_URL` 優先來源與命名（N8N_BASE_URL？）]

### Key Entities *(include if feature involves data)*

- 端點定義（Endpoint Definition）：由 `BASE_URL` 與特定路徑構成的實際 URL（UI、API、Webhook）。
- 憑證設定（Credential Settings）：API Key、（可選）HTTPS 憑證與 SMTP/Sheets 等相依服務憑證。

## Assumptions

- 使用者已可啟動 n8n 服務，知道基本管理介面位置。
- 若對外提供 Webhook，部署端將自行處理防火牆、反向代理與憑證設定。
- 測試時以最小可行範例（單一 Webhook 與簡單 JSON）驗證可用性。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- SC-001：使用者可在 2 分鐘內確認 UI 是否可用（能開啟 `BASE_URL/home/workflows`）。
- SC-002：90% 使用者可在首次嘗試內透過提供的範例成功觸發 Webhook（返回 200），或在 5 分鐘內依排查建議解決。
- SC-003：使用者能以 API Key 成功呼叫 `GET /rest/workflows`，驗證自動化可行（成功率 95% 以上）。
- SC-004：對於錯誤情況，使用者可依文件完成至少一項有效排查（如修正 `BASE_URL`、設定 API Key、啟用工作流）。

