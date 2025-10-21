# Feature Specification: 可匯入的 n8n Workflow（Webhook → Google Sheets → Email）

**Feature Branch**: `002-a-n8n-workflow`  
**Created**: 2025-10-20  
**Status**: Draft  
**Input**: User description: "A) 提供可匯入的 n8n Workflow（Webhook → Google Sheets → Email）"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Webhook 事件寫入試算表 (Priority: P1)

身為營運/分析人員，我希望有一個可公開（或受限制）的 Webhook 入口，能將外部送進來的事件資料（JSON）可靠地追加寫入 Google 試算表，讓我即時彙整並後續分析。

**Why this priority**: 事件資料的可靠入庫是整體流程的基礎，若無法落地至表單，後續通知與分析皆無從談起。

**Independent Test**: 只導入 Webhook→Sheets 流程即可驗證：對 Webhook 送出一筆有效 JSON，於目標試算表新增一列對應欄位，並回應成功狀態。

**Acceptance Scenarios**:

1. Given 試算表已設定且可寫入，When 以 JSON POST 至 Webhook，Then 試算表在 3 秒內新增一列包含必要欄位與接收時間。
2. Given 缺少必要欄位，When 送出請求，Then 回應明確錯誤與缺失欄位列表，且不寫入試算表。
3. Given 重複事件（相同事件識別），When 送出請求，Then 系統去重後僅保留一列紀錄並回應成功或已處理狀態。

---

### User Story 2 - 新增資料即時郵件通知 (Priority: P2)

身為通知接收者，我希望在新資料寫入試算表後，能收到包含重點欄位的郵件通知，以便即時跟進。

**Why this priority**: 及時知會可提升流轉效率，避免重要事件被忽略。

**Independent Test**: 僅啟用 Sheets→Email 流程：對試算表新增一列（或模擬 Webhook 輸入），驗證郵件於合理時間內送達指定收件者，內容包含對應欄位值。

**Acceptance Scenarios**:

1. Given 有效的收件者清單，When 新增一列資料，Then 於 1 分鐘內寄出郵件，主旨與內文套用已定義模板與欄位。
2. Given 部分收件者信箱無效，When 寄送發生退信，Then 系統保留退信資訊於流程紀錄，不重複觸發同筆通知。
3. Given 郵件服務暫時不可用，When 追加寫入成功，Then 系統應提供重試或降級策略（例如暫存佇列、延時重試），不影響 Webhook 回應。

---

### User Story 3 - 可匯入與可設定的工作流程 (Priority: P3)

身為維運人員，我希望能直接匯入既定的 n8n Workflow JSON，並透過少量環境變數/憑證設定即可運作，降低部署與複製成本。

**Why this priority**: 降低導入阻力，提高跨環境複用與一致性。

**Independent Test**: 於乾淨的 n8n 環境，成功匯入 workflow，僅設定必要變數（例如試算表 ID、表單分頁名、收件者清單）後即可通過上述 P1/P2 測試。

**Acceptance Scenarios**:

1. Given 標準 n8n 節點可用，When 匯入 workflow，Then 所有節點狀態為有效（未缺少必要憑證/變數），且有清楚的設定說明。
2. Given 尚未設定憑證或變數，When 啟動流程，Then 提示缺失項目與補救步驟，不進行外部 API 呼叫。

---

### Edge Cases

- Webhook 收到非 JSON 或內容無法解析：回應 4xx 與說明訊息，不寫入。
- 缺少必要欄位：回應 4xx 並列出缺少欄位，不寫入。
- 重覆事件（如 `event_id` 相同）：應去重，不重複寫入；回應為成功或已處理。
- 試算表不存在或無寫入權限：回應 5xx（或明確錯誤碼），並於流程紀錄指出權限/目標設定問題。
- API 配額/速率限制：採退避重試策略並記錄；重試上限後標註失敗，不阻塞 Webhook 即時回應。
- 郵件寄送失敗：記錄與重試（有上限）；不影響已成功的資料寫入。
- 欄位對應改動：需有可設定的欄位映射，避免直接耦合固定欄位。

## Requirements *(mandatory)*

### Functional Requirements

- FR-001：系統必須提供一份可匯入的 n8n Workflow JSON，涵蓋「Webhook → Google Sheets → Email」基本流程，且可於標準 n8n 環境成功匯入。
- FR-002：Webhook 端點必須接受 `application/json` 請求，驗證「必要欄位」並於失敗時回應 4xx 與缺失列表；成功時回應包含狀態與紀錄識別資訊。
- FR-003：系統必須支援欄位映射設定，將 Webhook 請求中的欄位對應到 Google 試算表的欄位（包含接收時間、來源 IP 等系統欄位）。
- FR-004：試算表目標（文件 ID/URL、工作表名稱）必須可設定；若目標不存在或無權限，需產生明確錯誤且不中斷其他請求。
- FR-005：系統必須提供去重機制以避免重覆寫入，去重依據可設定（如 `event_id` 或內容雜湊）。[NEEDS CLARIFICATION: 去重基準與時效窗口]
- FR-006：在成功寫入試算表後，系統必須寄出郵件通知至可設定之收件者清單；郵件主旨與內文可透過模板插入欄位值。[NEEDS CLARIFICATION: 收件者來源為固定清單或由 payload 提供?]
- FR-007：若寄信失敗，必須支援有限次重試與錯誤記錄；重試失敗後標註並不再重複通知。
- FR-008：系統必須於文件中清楚列出部署前置需求與設定項（變數/憑證），並附最小可行測試步驟（curl 測試樣例、欄位清單）。
- FR-009：流程執行紀錄必須可追蹤（例如：每次請求的處理狀態、錯誤原因、寄信結果）。
- FR-010：Webhook 回應時間應維持快速（例如 <2 秒）以利上游整合；後續動作可採非阻塞策略（不限定具體技術）。

### Key Entities *(include if feature involves data)*

- 事件（Event）：外部送入的一筆資料，包含必要欄位（例如：`event_id`、`timestamp`、`source`、`payload.*`）。
- 工作表列（Sheet Row）：映射後寫入的欄位集合，含系統欄位（接收時間、來源 IP、處理狀態）。
- 通知（Notification）：針對單筆（或多筆聚合）資料寄出的郵件，包含收件者、主旨、內文與寄送結果。

## Assumptions

- 已有可用之 n8n 執行個體與存取權限。
- 具備可寫入目標 Google 試算表之權限（服務帳戶或授權帳戶）。
- 郵件服務具備有效的寄信憑證與配額，允許通知型郵件寄送。
- Webhook 將於可信環境使用；若需公開網路曝露，將由部署方提供必要的安全機制（如 IP 白名單、金鑰/簽章）。
- 事件基本欄位具備：`event_id`（或可產生穩定雜湊）、`timestamp`、關鍵 payload 欄位（可由欄位映射設定）。
- 時區與日期格式以部署方標準為準（未指定時預設 UTC）。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- SC-001：90% 的 Webhook 請求在 2 秒內得到成功回應（資料有效且不須重試時）。
- SC-002：95% 的有效請求在 1 分鐘內完成「寫入試算表＋郵件寄出」；例外情況有清楚的失敗或延遲記錄。
- SC-003：於乾淨環境首次匯入 workflow 成功率達 95%（只需依文件設定變數/憑證）。
- SC-004：資料完整性：100% 成功寫入的列具備所有必要欄位，欄位缺漏比率為 0%。
- SC-005：重覆寫入率低於 1%（以去重規則衡量），且無用戶層面可見的重複通知。
- SC-006：使用者能在首次嘗試內完成從匯入到驗證（curl 測試＋收到郵件）的流程，無需修改流程內部結構。
