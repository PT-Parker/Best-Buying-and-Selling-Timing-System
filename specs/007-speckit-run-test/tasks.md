---
description: "Tasks for feature: Test n8n Webhook"
---

# Tasks: Test n8n Webhook

**Input**: Design documents from `specs/007-speckit-run-test/`
**Prerequisites**: spec.md present; plan.md not available (derive from repo context: Python CLI + PowerShell scripts + n8n)

**Tests**: Not explicitly requested; omit test files. Validate via manual commands and script outputs.

**Organization**: Tasks grouped by user story to enable independent implementation/testing.

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Ensure environment and baseline utilities for webhook testing.

- [ ] T001 Ensure logs directory exists in `logs/`
- [ ] T002 [P] Add `.gitignore` rule for `logs/n8n_webhook_tests.log` if missing
- [ ] T003 Document env vars in `README.md` (append n8n section)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Provide a cross-platform webhook test utility; fix existing task runner YAML so stories can run.

- [ ] T004 Create webhook tester `scripts/n8n_test_webhook.py` (CLI: `--url`, `--data`, `--data-file`, `--timeout`, `--mask-url`)
- [ ] T005 [P] Write sanitized log on each run to `logs/n8n_webhook_tests.log`
- [ ] T006 [P] Add Speckit task `test_n8n_webhook_py` in `.specify/memory/tasks-cli.yaml` to call the Python script
- [ ] T007 Fix YAML formatting issues in `.specify/memory/tasks-cli.yaml` here-doc blocks (ensure valid indentation and closing markers)
- [ ] T008 Add Speckit task `n8n_check_endpoints` dependency note under README (how to verify UI/API/webhook)

**Checkpoint**: Foundation ready — stories can be run independently via Python script and Speckit.

---

## Phase 3: User Story 1 - 以預設負載測試單一 Webhook (Priority: P1) 🎯 MVP

**Goal**: 向指定 Webhook URL 送出安全預設負載並回報狀態碼與延遲。

**Independent Test**: 執行 `python scripts/n8n_test_webhook.py --url "$N8N_WEBHOOK_URL"`；確認終端輸出成功/失敗、狀態碼、延遲，並寫入一筆日誌。

### Implementation for User Story 1

- [ ] T009 [US1] Implement default payload builder in `scripts/n8n_test_webhook.py`
- [ ] T010 [US1] Measure latency and print summary (status, ms) in `scripts/n8n_test_webhook.py`
- [ ] T011 [US1] Update Speckit `test_n8n_webhook` to use Python utility (fallback to existing PowerShell on Windows)
- [ ] T012 [US1] Mask URL in console/log output (domain + path prefix only)

**Checkpoint**: US1 independently verifiable via單一命令與日誌。

---

## Phase 4: User Story 2 - 使用自訂 JSON 負載測試 (Priority: P2)

**Goal**: 使用者可提供自訂 JSON；送出前做格式驗證與差異預覽。

**Independent Test**: 執行 `python scripts/n8n_test_webhook.py --url "$N8N_WEBHOOK_URL" --data '{"k":"v"}'`；看到有效性檢查與發送結果。

### Implementation for User Story 2

- [ ] T013 [US2] Add `--data`/`--data-file` parsing + JSON validation in `scripts/n8n_test_webhook.py`
- [ ] T014 [US2] Show diff vs default payload (keys added/overridden) in `scripts/n8n_test_webhook.py`
- [ ] T015 [P] [US2] Extend Speckit task to accept `DATA_JSON` or `DATA_FILE` env for custom payload

**Checkpoint**: US2可單獨驗證（不依賴 US1 以外模組）。

---

## Phase 5: User Story 3 - 產生可稽核的測試紀錄 (Priority: P3)

**Goal**: 為每次測試產生脫敏紀錄（時間、URL 掩碼、負載大小、結果、延遲）。

**Independent Test**: 檢查 `logs/n8n_webhook_tests.log` 最新一筆是否包含欄位且無敏感資訊。

### Implementation for User Story 3

- [ ] T016 [US3] Append structured JSON line to `logs/n8n_webhook_tests.log`（含 timestamp/url_mask/size/status/ms/success）
- [ ] T017 [US3] Redact secrets in URL/query/body before寫入日誌
- [ ] T018 [P] [US3] Add Speckit task `cat_last_webhook_test` 顯示最後一筆記錄（Windows/Posix）

**Checkpoint**: US3 實作完成後，審核可僅依日誌驗證測試歷史。

---

## Phase N: Polish & Cross-Cutting Concerns

- [ ] T019 [P] README: 新增「Webhook 測試快速指南」段落與風險提示
- [ ] T020 Harden error handling: timeouts/retries flags in `scripts/n8n_test_webhook.py`
- [ ] T021 Optional: `--headers key=value` 支援覆寫測試標頭於 `scripts/n8n_test_webhook.py`

---

## Dependencies & Execution Order

### Phase Dependencies

- Setup → Foundational → US1 → US2 → US3 → Polish（US2/US3 可在 Foundation 完成後平行，惟依 US1 稍有共用程式碼）

### User Story Dependencies

- US1 (P1): 基於 Foundational 工具，可獨立驗證
- US2 (P2): 依賴 US1 的工具介面，但測試不需 US3
- US3 (P3): 依賴 Foundation 與 US1 的送測介面

### Parallel Opportunities

- T001–T003 可平行
- T004–T006 可與 T007 平行（不同檔案）
- US2 的 T015 可在 US2 期中與 T013/T014 平行
- US3 的 T018 可與 T016/T017 平行

---

## Implementation Strategy

### MVP First（User Story 1）

1) 完成 Phase 1–2（建立測試腳本 + Speckit 任務）
2) 完成 US1：預設負載測試 + 延遲與狀態輸出 + 基本遮罩
3) 驗證：執行單一命令並檢視日誌

### Incremental Delivery

- 加入 US2 的自訂負載（強化可用性）
- 加入 US3 的稽核紀錄（提升可追溯性）

