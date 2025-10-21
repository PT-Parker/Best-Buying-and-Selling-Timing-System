# Feature Specification: Recreate n8n Workflow

**Feature Branch**: `011-recreate-n8n-workflow`  
**Created**: 2025-10-21  
**Status**: Draft  
**Input**: User description: "重新製作TW Live Signals ? Sheets + Email的n8n工作流程"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Reliable Signal Processing (Priority: P1)

As a user, I want a robust n8n workflow that reliably receives trading signals via a webhook, logs them to a Google Sheet, and sends an email notification, so that I can automate my trading alerts.

**Why this priority**: This is the core functionality of the feature. The entire purpose is to have a reliable, automated workflow.

**Independent Test**: A single webhook trigger should result in a new row in the Google Sheet and a received email.

**Acceptance Scenarios**:

1. **Given** the workflow is active, **When** a valid JSON payload is sent to the webhook, **Then** a new row with the correct data is created in the specified Google Sheet.
2. **Given** a new row is added to the Google Sheet, **When** the workflow continues, **Then** an email with the trading signal is sent to the specified recipient.

---

### Edge Cases

- What happens when the incoming JSON payload is malformed or missing fields?
- How does the system handle failures in the Google Sheets or Email nodes (e.g., API errors, invalid credentials)?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The workflow MUST be triggered by a POST request to a unique webhook URL.
- **FR-002**: The workflow MUST correctly parse the incoming JSON data, including `time`, `symbol`, `action`, `price`, and `reason`.
- **FR-003**: The workflow MUST append the parsed data as a new row to a Google Sheet.
- **FR-004**: The workflow MUST send an email containing the details of the trading signal.
- **FR-005**: The Google Sheet ID, sheet name, and email addresses (from/to) MUST be easily configurable.
- **FR-006**: The workflow MUST use existing, configured credentials for Google Sheets and SMTP.

### Key Entities *(include if feature involves data)*

- **TradingSignal**: Represents the incoming data with attributes: `time`, `symbol`, `action`, `price`, `reason`.
- **WorkflowExecution**: Represents a single run of the n8n workflow, with a status (success/failure) and associated logs.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of valid webhook triggers result in a new row in the Google Sheet.
- **SC-002**: 100% of successful Google Sheet appends result in an email notification being sent.
- **SC-003**: The end-to-end workflow execution time, from webhook trigger to email sent, is less than 10 seconds.
- **SC-004**: The workflow successfully executes with no errors when provided with valid credentials and a correctly formatted Google Sheet.