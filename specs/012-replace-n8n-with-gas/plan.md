# Implementation Plan: Replace n8n with Google Apps Script

**Feature Branch**: `012-replace-n8n-with-gas`  
**Feature Spec**: [spec.md](C:\Users\Parker\Best Buying and Selling Timing System\specs\012-replace-n8n-with-gas\spec.md)  
**Created**: 2025-10-21

## 1. Technical Context

- **Objective**: Replace the n8n workflow automation with a simpler solution using Google Apps Script for sending email notifications.
- **Primary Technologies**: Python, Google Apps Script, Google Sheets.
- **Integration Pattern**: The core Python application will make an HTTP POST request to a Google Apps Script web app URL. The payload will be a JSON object containing the trading signal.
- **Unknowns**:
  - [NEEDS CLARIFICATION: What is the best practice for securing the Google Apps Script web app? (e.g., API key in header, secret in payload)]

## 2. Constitution Check

*The project constitution is currently a template and does not contain specific principles. No violations detected.*

## 3. Implementation Phases

### Phase 0: Outline & Research

- **Task 1**: Research best practices for securing a public-facing Google Apps Script web app that is intended to be triggered by a server-side script.
- **Task 2**: Research robust error handling strategies for HTTP requests from Python to Google Apps Script, including retries and logging.

**Deliverable**: `research.md` documenting the chosen security model and error handling strategy.

### Phase 1: Design & Contracts

- **Task 1: Data Model**: Create `data-model.md` to formally define the `TradingSignal` and `Configuration` entities as specified in the feature spec.
- **Task 2: API Contract**: Create `contracts/gas_webhook.openapi.yaml` to define the HTTP POST request to the Google Apps Script, including the expected JSON payload and response.
- **Task 3: Quickstart Guide**: Create `quickstart.md` with instructions on how to set up the Google Apps Script, deploy it as a web app, and configure the URL in the Python application.
- **Task 4: Agent Context Update**: Run the script to add "Google Apps Script" to the agent's technical context.

**Deliverables**:
- `data-model.md`
- `contracts/gas_webhook.openapi.yaml`
- `quickstart.md`
- Updated agent context file.

### Phase 2: Implementation & Testing

- **Task 1: Remove n8n**: Delete the entire `/n8n` directory and any related scripts (e.g., `n8n_bootstrap.ps1`, `n8n_run_email_only_test.ps1`).
- **Task 2: Update Python App**: Modify the core Python application to:
  - Read the Google Apps Script URL from a configuration file.
  - Construct the JSON payload for the trading signal.
  - Make the HTTP POST request to the Google Apps Script URL.
  - Implement the error handling strategy defined in `research.md`.
- **Task 3: Unit Tests**: Write unit tests for the new HTTP request logic in the Python application.
- **Task 4: Integration Test**: Create an integration test that runs the core application and verifies that the Google Apps Script is called correctly (e.g., by checking for a logged success message).

**Deliverables**:
- Updated Python application code.
- New unit and integration tests.
- All n8n-related files removed.