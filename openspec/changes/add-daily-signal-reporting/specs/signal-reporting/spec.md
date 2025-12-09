## ADDED Requirements
### Requirement: Generate Daily Signal Report
The system SHALL produce a daily report of buy/sell candidates using the latest model or strategy outputs for configured tickers.

#### Scenario: CLI report generation succeeds
- **WHEN** the operator runs the daily signal report command with a target date (defaulting to today),
- **THEN** the system loads the latest signals for all configured tickers,
- **AND** outputs a ranked list with ticker, side, entry price, score/confidence, and rationale notes.

### Requirement: Export Daily Report
The system SHALL export the daily signal report to local CSV and optionally sync to a Google Sheets tab when credentials are configured.

#### Scenario: Local export only
- **WHEN** the operator runs the report without Google Sheets configuration,
- **THEN** a CSV is written to the outputs directory with all report rows and metadata.

#### Scenario: Google Sheets sync
- **WHEN** Google Sheets credentials and sheet/tab identifiers are provided,
- **THEN** the sheet is updated with the latest rows and timestamp while keeping a backup CSV.

### Requirement: Attach Metadata and QC Notes
The system SHALL attach generation metadata (data cutoff, model version or parameter set, run timestamp) and QC notes about anomalies to every report.

#### Scenario: Anomaly logging
- **WHEN** any ticker lacks fresh price data or a model score,
- **THEN** the report includes a QC section listing the issue per ticker,
- **AND** anomalies are logged without blocking the rest of the report.
