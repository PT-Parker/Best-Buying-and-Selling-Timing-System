## ADDED Requirements
### Requirement: Manual Decision Card Trigger
The system SHALL render the decision card only after the user clicks the refresh button.

#### Scenario: Decision card hidden on initial load
- **WHEN** the app loads without user action,
- **THEN** the decision card section is not shown.

#### Scenario: Decision card shown after refresh
- **WHEN** the user clicks "更新行情並生成今日決策卡",
- **THEN** the decision card is rendered with the latest summary.
