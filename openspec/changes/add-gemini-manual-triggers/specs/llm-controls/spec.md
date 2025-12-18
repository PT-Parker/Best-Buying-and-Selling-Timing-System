## ADDED Requirements
### Requirement: Manual Multi-Agent Trigger
The system SHALL run multi-agent Gemini reasoning only when the user clicks an explicit run button.

#### Scenario: Manual run
- **WHEN** the user enables multi-agent mode and clicks the run button,
- **THEN** the system calls Gemini and displays the result,
- **AND** no Gemini calls occur without a button click.

### Requirement: Manual Explanation Trigger
The system SHALL generate Gemini explanations for decision cards and forecasts only when the user clicks a generate button.

#### Scenario: Manual explanation
- **WHEN** the user clicks "生成 Gemini 解讀",
- **THEN** the system calls Gemini and shows the explanation,
- **AND** subsequent reruns reuse the cached explanation.
