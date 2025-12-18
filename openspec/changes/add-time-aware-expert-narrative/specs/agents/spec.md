## ADDED Requirements
### Requirement: Time-Aware Market Narrative
The system SHALL include a short narrative summary of recent price and volume behavior in expert scoring and decision prompts.

#### Scenario: Narrative provided to experts
- **WHEN** the orchestrator prepares market data for expert reasoning,
- **THEN** it constructs a recent time-window summary of price and volume behavior,
- **AND** passes the narrative alongside numeric indicators to the reasoning agent.

### Requirement: Confidence Calibration Guidance
The system SHALL instruct the expert decision prompt to scale confidence based on the degree of signal alignment or conflict.

#### Scenario: Confidence decreases on conflicting signals
- **WHEN** technical signals conflict or are mixed,
- **THEN** the prompt guidance directs confidence below 0.6 unless signals strongly agree.
