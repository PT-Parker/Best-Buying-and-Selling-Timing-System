## ADDED Requirements
### Requirement: Dynamic Expert Selection Reasoning
The system SHALL use an LLM-driven expert panel (bull, bear, neutral) to score market conditions and select an active expert to produce the final decision.

#### Scenario: Expert scores select active role
- **WHEN** the reasoning agent receives market data and model outputs,
- **THEN** it requests bull/bear/neutral suitability scores (0-100),
- **AND** selects the highest-scoring expert as the active role,
- **AND** returns action, confidence, reasoning, and active_role in strict JSON.

## MODIFIED Requirements
### Requirement: Fact-Subjectivity Split
The system SHALL prioritize LLM expert analysis of internal market data over external news sources, removing hard dependencies on third-party news APIs.

#### Scenario: News-free expert reasoning
- **WHEN** external news is unavailable or disabled,
- **THEN** the reasoning agent uses market data and model signals only,
- **AND** the decision flow remains functional without GNews inputs.
