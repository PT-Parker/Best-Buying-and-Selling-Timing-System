## ADDED Requirements
### Requirement: Gemini Expert Reasoning
The system SHALL use Gemini 2.5 Flash-Lite to score bull/bear/neutral experts and return a final decision based on market data only.

#### Scenario: Expert scoring with JSON output
- **WHEN** ReasoningAgent requests expert scoring with market data,
- **THEN** Gemini returns JSON with bull_score, bear_score, neutral_score.

### Requirement: Expert Decision JSON
The system SHALL prompt Gemini to return JSON with signal, confidence, reasoning, and active_role.

#### Scenario: Expert decision output
- **WHEN** the highest-scoring expert is selected,
- **THEN** Gemini returns a JSON decision with signal/confidence/reasoning/active_role.

## MODIFIED Requirements
### Requirement: Fact-Subjectivity Split
The system SHALL operate without external news APIs, using price action and technical indicators only.

#### Scenario: No external news dependency
- **WHEN** external news APIs are unavailable,
- **THEN** the system still produces a decision using technical market data only.
