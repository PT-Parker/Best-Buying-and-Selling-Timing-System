## ADDED Requirements
### Requirement: Decision Card Explanation
The system SHALL use Gemini to generate a 2-3 line Chinese explanation for the daily decision card and render it under an expander labeled "Gemini 解讀".

#### Scenario: Decision card explanation displayed
- **WHEN** a decision card is generated for a symbol,
- **THEN** the UI shows a Gemini explanation in the expander,
- **AND** the explanation references the model score and expected return.

### Requirement: Forecast Explanation
The system SHALL use Gemini to generate a 2-3 line Chinese explanation for the price forecast results and render it under an expander labeled "Gemini 解讀".

#### Scenario: Forecast explanation displayed
- **WHEN** a price forecast is generated,
- **THEN** the UI shows a Gemini explanation in the expander,
- **AND** the explanation references the forecast horizon and projected price.

### Requirement: Persist LLM Explanations
The system SHALL store Gemini explanations in MemoryDB with context metadata for later review.

#### Scenario: Explanation stored
- **WHEN** Gemini returns an explanation for a decision card or forecast,
- **THEN** the system stores it with symbol, timestamp, context, and input payload.
