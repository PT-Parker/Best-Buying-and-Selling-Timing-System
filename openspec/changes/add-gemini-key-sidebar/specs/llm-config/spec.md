## ADDED Requirements
### Requirement: Sidebar API Key Entry
The system SHALL provide a sidebar password input for GEMINI_API_KEY and an activation button to store it in session state.

#### Scenario: User activates Gemini key
- **WHEN** the user enters a key and presses activate,
- **THEN** the key is stored only in session state,
- **AND** the LLM status shows active for the current session.

### Requirement: Session Key Preference
The system SHALL prefer the session GEMINI_API_KEY over environment variables when creating the Gemini client.

#### Scenario: Session key overrides environment
- **WHEN** a session key is present,
- **THEN** Gemini client initialization uses the session key even if environment variables differ.
