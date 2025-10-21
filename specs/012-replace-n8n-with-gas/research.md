# Research: Replacing n8n with Google Apps Script

## 1. Securing the Google Apps Script Web App

**Decision**:

We will secure the Google Apps Script web app by requiring a secret token to be passed in the POST request body. The Apps Script will validate this token before processing any data.

**Rationale**:

- **Simplicity**: This method is straightforward to implement on both the Python (client) and Google Apps Script (server) side.
- **Effective for Server-to-Server**: Since the calls will only be coming from our trusted Python application, a shared secret is a sufficient and standard way to ensure the request is legitimate.
- **Avoids Over-complication**: Using OAuth2 would be overly complex for this internal use case, as we are not authenticating on behalf of a user but are simply verifying the identity of our own backend script.

**Implementation**:

1.  A secret token will be stored in a configuration file (e.g., `config/gas_secret.txt`) in the Python project.
2.  The Python script will read this token and include it in the JSON payload sent to the Google Apps Script.
3.  The Google Apps Script will have the same token hardcoded (or stored in its own Script Properties) and will check for its presence and correctness in the incoming request. If the token is missing or invalid, the script will return an error and stop execution.

**Alternatives Considered**:

- **OAuth2**: Deemed too complex for the current requirements.
- **No Security**: Unacceptable, as it would leave the web app endpoint exposed.

## 2. Error Handling for Python HTTP Requests

**Decision**:

The Python script will use the `requests` library and implement a retry mechanism with exponential backoff for handling transient HTTP errors.

**Rationale**:

- **Resilience**: Network requests can fail for many temporary reasons (e.g., Google's servers being momentarily busy, network hiccups). A retry mechanism ensures that the system can recover from these transient issues without manual intervention.
- **Efficiency**: Exponential backoff (waiting progressively longer between retries) is a best practice that prevents overwhelming a struggling server with rapid-fire requests.

**Implementation**:

- Use a `try...except` block to catch exceptions like `requests.exceptions.HTTPError`, `requests.exceptions.ConnectionError`, and `requests.exceptions.Timeout`.
- For transient errors (like 5xx server errors or timeouts), the script will retry the request up to 3 times, with an increasing delay between each attempt (e.g., 1s, 2s, 4s).
- For persistent errors or client-side errors (like 4xx errors, which might indicate a problem with our request), the script will log the error and not retry.

**Alternatives Considered**:

- **No Retries**: This is not robust and would lead to lost notifications on transient failures.
- **Simple Retries (No Backoff)**: Less effective and could contribute to server load during an outage.
