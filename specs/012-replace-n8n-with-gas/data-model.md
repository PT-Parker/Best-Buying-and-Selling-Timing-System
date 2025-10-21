# Data Model: Google Apps Script Email Notification

## 1. TradingSignal

Represents the trading signal data that is passed from the Python application to the Google Apps Script.

- **`time`** (string, mandatory): The timestamp of the signal (e.g., "2025-10-21 11:00:00").
- **`symbol`** (string, mandatory): The stock symbol (e.g., "2330").
- **`action`** (string, mandatory): The trading action, either "BUY" or "SELL".
- **`price`** (number, mandatory): The price at which the signal was generated.
- **`reason`** (string, optional): A brief description of the reason for the signal.
- **`secret`** (string, mandatory): A secret token for authenticating the request.

## 2. Configuration

Represents the configuration settings required by the Python application.

- **`gas_webhook_url`** (string, mandatory): The URL of the deployed Google Apps Script web app.
- **`email_recipient`** (string, mandatory): The email address to send the notification to.
- **`secret_token`** (string, mandatory): The secret token to be included in the payload to the Google Apps Script.
