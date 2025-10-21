# Quickstart: Google Apps Script Email Notification

This guide explains how to set up the Google Apps Script to send email notifications for trading signals.

## 1. Create the Google Apps Script

1.  Go to [script.google.com](https://script.google.com) and create a new project.
2.  Name the project (e.g., "Trading Signal Emailer").
3.  Paste the following code into the `Code.gs` file:

```javascript
function doPost(e) {
  // --- Configuration ---
  const SECRET_TOKEN = "your-secret-token"; // Replace with your own secret token
  const RECIPIENT_EMAIL = "your-email@example.com"; // Replace with the recipient's email

  try {
    const params = JSON.parse(e.postData.contents);

    // 1. Validate Secret Token
    if (params.secret !== SECRET_TOKEN) {
      return ContentService.createTextOutput(JSON.stringify({ status: "error", message: "Invalid secret token" })).setMimeType(ContentService.MimeType.JSON);
    }

    // 2. Prepare Email
    const subject = `【${params.symbol}】 ${params.action} @ ${params.price}`;
    const body = `
      Time: ${params.time}
      Symbol: ${params.symbol}
      Action: ${params.action}
      Price: ${params.price}
      Reason: ${params.reason || 'N/A'}
    `;

    // 3. Send Email
    MailApp.sendEmail(RECIPIENT_EMAIL, subject, body);

    // 4. Return Success Response
    return ContentService.createTextOutput(JSON.stringify({ status: "success" })).setMimeType(ContentService.MimeType.JSON);

  } catch (error) {
    // 5. Error Handling
    return ContentService.createTextOutput(JSON.stringify({ status: "error", message: error.toString() })).setMimeType(ContentService.MimeType.JSON);
  }
}
```

4.  **Important**: Replace `your-secret-token` and `your-email@example.com` with your actual secret token and recipient email address.

## 2. Deploy as a Web App

1.  Click on **Deploy** > **New deployment**.
2.  For **Select type**, choose **Web app**.
3.  In the **Deployment configuration**:
    -   **Description**: (Optional) e.g., "Trading Signal Webhook"
    -   **Execute as**: Me (your Google account)
    -   **Who has access**: Anyone
4.  Click **Deploy**.
5.  **Important**: Authorize the script to send emails on your behalf.
6.  Copy the **Web app URL**. This is your `gas_webhook_url`.

## 3. Configure the Python Application

1.  In the Python project, create or update your configuration file (e.g., `config.yaml` or `.env`) with the following:
    -   `gas_webhook_url`: The URL you copied from the Google Apps Script deployment.
    -   `secret_token`: The same secret token you set in the Google Apps Script.
2.  Ensure the Python script that generates the trading signal reads these configuration values and uses them to make the HTTP POST request.
