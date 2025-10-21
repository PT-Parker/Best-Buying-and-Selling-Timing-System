import os
import sys

# Add the root directory to the Python path to allow imports from the scripts folder
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from scripts.realtime_monitor import notify_gas

def main():
    print("Sending a test email via Google Apps Script...")
    
    test_payload = {
        "time": "2025-10-21 14:00:00",
        "symbol": "TEST",
        "action": "BUY",
        "price": 123.45,
        "reason": "Direct test of email functionality",
    }
    
    notify_gas(test_payload)
    
    print("Test email trigger has been sent. Please check your inbox.")
    print("Note: This only confirms the request was sent. Check your Google Apps Script execution logs for any errors on the script side.")

if __name__ == "__main__":
    main()
