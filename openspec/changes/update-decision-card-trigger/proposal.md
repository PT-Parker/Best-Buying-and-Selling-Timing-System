# Change: Decision Card Only After Manual Refresh

## Why
The decision card should appear only after the user explicitly clicks the refresh button to avoid confusion on initial load.

## What Changes
- Hide the decision card section until the user clicks "更新行情並生成今日決策卡".
- Keep existing functionality unchanged once the button is clicked.

## Impact
- Affected specs: ui
- Affected code: app.py
