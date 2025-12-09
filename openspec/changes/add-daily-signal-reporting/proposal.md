# Change: Daily Signal Reporting

## Why
Analysts want a repeatable way to publish the day's buy/sell signals with context so stakeholders can act quickly without digging through raw backtest outputs.

## What Changes
- Add a CLI workflow to assemble the latest model/strategy signals into a ranked daily report.
- Export the report to local CSV and optionally push to a configured Google Sheets tab for sharing.
- Attach metadata (data cutoff, model/model parameters, anomalies) to every run for auditability.

## Impact
- Affected specs: signal-reporting
- Affected code: signal generation loaders, report/export utilities, CLI entrypoints
