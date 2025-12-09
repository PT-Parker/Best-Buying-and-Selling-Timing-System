## 1. Implementation
- [ ] 1.1 Add a CLI entrypoint to generate a daily signal report from the latest model/strategy outputs.
- [ ] 1.2 Aggregate signals with ranking, side (buy/sell), confidence/score, and include data/model version metadata.
- [ ] 1.3 Export the report to CSV in an outputs directory and optionally sync to a configured Google Sheets tab.
- [ ] 1.4 Log anomalies (missing data, stale prices) and surface them in the report metadata/QC section.
- [ ] 1.5 Add tests using small fixtures to validate report content, ranking, and export formatting.
- [ ] 1.6 Document the workflow in quickstart/docs with command examples and configuration requirements.
