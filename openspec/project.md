# Project Context

## Purpose
Build the "Best Buying and Selling Timing System" for generating, evaluating, and communicating trading signals. The project centers on quantitative research workflows: ingesting market data, engineering features, backtesting strategies, training ML models, and producing reports that help decide when to buy or sell.

## Tech Stack
- Python 3.10 with pandas, numpy, pyarrow, matplotlib, scikit-learn, xgboost, yfinance, twstock, PyYAML
- CLI/automation scripts and notebooks under `src/` plus supporting scripts in `scripts/`
- Local artifacts (CSV, RDA-converted data, trained models) as primary data sources
- Presentation and docs tooling: PowerPoint or Google Slides for decks; Markdown/PDF for docs; assets stored under `deliverables/`

## Project Conventions

### Code Style
- Standard Python 3.10 conventions; prefer clear, typed functions and small modules
- Lint with `ruff check .` (run from `src/`); keep imports sorted and avoid unused code
- Configuration via YAML/CSV files when possible to keep strategies reproducible

### Architecture Patterns
- Specs-first via OpenSpec: author or update specs/changes before implementation
- CLI-first workflows for data prep, backtesting, and model inference; outputs written to `backtest_out/`, `metrics/`, and similar artifact folders
- Reuse shared data loading/feature engineering helpers; keep strategies pluggable so parameter sweeps and model variants stay isolated

### Testing Strategy
- Use `pytest` for unit and integration coverage; run from `src/` with `cd src; pytest`
- Validate linting via `ruff check .`
- Prefer deterministic fixtures for market data slices; persist small golden CSVs for repeatable backtest assertions

### Git Workflow
- Use feature branches named after the OpenSpec change ID (kebab-case, verb-led)
- Land code only after the corresponding proposal is approved; keep commits scoped to the tasks checklist
- Avoid history rewrites on shared branches; keep main stable and aligned with validated specs

## Domain Context
- Focus on timing for equity trades (TW/US tickers) using Yahoo Finance (`yfinance`) and Taiwan Stock Exchange data (`twstock`)
- Workflows include parameterized backtests, signal frequency vs. performance analysis, and ML models (e.g., XGBoost) for signal quality
- Outputs include CSV summaries, model artifacts, and presentation material for stakeholders

## Important Constraints
- Maintain compatibility with Python 3.10 runtime and existing dependencies
- Minimize new external dependencies; prefer offline/local data artifacts when possible
- Treat specs as the source of truth—no implementation without an approved proposal
- Document breaking changes explicitly and gate them behind proposals

## External Dependencies
- Market data: Yahoo Finance (`yfinance`), Taiwan Stock Exchange data (`twstock`)
- ML/analytics libraries: pandas, numpy, pyarrow, scikit-learn, xgboost, matplotlib
- Config and IO: PyYAML, CSV/Parquet files stored locally
- Presentation tooling: PowerPoint or Google Slides for decks; Markdown/PDF for docs
