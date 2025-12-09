<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

﻿# Best Buying and Selling Timing System Development Guidelines

Auto-generated from all feature plans. Last updated: 2025-10-22

## Active Technologies
- Python 3.10 (existing runtime from `.pyc` artifacts) + pandas, numpy, yfinance, twstock, PyYAML, matplotlib (015-batch-backtest-parameters)
- Python 3.10（現有環境） + pandas、numpy、pyarrow、xgboost、scikit-learn、yfinance、twstock、PyYAML (016-xgboost-usersparkerbest-buying)
- 本地檔案（CSV、RDA 轉換後資料、模型檔案） (016-xgboost-usersparkerbest-buying)
- 簡報（PowerPoint 2019+ 或 Google Slides）、文件（Markdown/PDF）、文本腳本（Markdown） + 無程式庫；可選用 PowerPoint、Google Slides、Figma 等設計工具 (017-20ppt-or-pdr)
- `deliverables/` 下的 `presentations/`、`docs/`、`multimedia/` (017-20ppt-or-pdr)

## Project Structure
```
src/
tests/
```

## Commands
cd src; pytest; ruff check .

## Code Style
Python 3.10 (existing runtime from `.pyc` artifacts): Follow standard conventions

## Recent Changes
- 017-20ppt-or-pdr: Added 簡報（PowerPoint 2019+ 或 Google Slides）、文件（Markdown/PDF）、文本腳本（Markdown） + 無程式庫；可選用 PowerPoint、Google Slides、Figma 等設計工具
- 016-xgboost-usersparkerbest-buying: Added Python 3.10（現有環境） + pandas、numpy、pyarrow、xgboost、scikit-learn、yfinance、twstock、PyYAML
- 015-batch-backtest-parameters: Added Python 3.10 (existing runtime from `.pyc` artifacts) + pandas, numpy, yfinance, twstock, PyYAML, matplotlib

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
