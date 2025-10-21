# Data Model: Add Volatile Stock Symbols

## 1. StockSymbol

Represents a unique identifier for a stock.

-   **`symbol`** (string, mandatory): The stock ticker symbol (e.g., "2330", "77868").

## 2. Watchlist

A collection of `StockSymbol` entities that the system monitors.

-   **`symbols`** (array of StockSymbol, mandatory): A list of stock ticker symbols.
