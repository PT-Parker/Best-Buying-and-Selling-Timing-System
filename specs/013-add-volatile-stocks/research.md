# Research: Adding Volatile Stock Symbols

## 1. Volatility Metrics

**Decision**:

For identifying "larger price fluctuations" in individual stocks, the **Standard Deviation of daily returns** is the most direct and commonly used metric. It quantifies the degree to which a stock's price or returns deviate from its average over a period. Higher standard deviation indicates higher volatility.

**Rationale**:

-   **Directness**: Directly measures price movement magnitude.
-   **Commonality**: Widely accepted and understood in financial analysis.
-   **Data Availability**: Can be calculated from historical daily price data, which is readily available.

**Alternatives Considered**:

-   **Beta**: Measures volatility relative to a market benchmark. While useful, it's less direct for identifying absolute price fluctuations of a single stock.
-   **Maximum Drawdown (MDD)**: Focuses on downside risk, not overall fluctuation.
-   **VIX Index / Implied Volatility**: These are market-wide or options-derived metrics, not directly applicable to individual stock selection for this purpose.

## 2. Proposed Volatile Stock Symbols

**Methodology**:

Given the user's request for a quick addition of volatile stocks and the context of a simple monitoring system, a pragmatic approach is to manually select well-known volatile stocks from the Taiwan stock market based on recent market data and common knowledge. This avoids the complexity of programmatic volatility calculation for this initial feature.

**Proposed Symbols (5 stocks)**:

Based on a web search for high-volatility stocks in the Taiwan market, the following symbols are proposed:

-   `77868` (Inforcom Technology Inc)
-   `77798` (Fabulous Global Holding Co Ltd)
-   `77866` (丹立)
-   `6403` (群登)
-   `3595` (山太士)

These symbols are identified as having higher daily price fluctuations according to recent market data.
