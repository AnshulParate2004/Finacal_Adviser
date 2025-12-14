# DSL Assumptions & Design Decisions

## Data Frequency
- Default timeframe: **Daily bars** (OHLCV)
- Time lags reference trading days, not calendar days
- `prev` = previous bar, `5d_ago` = 5 bars ago

## Indicator Handling
- Indicators return single values at current bar
- Insufficient data periods return `None` (strategy skips signal)
- Default parameters: SMA(20), RSI(14), MACD(12,26,9), BB(20,2)

## Cross Detection
- `crosses_above`: Previous bar below, current bar above
- `crosses_below`: Previous bar above, current bar below
- `crosses`: Either direction cross
- `touches`: Exact equality within 0.01% tolerance

## Time References
- `_prev`: 1 bar ago
- `_Nd_ago`: N days (bars) ago, where N is integer
- `_Nw_ago`: N weeks ago (N × 5 trading days)
- `_Nm_ago`: N months ago (N × 21 trading days)
- `[N]`: Alternative bracket syntax for N bars ago

## Execution Assumptions
- Entry signals evaluated at bar close
- Exit evaluated before next entry
- No position sizing specified (assumed external)
- No stop-loss/take-profit in DSL (strategy-level logic)
- Single position only (no pyramiding)

## Validation
- Case-insensitive keywords and indicators
- Whitespace ignored
- Entry block mandatory, exit optional
- Missing exit = hold until manual close
