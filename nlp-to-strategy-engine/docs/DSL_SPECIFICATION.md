# DSL Specification

## Grammar

```
strategy    ::= ENTRY ":" rule_block [EXIT ":" rule_block]
rule_block  ::= or_rule
or_rule     ::= and_rule | or_rule OR and_rule
and_rule    ::= comparison | and_rule AND comparison
comparison  ::= expr operator expr
expr        ::= series | indicator | time_ref | number | "(" or_rule ")"
```

## Operators

**Comparison:** `>` `<` `>=` `<=` `==` `!=`  
**Cross:** `crosses_above` `crosses_below` `crosses` `touches`  
**Boolean:** `AND` `OR`

## Data Types

**Series:** `close` `open` `high` `low` `volume`  
**Numbers:** `100` `1.5` `1K` `2.5M` `1B` (K=thousand, M=million, B=billion)  
**Time Refs:** `close_prev` `high_5d_ago` `low_1w_ago` `volume[3]`

## Indicators

| Indicator | Syntax | Parameters |
|-----------|--------|------------|
| SMA | `sma(series, period)` | series, period |
| EMA | `ema(series, period)` | series, period |
| RSI | `rsi(series, period)` | series, period |
| MACD | `macd(series, fast, slow, signal)` | series, 12, 26, 9 |
| Bollinger | `bb_upper/middle/lower(series, period, stddev)` | series, 20, 2 |
| ATR | `atr(period)` | period |
| ADX | `adx(period)` | period |
| Stochastic | `stoch(period, k, d, smooth)` | 14, 3, 3, 3 |

## Validation Rules

- All series/indicator names case-insensitive
- Entry block mandatory, exit optional
- Indicators validated for correct parameter count
- Time lags: prev, Nd_ago, Nw_ago, Nm_ago (N=number)
