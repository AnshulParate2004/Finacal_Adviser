# Trading Strategy DSL (Domain Specific Language) Specification

## Overview
This DSL enables traders to express entry/exit rules, technical indicators, boolean logic, and time-based conditions in a structured, unambiguous format.

---

## Grammar

### BNF Notation

```
<strategy>       ::= <entry_block> [<exit_block>]
<entry_block>    ::= "ENTRY:" <rule_list>
<exit_block>     ::= "EXIT:" <rule_list>
<rule_list>      ::= <rule> | <rule> "AND" <rule_list> | <rule> "OR" <rule_list> | "(" <rule_list> ")"
<rule>           ::= <condition>
<condition>      ::= <expression> <comparison_op> <expression>
<expression>     ::= <series> | <indicator> | <number> | <time_reference> | "(" <expression> ")"
<series>         ::= "close" | "open" | "high" | "low" | "volume"
<indicator>      ::= INDICATOR_NAME "(" <indicator_params> ")"
<indicator_params> ::= SERIES_NAME "," NUMBER | SERIES_NAME "," NUMBER "," NUMBER | SERIES_NAME "," NUMBER "," NUMBER "," NUMBER
<comparison_op>  ::= ">" | "<" | ">=" | "<=" | "==" | "!=" | "crosses_above" | "crosses_below" | "crosses" | "touches"
<time_reference> ::= SERIES_NAME "_" TIME_LAG | SERIES_NAME "[" NUMBER "]"
<TIME_LAG>       ::= "prev" | "1d_ago" | "5d_ago" | "1w_ago" | "1m_ago"
<number>         ::= INTEGER | FLOAT | FLOAT "K" | FLOAT "M" | FLOAT "B"
```

---

## Supported Components

### 1. Data Series
| Series | Description |
|--------|-------------|
| `close` | Closing price |
| `open` | Opening price |
| `high` | High price |
| `low` | Low price |
| `volume` | Trading volume |

### 2. Indicators

#### SMA (Simple Moving Average)
```
sma(close, 20)      → 20-period SMA of close
sma(high, 50)       → 50-period SMA of high
```

#### EMA (Exponential Moving Average)
```
ema(close, 12)      → 12-period EMA of close
ema(close, 26)      → 26-period EMA of close
```

#### RSI (Relative Strength Index)
```
rsi(close, 14)      → 14-period RSI of close
rsi(close, 21)      → 21-period RSI of close
```

#### MACD (Moving Average Convergence Divergence)
```
macd(close, 12, 26, 9)         → MACD line
macd_signal(close, 12, 26, 9)  → MACD signal line
macd_histogram(close, 12, 26, 9) → MACD histogram
```

#### Bollinger Bands
```
bb_upper(close, 20, 2)    → Upper band
bb_middle(close, 20, 2)   → Middle band (SMA)
bb_lower(close, 20, 2)    → Lower band
```

#### ATR (Average True Range)
```
atr(14)             → 14-period ATR
atr(21)             → 21-period ATR
```

#### ADX (Average Directional Index)
```
adx(14)             → 14-period ADX
adx(21)             → 21-period ADX
```

#### Stochastic
```
stoch(close, 14, 3, 3)    → Stochastic %K
stoch_d(close, 14, 3, 3)  → Stochastic %D
```

### 3. Comparison Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `>` | Greater than | `close > 100` |
| `<` | Less than | `rsi(close,14) < 30` |
| `>=` | Greater or equal | `close >= sma(close,20)` |
| `<=` | Less or equal | `volume <= 1000000` |
| `==` | Equal | `close == open` |
| `!=` | Not equal | `close != open` |
| `crosses_above` | Crosses above | `close crosses_above sma(close,20)` |
| `crosses_below` | Crosses below | `close crosses_below sma(close,50)` |
| `crosses` | Crosses (either dir) | `macd crosses 0` |
| `touches` | Touches (±tolerance) | `close touches bb_upper(close,20,2)` |

### 4. Time References

Refer to previous/past values:

```
close_prev          → Close from previous bar
high_1d_ago         → High from 1 day ago
low_5d_ago          → Low from 5 days ago
close_1w_ago        → Close from 1 week ago
close[1]            → Close 1 bar ago
close[5]            → Close 5 bars ago
close[20]           → Close 20 bars ago
```

### 5. Number Formats

| Format | Value | Example |
|--------|-------|---------|
| Integer | Direct | `100` |
| Float | Decimal | `1.5` |
| Thousands | ×1,000 | `1.5K` = 1,500 |
| Millions | ×1,000,000 | `1.5M` = 1,500,000 |
| Billions | ×1,000,000,000 | `1.5B` = 1,500,000,000 |

### 6. Boolean Operators

| Operator | Precedence | Example |
|----------|-----------|---------|
| `AND` | Higher | `close > 100 AND volume > 1M` |
| `OR` | Lower | `close > 100 OR rsi < 30` |
| `(...)` | Highest | `(close > 100 AND volume > 1M) OR rsi < 30` |

---

## DSL Examples

### Example 1: Simple Moving Average Crossover
```
ENTRY:
  close crosses_above sma(close, 20)

EXIT:
  close crosses_below sma(close, 20)
```

### Example 2: RSI + Volume Confirmation
```
ENTRY:
  RULE1: close > sma(close, 20)
  RULE2: rsi(close, 14) > 50
  RULE3: volume > 1M

EXIT:
  rsi(close, 14) < 30
```

### Example 3: Complex Logic with Parentheses
```
ENTRY:
  (close > sma(close, 20) AND volume > 1M) 
  OR (rsi(close, 14) > 70 AND close > close_1d_ago)

EXIT:
  rsi(close, 14) < 30 OR close < bb_lower(close, 20, 2)
```

### Example 4: Time-Based Conditions
```
ENTRY:
  close > high_prev AND close > high_1d_ago AND volume > volume_prev * 1.3

EXIT:
  close < close_5d_ago OR rsi(close, 14) < 20
```

### Example 5: Bollinger Bands Strategy
```
ENTRY:
  TOUCH_UPPER: close touches bb_upper(close, 20, 2)
  STRONG_MOMENTUM: rsi(close, 14) > 60

EXIT:
  close < bb_lower(close, 20, 2)
```

### Example 6: MACD Crossover
```
ENTRY:
  macd(close, 12, 26, 9) crosses_above macd_signal(close, 12, 26, 9)

EXIT:
  macd(close, 12, 26, 9) crosses_below macd_signal(close, 12, 26, 9)
```

---

## AST (Abstract Syntax Tree) Format

### Example Input
```
ENTRY: close > sma(close, 20) AND volume > 1M
EXIT: rsi(close, 14) < 30
```

### Corresponding AST
```json
{
  "type": "strategy",
  "entry": {
    "type": "boolean_op",
    "operator": "AND",
    "left": {
      "type": "comparison",
      "operator": ">",
      "left": {"type": "series", "name": "close"},
      "right": {"type": "indicator", "name": "sma", "params": ["close", 20]}
    },
    "right": {
      "type": "comparison",
      "operator": ">",
      "left": {"type": "series", "name": "volume"},
      "right": {"type": "number", "value": 1000000}
    }
  },
  "exit": {
    "type": "comparison",
    "operator": "<",
    "left": {"type": "indicator", "name": "rsi", "params": ["close", 14]},
    "right": {"type": "number", "value": 30}
  }
}
```

---

## Validation Rules

1. **Indicator Parameters**
   - `sma(series, period)`: period must be positive integer
   - `rsi(series, period)`: period must be positive integer
   - `macd(series, fast, slow, signal)`: fast < slow
   - `bb(series, period, std)`: period and std must be positive

2. **Series Validation**
   - Only recognized series names allowed: close, open, high, low, volume

3. **Comparison Operators**
   - `crosses_above`, `crosses_below` require comparable types
   - `==`, `!=` can compare any types

4. **Time References**
   - Must reference valid series or indicators
   - `_prev` suffix valid for any series
   - Valid lag periods: 1d, 5d, 1w, 1m

5. **Boolean Logic**
   - `AND` has higher precedence than `OR`
   - Parentheses override precedence
   - At least one rule required in ENTRY

---

## Design Assumptions

1. **Closing prices as default**: When no series specified in indicator, assume `close`
2. **AND precedence higher than OR**: Standard boolean precedence
3. **Single-bar lookback is default**: `close_prev` = previous bar (1-bar lookback)
4. **No arithmetic expressions**: Only direct comparisons allowed
5. **UTC/Standard time**: Time references assume standard trading calendar
6. **Daily bars by default**: Time references assume daily data
7. **Case-insensitive keywords**: ENTRY, entry, Entry all valid
8. **Whitespace-insensitive**: Extra spaces/newlines ignored

---

## Integration with NLP Parser

The DSL converts output from the NLP parser:

```python
from nlp_parser import parse_trading_rule
from dsl import DSLParser, validate_dsl

# Step 1: Parse natural language
result = parse_trading_rule("Buy when close above 20 SMA")

# Step 2: Convert to DSL
dsl_ast = DSLParser.from_json_rule(result.rule)

# Step 3: Validate
is_valid, errors, warnings = validate_dsl(dsl_ast)

# Step 4: Output
print(dsl_ast.to_dict())
```

---

## Error Handling

### Syntax Errors
```
Parse Error: Expected indicator or series
  at line 2, column 10

Parse Error: Unclosed parenthesis
  at line 1, column 45
```

### Validation Errors
```
Indicator sma expects 2 parameters, got 1
Invalid series name: price
Unknown time lag: 3d_ago
Invalid operator: ~~
```
