# Trading Strategy DSL - Quick Reference

## Installation
```bash
pip install lark
```

## Basic Usage

### Parse DSL
```python
from dsl import parse_dsl, validate_dsl

dsl_text = """
ENTRY:
    close > sma(close, 20) AND volume > 1M
EXIT:
    rsi(close, 14) < 30
"""

ast = parse_dsl(dsl_text)
is_valid, errors, warnings = validate_dsl(ast)

if is_valid:
    print(ast.to_dict())
```

### Convert from JSON Rule
```python
from dsl import DSLParser

rule_dict = {
    "entry": [
        {"left": "close", "operator": ">", "right": "sma(close,20)"},
        {"left": "volume", "operator": ">", "right": 1000000}
    ],
    "exit": [{"left": "rsi(close,14)", "operator": "<", "right": 30}]
}

ast = DSLParser.from_json_rule(rule_dict)
```

### Get Strategy Quality Metrics
```python
from dsl import get_strategy_quality

metrics = get_strategy_quality(ast)
print(f"Entry Complexity: {metrics['entry_complexity']}")
print(f"Has Exit: {metrics['has_exit']}")
print(f"Indicators: {metrics['indicator_count']}")
```

## Syntax Cheat Sheet

| Feature | Syntax | Example |
|---------|--------|---------|
| Entry | `ENTRY: <rule>` | `ENTRY: close > 100` |
| Exit | `EXIT: <rule>` | `EXIT: rsi < 30` |
| AND | `A AND B` | `close > 100 AND volume > 1M` |
| OR | `A OR B` | `close > 100 OR rsi < 30` |
| Groups | `(A AND B) OR C` | `(close > 100 AND vol > 1M) OR rsi > 70` |
| Greater | `A > B` | `close > sma(close,20)` |
| Less | `A < B` | `rsi(close,14) < 30` |
| Equal | `A == B` | `close == open` |
| Cross Above | `A crosses_above B` | `close crosses_above sma(close,20)` |
| Cross Below | `A crosses_below B` | `close crosses_below bb_lower(close,20,2)` |

## Series Names
- `close`, `open`, `high`, `low`, `volume`

## Indicators
```
sma(close, 20)              Simple Moving Average
ema(close, 12)              Exponential Moving Average
rsi(close, 14)              Relative Strength Index
macd(close, 12, 26, 9)      MACD
macd_signal(close, 12, 26, 9)
macd_histogram(close, 12, 26, 9)
bb_upper(close, 20, 2)      Bollinger Bands
bb_middle(close, 20, 2)
bb_lower(close, 20, 2)
atr(14)                     Average True Range
adx(14)                     Average Directional Index
stoch(close, 14, 3, 3)      Stochastic
stoch_d(close, 14, 3, 3)
```

## Time References
```
close_prev                  Previous bar
close_1d_ago                1 day ago
close_5d_ago                5 days ago
close_1w_ago                1 week ago
close[5]                    5 bars ago
```

## Number Formats
```
100                         100
1.5                         1.5
1K                          1,000
1.5K                        1,500
1M                          1,000,000
1.5M                        1,500,000
1B                          1,000,000,000
```

## Real Examples

### Simple Moving Average Crossover
```
ENTRY:
    close crosses_above sma(close, 20)

EXIT:
    close crosses_below sma(close, 20)
```

### RSI Overbought/Oversold
```
ENTRY:
    (rsi(close, 14) > 70 AND close > sma(close, 50))
    OR (rsi(close, 14) < 30 AND close < sma(close, 50))

EXIT:
    rsi(close, 14) >= 50 AND rsi(close, 14) <= 50
```

### Volume Confirmation
```
ENTRY:
    close > sma(close, 20)
    AND volume > sma(volume, 20)
    AND close > close_1d_ago

EXIT:
    volume < sma(volume, 20)
    OR close < bb_lower(close, 20, 2)
```

### Bollinger Bands Strategy
```
ENTRY:
    close < bb_lower(close, 20, 2)
    AND rsi(close, 14) < 30

EXIT:
    close > bb_middle(close, 20, 2)
    OR rsi(close, 14) > 50
```

### MACD Crossover
```
ENTRY:
    macd(close, 12, 26, 9) crosses_above macd_signal(close, 12, 26, 9)

EXIT:
    macd(close, 12, 26, 9) crosses_below macd_signal(close, 12, 26, 9)
```

## Integration with NLP

```python
from nlp_parser import parse_trading_rule
from dsl import DSLParser, validate_dsl

# 1. Parse natural language
nl_rule = "Buy when close above 20 SMA and volume above 1 million. Exit when RSI below 30."
result = parse_trading_rule(nl_rule)

# 2. Convert to DSL
dsl_ast = DSLParser.from_json_rule(result.rule)

# 3. Validate
is_valid, errors, warnings = validate_dsl(dsl_ast)

# 4. Use
if is_valid:
    strategy_dict = dsl_ast.to_dict()
```

## Error Messages

| Error | Cause | Fix |
|-------|-------|-----|
| `Unknown indicator: xxx` | Typo in indicator name | Check indicator name spelling |
| `Indicator X expects N params` | Wrong number of parameters | Add/remove parameters |
| `Invalid series name: xxx` | Unknown series | Use: close, open, high, low, volume |
| `Invalid time lag: xxx` | Unknown lag format | Use: prev, 1d_ago, 5d_ago, 1w_ago, 1m_ago |
| `Missing ENTRY block` | No entry rule defined | Add `ENTRY:` section |

## Performance Tips
- Keep entry rule < 5 conditions for clarity
- Define exit rule to avoid holding forever
- Use different indicators for entry vs exit
- Group conditions with parentheses for readability
