# DSL Examples

## Basic Entry/Exit

```
ENTRY: close > sma(close, 20)
EXIT: close < sma(close, 20)
```

## Boolean Logic

```
ENTRY: close > sma(close, 20) AND volume > 1M
EXIT: rsi(close, 14) > 70 OR close < sma(close, 50)
```

## Cross Events

```
ENTRY: sma(close, 20) crosses_above sma(close, 50)
EXIT: sma(close, 20) crosses_below sma(close, 50)
```

## Time References

```
ENTRY: close > close_prev AND volume > volume_5d_ago
EXIT: high < high_1w_ago
```

## Nested Logic

```
ENTRY: (close > sma(close, 20) AND rsi(close, 14) < 30) OR (volume > 2M AND close > open)
EXIT: rsi(close, 14) > 70
```

## Multiple Indicators

```
ENTRY: sma(close, 20) > sma(close, 50) AND rsi(close, 14) < 40 AND macd(close, 12, 26, 9) > 0
EXIT: bb_upper(close, 20, 2) touches close
```

## Scaled Numbers

```
ENTRY: volume > 1.5M AND close > 100
EXIT: volume < 500K OR close < 95
```

## Complete Strategy

```
ENTRY: close crosses_above sma(close, 20) AND volume > volume_prev AND rsi(close, 14) < 50
EXIT: close crosses_below sma(close, 20) OR rsi(close, 14) > 70
```
