# Trading Strategy DSL - Quick Reference

## What is it?
A domain-specific language for writing trading strategies in plain English-like syntax. Converts human-readable rules into executable code.

## Core Syntax

```
ENTRY: <condition>
EXIT: <condition>
```

## Operators
- **Compare:** `>` `<` `>=` `<=` `==` `!=`
- **Cross:** `crosses_above` `crosses_below` `crosses` `touches`
- **Logic:** `AND` `OR` `(parentheses)`

## Data
- **Price:** `close` `open` `high` `low` `volume`
- **Numbers:** `100` `1.5K` `2M` `1B`
- **Time:** `close_prev` `high_5d_ago` `low_1w_ago`

## Indicators
`sma(close, 20)` `ema(close, 12)` `rsi(close, 14)` `macd(close, 12, 26, 9)` `bb_upper(close, 20, 2)` `atr(14)` `adx(14)` `stoch(14, 3, 3, 3)`

## Example

```
ENTRY: sma(close, 20) crosses_above sma(close, 50) AND rsi(close, 14) < 40
EXIT: close crosses_below sma(close, 20) OR rsi(close, 14) > 70
```

## Files
- `DSL_SPECIFICATION.md` - Full grammar & syntax
- `EXAMPLES.md` - Usage examples
- `ASSUMPTIONS.md` - Design decisions
- `grammar.lark` - Parser grammar
- `ast_nodes.py` - AST structure
- `parser.py` - Text → AST converter
- `validator.py` - Validation & quality checks

## Usage

```python
from dsl import parse_dsl, validate_dsl

ast = parse_dsl("ENTRY: close > sma(close, 20)")
is_valid, errors, warnings = validate_dsl(ast)
```
