"""
Test DSL Parser with ParsedStrategy format
"""
import sys
import os
from typing import List
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsl import parse_dsl, validate_dsl, get_strategy_quality

# Mock NLP Output Models
class Condition(BaseModel):
    left: str
    operator: str
    right: float | str

class TradingRule(BaseModel):
    entry: List[Condition]
    exit: List[Condition]
    initial_capital: float
    position_size: float

class ParsedStrategy(BaseModel):
    rule: TradingRule
    original_text: str
    indicators_used: List[str]
    complexity: str
    initial_capital: float
    position_size: float

# Converter: ParsedStrategy → DSL
def convert_to_dsl(parsed_strategy: ParsedStrategy) -> str:
    def condition_to_dsl(condition: Condition) -> str:
        left = condition.left
        operator = condition.operator
        right = str(condition.right) if not isinstance(condition.right, str) else condition.right
        return f"{left} {operator} {right}"
    
    entry_dsl = " AND ".join([condition_to_dsl(c) for c in parsed_strategy.rule.entry])
    exit_dsl = " AND ".join([condition_to_dsl(c) for c in parsed_strategy.rule.exit])
    
    dsl_text = f"ENTRY: {entry_dsl}"
    if exit_dsl:
        dsl_text += f"\nEXIT: {exit_dsl}"
    return dsl_text


# Test Example
def main():
    print("\n" + "="*60)
    print("DSL Parser Test - Simple RSI Strategy")
    print("="*60)
    
    # Mock NLP output
    parsed = ParsedStrategy(
        rule=TradingRule(
            entry=[Condition(left="rsi(close, 14)", operator=">", right=70)],
            exit=[Condition(left="rsi(close, 14)", operator="<", right=30)],
            initial_capital=50000.0,
            position_size=0.5
        ),
        original_text="Buy when RSI > 70 with $50k invest 50%. Sell when RSI < 30.",
        indicators_used=["rsi"],
        complexity="simple",
        initial_capital=50000.0,
        position_size=0.5
    )
    
    print(f"\n Original: {parsed.original_text}")
    print(f"Capital: ${parsed.initial_capital:,.0f}")
    print(f"Position: {parsed.position_size * 100}%")
    
    # Convert to DSL
    dsl_text = convert_to_dsl(parsed)
    print(f"\n DSL Generated:\n{dsl_text}")
    
    # Parse & Validate
    ast = parse_dsl(dsl_text)
    print(f"\n Parsed: {type(ast).__name__}")
    
    is_valid, errors, warnings = validate_dsl(ast)
    print(f" Valid: {is_valid}")
    
    quality = get_strategy_quality(ast)
    print(f"\n Quality:")
    print(f"   Entry Complexity: {quality['entry_complexity']}")
    print(f"   Indicators: {quality['indicator_count']}")
    print(f"   Has Exit: {quality['has_exit']}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
