"""
Quick NLP Test - Single example runner
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nlp_parser import parse_trading_rule

# Your test text here
text = "Buy when RSI is above 70 with $50k and invest 50%. Sell when RSI drops below 30."

print("="*80)
print("QUICK NLP TEST")
print("="*80)
print(f"\nInput:\n  {text}\n")

try:
    result = parse_trading_rule(text)
    
    print("✓ SUCCESS!\n")
    print(f"Capital: ${result.initial_capital:,.2f}")
    print(f"Position: {result.position_size*100:.0f}%")
    print(f"Indicators: {result.indicators_used}")
    print(f"Complexity: {result.complexity}")
    
    print(f"\nEntry ({len(result.rule.entry)} conditions):")
    for cond in result.rule.entry:
        print(f"  • {cond.left} {cond.operator} {cond.right}")
    
    print(f"\nExit ({len(result.rule.exit)} conditions):")
    for cond in result.rule.exit:
        print(f"  • {cond.left} {cond.operator} {cond.right}")
    
except Exception as e:
    print(f"✗ ERROR: {e}")

print("\n" + "="*80)
