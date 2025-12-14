"""
Test script to verify capital and position size auto-extraction
"""
from nlp_parser import parse_trading_rule

# Test cases
test_cases = [
    {
        "name": "Capital with $50k notation",
        "text": "Buy when RSI is above 70 with $50k. Sell when RSI drops below 30.",
        "expected_capital": 50000.0,
        "expected_position": 1.0
    },
    {
        "name": "Capital with $50,000 notation",
        "text": "Buy when RSI is above 70 with $50,000 capital. Sell when RSI drops below 30.",
        "expected_capital": 50000.0,
        "expected_position": 1.0
    },
    {
        "name": "Capital with million notation",
        "text": "Start with $5m and buy when close crosses above SMA.",
        "expected_capital": 5000000.0,
        "expected_position": 1.0
    },
    {
        "name": "Position size with percentage",
        "text": "Buy when RSI > 70 and invest 50%. Sell when RSI < 30.",
        "expected_capital": 10000.0,
        "expected_position": 0.5
    },
    {
        "name": "Position size with 'half' keyword",
        "text": "Use half portfolio when MACD crosses signal. Exit when RSI < 30.",
        "expected_capital": 10000.0,
        "expected_position": 0.5
    },
    {
        "name": "Both capital and position",
        "text": "With $100k capital, invest 25% when RSI > 70. Exit when RSI < 30.",
        "expected_capital": 100000.0,
        "expected_position": 0.25
    },
    {
        "name": "Neither specified (defaults)",
        "text": "Buy when RSI is above 70. Sell when RSI drops below 30.",
        "expected_capital": 10000.0,
        "expected_position": 1.0
    },
    {
        "name": "Quarter position",
        "text": "Risk quarter of portfolio when close > SMA(20). Exit when RSI < 30.",
        "expected_capital": 10000.0,
        "expected_position": 0.25
    }
]

print("="*80)
print("TESTING CAPITAL AND POSITION SIZE AUTO-EXTRACTION")
print("="*80)

for i, test in enumerate(test_cases, 1):
    print(f"\n[Test {i}] {test['name']}")
    print(f"Input: \"{test['text']}\"")
    
    try:
        result = parse_trading_rule(test['text'])
        
        capital_match = result.initial_capital == test['expected_capital']
        position_match = abs(result.position_size - test['expected_position']) < 0.01
        
        print(f"Expected Capital: ${test['expected_capital']:,.2f}")
        print(f"Extracted Capital: ${result.initial_capital:,.2f} {'✓' if capital_match else '✗'}")
        
        print(f"Expected Position: {test['expected_position']*100:.0f}%")
        print(f"Extracted Position: {result.position_size*100:.0f}% {'✓' if position_match else '✗'}")
        
        if capital_match and position_match:
            print("Status: ✓ PASSED")
        else:
            print("Status: ✗ FAILED")
            
    except Exception as e:
        print(f"Status: ✗ ERROR - {str(e)}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
