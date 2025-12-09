"""
Test examples that should pass the completeness check and parse successfully.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp_parser import check_completeness, parse_trading_rule


def test_examples():
    """Test examples that should pass."""
    
    print("=" * 80)
    print("Testing Complete Trading Rules")
    print("=" * 80)
    print()
    
    # Examples that should PASS
    passing_examples = [
        # Simple entry only
        "Buy when close is above 20-day moving average",
        
        # Entry with multiple conditions
        "Buy when close is above SMA(20) and RSI(14) is above 50",
        
        # Entry with volume
        "Buy when close is above 20-day moving average and volume is above 1 million",
        
        # Entry and exit
        "Buy when close is above SMA(20). Exit when RSI(14) is below 30",
        
        # Crosses pattern
        "Enter when price crosses above yesterday's high",
        
        # Complex with exit
        "Buy when close is above 20-day moving average and RSI(14) is above 50 and volume is above 1M. Exit when RSI(14) falls below 30.",
        
        # Alternative phrasing
        "Enter long when close price is above the 20-day SMA",
        
        # With specific values
        "Buy when RSI is above 50 and MACD crosses above signal line",
    ]
    
    for i, rule_text in enumerate(passing_examples, 1):
        print(f"\n{'=' * 80}")
        print(f"Example {i}: {rule_text}")
        print(f"{'=' * 80}")
        
        try:
            # Check completeness (hybrid: offline + LLM)
            print("\n[Step 1] Checking completeness (Hybrid: Offline + LLM)...")
            is_complete, response, used_llm = check_completeness(rule_text)
            
            llm_status = "✓ LLM" if used_llm else "✓ OFFLINE"
            print(f"  Method: {llm_status} | Status: {response.status.upper()} | Complete: {is_complete}")
            print(f"  Confidence: {response.confidence}")
            
            if not is_complete:
                print(f"  ❌ FAILED - Should have passed!")
                print(f"  Missing: {', '.join(response.missing_elements)}")
                print(f"  Suggestion: {response.suggestion}")
                continue
            
            # Parse the rule
            print("\n[Step 2] Parsing rule...")
            parsed_strategy = parse_trading_rule(rule_text)
            
            print(f"  ✓ SUCCESS")
            print(f"  Complexity: {parsed_strategy.complexity}")
            print(f"  Indicators used: {', '.join(parsed_strategy.indicators_used) or 'None'}")
            
            print("\n  Entry conditions:")
            if parsed_strategy.rule.entry:
                for j, condition in enumerate(parsed_strategy.rule.entry, 1):
                    print(f"    {j}. {condition.left} {condition.operator} {condition.right}")
            else:
                print("    None")
            
            print("\n  Exit conditions:")
            if parsed_strategy.rule.exit:
                for j, condition in enumerate(parsed_strategy.rule.exit, 1):
                    print(f"    {j}. {condition.left} {condition.operator} {condition.right}")
            else:
                print("    None (optional)")
                
        except ValueError as e:
            print(f"  ❌ ERROR: {e}")
        except Exception as e:
            print(f"  ❌ UNEXPECTED ERROR: {e}")
    
    print("\n" + "=" * 80)
    print("Testing Incomplete Rules (Should Fail)")
    print("=" * 80)
    print()
    
    # Examples that should FAIL
    failing_examples = [
        "Buy when close is above",  # Missing threshold
        "When RSI is below 30",     # Missing action
        "close is above SMA(20)",   # Missing action
        "above 50",                 # Missing everything
        "Buy",                      # Missing condition
    ]
    
    for i, rule_text in enumerate(failing_examples, 1):
        print(f"\n{'=' * 80}")
        print(f"Fail Example {i}: {rule_text}")
        print(f"{'=' * 80}")
        
        try:
            is_complete, response, used_llm = check_completeness(rule_text)
            
            llm_status = "✓ LLM" if used_llm else "✓ OFFLINE"
            
            if not is_complete:
                print(f"  ✓ CORRECTLY REJECTED | Method: {llm_status}")
                print(f"  Missing: {', '.join(response.missing_elements)}")
                print(f"  Suggestion: {response.suggestion}")
            else:
                print(f"  ❌ SHOULD HAVE FAILED - Marked as complete!")
                
        except Exception as e:
            print(f"  ✓ CORRECTLY REJECTED WITH ERROR: {e}")
    
    print("\n" + "=" * 80)
    print("Tests completed!")
    print("=" * 80)
    print("\n📊 OPTIMIZATION SUMMARY:")
    print("  • Offline checks: Reduce API calls for well-formed rules")
    print("  • LLM fallback: Validate ambiguous/unusual wording")
    print("  • Result: Most rules = 1 API call (parsing only)")
    print("           Ambiguous rules = 2 API calls (completeness + parsing)")
    print("=" * 80)


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not found in environment variables!")
        print("Please create a .env file with your Google API key.")
        sys.exit(1)
    
    test_examples()
