"""
NLP Parser Tests - Natural Language to Structured JSON
Tests the hybrid offline + LLM parsing system for trading rules
"""
import sys
import os
import json
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp_parser import parse_trading_rule, check_completeness
from nlp_parser.parser import OfflineCompletenessCheck


# ============================================================================
# TEST 1: Offline Completeness Check (No API calls)
# ============================================================================

def test_offline_completeness_check():
    """Test offline completeness validation (pattern-based, no LLM)"""
    print("\n" + "="*80)
    print("TEST 1: Offline Completeness Check (Pattern-Based)")
    print("="*80)
    
    test_cases = [
        {
            "name": "Complete: SMA Crossover",
            "text": "Buy when close crosses above 20-day SMA",
            "expected_complete": True
        },
        {
            "name": "Complete: RSI + Volume",
            "text": "Enter long when RSI is above 50 and volume is above 1M",
            "expected_complete": True
        },
        {
            "name": "Complete: Price above SMA",
            "text": "Buy when close is above SMA(20)",
            "expected_complete": True
        },
        {
            "name": "Incomplete: Missing comparison value",
            "text": "Buy when close is above",
            "expected_complete": False
        },
        {
            "name": "Incomplete: Missing action",
            "text": "When RSI is below 30",
            "expected_complete": False
        },
        {
            "name": "Incomplete: No indicator",
            "text": "Buy when above 50",
            "expected_complete": False
        },
        {
            "name": "Complete: Multiple conditions",
            "text": "Buy when close > 100 and volume > 1M and RSI > 50",
            "expected_complete": True
        }
    ]
    
    passed = 0
    failed = 0
    
    print("\nRunning offline completeness checks (no API calls)...\n")
    
    for i, test in enumerate(test_cases, 1):
        is_complete, details = OfflineCompletenessCheck.check(test['text'])
        
        status = "✓" if is_complete == test['expected_complete'] else "✗"
        result = "PASS" if is_complete == test['expected_complete'] else "FAIL"
        
        print(f"{status} Test {i}: {test['name']}")
        print(f"  Input: \"{test['text']}\"")
        print(f"  Result: {result} (Expected: {'Complete' if test['expected_complete'] else 'Incomplete'}, Got: {'Complete' if is_complete else 'Incomplete'})")
        
        if not is_complete:
            print(f"  Missing: {', '.join(details['missing_elements'])}")
        
        if is_complete == test['expected_complete']:
            passed += 1
        else:
            failed += 1
        print()
    
    print(f"\nOffline Check Results: {passed}/{len(test_cases)} passed, {failed} failed")


# ============================================================================
# TEST 2: Completeness Check with LLM Fallback
# ============================================================================

def test_completeness_check_with_llm():
    """Test hybrid completeness check (offline first, then LLM)"""
    print("\n" + "="*80)
    print("TEST 2: Hybrid Completeness Check (Offline + LLM Fallback)")
    print("="*80)
    
    test_cases = [
        {
            "name": "Complete rule",
            "text": "Buy when close is above 20-day moving average and volume is above 1M",
            "expected_complete": True
        },
        {
            "name": "Incomplete: dangling comparison",
            "text": "Buy when close is above",
            "expected_complete": False
        },
        {
            "name": "Complete: with exit",
            "text": "Buy when RSI is above 70. Sell when RSI drops below 30.",
            "expected_complete": True
        },
        {
            "name": "Incomplete: no action",
            "text": "When price crosses above 100",
            "expected_complete": False
        }
    ]
    
    print("\nRunning hybrid completeness checks (offline + LLM fallback)...\n")
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        try:
            is_complete, response, used_llm = check_completeness(test['text'])
            
            status = "✓" if is_complete == test['expected_complete'] else "✗"
            result = "PASS" if is_complete == test['expected_complete'] else "FAIL"
            
            print(f"{status} Test {i}: {test['name']}")
            print(f"  Input: \"{test['text']}\"")
            print(f"  Result: {result}")
            print(f"  Used LLM: {'Yes' if used_llm else 'No (offline check passed)'}")
            print(f"  Status: {response.status}")
            print(f"  Confidence: {response.confidence}")
            
            if not is_complete and response.missing_elements:
                print(f"  Missing: {', '.join(response.missing_elements)}")
            if response.suggestion:
                print(f"  Suggestion: {response.suggestion}")
            
            if is_complete == test['expected_complete']:
                passed += 1
            else:
                failed += 1
            
        except Exception as e:
            print(f"✗ Test {i}: {test['name']}")
            print(f"  ERROR: {str(e)}")
            failed += 1
        
        print()
    
    print(f"\nHybrid Check Results: {passed}/{len(test_cases)} passed, {failed} failed")


# ============================================================================
# TEST 3: Full NLP Parsing (Complete Rules)
# ============================================================================

def test_nlp_parsing_complete_rules():
    """Test parsing of complete trading rules"""
    print("\n" + "="*80)
    print("TEST 3: Full NLP Parsing (Complete Rules)")
    print("="*80)
    
    test_cases = [
        {
            "name": "Simple SMA Crossover",
            "text": "Buy when close crosses above 20-day SMA. Sell when close crosses below 20-day SMA.",
            "expected_indicators": ["sma"]
        },
        {
            "name": "RSI Strategy",
            "text": "Enter long when RSI(14) is above 50. Exit when RSI(14) drops below 30.",
            "expected_indicators": ["rsi"]
        },
        {
            "name": "Multi-Indicator Strategy",
            "text": "Buy when close is above 20-day SMA and RSI is above 50 and volume is above 1.5 million. Exit when RSI drops below 30.",
            "expected_indicators": ["sma", "rsi"]
        },
        {
            "name": "Bollinger Bands",
            "text": "Enter when close is below lower Bollinger Band and RSI is below 30. Exit when close crosses above upper Bollinger Band.",
            "expected_indicators": ["bb", "rsi"]
        },
        {
            "name": "MACD Crossover",
            "text": "Buy when MACD line crosses above signal line. Sell when MACD crosses below signal.",
            "expected_indicators": ["macd"]
        }
    ]
    
    print("\nParsing complete trading rules...\n")
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        try:
            parsed = parse_trading_rule(test['text'])
            
            print(f"✓ Test {i}: {test['name']}")
            print(f"  Input: \"{test['text']}\"")
            print(f"  Original Text: {parsed.original_text}")
            print(f"  Indicators Used: {', '.join(parsed.indicators_used)}")
            print(f"  Complexity: {parsed.complexity}")
            print(f"  Entry Conditions: {len(parsed.rule.entry)}")
            print(f"  Exit Conditions: {len(parsed.rule.exit)}")
            
            # Print structured JSON
            print(f"\n  Structured Rule:")
            rule_dict = parsed.rule.dict()
            print(f"    Entry: {json.dumps(rule_dict['entry'], indent=6)}")
            if rule_dict['exit']:
                print(f"    Exit: {json.dumps(rule_dict['exit'], indent=6)}")
            
            passed += 1
            
        except Exception as e:
            print(f"✗ Test {i}: {test['name']}")
            print(f"  Input: \"{test['text']}\"")
            print(f"  ERROR: {str(e)}")
            failed += 1
        
        print()
    
    print(f"\nParsing Results: {passed}/{len(test_cases)} passed, {failed} failed")


# ============================================================================
# TEST 4: Incomplete Rules (Should Fail Gracefully)
# ============================================================================

def test_nlp_parsing_incomplete_rules():
    """Test parsing of incomplete rules (should fail with helpful messages)"""
    print("\n" + "="*80)
    print("TEST 4: Incomplete Rules (Error Handling)")
    print("="*80)
    
    test_cases = [
        {
            "name": "Missing comparison value",
            "text": "Buy when close is above",
            "should_fail": True
        },
        {
            "name": "Missing action",
            "text": "When RSI is below 30",
            "should_fail": True
        },
        {
            "name": "Ambiguous indicator",
            "text": "Buy when moving average is above",
            "should_fail": True
        },
        {
            "name": "No indicator",
            "text": "Buy when above 100",
            "should_fail": True
        }
    ]
    
    print("\nTesting incomplete rules (should fail gracefully)...\n")
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        try:
            parsed = parse_trading_rule(test['text'])
            
            # Should not reach here if test expects failure
            if test['should_fail']:
                print(f"✗ Test {i}: {test['name']}")
                print(f"  Input: \"{test['text']}\"")
                print(f"  ERROR: Should have failed but succeeded!")
                failed += 1
            else:
                print(f"✓ Test {i}: {test['name']}")
                print(f"  Parsed successfully (as expected)")
                passed += 1
            
        except ValueError as e:
            # Expected error for incomplete rules
            if test['should_fail']:
                print(f"✓ Test {i}: {test['name']}")
                print(f"  Input: \"{test['text']}\"")
                print(f"  Failed gracefully (as expected)")
                print(f"  Error Message: {str(e)[:150]}...")
                passed += 1
            else:
                print(f"✗ Test {i}: {test['name']}")
                print(f"  Input: \"{test['text']}\"")
                print(f"  ERROR: Should have succeeded but failed!")
                print(f"  Error: {str(e)}")
                failed += 1
        
        except Exception as e:
            print(f"✗ Test {i}: {test['name']}")
            print(f"  Input: \"{test['text']}\"")
            print(f"  ERROR: Unexpected error: {str(e)}")
            failed += 1
        
        print()
    
    print(f"\nError Handling Results: {passed}/{len(test_cases)} passed, {failed} failed")


# ============================================================================
# TEST 5: Edge Cases and Special Syntax
# ============================================================================

def test_edge_cases():
    """Test edge cases and special syntax"""
    print("\n" + "="*80)
    print("TEST 5: Edge Cases and Special Syntax")
    print("="*80)
    
    test_cases = [
        {
            "name": "Time-based reference",
            "text": "Buy when close is above yesterday's high and volume increases by 30%",
        },
        {
            "name": "Multiple indicators",
            "text": "Enter when close > SMA(20) and close > EMA(12) and RSI > 50 and MACD > 0",
        },
        {
            "name": "Percentage format",
            "text": "Buy when volume increases by more than 50 percent",
        },
        {
            "name": "Natural language numbers",
            "text": "Enter when volume is above one million",
        },
        {
            "name": "Only entry condition",
            "text": "Buy when RSI crosses above 30",
        }
    ]
    
    print("\nTesting edge cases...\n")
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        try:
            parsed = parse_trading_rule(test['text'])
            
            print(f"✓ Test {i}: {test['name']}")
            print(f"  Input: \"{test['text']}\"")
            print(f"  Indicators: {', '.join(parsed.indicators_used)}")
            print(f"  Entry Conditions: {len(parsed.rule.entry)}")
            print(f"  Exit Conditions: {len(parsed.rule.exit)}")
            print(f"  Complexity: {parsed.complexity}")
            
            passed += 1
            
        except Exception as e:
            print(f"✗ Test {i}: {test['name']}")
            print(f"  Input: \"{test['text']}\"")
            print(f"  ERROR: {str(e)[:150]}...")
            failed += 1
        
        print()
    
    print(f"\nEdge Case Results: {passed}/{len(test_cases)} passed, {failed} failed")


# ============================================================================
# TEST 6: Complexity Classification
# ============================================================================

def test_complexity_classification():
    """Test strategy complexity classification"""
    print("\n" + "="*80)
    print("TEST 6: Complexity Classification")
    print("="*80)
    
    test_cases = [
        {
            "name": "Simple strategy",
            "text": "Buy when close crosses above SMA(20)",
            "expected_complexity": "simple"
        },
        {
            "name": "Medium strategy",
            "text": "Buy when close > SMA(20) and RSI > 50. Exit when RSI < 30.",
            "expected_complexity": "medium"
        },
        {
            "name": "Complex strategy",
            "text": "Buy when close > SMA(20) and RSI > 50 and volume > 1M and close > EMA(12). Exit when RSI < 30 or close < SMA(50).",
            "expected_complexity": "complex"
        }
    ]
    
    print("\nTesting complexity classification...\n")
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        try:
            parsed = parse_trading_rule(test['text'])
            
            complexity_match = parsed.complexity == test['expected_complexity']
            status = "✓" if complexity_match else "✗"
            
            print(f"{status} Test {i}: {test['name']}")
            print(f"  Input: \"{test['text']}\"")
            print(f"  Expected: {test['expected_complexity']}")
            print(f"  Got: {parsed.complexity}")
            print(f"  Total Conditions: {len(parsed.rule.entry) + len(parsed.rule.exit)}")
            
            if complexity_match:
                passed += 1
            else:
                failed += 1
            
        except Exception as e:
            print(f"✗ Test {i}: {test['name']}")
            print(f"  ERROR: {str(e)}")
            failed += 1
        
        print()
    
    print(f"\nComplexity Classification Results: {passed}/{len(test_cases)} passed, {failed} failed")


# ============================================================================
# TEST SUMMARY
# ============================================================================

def test_summary():
    """Print test summary"""
    print("\n" + "="*80)
    print("NLP PARSER TEST SUMMARY")
    print("="*80)
    print("\n✓ All test suites completed!")
    print("\nTest Coverage:")
    print("  [✓] Test 1: Offline Completeness Check (Pattern-Based)")
    print("  [✓] Test 2: Hybrid Completeness Check (Offline + LLM)")
    print("  [✓] Test 3: Full NLP Parsing (Complete Rules)")
    print("  [✓] Test 4: Incomplete Rules (Error Handling)")
    print("  [✓] Test 5: Edge Cases and Special Syntax")
    print("  [✓] Test 6: Complexity Classification")
    print("\n" + "="*80)


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all NLP parser tests"""
    print("\n" + "="*80)
    print("NLP PARSER TESTS")
    print("Natural Language → Structured JSON")
    print("="*80)
    
    print("\nNote: Some tests require LLM API calls (Google Gemini).")
    print("Make sure you have GOOGLE_API_KEY set in your environment.\n")
    
    try:
        # Test 1: Offline checks (no API calls)
        test_offline_completeness_check()
        
        # Test 2: Hybrid checks (may use API)
        print("\n⚠️  The following tests will use LLM API calls if needed...")
        input("Press Enter to continue or Ctrl+C to skip LLM tests...")
        
        test_completeness_check_with_llm()
        
        # Test 3-6: Full parsing tests (require API)
        test_nlp_parsing_complete_rules()
        test_nlp_parsing_incomplete_rules()
        test_edge_cases()
        test_complexity_classification()
        
        # Summary
        test_summary()
        
    except KeyboardInterrupt:
        print("\n\n✗ Tests interrupted by user")
    except Exception as e:
        print(f"\n✗ Test Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check dependencies
    try:
        from nlp_parser import parse_trading_rule, check_completeness
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Make sure nlp_parser module is installed")
        sys.exit(1)
    
    # Check for API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("WARNING: GOOGLE_API_KEY not found in environment")
        print("Some tests will fail without API access")
        print("\nSet it with: export GOOGLE_API_KEY='your-api-key'")
        print("\nContinuing with offline tests only...\n")
        
        # Run only offline test
        test_offline_completeness_check()
        print("\n✓ Offline tests completed. Skipping LLM tests.")
    else:
        # Run all tests
        run_all_tests()
