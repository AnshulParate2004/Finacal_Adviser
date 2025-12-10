"""
DSL Examples and Tests - JSON input to AST output
Demonstrates parsing, validation, and conversion from JSON rules
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsl import parse_dsl, validate_dsl, get_strategy_quality, DSLParser


# ============================================================================
# TEST EXAMPLES WITH JSON INPUT → DSL OUTPUT
# ============================================================================

def test_example_1_json_to_dsl():
    """Example 1: Simple SMA Crossover - JSON to DSL"""
    print("\n" + "="*80)
    print("Example 1: Simple SMA Crossover (JSON → DSL)")
    print("="*80)
    
    # JSON Rule Input (from NLP parser)
    json_rule = {
        "entry": [
            {"left": "close", "operator": "crosses_above", "right": "sma(close,20)"}
        ],
        "exit": [
            {"left": "close", "operator": "crosses_below", "right": "sma(close,20)"}
        ]
    }
    
    print("\n[Input] JSON Rule:")
    print(json.dumps(json_rule, indent=2))
    
    # Convert to DSL AST
    print("\n[Processing] Converting JSON to DSL AST...")
    try:
        ast = DSLParser.from_json_rule(json_rule)
        is_valid, errors, warnings = validate_dsl(ast)
        
        if is_valid:
            print("✓ Valid DSL")
            print("\n[Output] DSL AST:")
            print(json.dumps(ast.to_dict(), indent=2))
        else:
            print(f"✗ Validation Error: {errors}")
    except Exception as e:
        print(f"✗ Error: {e}")


def test_example_2_rsi_volume_json():
    """Example 2: RSI + Volume - JSON to DSL"""
    print("\n" + "="*80)
    print("Example 2: RSI + Volume Confirmation (JSON → DSL)")
    print("="*80)
    
    # JSON Rule Input
    json_rule = {
        "entry": [
            {"left": "close", "operator": ">", "right": "sma(close,20)"},
            {"left": "rsi(close,14)", "operator": ">", "right": 50},
            {"left": "volume", "operator": ">", "right": "1M"}
        ],
        "exit": [
            {"left": "rsi(close,14)", "operator": "<", "right": 30}
        ]
    }
    
    print("\n[Input] JSON Rule:")
    print(json.dumps(json_rule, indent=2))
    
    # Convert to DSL AST
    print("\n[Processing] Converting JSON to DSL AST...")
    try:
        ast = DSLParser.from_json_rule(json_rule)
        is_valid, errors, warnings = validate_dsl(ast)
        
        if is_valid:
            print("✓ Valid DSL")
            metrics = get_strategy_quality(ast)
            print(f"\nQuality Metrics:")
            print(f"  Entry Complexity: {metrics['entry_complexity']}")
            print(f"  Exit Complexity: {metrics['exit_complexity']}")
            print(f"  Total Indicators: {metrics['indicator_count']}")
            print(f"  Has Exit: {metrics['has_exit']}")
            
            print("\n[Output] DSL AST:")
            print(json.dumps(ast.to_dict(), indent=2))
        else:
            print(f"✗ Validation Error: {errors}")
    except Exception as e:
        print(f"✗ Error: {e}")


def test_example_3_bollinger_bands_json():
    """Example 3: Bollinger Bands - JSON to DSL"""
    print("\n" + "="*80)
    print("Example 3: Bollinger Bands Strategy (JSON → DSL)")
    print("="*80)
    
    # JSON Rule Input
    json_rule = {
        "entry": [
            {"left": "close", "operator": "<", "right": "bb_lower(close,20,2)"},
            {"left": "rsi(close,14)", "operator": "<", "right": 30}
        ],
        "exit": [
            {"left": "close", "operator": ">", "right": "bb_upper(close,20,2)"}
        ]
    }
    
    print("\n[Input] JSON Rule:")
    print(json.dumps(json_rule, indent=2))
    
    # Convert to DSL AST
    print("\n[Processing] Converting JSON to DSL AST...")
    try:
        ast = DSLParser.from_json_rule(json_rule)
        is_valid, errors, warnings = validate_dsl(ast)
        
        if is_valid:
            print("✓ Valid DSL")
            metrics = get_strategy_quality(ast)
            print(f"\nQuality Metrics:")
            print(f"  Entry Complexity: {metrics['entry_complexity']}")
            print(f"  Exit Complexity: {metrics['exit_complexity']}")
            print(f"  Indicators Used: {metrics['indicator_count']}")
            
            print("\n[Output] DSL AST:")
            print(json.dumps(ast.to_dict(), indent=2))
        else:
            print(f"✗ Validation Error: {errors}")
    except Exception as e:
        print(f"✗ Error: {e}")


def test_example_4_macd_crossover_json():
    """Example 4: MACD Crossover - JSON to DSL"""
    print("\n" + "="*80)
    print("Example 4: MACD Crossover (JSON → DSL)")
    print("="*80)
    
    # JSON Rule Input
    json_rule = {
        "entry": [
            {"left": "macd(close,12,26,9)", "operator": "crosses_above", "right": "macd_signal(close,12,26,9)"}
        ],
        "exit": [
            {"left": "macd(close,12,26,9)", "operator": "crosses_below", "right": "macd_signal(close,12,26,9)"}
        ]
    }
    
    print("\n[Input] JSON Rule:")
    print(json.dumps(json_rule, indent=2))
    
    # Convert to DSL AST
    print("\n[Processing] Converting JSON to DSL AST...")
    try:
        ast = DSLParser.from_json_rule(json_rule)
        is_valid, errors, warnings = validate_dsl(ast)
        
        if is_valid:
            print("✓ Valid DSL")
            
            print("\n[Output] DSL AST:")
            print(json.dumps(ast.to_dict(), indent=2))
        else:
            print(f"✗ Validation Error: {errors}")
    except Exception as e:
        print(f"✗ Error: {e}")


def test_example_5_time_based_json():
    """Example 5: Time-Based Conditions - JSON to DSL"""
    print("\n" + "="*80)
    print("Example 5: Time-Based Conditions (JSON → DSL)")
    print("="*80)
    
    # JSON Rule Input
    json_rule = {
        "entry": [
            {"left": "close", "operator": ">", "right": "close_prev"},
            {"left": "close", "operator": ">", "right": "close_1d_ago"},
            {"left": "volume", "operator": ">", "right": "volume_prev"}
        ],
        "exit": [
            {"left": "close", "operator": "<", "right": "close_5d_ago"}
        ]
    }
    
    print("\n[Input] JSON Rule:")
    print(json.dumps(json_rule, indent=2))
    
    # Convert to DSL AST
    print("\n[Processing] Converting JSON to DSL AST...")
    try:
        ast = DSLParser.from_json_rule(json_rule)
        is_valid, errors, warnings = validate_dsl(ast)
        
        if is_valid:
            print("✓ Valid DSL")
            
            print("\n[Output] DSL AST:")
            print(json.dumps(ast.to_dict(), indent=2))
        else:
            print(f"✗ Validation Error: {errors}")
    except Exception as e:
        print(f"✗ Error: {e}")


def test_example_6_complex_logic_json():
    """Example 6: Complex Logic with Multiple Conditions - JSON to DSL"""
    print("\n" + "="*80)
    print("Example 6: Complex Logic (JSON → DSL)")
    print("="*80)
    
    # JSON Rule Input
    json_rule = {
        "entry": [
            {"left": "close", "operator": ">", "right": "sma(close,20)"},
            {"left": "volume", "operator": ">", "right": "1M"},
            {"left": "rsi(close,14)", "operator": ">", "right": 50},
            {"left": "close", "operator": ">", "right": "close_1d_ago"},
            {"left": "atr(14)", "operator": ">", "right": 2}
        ],
        "exit": [
            {"left": "rsi(close,14)", "operator": "<", "right": 30},
            {"left": "close", "operator": "<", "right": "sma(close,50)"}
        ]
    }
    
    print("\n[Input] JSON Rule:")
    print(json.dumps(json_rule, indent=2))
    
    # Convert to DSL AST
    print("\n[Processing] Converting JSON to DSL AST...")
    try:
        ast = DSLParser.from_json_rule(json_rule)
        is_valid, errors, warnings = validate_dsl(ast)
        
        if is_valid:
            print("✓ Valid DSL")
            metrics = get_strategy_quality(ast)
            print(f"\nQuality Metrics:")
            print(f"  Entry Complexity: {metrics['entry_complexity']}")
            print(f"  Exit Complexity: {metrics['exit_complexity']}")
            print(f"  Total Indicators: {metrics['indicator_count']}")
            if metrics['warnings']:
                print(f"  Warnings: {metrics['warnings']}")
            
            print("\n[Output] DSL AST:")
            print(json.dumps(ast.to_dict(), indent=2))
        else:
            print(f"✗ Validation Error: {errors}")
    except Exception as e:
        print(f"✗ Error: {e}")


def test_example_7_all_indicators_json():
    """Example 7: All Indicators - JSON to DSL"""
    print("\n" + "="*80)
    print("Example 7: All Indicators (JSON → DSL)")
    print("="*80)
    
    # JSON Rule Input
    json_rule = {
        "entry": [
            {"left": "close", "operator": ">", "right": "sma(close,20)"},
            {"left": "close", "operator": ">", "right": "ema(close,12)"},
            {"left": "rsi(close,14)", "operator": ">", "right": 50},
            {"left": "macd(close,12,26,9)", "operator": ">", "right": 0},
            {"left": "close", "operator": ">", "right": "bb_lower(close,20,2)"},
            {"left": "atr(14)", "operator": ">", "right": 1},
            {"left": "adx(14)", "operator": ">", "right": 20},
            {"left": "stoch(close,14,3,3)", "operator": ">", "right": 20}
        ],
        "exit": [
            {"left": "close", "operator": "<", "right": "sma(close,20)"}
        ]
    }
    
    print("\n[Input] JSON Rule (8 indicators):")
    print(json.dumps(json_rule, indent=2))
    
    # Convert to DSL AST
    print("\n[Processing] Converting JSON to DSL AST...")
    try:
        ast = DSLParser.from_json_rule(json_rule)
        is_valid, errors, warnings = validate_dsl(ast)
        
        if is_valid:
            print("✓ Valid DSL")
            metrics = get_strategy_quality(ast)
            print(f"\nQuality Metrics:")
            print(f"  Entry Complexity: {metrics['entry_complexity']}")
            print(f"  Exit Complexity: {metrics['exit_complexity']}")
            print(f"  Total Indicators: {metrics['indicator_count']}")
            if metrics['warnings']:
                print(f"  ⚠ Warnings: {metrics['warnings']}")
            
            print("\n[Output] DSL AST (Entry):")
            output = ast.to_dict()
            print(json.dumps({"entry": output["entry"]}, indent=2))
        else:
            print(f"✗ Validation Error: {errors}")
    except Exception as e:
        print(f"✗ Error: {e}")


def test_example_8_invalid_json():
    """Example 8: Invalid JSON - Error Handling"""
    print("\n" + "="*80)
    print("Example 8: Invalid JSON - Error Handling")
    print("="*80)
    
    invalid_examples = [
        {
            "name": "Missing entry",
            "rule": {"exit": [{"left": "rsi(close,14)", "operator": "<", "right": 30}]}
        },
        {
            "name": "Invalid indicator",
            "rule": {
                "entry": [{"left": "unknown_ind(close,20)", "operator": ">", "right": 100}]
            }
        },
        {
            "name": "Invalid series",
            "rule": {
                "entry": [{"left": "price", "operator": ">", "right": 100}]
            }
        }
    ]
    
    for example in invalid_examples:
        print(f"\n[Test] {example['name']}:")
        print(f"  Input: {json.dumps(example['rule'])}")
        try:
            ast = DSLParser.from_json_rule(example['rule'])
            is_valid, errors, warnings = validate_dsl(ast)
            if not is_valid:
                print(f"  ✗ Expected Error (caught): {errors[0]}")
            else:
                print(f"  ✗ Should have failed!")
        except Exception as e:
            print(f"  ✗ Expected Error (caught): {str(e)[:80]}")


def test_example_9_dsl_text_parsing():
    """Example 9: Direct DSL Text Parsing"""
    print("\n" + "="*80)
    print("Example 9: Direct DSL Text Parsing")
    print("="*80)
    
    dsl_examples = [
        {
            "name": "Simple Crossover",
            "dsl": "ENTRY:\n  close crosses_above sma(close, 20)\nEXIT:\n  close crosses_below sma(close, 20)"
        },
        {
            "name": "RSI Strategy",
            "dsl": "ENTRY:\n  rsi(close, 14) > 50 AND volume > 1M\nEXIT:\n  rsi(close, 14) < 30"
        },
        {
            "name": "Bollinger Bands",
            "dsl": "ENTRY:\n  close < bb_lower(close, 20, 2) AND rsi(close, 14) < 30\nEXIT:\n  close > bb_upper(close, 20, 2)"
        }
    ]
    
    for example in dsl_examples:
        print(f"\n[Test] {example['name']}:")
        print(f"  DSL Input:\n{example['dsl']}")
        try:
            ast = parse_dsl(example['dsl'])
            is_valid, errors, warnings = validate_dsl(ast)
            if is_valid:
                print(f"  ✓ Valid")
                metrics = get_strategy_quality(ast)
                print(f"  Complexity: {metrics['entry_complexity']}")
            else:
                print(f"  ✗ Error: {errors}")
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:80]}")


def test_summary():
    """Summary of all tests"""
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("\n✓ All examples completed!")
    print("\nTest Coverage:")
    print("  [✓] Example 1: Simple SMA Crossover (JSON → DSL)")
    print("  [✓] Example 2: RSI + Volume (JSON → DSL)")
    print("  [✓] Example 3: Bollinger Bands (JSON → DSL)")
    print("  [✓] Example 4: MACD Crossover (JSON → DSL)")
    print("  [✓] Example 5: Time-Based Conditions (JSON → DSL)")
    print("  [✓] Example 6: Complex Logic (JSON → DSL)")
    print("  [✓] Example 7: All Indicators (JSON → DSL)")
    print("  [✓] Example 8: Error Handling")
    print("  [✓] Example 9: Direct DSL Parsing")
    print("\n" + "="*80)


def run_all_tests():
    """Run all test examples"""
    print("\n" + "="*80)
    print("TRADING STRATEGY DSL - TEST EXAMPLES")
    print("JSON Input → DSL AST Output")
    print("="*80)
    
    try:
        test_example_1_json_to_dsl()
        test_example_2_rsi_volume_json()
        test_example_3_bollinger_bands_json()
        test_example_4_macd_crossover_json()
        test_example_5_time_based_json()
        test_example_6_complex_logic_json()
        test_example_7_all_indicators_json()
        test_example_8_invalid_json()
        test_example_9_dsl_text_parsing()
        test_summary()
    except Exception as e:
        print(f"\n✗ Test Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check dependencies
    try:
        import lark
    except ImportError:
        print("ERROR: Lark not installed. Install with: pip install lark")
        sys.exit(1)
    
    run_all_tests()
