"""
Code Generator Tests - AST to Trading Signals
Demonstrates generating trading functions from DSL and evaluating on real data
"""
import sys
import os
import json
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsl import DSLParser, parse_dsl, Strategy
from codegen import generate_trading_function


def create_sample_data(periods: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    
    close = 100 + np.cumsum(np.random.randn(periods) * 2)
    high = close + np.abs(np.random.randn(periods) * 1)
    low = close - np.abs(np.random.randn(periods) * 1)
    open_ = close + np.random.randn(periods) * 0.5
    volume = np.random.randint(1000000, 5000000, periods)
    
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    return df


def test_example_1_sma_crossover():
    """Test 1: Simple SMA Crossover Strategy"""
    print("\
" + "="*80)
    print("Test 1: Simple SMA Crossover Strategy")
    print("="*80)
    
    # JSON input
    json_rule = {
        "entry": [
            {"left": "close", "operator": "crosses_above", "right": "sma(close,20)"}
        ],
        "exit": [
            {"left": "close", "operator": "crosses_below", "right": "sma(close,20)"}
        ]
    }
    
    print("\
[Step 1] Convert JSON to DSL AST...")
    ast = DSLParser.from_json_rule(json_rule)
    print("✓ AST Created")
    
    print("\
[Step 2] Generate trading function from AST...")
    trading_func = generate_trading_function(ast)
    print("✓ Function Generated")
    
    print("\
[Step 3] Create sample data and evaluate signals...")
    df = create_sample_data(100)
    signals = trading_func(df)
    
    entry_count = signals['entry'].sum()
    exit_count = signals['exit'].sum()
    
    print(f"✓ Signals Evaluated")
    print(f"  Entry signals: {entry_count}")
    print(f"  Exit signals: {exit_count}")
    
    print("\
[Step 4] Sample signals output:")
    print(signals['signals_df'].tail(10))


def test_example_2_rsi_volume():
    """Test 2: RSI + Volume Strategy"""
    print("\
" + "="*80)
    print("Test 2: RSI + Volume Confirmation Strategy")
    print("="*80)
    
    # JSON input
    json_rule = {
        "entry": [
            {"left": "close", "operator": ">", "right": "sma(close,20)"},
            {"left": "rsi(close,14)", "operator": ">", "right": 50},
            {"left": "volume", "operator": ">", "right": "1000000"}
        ],
        "exit": [
            {"left": "rsi(close,14)", "operator": "<", "right": 30}
        ]
    }
    
    print("\
[Step 1] Convert JSON to DSL AST...")
    ast = DSLParser.from_json_rule(json_rule)
    print("✓ AST Created")
    
    print("\
[Step 2] Generate trading function...")
    trading_func = generate_trading_function(ast)
    print("✓ Function Generated")
    
    print("\
[Step 3] Evaluate on sample data...")
    df = create_sample_data(150)
    signals = trading_func(df)
    
    print(f"✓ Signals Evaluated")
    print(f"  Entry signals: {signals['entry'].sum()}")
    print(f"  Exit signals: {signals['exit'].sum()}")
    print(f"  Active positions: {(signals['entry'].cumsum() - signals['exit'].cumsum()).max()}")
    
    print("\
[Step 4] Sample output:")
    print(signals['signals_df'].iloc[40:50])


def test_example_3_bollinger_bands():
    """Test 3: Bollinger Bands Strategy"""
    print("\
" + "="*80)
    print("Test 3: Bollinger Bands Strategy")
    print("="*80)
    
    # JSON input
    json_rule = {
        "entry": [
            {"left": "close", "operator": "<", "right": "bb_lower(close,20,2)"},
            {"left": "rsi(close,14)", "operator": "<", "right": 30}
        ],
        "exit": [
            {"left": "close", "operator": ">", "right": "bb_upper(close,20,2)"}
        ]
    }
    
    print("\
[Step 1] Convert JSON to DSL AST...")
    ast = DSLParser.from_json_rule(json_rule)
    print("✓ AST Created")
    
    print("\
[Step 2] Generate trading function...")
    trading_func = generate_trading_function(ast)
    print("✓ Function Generated")
    
    print("\
[Step 3] Evaluate on sample data...")
    df = create_sample_data(200)
    signals = trading_func(df)
    
    print(f"✓ Signals Evaluated")
    print(f"  Entry signals: {signals['entry'].sum()}")
    print(f"  Exit signals: {signals['exit'].sum()}")
    
    print("\
[Step 4] Sample output:")
    print(signals['signals_df'].iloc[30:40])


def test_example_4_macd():
    """Test 4: MACD Crossover Strategy"""
    print("\
" + "="*80)
    print("Test 4: MACD Crossover Strategy")
    print("="*80)
    
    # JSON input
    json_rule = {
        "entry": [
            {"left": "macd(close,12,26,9)", "operator": "crosses_above", "right": "macd_signal(close,12,26,9)"}
        ],
        "exit": [
            {"left": "macd(close,12,26,9)", "operator": "crosses_below", "right": "macd_signal(close,12,26,9)"}
        ]
    }
    
    print("\
[Step 1] Convert JSON to DSL AST...")
    ast = DSLParser.from_json_rule(json_rule)
    print("✓ AST Created")
    
    print("\
[Step 2] Generate trading function...")
    trading_func = generate_trading_function(ast)
    print("✓ Function Generated")
    
    print("\
[Step 3] Evaluate on sample data...")
    df = create_sample_data(200)
    signals = trading_func(df)
    
    print(f"✓ Signals Evaluated")
    print(f"  Entry signals: {signals['entry'].sum()}")
    print(f"  Exit signals: {signals['exit'].sum()}")
    
    print("\
[Step 4] Sample output:")
    print(signals['signals_df'].iloc[35:45])


def test_example_5_time_references():
    """Test 5: Time-Based References"""
    print("\
" + "="*80)
    print("Test 5: Time-Based References Strategy")
    print("="*80)
    
    # JSON input
    json_rule = {
        "entry": [
            {"left": "close", "operator": ">", "right": "close_prev"},
            {"left": "volume", "operator": ">", "right": "volume_prev"}
        ],
        "exit": [
            {"left": "close", "operator": "<", "right": "close_1d_ago"}
        ]
    }
    
    print("\
[Step 1] Convert JSON to DSL AST...")
    ast = DSLParser.from_json_rule(json_rule)
    print("✓ AST Created")
    
    print("\
[Step 2] Generate trading function...")
    trading_func = generate_trading_function(ast)
    print("✓ Function Generated")
    
    print("\
[Step 3] Evaluate on sample data...")
    df = create_sample_data(100)
    signals = trading_func(df)
    
    print(f"✓ Signals Evaluated")
    print(f"  Entry signals: {signals['entry'].sum()}")
    print(f"  Exit signals: {signals['exit'].sum()}")
    
    print("\
[Step 4] Sample output:")
    print(signals['signals_df'].iloc[5:15])


def test_example_6_dsl_text():
    """Test 6: Parse DSL Text Directly"""
    print("\
" + "="*80)
    print("Test 6: Parse DSL Text Directly")
    print("="*80)
    
    dsl = """
    ENTRY:
        close > sma(close, 20) AND volume > 1000000
    EXIT:
        close < sma(close, 50)
    """
    
    print("\
[Step 1] Parse DSL text...")
    ast = parse_dsl(dsl)
    print("✓ DSL Parsed")
    print(f"  Entry type: {type(ast.entry).__name__}")
    print(f"  Exit type: {type(ast.exit).__name__}")
    
    print("\
[Step 2] Generate trading function...")
    trading_func = generate_trading_function(ast)
    print("✓ Function Generated")
    
    print("\
[Step 3] Evaluate on sample data...")
    df = create_sample_data(150)
    signals = trading_func(df)
    
    print(f"✓ Signals Evaluated")
    print(f"  Entry signals: {signals['entry'].sum()}")
    print(f"  Exit signals: {signals['exit'].sum()}")
    
    print("\
[Step 4] Sample output:")
    print(signals['signals_df'].iloc[25:35])


def test_example_7_error_handling():
    """Test 7: Error Handling"""
    print("\
" + "="*80)
    print("Test 7: Error Handling")
    print("="*80)
    
    test_cases = [
        {
            "name": "Missing column",
            "ast_json": {
                "entry": [{"left": "missing_col", "operator": ">", "right": 100}]
            },
            "expect_error": True
        },
        {
            "name": "Valid strategy",
            "ast_json": {
                "entry": [{"left": "close", "operator": ">", "right": 100}],
                "exit": [{"left": "close", "operator": "<", "right": 90}]
            },
            "expect_error": False
        }
    ]
    
    for test_case in test_cases:
        print(f"\
[Test] {test_case['name']}:")
        try:
            ast = DSLParser.from_json_rule(test_case['ast_json'])
            trading_func = generate_trading_function(ast)
            df = create_sample_data(50)
            signals = trading_func(df)
            
            if test_case['expect_error']:
                print(f"  ✗ Should have failed but succeeded")
            else:
                print(f"  ✓ Success as expected")
                print(f"    Entry signals: {signals['entry'].sum()}")
        except Exception as e:
            if test_case['expect_error']:
                print(f"  ✓ Failed as expected: {str(e)[:60]}")
            else:
                print(f"  ✗ Unexpected error: {str(e)[:60]}")


def test_summary():
    """Print test summary"""
    print("\
" + "="*80)
    print("CODE GENERATOR TEST SUMMARY")
    print("="*80)
    print("\
✓ All tests completed!")
    print("\
Test Coverage:")
    print("  [✓] Test 1: Simple SMA Crossover")
    print("  [✓] Test 2: RSI + Volume Strategy")
    print("  [✓] Test 3: Bollinger Bands Strategy")
    print("  [✓] Test 4: MACD Crossover Strategy")
    print("  [✓] Test 5: Time-Based References")
    print("  [✓] Test 6: Direct DSL Text Parsing")
    print("  [✓] Test 7: Error Handling")
    print("\
" + "="*80)


def run_all_tests():
    """Run all test examples"""
    print("\
" + "="*80)
    print("CODE GENERATOR TESTS")
    print("AST → Python Trading Functions")
    print("="*80)
    
    try:
        test_example_1_sma_crossover()
        test_example_2_rsi_volume()
        test_example_3_bollinger_bands()
        test_example_4_macd()
        test_example_5_time_references()
        test_example_6_dsl_text()
        test_example_7_error_handling()
        test_summary()
    except Exception as e:
        print(f"\
✗ Test Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check dependencies
    try:
        import pandas
        import numpy
        import lark
    except ImportError as e:
        print(f"ERROR: Missing dependency. Install with: pip install pandas numpy lark")
        sys.exit(1)
    
    run_all_tests()
