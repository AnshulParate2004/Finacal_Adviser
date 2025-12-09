"""
Backtest Examples and Tests
Demonstrates running backtests on trading strategies
"""
import sys
import os
import pandas as pd
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsl import DSLParser, parse_dsl
from backtester import BacktestEngine


def load_sample_data() -> pd.DataFrame:
    """Load sample OHLCV data"""
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sample_data.csv')
    df = pd.read_csv(data_path, index_col='date', parse_dates=True)
    return df


def test_backtest_1_sma_crossover():
    """Test 1: SMA Crossover Strategy"""
    print("\n" + "="*80)
    print("Test 1: SMA Crossover Strategy Backtest")
    print("="*80)
    
    # Load data
    df = load_sample_data()
    print(f"\n[Data] Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # Create strategy
    json_rule = {
        "entry": [
            {"left": "close", "operator": "crosses_above", "right": "sma(close,10)"}
        ],
        "exit": [
            {"left": "close", "operator": "crosses_below", "right": "sma(close,10)"}
        ]
    }
    
    print("\n[Strategy] SMA(10) Crossover")
    print("  Entry: close crosses above SMA(10)")
    print("  Exit: close crosses below SMA(10)")
    
    # Convert to AST
    ast = DSLParser.from_json_rule(json_rule)
    
    # Run backtest
    print("\n[Backtest] Running...")
    engine = BacktestEngine(initial_capital=10000.0, position_size=1.0)
    result = engine.run(df, ast)
    
    # Print summary
    engine.print_summary()
    
    print(f"\n[Results] JSON:")
    print(json.dumps(result.to_dict(), indent=2, default=str)[:500] + "...")


def test_backtest_2_rsi_strategy():
    """Test 2: RSI Strategy"""
    print("\n" + "="*80)
    print("Test 2: RSI Strategy Backtest")
    print("="*80)
    
    # Load data
    df = load_sample_data()
    
    # Create strategy
    json_rule = {
        "entry": [
            {"left": "rsi(close,14)", "operator": "<", "right": 30}
        ],
        "exit": [
            {"left": "rsi(close,14)", "operator": ">", "right": 70}
        ]
    }
    
    print("\n[Strategy] RSI Oversold/Overbought")
    print("  Entry: RSI(14) < 30 (oversold)")
    print("  Exit: RSI(14) > 70 (overbought)")
    
    # Convert to AST
    ast = DSLParser.from_json_rule(json_rule)
    
    # Run backtest
    print("\n[Backtest] Running...")
    engine = BacktestEngine(initial_capital=10000.0, position_size=1.0)
    result = engine.run(df, ast)
    
    # Print summary
    engine.print_summary()


def test_backtest_3_dsl_text():
    """Test 3: DSL Text Parsing and Backtest"""
    print("\n" + "="*80)
    print("Test 3: DSL Text Strategy Backtest")
    print("="*80)
    
    # Load data
    df = load_sample_data()
    
    # Create strategy in DSL
    dsl = """
    ENTRY:
        close > sma(close, 20) AND volume > 1200000
    EXIT:
        close < sma(close, 20)
    """
    
    print("\n[Strategy] SMA + Volume Filter")
    print("  Entry: close > SMA(20) AND volume > 1.2M")
    print("  Exit: close < SMA(20)")
    
    # Parse DSL
    ast = parse_dsl(dsl)
    
    # Run backtest
    print("\n[Backtest] Running...")
    engine = BacktestEngine(initial_capital=10000.0, position_size=1.0)
    result = engine.run(df, ast)
    
    # Print summary
    engine.print_summary()


def test_backtest_4_bollinger_bands():
    """Test 4: Bollinger Bands Strategy"""
    print("\n" + "="*80)
    print("Test 4: Bollinger Bands Strategy Backtest")
    print("="*80)
    
    # Load data
    df = load_sample_data()
    
    # Create strategy
    json_rule = {
        "entry": [
            {"left": "close", "operator": "<", "right": "bb_lower(close,20,2)"}
        ],
        "exit": [
            {"left": "close", "operator": ">", "right": "bb_middle(close,20,2)"}
        ]
    }
    
    print("\n[Strategy] Bollinger Bands Reversal")
    print("  Entry: close < BB_Lower(20, 2std)")
    print("  Exit: close > BB_Middle(20, 2std)")
    
    # Convert to AST
    ast = DSLParser.from_json_rule(json_rule)
    
    # Run backtest
    print("\n[Backtest] Running...")
    engine = BacktestEngine(initial_capital=10000.0, position_size=1.0)
    result = engine.run(df, ast)
    
    # Print summary
    engine.print_summary()


def test_backtest_5_complex_strategy():
    """Test 5: Complex Multi-Indicator Strategy"""
    print("\n" + "="*80)
    print("Test 5: Complex Multi-Indicator Strategy Backtest")
    print("="*80)
    
    # Load data
    df = load_sample_data()
    
    # Create complex strategy
    json_rule = {
        "entry": [
            {"left": "close", "operator": ">", "right": "sma(close,20)"},
            {"left": "rsi(close,14)", "operator": ">", "right": 50},
            {"left": "volume", "operator": ">", "right": 1500000}
        ],
        "exit": [
            {"left": "rsi(close,14)", "operator": "<", "right": 30}
        ]
    }
    
    print("\n[Strategy] SMA + RSI + Volume")
    print("  Entry: close > SMA(20) AND RSI > 50 AND volume > 1.5M")
    print("  Exit: RSI < 30")
    
    # Convert to AST
    ast = DSLParser.from_json_rule(json_rule)
    
    # Run backtest
    print("\n[Backtest] Running...")
    engine = BacktestEngine(initial_capital=10000.0, position_size=1.0)
    result = engine.run(df, ast)
    
    # Print summary
    engine.print_summary()


def test_backtest_summary():
    """Summary of all backtests"""
    print("\n" + "="*80)
    print("BACKTEST SUMMARY")
    print("="*80)
    print("\n✓ All backtest examples completed!")
    print("\nTest Coverage:")
    print("  [✓] Test 1: SMA Crossover Strategy")
    print("  [✓] Test 2: RSI Strategy")
    print("  [✓] Test 3: DSL Text Strategy")
    print("  [✓] Test 4: Bollinger Bands Strategy")
    print("  [✓] Test 5: Complex Multi-Indicator Strategy")
    print("\n" + "="*80)


def run_all_backtests():
    """Run all backtest examples"""
    print("\n" + "="*80)
    print("BACKTEST ENGINE TESTS")
    print("Strategy Execution Simulation")
    print("="*80)
    
    try:
        test_backtest_1_sma_crossover()
        test_backtest_2_rsi_strategy()
        test_backtest_3_dsl_text()
        test_backtest_4_bollinger_bands()
        test_backtest_5_complex_strategy()
        test_backtest_summary()
    except Exception as e:
        print(f"\n✗ Backtest Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check dependencies
    try:
        import pandas
        import numpy
    except ImportError:
        print("ERROR: Missing dependencies. Install with: pip install pandas numpy")
        sys.exit(1)
    
    run_all_backtests()
