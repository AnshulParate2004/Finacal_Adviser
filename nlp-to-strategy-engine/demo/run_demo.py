"""
Complete Demo: NLP → JSON → DSL → Backtest Pipeline
Demonstrates the full flow from natural language to backtest results
"""
import sys
import os
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nlp_parser import parse_trading_rule
from dsl import DSLParser, validate_dsl
from codegen import generate_trading_function
from backtester import BacktestEngine


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(title)
    print("="*80)


def run_complete_demo():
    """Run complete pipeline demo"""
    
    print_section("NLP-TO-STRATEGY-ENGINE COMPLETE DEMO")
    print("\nThis demo shows the complete flow:")
    print("  1. Natural Language → Structured JSON (NLP Parser)")
    print("  2. JSON → DSL AST (DSL Parser)")
    print("  3. DSL → Trading Signals (Code Generator)")
    print("  4. Signals → Backtest Results (Backtest Engine)")
    
    # Example natural language rules
    examples = [
        {
            "name": "Simple SMA Crossover",
            "text": "Buy when close crosses above 10-day SMA. Sell when close crosses below 10-day SMA."
        },
        {
            "name": "RSI + Volume Strategy",
            "text": "Buy when close is above 20-day SMA and RSI is above 50 and volume is above 1.5 million. Exit when RSI drops below 30."
        },
        {
            "name": "Bollinger Bands Mean Reversion",
            "text": "Enter long when close is below lower Bollinger Band and RSI is below 30. Exit when close crosses above upper Bollinger Band."
        }
    ]
    
    # Load sample data
    data_path = Path(__file__).parent.parent / 'data' / 'sample_data.csv'
    
    print_section("STEP 0: Load Market Data")
    try:
        data = pd.read_csv(data_path, index_col='date', parse_dates=True)
        
        # Convert to float64 for TA-Lib
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = data[col].astype('float64')
        
        print(f"✓ Loaded {len(data)} bars from {data.index[0].date()} to {data.index[-1].date()}")
        print(f"\nSample data:")
        print(data.head())
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Run each example
    for i, example in enumerate(examples, 1):
        print_section(f"EXAMPLE {i}: {example['name']}")
        print(f"\nNatural Language Rule:")
        print(f'  "{example["text"]}"')
        
        try:
            # Step 1: NLP → JSON
            print("\n[STEP 1] Natural Language → JSON")
            parsed_strategy = parse_trading_rule(example['text'])
            
            print(f"✓ Parsed successfully")
            print(f"  Indicators used: {', '.join(parsed_strategy.indicators_used) if parsed_strategy.indicators_used else 'None'}")
            print(f"  Complexity: {parsed_strategy.complexity}")
            print(f"\nStructured JSON:")
            print(json.dumps(parsed_strategy.rule.dict(), indent=2))
            
            # Step 2: JSON → DSL AST
            print("\n[STEP 2] JSON → DSL AST")
            ast = DSLParser.from_json_rule(parsed_strategy.rule.dict())
            
            # Validate AST
            is_valid, errors, warnings = validate_dsl(ast)
            
            if not is_valid:
                print(f"✗ Validation failed: {errors}")
                continue
            
            print(f"✓ AST created and validated")
            if warnings:
                print(f"  Warnings: {warnings}")
            
            # Step 3: DSL → Trading Signals
            print("\n[STEP 3] Generate Trading Signals")
            trading_func = generate_trading_function(ast)
            signals = trading_func(data)
            
            entry_count = int(signals['entry'].sum())
            exit_count = int(signals['exit'].sum())
            
            print(f"✓ Signals generated")
            print(f"  Entry signals: {entry_count}")
            print(f"  Exit signals: {exit_count}")
            
            # Step 4: Signals → Backtest
            print("\n[STEP 4] Run Backtest Simulation")
            engine = BacktestEngine(initial_capital=10000.0, position_size=1.0)
            result = engine.run(data, ast)
            
            print(f"✓ Backtest completed")
            print(f"  Trades executed: {result.total_trades}")
            
            # Print results
            print("\n[STEP 5] Backtest Results")
            print(f"\nPerformance Metrics:")
            print(f"  Total Trades: {result.total_trades}")
            print(f"  Win Rate: {result.win_rate:.2f}%")
            print(f"  Total Profit: ${result.total_profit:.2f}")
            print(f"  Total Return: {result.total_return_pct:.2f}%")
            print(f"  Max Drawdown: {result.max_drawdown:.2f}%")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.4f}")
            
            if result.trades:
                print(f"\nTrade History (first 3):")
                for j, trade in enumerate(result.trades[:3], 1):
                    print(f"  {j}. {trade.entry_date} → {trade.exit_date}")
                    print(f"     ${trade.entry_price:.2f} → ${trade.exit_price:.2f}")
                    print(f"     P&L: ${trade.profit:+.2f} ({trade.return_pct:+.2f}%)")
            
            print(f"\n✓ Pipeline completed successfully for '{example['name']}'")
            
        except Exception as e:
            print(f"\n✗ Error in pipeline: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print_section("DEMO COMPLETE")
    print("\n✓ All examples processed")
    print("\nThe pipeline successfully:")
    print("  • Parsed natural language trading rules")
    print("  • Converted to structured JSON")
    print("  • Built DSL abstract syntax trees")
    print("  • Generated trading signals")
    print("  • Executed backtest simulations")
    print("  • Calculated performance metrics")


def run_interactive_demo():
    """Interactive demo - user can input their own rule"""
    print_section("INTERACTIVE MODE - Enter Your Trading Rule")
    
    print("\nEnter a natural language trading rule:")
    print("Example: 'Buy when RSI is above 70 and volume is high. Sell when RSI drops below 30.'")
    print("\nYour rule: ", end='')
    
    user_input = input().strip()
    
    if not user_input:
        print("No input provided. Exiting.")
        return
    
    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'sample_data.csv'
    
    try:
        data = pd.read_csv(data_path, index_col='date', parse_dates=True)
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = data[col].astype('float64')
        
        print(f"\n✓ Data loaded: {len(data)} bars")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    try:
        # Parse
        print("\n[1/4] Parsing natural language...")
        parsed_strategy = parse_trading_rule(user_input)
        print(f"✓ Parsed: {len(parsed_strategy.rule.entry)} entry conditions, {len(parsed_strategy.rule.exit)} exit conditions")
        
        # Convert to AST
        print("\n[2/4] Building DSL AST...")
        ast = DSLParser.from_json_rule(parsed_strategy.rule.dict())
        is_valid, errors, warnings = validate_dsl(ast)
        
        if not is_valid:
            print(f"✗ Validation failed: {errors}")
            return
        
        print(f"✓ AST validated")
        
        # Generate signals
        print("\n[3/4] Generating signals...")
        trading_func = generate_trading_function(ast)
        signals = trading_func(data)
        print(f"✓ Entry: {int(signals['entry'].sum())}, Exit: {int(signals['exit'].sum())}")
        
        # Backtest
        print("\n[4/4] Running backtest...")
        engine = BacktestEngine(initial_capital=10000.0, position_size=1.0)
        result = engine.run(data, ast)
        
        print("\n" + "="*80)
        print("BACKTEST RESULTS")
        print("="*80)
        print(f"\nTrades: {result.total_trades}")
        print(f"Win Rate: {result.win_rate:.2f}%")
        print(f"Total Return: {result.total_return_pct:.2f}%")
        print(f"Profit: ${result.total_profit:.2f}")
        print(f"Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.4f}")
        
        if result.trades:
            print(f"\nTrades:")
            for i, trade in enumerate(result.trades, 1):
                print(f"  {i}. ${trade.entry_price:.2f} → ${trade.exit_price:.2f} "
                      f"({trade.return_pct:+.2f}%)")
        
        print("\n✓ Complete!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point"""
    print("\nNLP-TO-STRATEGY-ENGINE DEMO")
    print("="*80)
    print("\nChoose demo mode:")
    print("  1. Complete Demo (run predefined examples)")
    print("  2. Interactive Mode (enter your own rule)")
    print("  3. Both")
    print("\nChoice (1/2/3): ", end='')
    
    choice = input().strip()
    
    if choice == '1':
        run_complete_demo()
    elif choice == '2':
        run_interactive_demo()
    elif choice == '3':
        run_complete_demo()
        run_interactive_demo()
    else:
        print("Invalid choice. Running complete demo...")
        run_complete_demo()


if __name__ == "__main__":
    main()
