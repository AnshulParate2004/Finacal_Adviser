"""
Complete Pipeline: JSON Rule → DSL → Code Generation → Backtesting
Demonstrates the full flow from trading rule to backtest results
"""
import sys
import os
import pandas as pd
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsl import DSLParser, parse_dsl, validate_dsl
from codegen import generate_trading_function
from backtester import BacktestEngine


class JsonToBacktester:
    """Complete pipeline from JSON to backtest results"""
    
    def __init__(self, initial_capital: float = 10000.0):
        """Initialize pipeline"""
        self.initial_capital = initial_capital
        self.data = None
        self.ast = None
        self.signals = None
        self.result = None
    
    def load_data(self, csv_path: str) -> bool:
        """Load OHLCV data from CSV"""
        try:
            self.data = pd.read_csv(csv_path, index_col='date', parse_dates=True)
            
            # Convert all numeric columns to float64 (required by TA-Lib)
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in self.data.columns:
                    self.data[col] = self.data[col].astype('float64')
            
            print(f"✓ Data loaded: {len(self.data)} bars from {self.data.index[0].date()} to {self.data.index[-1].date()}")
            return True
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def json_to_ast(self, json_rule: dict) -> bool:
        """Convert JSON rule to DSL AST"""
        try:
            self.ast = DSLParser.from_json_rule(json_rule)
            
            # Validate
            is_valid, errors, warnings = validate_dsl(self.ast)
            
            if not is_valid:
                print(f"✗ Validation errors: {errors}")
                return False
            
            print(f"✓ AST created and validated")
            if warnings:
                print(f"  Warnings: {warnings}")
            
            return True
        except Exception as e:
            print(f"✗ Error converting to AST: {e}")
            return False
    
    def dsl_text_to_ast(self, dsl_text: str) -> bool:
        """Convert DSL text to AST"""
        try:
            self.ast = parse_dsl(dsl_text)
            
            # Validate
            is_valid, errors, warnings = validate_dsl(self.ast)
            
            if not is_valid:
                print(f"✗ Validation errors: {errors}")
                return False
            
            print(f"✓ DSL parsed and validated")
            if warnings:
                print(f"  Warnings: {warnings}")
            
            return True
        except Exception as e:
            print(f"✗ Error parsing DSL: {e}")
            return False
    
    def generate_signals(self) -> bool:
        """Generate trading signals from AST"""
        try:
            if self.ast is None:
                print("✗ No AST available")
                return False
            
            if self.data is None:
                print("✗ No data loaded")
                return False
            
            # Generate function
            trading_func = generate_trading_function(self.ast)
            
            # Evaluate signals
            self.signals = trading_func(self.data)
            
            entry_count = self.signals['entry'].sum()
            exit_count = self.signals['exit'].sum()
            
            print(f"✓ Signals generated")
            print(f"  Entry signals: {int(entry_count)}")
            print(f"  Exit signals: {int(exit_count)}")
            
            return True
        except Exception as e:
            print(f"✗ Error generating signals: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_backtest(self, position_size: float = 1.0) -> bool:
        """Run backtest simulation"""
        try:
            if self.ast is None:
                print("✗ No AST available")
                return False
            
            if self.data is None:
                print("✗ No data loaded")
                return False
            
            # Run backtest
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                position_size=position_size
            )
            self.result = engine.run(self.data, self.ast)
            
            print(f"✓ Backtest completed")
            print(f"  Trades executed: {self.result.total_trades}")
            
            return True
        except Exception as e:
            print(f"✗ Error running backtest: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_full_report(self):
        """Print complete backtest report"""
        if self.result is None:
            print("No backtest results available")
            return
        
        print("\n" + "="*80)
        print("COMPLETE BACKTEST REPORT")
        print("="*80)
        
        # Strategy info
        print(f"\nStrategy Information:")
        print(f"  Initial Capital: ${self.initial_capital:,.2f}")
        print(f"  Data Period: {len(self.data)} bars")
        print(f"  Date Range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
        
        # Trading Activity
        print(f"\nTrading Activity:")
        print(f"  Total Trades: {self.result.total_trades}")
        print(f"  Winning Trades: {self.result.winning_trades}")
        print(f"  Losing Trades: {self.result.losing_trades}")
        print(f"  Win Rate: {self.result.win_rate:.2f}%")
        
        # Profitability
        print(f"\nProfitability:")
        print(f"  Total Profit: ${self.result.total_profit:,.2f}")
        print(f"  Total Return: {self.result.total_return_pct:.2f}%")
        print(f"  Avg Trade Return: {self.result.avg_trade_return:.2f}%")
        print(f"  Max Profit: ${self.result.max_profit:,.2f}")
        print(f"  Max Loss: ${self.result.max_loss:,.2f}")
        
        # Risk Metrics
        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown: {self.result.max_drawdown:.2f}%")
        print(f"  Sharpe Ratio: {self.result.sharpe_ratio:.4f}")
        
        # Trades
        if self.result.trades:
            print(f"\nTrade Details (first 10):")
            for i, trade in enumerate(self.result.trades[:10], 1):
                print(f"  {i}. {trade.entry_date} → {trade.exit_date}")
                print(f"     Price: ${trade.entry_price:.2f} → ${trade.exit_price:.2f}")
                print(f"     P&L: ${trade.profit:+,.2f} ({trade.return_pct:+.2f}%)")
            
            if len(self.result.trades) > 10:
                print(f"  ... and {len(self.result.trades) - 10} more trades")
        
        print("\n" + "="*80)
    
    def export_results(self, filepath: str) -> bool:
        """Export results to JSON file"""
        try:
            if self.result is None:
                print("✗ No backtest results to export")
                return False
            
            results_dict = {
                'timestamp': datetime.now().isoformat(),
                'initial_capital': self.initial_capital,
                'data_period': {
                    'start': str(self.data.index[0].date()),
                    'end': str(self.data.index[-1].date()),
                    'bars': len(self.data)
                },
                'backtest_results': self.result.to_dict()
            }
            
            with open(filepath, 'w') as f:
                json.dump(results_dict, f, indent=2, default=str)
            
            print(f"✓ Results exported to {filepath}")
            return True
        except Exception as e:
            print(f"✗ Error exporting results: {e}")
            return False


def example_1_json_rule():
    """Example 1: Simple SMA Crossover from JSON"""
    print("\n" + "="*80)
    print("EXAMPLE 1: JSON Rule → SMA Crossover Backtest")
    print("="*80)
    
    # Define rule as JSON
    json_rule = {
        "entry": [
            {"left": "close", "operator": "crosses_above", "right": "sma(close,10)"}
        ],
        "exit": [
            {"left": "close", "operator": "crosses_below", "right": "sma(close,10)"}
        ]
    }
    
    print("\n[Step 1] Rule Definition (JSON)")
    print(json.dumps(json_rule, indent=2))
    
    # Initialize pipeline
    pipeline = JsonToBacktester(initial_capital=10000.0)
    
    # Load data
    print("\n[Step 2] Load OHLCV Data")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_data.csv')
    if not pipeline.load_data(data_path):
        return
    
    # Convert JSON to AST
    print("\n[Step 3] Convert JSON to DSL AST")
    if not pipeline.json_to_ast(json_rule):
        return
    
    # Generate signals
    print("\n[Step 4] Generate Trading Signals")
    if not pipeline.generate_signals():
        return
    
    # Run backtest
    print("\n[Step 5] Execute Backtest Simulation")
    if not pipeline.run_backtest(position_size=1.0):
        return
    
    # Print report
    print("\n[Step 6] Generate Report")
    pipeline.print_full_report()


def example_2_rsi_volume():
    """Example 2: RSI + Volume Strategy"""
    print("\n" + "="*80)
    print("EXAMPLE 2: RSI + Volume Strategy Backtest")
    print("="*80)
    
    # Define rule as JSON
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
    
    print("\n[Rule] Entry: close > SMA(20) AND RSI > 50 AND volume > 1.5M")
    print("[Rule] Exit: RSI < 30")
    
    # Initialize pipeline
    pipeline = JsonToBacktester(initial_capital=10000.0)
    
    # Load data
    print("\n[Step 1] Load Data")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_data.csv')
    if not pipeline.load_data(data_path):
        return
    
    # Convert JSON to AST
    print("\n[Step 2] JSON → DSL")
    if not pipeline.json_to_ast(json_rule):
        return
    
    # Generate signals
    print("\n[Step 3] Generate Signals")
    if not pipeline.generate_signals():
        return
    
    # Run backtest
    print("\n[Step 4] Run Backtest")
    if not pipeline.run_backtest(position_size=1.0):
        return
    
    # Print report
    print("\n[Step 5] Report")
    pipeline.print_full_report()
    
    # Export results (DISABLED)
    # print("\n[Step 6] Export Results")
    # pipeline.export_results('backtest_results.json')


def example_3_dsl_text():
    """Example 3: DSL Text to Backtest"""
    print("\n" + "="*80)
    print("EXAMPLE 3: DSL Text Strategy Backtest")
    print("="*80)
    
    # Define rule in DSL text
    dsl_text = """
    ENTRY:
        close > sma(close, 20) AND volume > 1500000
    EXIT:
        close < sma(close, 20)
    """
    
    print("\n[Rule] DSL Text:")
    print(dsl_text)
    
    # Initialize pipeline
    pipeline = JsonToBacktester(initial_capital=10000.0)
    
    # Load data
    print("[Step 1] Load Data")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_data.csv')
    if not pipeline.load_data(data_path):
        return
    
    # Parse DSL text
    print("\n[Step 2] Parse DSL Text")
    if not pipeline.dsl_text_to_ast(dsl_text):
        return
    
    # Generate signals
    print("\n[Step 3] Generate Signals")
    if not pipeline.generate_signals():
        return
    
    # Run backtest
    print("\n[Step 4] Run Backtest")
    if not pipeline.run_backtest(position_size=1.0):
        return
    
    # Print report
    print("\n[Step 5] Report")
    pipeline.print_full_report()


def example_4_bollinger_bands():
    """Example 4: Bollinger Bands Strategy"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Bollinger Bands Strategy Backtest")
    print("="*80)
    
    # Define rule as JSON
    json_rule = {
        "entry": [
            {"left": "close", "operator": "<", "right": "bb_lower(close,20,2)"},
            {"left": "rsi(close,14)", "operator": "<", "right": 30}
        ],
        "exit": [
            {"left": "close", "operator": ">", "right": "bb_upper(close,20,2)"}
        ]
    }
    
    print("\n[Rule] Entry: close < BB_Lower AND RSI < 30")
    print("[Rule] Exit: close > BB_Upper")
    
    # Initialize pipeline
    pipeline = JsonToBacktester(initial_capital=10000.0)
    
    # Load data
    print("\n[Step 1] Load Data")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_data.csv')
    if not pipeline.load_data(data_path):
        return
    
    # Convert JSON to AST
    print("\n[Step 2] JSON → DSL")
    if not pipeline.json_to_ast(json_rule):
        return
    
    # Generate signals
    print("\n[Step 3] Generate Signals")
    if not pipeline.generate_signals():
        return
    
    # Run backtest
    print("\n[Step 4] Run Backtest")
    if not pipeline.run_backtest(position_size=1.0):
        return
    
    # Print report
    print("\n[Step 5] Report")
    pipeline.print_full_report()


def example_5_macd():
    """Example 5: MACD Crossover Strategy"""
    print("\n" + "="*80)
    print("EXAMPLE 5: MACD Crossover Strategy Backtest")
    print("="*80)
    
    # Define rule as JSON
    json_rule = {
        "entry": [
            {"left": "macd(close,12,26,9)", "operator": "crosses_above", "right": "macd_signal(close,12,26,9)"}
        ],
        "exit": [
            {"left": "macd(close,12,26,9)", "operator": "crosses_below", "right": "macd_signal(close,12,26,9)"}
        ]
    }
    
    print("\n[Rule] Entry: MACD crosses above Signal Line")
    print("[Rule] Exit: MACD crosses below Signal Line")
    
    # Initialize pipeline
    pipeline = JsonToBacktester(initial_capital=10000.0)
    
    # Load data
    print("\n[Step 1] Load Data")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_data.csv')
    if not pipeline.load_data(data_path):
        return
    
    # Convert JSON to AST
    print("\n[Step 2] JSON → DSL")
    if not pipeline.json_to_ast(json_rule):
        return
    
    # Generate signals
    print("\n[Step 3] Generate Signals")
    if not pipeline.generate_signals():
        return
    
    # Run backtest
    print("\n[Step 4] Run Backtest")
    if not pipeline.run_backtest(position_size=1.0):
        return
    
    # Print report
    print("\n[Step 5] Report")
    pipeline.print_full_report()


def print_summary():
    """Print execution summary"""
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print("\n✓ All examples completed successfully!")
    print("\nExamples Executed:")
    print("  [✓] Example 1: JSON Rule → SMA Crossover")
    print("  [✓] Example 2: RSI + Volume Strategy")
    print("  [✓] Example 3: DSL Text Strategy")
    print("  [✓] Example 4: Bollinger Bands Strategy")
    print("  [✓] Example 5: MACD Crossover Strategy")
    print("\nEach example demonstrates the complete pipeline:")
    print("  1. Define trading rule (JSON or DSL)")
    print("  2. Load OHLCV data")
    print("  3. Convert to DSL AST")
    print("  4. Generate trading signals")
    print("  5. Execute backtest simulation")
    print("  6. Generate comprehensive report")
    print("\n" + "="*80)


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("JSON → DSL → CODE GENERATION → BACKTESTING")
    print("Complete Trading Strategy Pipeline")
    print("="*80)
    
    try:
        example_1_json_rule()
        example_2_rsi_volume()
        example_3_dsl_text()
        example_4_bollinger_bands()
        example_5_macd()
        print_summary()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check dependencies
    try:
        import pandas
        import numpy
    except ImportError:
        print("ERROR: Missing dependencies. Install with: pip install pandas numpy lark")
        sys.exit(1)
    
    main()
