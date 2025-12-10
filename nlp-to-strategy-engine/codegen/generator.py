"""
Code Generator - Converts DSL AST to Python trading functions using TA-Lib.

Note: Indicators are implemented using TA-Lib and require `ta-lib` to be installed.
      Install with: pip install TA-Lib (after installing the C library)
"""
from typing import Callable, Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import sys
import os
import talib

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsl.ast_nodes import (
    ASTNode, Strategy, Series, Number, Indicator, TimeReference,
    Comparison, BooleanOp
)


class IndicatorCalculator:
    """Helper class to calculate technical indicators using TA-Lib"""
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average using TA-Lib"""
        result = talib.SMA(series.values, timeperiod=period)
        return pd.Series(result, index=series.index)
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average using TA-Lib"""
        result = talib.EMA(series.values, timeperiod=period)
        return pd.Series(result, index=series.index)
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index using TA-Lib"""
        result = talib.RSI(series.values, timeperiod=period)
        return pd.Series(result, index=series.index)
    
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence) using TA-Lib"""
        macd_line, signal_line, histogram = talib.MACD(
            series.values,
            fastperiod=fast,
            slowperiod=slow,
            signalperiod=signal
        )
        return (
            pd.Series(macd_line, index=series.index),
            pd.Series(signal_line, index=series.index),
            pd.Series(histogram, index=series.index)
        )
    
    @staticmethod
    def bb_upper(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """Bollinger Bands Upper using TA-Lib"""
        upper, middle, lower = talib.BBANDS(
            series.values,
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev,
            matype=0
        )
        return pd.Series(upper, index=series.index)
    
    @staticmethod
    def bb_middle(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """Bollinger Bands Middle using TA-Lib"""
        upper, middle, lower = talib.BBANDS(
            series.values,
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev,
            matype=0
        )
        return pd.Series(middle, index=series.index)
    
    @staticmethod
    def bb_lower(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """Bollinger Bands Lower using TA-Lib"""
        upper, middle, lower = talib.BBANDS(
            series.values,
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev,
            matype=0
        )
        return pd.Series(lower, index=series.index)
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range using TA-Lib"""
        result = talib.ATR(high.values, low.values, close.values, timeperiod=period)
        return pd.Series(result, index=close.index)
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index using TA-Lib"""
        result = talib.ADX(high.values, low.values, close.values, timeperiod=period)
        return pd.Series(result, index=close.index)
    
    @staticmethod
    def stoch(close: pd.Series, high: pd.Series, low: pd.Series, 
              period: int = 14, k_period: int = 3, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator using TA-Lib"""
        slowk, slowd = talib.STOCH(
            high.values,
            low.values,
            close.values,
            fastk_period=period,
            slowk_period=k_period,
            slowk_matype=0,
            slowd_period=d_period,
            slowd_matype=0
        )
        return (
            pd.Series(slowk, index=close.index),
            pd.Series(slowd, index=close.index)
        )
    
    @staticmethod
    def stoch_d(close: pd.Series, high: pd.Series, low: pd.Series,
                period: int = 14, k_period: int = 3, d_period: int = 3) -> pd.Series:
        """Stochastic %D line using TA-Lib"""
        _, d = IndicatorCalculator.stoch(close, high, low, period, k_period, d_period)
        return d


class CodeGenerator:
    """Converts DSL AST to Python trading logic"""
    
    def __init__(self):
        self.indicators_cache: Dict[str, pd.Series] = {}
        self.df: Optional[pd.DataFrame] = None
    
    def generate_function(self, ast: Strategy) -> Callable:
        """
        Generate a Python function from Strategy AST
        
        Args:
            ast: Strategy AST node
            
        Returns:
            Function that takes DataFrame and returns signals
        """
        def evaluate_signals(df: pd.DataFrame) -> Dict[str, pd.Series]:
            """
            Evaluate entry/exit signals on OHLCV data
            
            Args:
                df: DataFrame with columns: open, high, low, close, volume
                
            Returns:
                Dict with 'entry' and 'exit' boolean Series
            """
            self.df = df.copy()
            self.indicators_cache = {}
            
            # Ensure required columns
            required = {'open', 'high', 'low', 'close', 'volume'}
            if not required.issubset(set(df.columns)):
                raise ValueError(f"DataFrame missing columns. Required: {required}")
            
            # Convert all numeric columns to float64 (TA-Lib requirement)
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype('float64')
            
            # Generate entry signal
            entry_signal = self._evaluate_node(ast.entry)
            
            # Generate exit signal if present
            exit_signal = self._evaluate_node(ast.exit) if ast.exit else pd.Series(False, index=df.index)
            
            return {
                'entry': entry_signal,
                'exit': exit_signal,
                'signals_df': pd.DataFrame({
                    'entry': entry_signal,
                    'exit': exit_signal
                }, index=df.index)
            }
        
        return evaluate_signals
    
    def _evaluate_node(self, node: ASTNode) -> pd.Series:
        """Recursively evaluate AST node to boolean Series"""
        if isinstance(node, Comparison):
            return self._evaluate_comparison(node)
        elif isinstance(node, BooleanOp):
            return self._evaluate_boolean_op(node)
        else:
            raise ValueError(f"Cannot evaluate node type: {type(node)}")
    
    def _evaluate_comparison(self, node: Comparison) -> pd.Series:
        """Evaluate comparison expression"""
        left = self._evaluate_expression(node.left)
        right = self._evaluate_expression(node.right)
        
        op = node.operator.lower()
        
        if op == ">":
            return left > right
        elif op == "<":
            return left < right
        elif op == ">=":
            return left >= right
        elif op == "<=":
            return left <= right
        elif op == "==":
            return left == right
        elif op == "!=":
            return left != right
        elif op == "crosses_above":
            return (left > right) & (left.shift(1) <= right.shift(1))
        elif op == "crosses_below":
            return (left < right) & (left.shift(1) >= right.shift(1))
        elif op == "crosses":
            crosses_above = (left > right) & (left.shift(1) <= right.shift(1))
            crosses_below = (left < right) & (left.shift(1) >= right.shift(1))
            return crosses_above | crosses_below
        elif op == "touches":
            # Touches with ±5% tolerance
            tolerance = right * 0.05
            return (left >= (right - tolerance)) & (left <= (right + tolerance))
        else:
            raise ValueError(f"Unknown operator: {op}")
    
    def _evaluate_boolean_op(self, node: BooleanOp) -> pd.Series:
        """Evaluate boolean operation (AND/OR)"""
        left = self._evaluate_node(node.left)
        right = self._evaluate_node(node.right)
        
        if node.operator.upper() == "AND":
            return left & right
        elif node.operator.upper() == "OR":
            return left | right
        else:
            raise ValueError(f"Unknown boolean operator: {node.operator}")
    
    def _evaluate_expression(self, expr: ASTNode) -> pd.Series:
        """Evaluate expression to get Series values"""
        if isinstance(expr, Series):
            return self._evaluate_series(expr)
        elif isinstance(expr, Number):
            return self._evaluate_number(expr)
        elif isinstance(expr, Indicator):
            return self._evaluate_indicator(expr)
        elif isinstance(expr, TimeReference):
            return self._evaluate_time_ref(expr)
        else:
            raise ValueError(f"Unknown expression type: {type(expr)}")
    
    def _evaluate_series(self, node: Series) -> pd.Series:
        """Get series from DataFrame"""
        if node.name not in self.df.columns:
            raise ValueError(f"Column not found: {node.name}")
        return self.df[node.name]
    
    def _evaluate_number(self, node: Number) -> pd.Series:
        """Create constant Series from number"""
        return pd.Series(node.value, index=self.df.index)
    
    def _evaluate_indicator(self, node: Indicator) -> pd.Series:
        """Calculate technical indicator using TA-Lib"""
        cache_key = f"{node.name}({','.join(str(p) for p in node.params)})"
        
        if cache_key in self.indicators_cache:
            return self.indicators_cache[cache_key]
        
        name = node.name.lower()
        
        if name == "sma":
            series_name, period = node.params[0], int(node.params[1])
            result = IndicatorCalculator.sma(self.df[series_name], period)
        elif name == "ema":
            series_name, period = node.params[0], int(node.params[1])
            result = IndicatorCalculator.ema(self.df[series_name], period)
        elif name == "rsi":
            series_name, period = node.params[0], int(node.params[1])
            result = IndicatorCalculator.rsi(self.df[series_name], period)
        elif name == "macd":
            series_name, fast, slow, signal_p = (node.params[0], int(node.params[1]), 
                                                  int(node.params[2]), int(node.params[3]))
            macd_line, _, _ = IndicatorCalculator.macd(self.df[series_name], fast, slow, signal_p)
            result = macd_line
        elif name == "macd_signal":
            series_name, fast, slow, signal_p = (node.params[0], int(node.params[1]), 
                                                  int(node.params[2]), int(node.params[3]))
            _, signal_line, _ = IndicatorCalculator.macd(self.df[series_name], fast, slow, signal_p)
            result = signal_line
        elif name == "macd_histogram":
            series_name, fast, slow, signal_p = (node.params[0], int(node.params[1]), 
                                                  int(node.params[2]), int(node.params[3]))
            _, _, histogram = IndicatorCalculator.macd(self.df[series_name], fast, slow, signal_p)
            result = histogram
        elif name == "bb_upper":
            series_name, period, std = node.params[0], int(node.params[1]), float(node.params[2])
            result = IndicatorCalculator.bb_upper(self.df[series_name], period, std)
        elif name == "bb_middle":
            series_name, period, std = node.params[0], int(node.params[1]), float(node.params[2])
            result = IndicatorCalculator.bb_middle(self.df[series_name], period, std)
        elif name == "bb_lower":
            series_name, period, std = node.params[0], int(node.params[1]), float(node.params[2])
            result = IndicatorCalculator.bb_lower(self.df[series_name], period, std)
        elif name == "atr":
            period = int(node.params[0])
            result = IndicatorCalculator.atr(self.df['high'], self.df['low'], self.df['close'], period)
        elif name == "adx":
            period = int(node.params[0])
            result = IndicatorCalculator.adx(self.df['high'], self.df['low'], self.df['close'], period)
        elif name == "stoch":
            series_name, period, k_p, d_p = (node.params[0], int(node.params[1]), 
                                             int(node.params[2]), int(node.params[3]))
            k, _ = IndicatorCalculator.stoch(self.df[series_name], self.df['high'], 
                                            self.df['low'], period, k_p, d_p)
            result = k
        elif name == "stoch_d":
            series_name, period, k_p, d_p = (node.params[0], int(node.params[1]), 
                                             int(node.params[2]), int(node.params[3]))
            result = IndicatorCalculator.stoch_d(self.df[series_name], self.df['high'], 
                                                self.df['low'], period, k_p, d_p)
        else:
            raise ValueError(f"Unknown indicator: {name}")
        
        self.indicators_cache[cache_key] = result
        return result
    
    def _evaluate_time_ref(self, node: TimeReference) -> pd.Series:
        """Evaluate time reference (e.g., close_prev, high_1d_ago)"""
        series = self.df[node.series]
        
        lag = node.lag.lower()
        
        if lag == "prev":
            return series.shift(1)
        elif lag.endswith("d_ago"):
            days = int(lag.replace("d_ago", ""))
            return series.shift(days)
        elif lag.endswith("w_ago"):
            weeks = int(lag.replace("w_ago", ""))
            return series.shift(weeks * 5)  # Assume 5 trading days per week
        elif lag.endswith("m_ago"):
            months = int(lag.replace("m_ago", ""))
            return series.shift(months * 21)  # Assume 21 trading days per month
        else:
            # Try parsing as numeric lag
            try:
                shift_amount = int(lag)
                return series.shift(shift_amount)
            except ValueError:
                raise ValueError(f"Unknown time lag: {lag}")


def generate_trading_function(ast: Strategy) -> Callable:
    """
    Convenience function to generate trading function from AST
    
    Args:
        ast: Strategy AST node
        
    Returns:
        Function that evaluates signals on DataFrame
    """
    generator = CodeGenerator()
    return generator.generate_function(ast)
