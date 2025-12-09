"""Backtesting Engine - Simulates trading strategy execution"""
import sys
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsl import Strategy
from codegen import generate_trading_function


@dataclass
class Trade:
    """Represents a single trade"""
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    quantity: float = 1.0
    
    @property
    def profit(self) -> float:
        """Calculate absolute profit"""
        return (self.exit_price - self.entry_price) * self.quantity
    
    @property
    def return_pct(self) -> float:
        """Calculate return percentage"""
        if self.entry_price == 0:
            return 0
        return ((self.exit_price - self.entry_price) / self.entry_price) * 100
    
    def to_dict(self) -> Dict:
        """Convert trade to dictionary"""
        return {
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'exit_date': self.exit_date,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'profit': self.profit,
            'return_pct': self.return_pct
        }


@dataclass
class BacktestResult:
    """Results of backtest execution"""
    trades: List[Trade]
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    total_return_pct: float
    avg_trade_return: float
    max_profit: float
    max_loss: float
    max_drawdown: float
    sharpe_ratio: float
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_profit': self.total_profit,
            'total_return_pct': self.total_return_pct,
            'avg_trade_return': self.avg_trade_return,
            'max_profit': self.max_profit,
            'max_loss': self.max_loss,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'trades': [t.to_dict() for t in self.trades]
        }


class BacktestEngine:
    """Executes backtest simulation based on trading signals"""
    
    def __init__(self, initial_capital: float = 10000.0, position_size: float = 1.0):
        """
        Initialize backtest engine
        
        Args:
            initial_capital: Starting capital
            position_size: Position size per trade (1.0 = full available capital)
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.trades: List[Trade] = []
        self.df: Optional[pd.DataFrame] = None
        self.signals: Optional[Dict] = None
    
    def run(self, df: pd.DataFrame, ast: Strategy) -> BacktestResult:
        """
        Run backtest on DataFrame with strategy AST
        
        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            ast: Strategy AST from DSL parser
            
        Returns:
            BacktestResult with trades and metrics
        """
        self.df = df.copy()
        
        # Generate trading function from AST
        trading_func = generate_trading_function(ast)
        
        # Evaluate signals
        self.signals = trading_func(self.df)
        entry_signals = self.signals['entry'].astype(bool)
        exit_signals = self.signals['exit'].astype(bool)
        
        # Execute trades
        self.trades = []
        in_position = False
        entry_idx = -1
        
        for i in range(len(self.df)):
            # Check for entry
            if not in_position and entry_signals.iloc[i]:
                entry_idx = i
                in_position = True
            
            # Check for exit
            if in_position and exit_signals.iloc[i]:
                # Create trade
                trade = Trade(
                    entry_date=str(self.df.index[entry_idx]),
                    entry_price=float(self.df['close'].iloc[entry_idx]),
                    exit_date=str(self.df.index[i]),
                    exit_price=float(self.df['close'].iloc[i]),
                    quantity=self.position_size
                )
                self.trades.append(trade)
                in_position = False
        
        # Close any open position at end
        if in_position:
            trade = Trade(
                entry_date=str(self.df.index[entry_idx]),
                entry_price=float(self.df['close'].iloc[entry_idx]),
                exit_date=str(self.df.index[-1]),
                exit_price=float(self.df['close'].iloc[-1]),
                quantity=self.position_size
            )
            self.trades.append(trade)
        
        # Calculate metrics
        return self._calculate_metrics()
    
    def _calculate_metrics(self) -> BacktestResult:
        """Calculate backtest metrics"""
        if not self.trades:
            return BacktestResult(
                trades=[],
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_profit=0.0,
                total_return_pct=0.0,
                avg_trade_return=0.0,
                max_profit=0.0,
                max_loss=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0
            )
        
        # Basic metrics
        total_trades = len(self.trades)
        profits = [t.profit for t in self.trades]
        returns = [t.return_pct for t in self.trades]
        
        winning_trades = sum(1 for p in profits if p > 0)
        losing_trades = sum(1 for p in profits if p < 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = sum(profits)
        total_return_pct = (total_profit / self.initial_capital) * 100
        avg_trade_return = np.mean(returns) if returns else 0
        
        max_profit = max(profits) if profits else 0
        max_loss = min(profits) if profits else 0
        
        # Calculate max drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Calculate Sharpe ratio (annualized)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        
        return BacktestResult(
            trades=self.trades,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_profit=total_profit,
            total_return_pct=total_return_pct,
            avg_trade_return=avg_trade_return,
            max_profit=max_profit,
            max_loss=max_loss,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from trades"""
        if not self.trades:
            return 0.0
        
        cumulative = self.initial_capital
        peak = cumulative
        max_dd = 0.0
        
        for trade in self.trades:
            cumulative += trade.profit
            if cumulative > peak:
                peak = cumulative
            dd = (peak - cumulative) / peak * 100 if peak != 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns) / 100  # Convert to decimal
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        # Annualized Sharpe ratio
        sharpe = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)
        return sharpe
    
    def print_summary(self):
        """Print backtest summary"""
        if not self.trades:
            print("No trades executed")
            return
        
        result = self._calculate_metrics()
        
        print("\n" + "="*80)
        print("BACKTEST SUMMARY")
        print("="*80)
        print(f"\nTrading Activity:")
        print(f"  Total Trades: {result.total_trades}")
        print(f"  Winning Trades: {result.winning_trades}")
        print(f"  Losing Trades: {result.losing_trades}")
        print(f"  Win Rate: {result.win_rate:.2f}%")
        
        print(f"\nProfitability:")
        print(f"  Total Profit: ${result.total_profit:.2f}")
        print(f"  Total Return: {result.total_return_pct:.2f}%")
        print(f"  Avg Trade Return: {result.avg_trade_return:.2f}%")
        print(f"  Max Profit: ${result.max_profit:.2f}")
        print(f"  Max Loss: ${result.max_loss:.2f}")
        
        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.4f}")
        
        print(f"\nTrades:")
        for i, trade in enumerate(self.trades[:5], 1):  # Show first 5
            print(f"  {i}. {trade.entry_date} → {trade.exit_date}: "
                  f"${trade.entry_price:.2f} → ${trade.exit_price:.2f} "
                  f"({trade.return_pct:+.2f}%)")
        
        if len(self.trades) > 5:
            print(f"  ... and {len(self.trades) - 5} more trades")
        
        print("="*80)
