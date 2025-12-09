"""Backtester Module - Simulate trading strategy execution"""
from .engine import BacktestEngine, BacktestResult, Trade
from .indicators import TechnicalIndicators, calculate_returns, calculate_cumulative_returns

__all__ = [
    'BacktestEngine',
    'BacktestResult',
    'Trade',
    'TechnicalIndicators',
    'calculate_returns',
    'calculate_cumulative_returns',
]
