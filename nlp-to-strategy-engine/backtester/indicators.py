"""
Indicator helper functions for backtesting using TA-Lib.

Note: Indicators are implemented using TA-Lib and require `ta-lib` to be installed.
      Install with: pip install TA-Lib (after installing the C library)
"""
import pandas as pd
import numpy as np
import talib


class TechnicalIndicators:
    """Technical indicator calculations for backtesting using TA-Lib"""
    
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
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, 
             signal_period: int = 9) -> tuple:
        """MACD (Moving Average Convergence Divergence) using TA-Lib"""
        macd_line, signal_line, histogram = talib.MACD(
            series.values,
            fastperiod=fast,
            slowperiod=slow,
            signalperiod=signal_period
        )
        return (
            pd.Series(macd_line, index=series.index),
            pd.Series(signal_line, index=series.index),
            pd.Series(histogram, index=series.index)
        )
    
    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, 
                       std_dev: float = 2.0) -> tuple:
        """Bollinger Bands using TA-Lib"""
        upper, middle, lower = talib.BBANDS(
            series.values,
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev,
            matype=0
        )
        return (
            pd.Series(upper, index=series.index),
            pd.Series(middle, index=series.index),
            pd.Series(lower, index=series.index)
        )
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = 14) -> pd.Series:
        """Average True Range using TA-Lib"""
        result = talib.ATR(high.values, low.values, close.values, timeperiod=period)
        return pd.Series(result, index=close.index)
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = 14) -> pd.Series:
        """Average Directional Index using TA-Lib"""
        result = talib.ADX(high.values, low.values, close.values, timeperiod=period)
        return pd.Series(result, index=close.index)
    
    @staticmethod
    def stochastic(close: pd.Series, high: pd.Series, low: pd.Series,
                  period: int = 14, k_period: int = 3, 
                  d_period: int = 3) -> tuple:
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
    def momentum(series: pd.Series, period: int = 10) -> pd.Series:
        """Momentum indicator using TA-Lib"""
        result = talib.MOM(series.values, timeperiod=period)
        return pd.Series(result, index=series.index)
    
    @staticmethod
    def rate_of_change(series: pd.Series, period: int = 10) -> pd.Series:
        """Rate of Change (ROC) using TA-Lib"""
        result = talib.ROC(series.values, timeperiod=period)
        return pd.Series(result, index=series.index)


def calculate_returns(df: pd.DataFrame) -> pd.Series:
    """Calculate daily returns"""
    return df['close'].pct_change() * 100


def calculate_cumulative_returns(df: pd.DataFrame) -> pd.Series:
    """Calculate cumulative returns"""
    returns = calculate_returns(df)
    return (1 + returns / 100).cumprod() - 1


def calculate_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate rolling volatility"""
    returns = calculate_returns(df)
    volatility = returns.rolling(window=period).std()
    return volatility


def calculate_correlation(df: pd.DataFrame, other_df: pd.DataFrame, 
                         period: int = 30) -> pd.Series:
    """Calculate rolling correlation between two series"""
    returns1 = calculate_returns(df)
    returns2 = calculate_returns(other_df)
    correlation = returns1.rolling(window=period).corr(returns2)
    return correlation
