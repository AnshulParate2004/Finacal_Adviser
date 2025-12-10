"""Indicator helper functions for backtesting with optional TA-Lib"""
import pandas as pd
import numpy as np

# Try to import TA-Lib, fallback to pandas if not available
try:
    import talib
    USE_TALIB = True
except ImportError:
    USE_TALIB = False


class TechnicalIndicators:
    """Technical indicator calculations for backtesting (TA-Lib or pandas fallback)"""
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        if USE_TALIB:
            result = talib.SMA(series.values, timeperiod=period)
            return pd.Series(result, index=series.index)
        else:
            return series.rolling(window=period).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        if USE_TALIB:
            result = talib.EMA(series.values, timeperiod=period)
            return pd.Series(result, index=series.index)
        else:
            return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        if USE_TALIB:
            result = talib.RSI(series.values, timeperiod=period)
            return pd.Series(result, index=series.index)
        else:
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
    
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, 
             signal_period: int = 9) -> tuple:
        """MACD (Moving Average Convergence Divergence)"""
        if USE_TALIB:
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
        else:
            ema_fast = series.ewm(span=fast, adjust=False).mean()
            ema_slow = series.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, 
                       std_dev: float = 2.0) -> tuple:
        """Bollinger Bands"""
        if USE_TALIB:
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
        else:
            sma = series.rolling(window=period).mean()
            std = series.rolling(window=period).std()
            upper = sma + (std * std_dev)
            middle = sma
            lower = sma - (std * std_dev)
            return upper, middle, lower
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = 14) -> pd.Series:
        """Average True Range"""
        if USE_TALIB:
            result = talib.ATR(high.values, low.values, close.values, timeperiod=period)
            return pd.Series(result, index=close.index)
        else:
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = 14) -> pd.Series:
        """Average Directional Index"""
        if USE_TALIB:
            result = talib.ADX(high.values, low.values, close.values, timeperiod=period)
            return pd.Series(result, index=close.index)
        else:
            plus_dm = (high.diff()).clip(lower=0)
            minus_dm = (-low.diff()).clip(lower=0)
            atr = TechnicalIndicators.atr(high, low, close, period)
            
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            return adx
    
    @staticmethod
    def stochastic(close: pd.Series, high: pd.Series, low: pd.Series,
                  period: int = 14, k_period: int = 3, 
                  d_period: int = 3) -> tuple:
        """Stochastic Oscillator"""
        if USE_TALIB:
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
        else:
            low_min = low.rolling(window=period).min()
            high_max = high.rolling(window=period).max()
            k_percent = 100 * ((close - low_min) / (high_max - low_min))
            k = k_percent.rolling(window=k_period).mean()
            d = k.rolling(window=d_period).mean()
            return k, d
    
    @staticmethod
    def momentum(series: pd.Series, period: int = 10) -> pd.Series:
        """Momentum indicator"""
        if USE_TALIB:
            result = talib.MOM(series.values, timeperiod=period)
            return pd.Series(result, index=series.index)
        else:
            return series - series.shift(period)
    
    @staticmethod
    def rate_of_change(series: pd.Series, period: int = 10) -> pd.Series:
        """Rate of Change (ROC)"""
        if USE_TALIB:
            result = talib.ROC(series.values, timeperiod=period)
            return pd.Series(result, index=series.index)
        else:
            roc = ((series - series.shift(period)) / series.shift(period)) * 100
            return roc


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
