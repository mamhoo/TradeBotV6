"""
indicators.py — Technical indicators using pandas + ta library
pip install ta
"""

import pandas as pd
import numpy as np


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period=20, std=2.0):
    mid = series.rolling(period).mean()
    std_val = series.rolling(period).std()
    upper = mid + std * std_val
    lower = mid - std * std_val
    return upper, mid, lower


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def is_bullish_engulfing(df: pd.DataFrame) -> bool:
    """Check last 2 candles for bullish engulfing pattern."""
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    return (prev["close"] < prev["open"] and      # prev bearish
            curr["close"] > curr["open"] and       # curr bullish
            curr["open"] < prev["close"] and       # opens below prev close
            curr["close"] > prev["open"])          # closes above prev open


def is_bearish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    return (prev["close"] > prev["open"] and
            curr["close"] < curr["open"] and
            curr["open"] > prev["close"] and
            curr["close"] < prev["open"])


def is_hammer(df: pd.DataFrame) -> bool:
    """Bullish hammer: small body, long lower wick."""
    candle = df.iloc[-1]
    body = abs(candle["close"] - candle["open"])
    lower_wick = min(candle["open"], candle["close"]) - candle["low"]
    upper_wick = candle["high"] - max(candle["open"], candle["close"])
    return (lower_wick >= 2 * body and upper_wick <= body * 0.5 and
            candle["close"] > candle["open"])


def is_shooting_star(df: pd.DataFrame) -> bool:
    """Bearish shooting star: small body, long upper wick."""
    candle = df.iloc[-1]
    body = abs(candle["close"] - candle["open"])
    upper_wick = candle["high"] - max(candle["open"], candle["close"])
    lower_wick = min(candle["open"], candle["close"]) - candle["low"]
    return (upper_wick >= 2 * body and lower_wick <= body * 0.5 and
            candle["close"] < candle["open"])


def get_trend(df: pd.DataFrame, fast: int = 21, slow: int = 55) -> str:
    """Determine trend from EMA crossover."""
    close = df["close"]
    ema_f = ema(close, fast).iloc[-1]
    ema_s = ema(close, slow).iloc[-1]
    if ema_f > ema_s * 1.0003:
        return "UP"
    elif ema_f < ema_s * 0.9997:
        return "DOWN"
    return "NEUTRAL"


def rsi_divergence(df: pd.DataFrame, period: int = 14, lookback: int = 20) -> str:
    """
    Detect RSI divergence.
    Returns: "BULLISH_DIV", "BEARISH_DIV", or "NONE"
    """
    if len(df) < lookback + period:
        return "NONE"

    close = df["close"].values[-lookback:]
    rsi_vals = rsi(df["close"], period).values[-lookback:]

    # Bullish divergence: price makes lower low, RSI makes higher low
    price_lower_low = close[-1] < min(close[:-5])
    rsi_higher_low = rsi_vals[-1] > min(rsi_vals[:-5])
    if price_lower_low and rsi_higher_low:
        return "BULLISH_DIV"

    # Bearish divergence: price makes higher high, RSI makes lower high
    price_higher_high = close[-1] > max(close[:-5])
    rsi_lower_high = rsi_vals[-1] < max(rsi_vals[:-5])
    if price_higher_high and rsi_lower_high:
        return "BEARISH_DIV"

    return "NONE"
