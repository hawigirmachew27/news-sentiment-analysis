"""
src/indicators.py
-----------------
Technical indicator functions for stock price analysis.

What is a technical indicator?
  Traders use math formulas on past price/volume data to spot trends,
  momentum, and potential buy/sell signals. This module implements the
  most common ones from scratch so you can see exactly how they work.

Each function takes a pandas Series (usually the 'Close' price column)
and returns a new Series with the indicator values.
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# MOVING AVERAGES
# ─────────────────────────────────────────────────────────────────────────────

def simple_moving_average(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Simple Moving Average (SMA)
    ---------------------------
    The average closing price over the last `window` days.
    Think of it as a "smoothed" version of the price line.

    Example: SMA(20) on day 20 = average of the last 20 closing prices.
    The first (window-1) values will be NaN because we don't have
    enough history yet.

    Parameters
    ----------
    close  : pd.Series  — daily closing prices
    window : int        — how many days to average (default: 20)

    Returns
    -------
    pd.Series with the same index as `close`
    """
    return close.rolling(window=window).mean()


def exponential_moving_average(close: pd.Series, span: int = 20) -> pd.Series:
    """
    Exponential Moving Average (EMA)
    ---------------------------------
    Like SMA but more recent prices get higher weight.
    This makes EMA react faster to price changes than SMA.

    Formula: EMA_today = alpha * Price_today + (1-alpha) * EMA_yesterday
             where alpha = 2 / (span + 1)

    Parameters
    ----------
    close : pd.Series — daily closing prices
    span  : int       — lookback period (default: 20)

    Returns
    -------
    pd.Series with the same index as `close`
    """
    return close.ewm(span=span, adjust=False).mean()


# ─────────────────────────────────────────────────────────────────────────────
# MOMENTUM INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI)
    ------------------------------
    Measures whether a stock is overbought (>70) or oversold (<30).
    RSI is always between 0 and 100.

    How it works:
      1. Calculate daily price changes (gains and losses).
      2. Average gains over `window` days, average losses over `window` days.
      3. RSI = 100 - (100 / (1 + avg_gain / avg_loss))

    Interpretation:
      RSI > 70  → overbought (price may pull back)
      RSI < 30  → oversold  (price may bounce up)
      RSI = 50  → neutral

    Parameters
    ----------
    close  : pd.Series — daily closing prices
    window : int       — lookback period (default: 14 days)

    Returns
    -------
    pd.Series with RSI values (0–100)
    """
    # Step 1: daily price change
    delta = close.diff()

    # Step 2: separate gains (positive changes) and losses (negative changes)
    gains  = delta.clip(lower=0)          # keep only positive values
    losses = (-delta).clip(lower=0)       # keep only negative values (made positive)

    # Step 3: exponential average of gains and losses
    avg_gain = gains.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = losses.ewm(com=window - 1, min_periods=window).mean()

    # Step 4: calculate RSI
    rs  = avg_gain / avg_loss.replace(0, np.nan)   # avoid divide-by-zero
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD — Moving Average Convergence/Divergence
    ---------------------------------------------
    One of the most popular momentum indicators. It shows the relationship
    between two EMAs of the closing price.

    Three lines:
      - MACD Line   = EMA(12) − EMA(26)   (the "fast minus slow" line)
      - Signal Line = EMA(9) of MACD line  (smoother version of MACD)
      - Histogram   = MACD Line − Signal Line  (the "gap" between them)

    Signals:
      • MACD crosses above Signal → bullish (potential buy)
      • MACD crosses below Signal → bearish (potential sell)
      • Histogram shrinking toward zero → trend losing momentum

    Parameters
    ----------
    close  : pd.Series — daily closing prices
    fast   : int       — fast EMA period   (default: 12)
    slow   : int       — slow EMA period   (default: 26)
    signal : int       — signal EMA period (default: 9)

    Returns
    -------
    pd.DataFrame with columns: ['macd', 'signal', 'histogram']
    """
    ema_fast   = exponential_moving_average(close, span=fast)
    ema_slow   = exponential_moving_average(close, span=slow)
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line

    return pd.DataFrame({
        'macd':      macd_line,
        'signal':    signal_line,
        'histogram': histogram,
    }, index=close.index)


# ─────────────────────────────────────────────────────────────────────────────
# VOLATILITY INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands
    ---------------
    Three lines around the price that expand when volatility is high
    and contract when volatility is low.

    - Middle Band = SMA(20)
    - Upper Band  = SMA(20) + 2 × standard deviation
    - Lower Band  = SMA(20) − 2 × standard deviation

    Interpretation:
      • Price near upper band → potentially overbought
      • Price near lower band → potentially oversold
      • Bands widening → increasing volatility
      • Bands squeezing → low volatility (often precedes a big move)

    Parameters
    ----------
    close   : pd.Series — daily closing prices
    window  : int       — rolling window for SMA (default: 20)
    num_std : float     — how many standard deviations wide the bands are

    Returns
    -------
    pd.DataFrame with columns: ['middle', 'upper', 'lower', 'bandwidth']
    """
    middle    = close.rolling(window=window).mean()
    std       = close.rolling(window=window).std()
    upper     = middle + num_std * std
    lower     = middle - num_std * std
    bandwidth = (upper - lower) / middle   # relative width of the bands

    return pd.DataFrame({
        'middle':    middle,
        'upper':     upper,
        'lower':     lower,
        'bandwidth': bandwidth,
    }, index=close.index)


# ─────────────────────────────────────────────────────────────────────────────
# RETURNS & RISK
# ─────────────────────────────────────────────────────────────────────────────

def daily_returns(close: pd.Series) -> pd.Series:
    """
    Daily Percentage Returns
    ------------------------
    How much did the price change (%) from yesterday to today?

    Formula: (today's price - yesterday's price) / yesterday's price

    Example: if price goes from $100 → $103, return = 3%.

    Parameters
    ----------
    close : pd.Series — daily closing prices

    Returns
    -------
    pd.Series of percentage returns (e.g. 0.03 means +3%)
    """
    return close.pct_change()


def cumulative_returns(close: pd.Series) -> pd.Series:
    """
    Cumulative Returns
    ------------------
    Total growth of a $1 investment from the start of the period.

    Example: value of 1.5 means the investment grew 50% since day 1.

    Parameters
    ----------
    close : pd.Series — daily closing prices

    Returns
    -------
    pd.Series (starts near 1.0, grows/shrinks over time)
    """
    returns = daily_returns(close)
    return (1 + returns).cumprod()


def rolling_volatility(close: pd.Series, window: int = 30) -> pd.Series:
    """
    Rolling Volatility (Annualised)
    --------------------------------
    Standard deviation of daily returns over a rolling window,
    scaled to annual terms (multiply by √252 because there are
    ~252 trading days per year).

    Higher volatility = bigger price swings = more risk (and opportunity).

    Parameters
    ----------
    close  : pd.Series — daily closing prices
    window : int       — rolling window in days (default: 30)

    Returns
    -------
    pd.Series of annualised volatility values
    """
    returns = daily_returns(close)
    return returns.rolling(window=window).std() * np.sqrt(252)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: add all indicators to a DataFrame at once
# ─────────────────────────────────────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to a stock DataFrame in one call.

    Expects a DataFrame with at least a 'Close' column.
    Returns a copy with new indicator columns added.

    New columns added:
      sma_20, sma_50, ema_20, ema_50,
      rsi_14,
      macd, macd_signal, macd_hist,
      bb_middle, bb_upper, bb_lower, bb_bandwidth,
      daily_return, cumulative_return, volatility_30d

    Parameters
    ----------
    df : pd.DataFrame — must contain a 'Close' column

    Returns
    -------
    pd.DataFrame — original data + indicator columns
    """
    df = df.copy()
    close = df['Close']

    # Moving averages
    df['sma_20'] = simple_moving_average(close, window=20)
    df['sma_50'] = simple_moving_average(close, window=50)
    df['ema_20'] = exponential_moving_average(close, span=20)
    df['ema_50'] = exponential_moving_average(close, span=50)

    # RSI
    df['rsi_14'] = relative_strength_index(close, window=14)

    # MACD
    macd_df = macd(close)
    df['macd']        = macd_df['macd']
    df['macd_signal'] = macd_df['signal']
    df['macd_hist']   = macd_df['histogram']

    # Bollinger Bands
    bb_df = bollinger_bands(close)
    df['bb_middle']    = bb_df['middle']
    df['bb_upper']     = bb_df['upper']
    df['bb_lower']     = bb_df['lower']
    df['bb_bandwidth'] = bb_df['bandwidth']

    # Returns & Risk
    df['daily_return']      = daily_returns(close)
    df['cumulative_return'] = cumulative_returns(close)
    df['volatility_30d']    = rolling_volatility(close, window=30)

    return df