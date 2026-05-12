"""
src/correlation.py
------------------
Statistical correlation helpers for linking news sentiment to
stock price returns.

What is Pearson correlation?
  It measures how strongly two variables move together.
  The result (r) is always between -1 and +1:

    r = +1.0  perfect positive relationship (both go up together)
    r = 0.0   no linear relationship
    r = -1.0  perfect negative relationship (one up, other down)

  A p-value tells you whether r is statistically significant:
    p < 0.05  → significant at the 95% confidence level
    p < 0.01  → significant at the 99% confidence level
    p > 0.05  → not significant (could be random noise)

Rule of thumb for r magnitude:
    |r| < 0.1   → negligible
    |r| < 0.3   → weak
    |r| < 0.5   → moderate
    |r| >= 0.5  → strong
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional


# ─── Merge sentiment with price returns ───────────────────────────────────────

def merge_sentiment_returns(
    daily_sent: pd.DataFrame,
    price_df: pd.DataFrame,
    stock: str,
    lag: int = 0,
    stock_col: str = 'stock',
    date_col: str = 'trading_date',
    sentiment_col: str = 'mean_sentiment',
    return_col: str = 'daily_return',
) -> pd.DataFrame:
    """
    Join daily sentiment scores with daily stock returns for one stock.

    Lag parameter:
      lag=0  → same-day  (does today's news predict today's return?)
      lag=1  → next-day  (does today's news predict tomorrow's return?)
      lag=2  → two days later

    Parameters
    ----------
    daily_sent    : pd.DataFrame  — output of sentiment.daily_sentiment()
    price_df      : pd.DataFrame  — price data with 'daily_return' column,
                                    indexed by Date
    stock         : str           — ticker to filter
    lag           : int           — days to shift sentiment forward (0, 1, 2)

    Returns
    -------
    pd.DataFrame with columns: ['trading_date', sentiment_col, return_col]
    """
    # Filter to the selected stock
    sent = daily_sent[daily_sent[stock_col] == stock].copy()
    sent[date_col] = pd.to_datetime(sent[date_col])
    sent = sent.set_index(date_col)[[sentiment_col, 'article_count', 'pct_positive', 'pct_negative']]

    # Prepare returns — align index to date only (no time)
    rets = price_df[[return_col]].copy()
    rets.index = pd.to_datetime(rets.index).normalize()

    # Apply lag: shift sentiment forward by `lag` trading days
    if lag > 0:
        combined = sent.join(rets, how='inner')
        # shift the return column by -lag rows (look ahead into future returns)
        combined[return_col] = combined[return_col].shift(-lag)
        combined = combined.dropna(subset=[sentiment_col, return_col])
    else:
        combined = sent.join(rets, how='inner').dropna(subset=[sentiment_col, return_col])

    combined.index.name = 'date'
    return combined.reset_index()


# ─── Pearson correlation ──────────────────────────────────────────────────────

def pearson_correlation(
    merged: pd.DataFrame,
    sentiment_col: str = 'mean_sentiment',
    return_col: str = 'daily_return',
) -> dict:
    """
    Compute Pearson r between sentiment and daily return.

    Parameters
    ----------
    merged        : pd.DataFrame  — output of merge_sentiment_returns()
    sentiment_col : str
    return_col    : str

    Returns
    -------
    dict with keys: r, p_value, n, significant_05, significant_01,
                    strength, direction
    """
    # Guard against duplicate column names that arise from joins
    s = merged[sentiment_col]
    r_ = merged[return_col]
    if isinstance(s, pd.DataFrame):  s  = s.iloc[:, 0]
    if isinstance(r_, pd.DataFrame): r_ = r_.iloc[:, 0]
    x = s.values.astype(float)
    y = r_.values.astype(float)
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]

    if len(x) < 10:
        return {'r': np.nan, 'p_value': np.nan, 'n': len(x),
                'significant_05': False, 'significant_01': False,
                'strength': 'insufficient data', 'direction': 'N/A'}

    r, p = stats.pearsonr(x, y)

    # Classify strength
    abs_r = abs(r)
    if abs_r < 0.1:
        strength = 'Negligible'
    elif abs_r < 0.3:
        strength = 'Weak'
    elif abs_r < 0.5:
        strength = 'Moderate'
    else:
        strength = 'Strong'

    return {
        'r': round(r, 4),
        'p_value': round(p, 4),
        'n': len(x),
        'significant_05': p < 0.05,
        'significant_01': p < 0.01,
        'strength': strength,
        'direction': 'Positive' if r > 0 else 'Negative',
    }


# ─── Lag sweep ────────────────────────────────────────────────────────────────

def lag_correlation_sweep(
    daily_sent: pd.DataFrame,
    price_df: pd.DataFrame,
    stock: str,
    lags: list = [0, 1, 2],
) -> pd.DataFrame:
    """
    Run Pearson correlation at multiple lags and return a summary table.

    Parameters
    ----------
    daily_sent : pd.DataFrame
    price_df   : pd.DataFrame
    stock      : str
    lags       : list of int

    Returns
    -------
    pd.DataFrame with one row per lag
    """
    rows = []
    for lag in lags:
        merged = merge_sentiment_returns(daily_sent, price_df, stock, lag=lag)
        result = pearson_correlation(merged)
        result['lag'] = lag
        result['stock'] = stock
        rows.append(result)
    return pd.DataFrame(rows)[['stock', 'lag', 'r', 'p_value', 'n',
                                'significant_05', 'strength', 'direction']]


# ─── Full correlation table (all stocks, one lag) ────────────────────────────

def correlation_table(
    daily_sent: pd.DataFrame,
    price_dfs: dict,
    stocks: list,
    lag: int = 0,
    ticker_map: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build a correlation summary table for all stocks at a given lag.

    Parameters
    ----------
    daily_sent : pd.DataFrame
    price_dfs  : dict  — {ticker: DataFrame}
    stocks     : list  — tickers to include
    lag        : int
    ticker_map : dict  — optional mapping e.g. {'GOOGL': 'GOOG'}

    Returns
    -------
    pd.DataFrame with columns: stock, r, p_value, n, strength, direction
    """
    rows = []
    for stock in stocks:
        price_key = (ticker_map or {}).get(stock, stock)
        if price_key not in price_dfs:
            continue
        merged = merge_sentiment_returns(daily_sent, price_dfs[price_key], stock, lag=lag)
        result = pearson_correlation(merged)
        result['stock'] = stock
        rows.append(result)
    df = pd.DataFrame(rows)
    df = df.sort_values('r', ascending=False)
    return df[['stock', 'r', 'p_value', 'n', 'significant_05', 'strength', 'direction']]