"""
src/sentiment.py
----------------
Sentiment analysis helpers for financial news headlines.

What is sentiment analysis?
  We read each news headline and assign it a score that captures
  whether the language is positive, negative, or neutral.

  We use VADER (Valence Aware Dictionary and sEntiment Reasoner),
  a rule-based tool that works well on short, punchy text like
  financial headlines without needing any model training.

VADER output — four scores per headline:
  pos      : proportion of text that is positive  (0 to 1)
  neg      : proportion of text that is negative  (0 to 1)
  neu      : proportion of text that is neutral   (0 to 1)
  compound : overall score, rescaled to -1 to +1
             > +0.05  → Positive
             < -0.05  → Negative
             otherwise → Neutral
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# One shared analyser instance (it is stateless so safe to reuse)
_analyser = SentimentIntensityAnalyzer()


# ─── Scoring ──────────────────────────────────────────────────────────────────

def score_headline(headline: str) -> dict:
    """
    Return VADER scores for a single headline string.

    Parameters
    ----------
    headline : str

    Returns
    -------
    dict with keys: pos, neg, neu, compound
    """
    return _analyser.polarity_scores(str(headline))


def score_dataframe(df: pd.DataFrame, headline_col: str = 'headline') -> pd.DataFrame:
    """
    Add VADER sentiment columns to a DataFrame of headlines.

    New columns added:
      vader_pos      positive score  (0-1)
      vader_neg      negative score  (0-1)
      vader_neu      neutral score   (0-1)
      vader_compound compound score  (-1 to +1)
      sentiment      label: 'Positive' / 'Neutral' / 'Negative'

    Parameters
    ----------
    df           : pd.DataFrame  — must contain headline_col
    headline_col : str           — name of the headline column

    Returns
    -------
    pd.DataFrame — copy with new sentiment columns added
    """
    df = df.copy()
    scores = df[headline_col].apply(score_headline)
    df['vader_pos']      = scores.apply(lambda s: s['pos'])
    df['vader_neg']      = scores.apply(lambda s: s['neg'])
    df['vader_neu']      = scores.apply(lambda s: s['neu'])
    df['vader_compound'] = scores.apply(lambda s: s['compound'])
    df['sentiment']      = df['vader_compound'].apply(label_sentiment)
    return df


def label_sentiment(compound: float) -> str:
    """
    Convert a VADER compound score to a human-readable label.

    Parameters
    ----------
    compound : float  — value between -1 and +1

    Returns
    -------
    'Positive', 'Neutral', or 'Negative'
    """
    if compound > 0.05:
        return 'Positive'
    elif compound < -0.05:
        return 'Negative'
    else:
        return 'Neutral'


# ─── Date Alignment ───────────────────────────────────────────────────────────

def align_to_trading_day(
    df: pd.DataFrame,
    date_col: str = 'date',
    market_open_hour: int = 9,
    market_close_hour: int = 16,
) -> pd.DataFrame:
    """
    Map each article's timestamp to the correct trading day for
    correlation with that day's stock return.

    Rules applied:
      1. Strip timezone info (normalise to EST).
      2. Pre-market  (hour < 9:30)  → same calendar date.
      3. During market (9:30–16:00) → same calendar date.
      4. Post-close  (hour >= 16)   → next calendar date.
      5. Weekend / non-trading days → shift forward to next Monday
         (handled downstream when joining to price data which only
         has trading days).

    Parameters
    ----------
    df                : pd.DataFrame
    date_col          : str  — column containing publication datetime
    market_open_hour  : int  — hour of market open (default 9)
    market_close_hour : int  — hour of market close (default 16)

    Returns
    -------
    pd.DataFrame with a new 'trading_date' column (date only, no time)
    """
    df = df.copy()
    # Parse and strip timezone
    dt = pd.to_datetime(df[date_col].astype(str).str[:19])
    
    # Post-close articles map to next calendar day
    post_close = dt.dt.hour >= market_close_hour
    trading_date = dt.dt.normalize()                      # midnight same day
    trading_date = trading_date + pd.to_timedelta(        # shift post-close +1 day
        post_close.astype(int), unit='D'
    )
    df['trading_date'] = trading_date.dt.date
    return df


# ─── Aggregation ─────────────────────────────────────────────────────────────

def daily_sentiment(
    df: pd.DataFrame,
    stock_col: str = 'stock',
    date_col: str = 'trading_date',
    score_col: str = 'vader_compound',
) -> pd.DataFrame:
    """
    Compute the daily average VADER compound score per stock.

    Multiple articles about the same stock on the same day are
    averaged. This gives one sentiment value per (stock, date) pair
    to match against the daily price return.

    Parameters
    ----------
    df        : pd.DataFrame  — scored and date-aligned news
    stock_col : str           — column with stock ticker
    date_col  : str           — column with trading date (date objects)
    score_col : str           — column with compound scores

    Returns
    -------
    pd.DataFrame with columns: [stock_col, date_col, 'mean_sentiment',
                                 'article_count', 'pct_positive',
                                 'pct_negative']
    """
    grp = df.groupby([stock_col, date_col])
    out = grp[score_col].agg(
        mean_sentiment='mean',
        article_count='count',
    ).reset_index()
    # Percentage positive / negative per day
    pos = grp.apply(lambda g: (g[score_col] > 0.05).mean()).reset_index(name='pct_positive')
    neg = grp.apply(lambda g: (g[score_col] < -0.05).mean()).reset_index(name='pct_negative')
    out = out.merge(pos, on=[stock_col, date_col]).merge(neg, on=[stock_col, date_col])
    return out