"""
src/eda_utils.py
----------------
Reusable EDA helper functions for the Nova Financial Solutions
News Sentiment Analysis project.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

import pandas as pd
import numpy as np


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_news_data(filepath: str) -> pd.DataFrame:
    """Load and perform initial cleaning of the FNSPID news CSV.

    Parameters
    ----------
    filepath : str
        Path to raw_analyst_ratings.csv

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with typed columns.
    """
    df = pd.read_csv(filepath)

    # Parse dates (strip timezone suffix for compatibility)
    df['date'] = pd.to_datetime(df['date'].str[:19], errors='coerce')

    # Derived columns
    df['headline_len'] = df['headline'].str.len()
    df['hour']         = df['date'].dt.hour
    df['day_of_week']  = df['date'].dt.day_name()
    df['year_month']   = df['date'].dt.to_period('M')
    df['date_only']    = df['date'].dt.normalize()

    return df


# ─── Descriptive Statistics ───────────────────────────────────────────────────

def headline_stats(df: pd.DataFrame) -> pd.Series:
    """Return descriptive statistics for headline character lengths."""
    return df['headline_len'].describe()


def articles_per_publisher(df: pd.DataFrame, top_n: int = 15) -> pd.Series:
    """Return article counts per publisher, sorted descending."""
    return df['publisher'].value_counts().head(top_n)


def extract_domain(publisher: str) -> str:
    """Extract domain from email-style publisher names."""
    if '@' in publisher:
        return publisher.split('@')[-1]
    return publisher


def publisher_domains(df: pd.DataFrame) -> pd.Series:
    """Map publishers to domains and return counts."""
    return df['publisher'].apply(extract_domain).value_counts()


# ─── Text Analysis ────────────────────────────────────────────────────────────

_STOPWORDS = {
    'a','an','the','in','on','at','for','of','by','to','is','are','was',
    'be','been','with','and','or','its','it','as','up','from','after',
    'than','that','this','into','amid','over','about','after','before',
    'between','has','have','had','not','no','but','will','would','could',
    'should','may','might','their','they','them','these','those',
}


def top_keywords(
    headlines: pd.Series,
    top_n: int = 30,
    extra_stop: Optional[set] = None,
    min_len: int = 3,
) -> List[tuple]:
    """Extract most frequent non-stopword tokens from headlines.

    Parameters
    ----------
    headlines : pd.Series
        Series of headline strings.
    top_n : int
        Number of top keywords to return.
    extra_stop : set, optional
        Additional stop words to exclude.
    min_len : int
        Minimum token length to consider.

    Returns
    -------
    List[tuple]
        List of (word, frequency) tuples.
    """
    stop = _STOPWORDS.copy()
    if extra_stop:
        stop.update(extra_stop)

    all_words = []
    for hl in headlines.dropna():
        tokens = re.findall(r"[A-Za-z']+", hl.lower())
        all_words.extend([t for t in tokens if t not in stop and len(t) >= min_len])

    return Counter(all_words).most_common(top_n)


# ─── Time-Series Helpers ──────────────────────────────────────────────────────

def monthly_volume(df: pd.DataFrame, stock: Optional[str] = None) -> pd.DataFrame:
    """Compute monthly article volume, optionally filtered by stock.

    Returns a DataFrame with columns ['year_month_dt', 'count'].
    """
    subset = df if stock is None else df[df['stock'] == stock]
    monthly = subset.groupby('year_month').size().reset_index(name='count')
    monthly['year_month_dt'] = monthly['year_month'].dt.to_timestamp()
    return monthly


def hourly_volume(df: pd.DataFrame) -> pd.Series:
    """Return article counts grouped by hour of day."""
    return df.groupby('hour').size()


def day_of_week_volume(df: pd.DataFrame) -> pd.Series:
    """Return article counts grouped by day of week."""
    order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    counts = df.groupby('day_of_week').size()
    return counts.reindex([d for d in order if d in counts.index])


# ─── Quality Checks ───────────────────────────────────────────────────────────

def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarising missing values and dtypes."""
    report = pd.DataFrame({
        'dtype':    df.dtypes,
        'missing':  df.isnull().sum(),
        'missing%': (df.isnull().mean() * 100).round(2),
        'unique':   df.nunique(),
    })
    return report