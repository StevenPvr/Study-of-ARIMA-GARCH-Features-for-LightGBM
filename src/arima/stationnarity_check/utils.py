"""Utility functions for stationarity checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _validate_series(series: pd.Series) -> pd.Series:
    """Return a clean Series (datetime index not enforced).

    Args:
        series: Input time series.

    Returns:
        Cleaned Series with NaN values removed and converted to float.

    Raises:
        ValueError: If series is None or empty after dropna.
    """
    if series is None:
        raise ValueError("series is None")
    s = pd.Series(series).dropna().astype(float)
    if s.empty:
        raise ValueError("series is empty after dropna")
    return s


def _load_csv_series(*, data_file: str, column: str, date_col: str = "date") -> pd.Series:
    """Load a column as Series from a CSV with a date index.

    Args:
        data_file: Path to CSV file.
        column: Column name to extract.
        date_col: Name of the date column. Defaults to "date".

    Returns:
        Series with date index and NaN values removed.

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If column is missing or does not return a Series.
    """
    path = Path(data_file)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    df = pd.read_csv(path, parse_dates=[date_col]).set_index(date_col)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {data_file}")
    series = df[column]
    if not isinstance(series, pd.Series):
        raise ValueError(f"Column '{column}' did not return a Series")
    return series.dropna()

