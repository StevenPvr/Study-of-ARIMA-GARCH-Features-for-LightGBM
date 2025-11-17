"""Shared fixtures and configuration for data_fetching tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


@pytest.fixture
def sample_ticker_data() -> pd.DataFrame:
    """Create sample ticker data with all required columns.

    Returns:
        DataFrame with Date, Open, High, Low, Close, Volume columns.
    """
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        },
    )


@pytest.fixture
def sample_processed_data() -> pd.DataFrame:
    """Create sample processed ticker data.

    Returns:
        DataFrame with normalized column names and ticker column.
    """
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200],
            "tickers": ["TEST", "TEST", "TEST"],
        }
    )
