"""Shared fixtures for data_conversion tests."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Sample raw DataFrame with stock data."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
            "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "open": [100.0, 102.0, 200.0, 204.0],
            "closing": [101.0, 103.0, 201.0, 205.0],
            "volume": [1000000, 1100000, 2000000, 2100000],
        }
    )


@pytest.fixture
def sample_returns_df() -> pd.DataFrame:
    """Sample DataFrame with log returns."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "log_return": [0.01, 0.02, 0.015, 0.025],
            "open": [100.0, 200.0, 102.0, 204.0],
            "closing": [101.0, 201.0, 103.0, 205.0],
        }
    )


@pytest.fixture
def sample_liquidity_metrics() -> pd.DataFrame:
    """Sample liquidity metrics with weights."""
    return pd.DataFrame(
        {"weight": [0.4, 0.6], "liquidity_score": [40000000.0, 60000000.0]},
        index=pd.Index(["AAPL", "MSFT"]),
    )


@pytest.fixture
def sample_daily_weight_totals() -> pd.DataFrame:
    """Sample daily weight totals."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "weight_sum": [1.0, 1.0],
        }
    )


@pytest.fixture
def sample_aggregated_returns() -> pd.DataFrame:
    """Sample aggregated weighted returns."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "weighted_log_return": [0.016, 0.021],
        }
    )
