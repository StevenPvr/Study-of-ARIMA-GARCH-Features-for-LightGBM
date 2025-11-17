"""Unit tests for metrics computation functions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data_cleaning.metrics import (
    compute_empty_quality_metrics,
    compute_missing_business_days,
    compute_quality_metrics,
)


class TestComputeEmptyQualityMetrics:
    """Tests for compute_empty_quality_metrics function."""

    def test_empty_metrics_structure(self) -> None:
        """Test that empty metrics have correct structure."""
        metrics = compute_empty_quality_metrics()

        assert isinstance(metrics, dict)
        assert "na_by_column" in metrics
        assert "duplicate_rows_on_date_ticker" in metrics
        assert "rows_with_nonpositive_volume" in metrics
        assert "non_monotonic_ticker_dates" in metrics
        assert "top_missing_business_days" in metrics
        assert metrics["na_by_column"] == {}
        assert metrics["duplicate_rows_on_date_ticker"] == 0
        assert metrics["rows_with_nonpositive_volume"] == 0
        assert metrics["non_monotonic_ticker_dates"] == 0
        assert metrics["top_missing_business_days"] == []


class TestComputeMissingBusinessDays:
    """Tests for compute_missing_business_days function."""

    def test_no_missing_days(self) -> None:
        """Test with no missing business days."""
        df = pd.DataFrame(
            {
                "tickers": ["AAPL"] * 5,
                "date": pd.bdate_range("2020-01-01", periods=5),
            }
        )
        result = compute_missing_business_days(df)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_with_missing_days(self) -> None:
        """Test with missing business days."""
        # Create data with missing days (skip weekends and one business day)
        dates = pd.bdate_range("2020-01-01", periods=5)
        # Remove one business day
        dates = dates.drop(dates[2])
        df = pd.DataFrame(
            {
                "tickers": ["AAPL"] * len(dates),
                "date": dates,
            }
        )
        result = compute_missing_business_days(df)

        assert isinstance(result, list)
        assert len(result) > 0
        assert "ticker" in result[0]
        assert "missing_business_days" in result[0]

    def test_empty_dataframe(self) -> None:
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = compute_missing_business_days(df)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_missing_columns(self) -> None:
        """Test with missing columns."""
        df = pd.DataFrame({"tickers": ["AAPL"]})
        result = compute_missing_business_days(df)

        assert isinstance(result, list)
        assert len(result) == 0


class TestComputeQualityMetrics:
    """Tests for compute_quality_metrics function."""

    def test_empty_dataframe(self) -> None:
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        metrics = compute_quality_metrics(df)

        assert metrics == compute_empty_quality_metrics()

    def test_basic_metrics(self) -> None:
        """Test basic quality metrics computation."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5, freq="D"),
                "tickers": ["AAPL"] * 5,
                "open": [100.0] * 5,
                "close": [100.5] * 5,
                "volume": [1000] * 5,
            }
        )
        metrics = compute_quality_metrics(df)

        assert "na_by_column" in metrics
        assert "duplicate_rows_on_date_ticker" in metrics
        assert "rows_with_nonpositive_volume" in metrics
        assert "non_monotonic_ticker_dates" in metrics
        assert "top_missing_business_days" in metrics

    def test_missing_values_metrics(self) -> None:
        """Test metrics with missing values."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5, freq="D"),
                "tickers": ["AAPL"] * 5,
                "open": [100.0, np.nan, 102.0, np.nan, 104.0],
                "close": [100.5] * 5,
                "volume": [1000] * 5,
            }
        )
        metrics = compute_quality_metrics(df)

        assert metrics["na_by_column"]["open"] == 2
        assert metrics["na_by_column"]["close"] == 0

    def test_duplicate_metrics(self) -> None:
        """Test metrics with duplicates."""
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02", "2020-01-01"],
                "tickers": ["AAPL", "AAPL", "AAPL"],
                "open": [100.0, 101.0, 100.0],
                "close": [100.5, 101.5, 100.5],
                "volume": [1000, 1100, 1000],
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        metrics = compute_quality_metrics(df)

        assert metrics["duplicate_rows_on_date_ticker"] == 1

    def test_nonpositive_volume_metrics(self) -> None:
        """Test metrics with non-positive volume."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5, freq="D"),
                "tickers": ["AAPL"] * 5,
                "open": [100.0] * 5,
                "close": [100.5] * 5,
                "volume": [1000, 0, -100, 1200, 1300],
            }
        )
        metrics = compute_quality_metrics(df)

        assert metrics["rows_with_nonpositive_volume"] == 2

    def test_monotonicity_violations(self) -> None:
        """Test metrics with monotonicity violations (duplicate dates)."""
        df = pd.DataFrame(
            {
                "date": [
                    pd.Timestamp("2020-01-01"),
                    pd.Timestamp("2020-01-01"),  # Duplicate date
                    pd.Timestamp("2020-01-02"),
                ],
                "tickers": ["AAPL"] * 3,
                "open": [100.0] * 3,
                "close": [100.5] * 3,
                "volume": [1000] * 3,
            }
        )
        metrics = compute_quality_metrics(df)

        assert metrics["non_monotonic_ticker_dates"] == 1

    def test_missing_business_days_included(self) -> None:
        """Test that missing business days are included in metrics."""
        # Create data with missing business days
        dates = pd.bdate_range("2020-01-01", periods=5)
        dates = dates.drop(dates[2])
        df = pd.DataFrame(
            {
                "tickers": ["AAPL"] * len(dates),
                "date": dates,
                "open": [100.0] * len(dates),
                "close": [100.5] * len(dates),
                "volume": [1000] * len(dates),
            }
        )
        metrics = compute_quality_metrics(df)

        assert "top_missing_business_days" in metrics
        assert isinstance(metrics["top_missing_business_days"], list)

    def test_missing_volume_column(self) -> None:
        """Test that missing volume column is handled."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5, freq="D"),
                "tickers": ["AAPL"] * 5,
                "open": [100.0] * 5,
                "close": [100.5] * 5,
            }
        )
        metrics = compute_quality_metrics(df)

        assert metrics["rows_with_nonpositive_volume"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
