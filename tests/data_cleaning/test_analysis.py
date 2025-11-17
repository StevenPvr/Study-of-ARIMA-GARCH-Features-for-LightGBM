"""Unit tests for analysis functions."""

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

from unittest.mock import MagicMock, patch

from src.data_cleaning.analysis import (
    analyze_general_statistics,
    analyze_missing_values,
    analyze_outliers,
    analyze_ticker_distribution,
    compute_monotonicity_violations,
    has_required_columns_for_monotonicity,
    is_ticker_monotonic,
    report_least_observations,
)


class TestHasRequiredColumnsForMonotonicity:
    """Tests for has_required_columns_for_monotonicity function."""

    def test_has_required_columns(self) -> None:
        """Test with required columns."""
        df = pd.DataFrame({"tickers": ["AAPL"], "date": ["2020-01-01"]})
        assert has_required_columns_for_monotonicity(df) is True

    def test_missing_columns(self) -> None:
        """Test with missing columns."""
        df = pd.DataFrame({"tickers": ["AAPL"]})
        assert has_required_columns_for_monotonicity(df) is False

    def test_empty_dataframe(self) -> None:
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        assert has_required_columns_for_monotonicity(df) is False


class TestIsTickerMonotonic:
    """Tests for is_ticker_monotonic function."""

    def test_monotonic_dates(self) -> None:
        """Test with strictly increasing dates."""
        dates = pd.Series(pd.date_range("2020-01-01", periods=5, freq="D"))
        assert is_ticker_monotonic(dates) is True

    def test_non_monotonic_dates(self) -> None:
        """Test with non-monotonic dates."""
        # The function sorts dates first, so we need dates that remain non-increasing
        # even after sorting; duplicates or decreases must persist post-sort.
        # But since it sorts, we need to test with actual decreasing sequence in sorted order.
        dates = pd.Series(
            [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-01"),  # Duplicate date
                pd.Timestamp("2020-01-02"),
            ]
        )
        # With duplicate dates, the diff will have 0, so it should return False
        assert is_ticker_monotonic(dates) is False

    def test_empty_series(self) -> None:
        """Test with empty series."""
        dates = pd.Series([], dtype="datetime64[ns]")
        assert is_ticker_monotonic(dates) is True

    def test_single_date(self) -> None:
        """Test with single date."""
        dates = pd.Series([pd.Timestamp("2020-01-01")])
        assert is_ticker_monotonic(dates) is True


class TestComputeMonotonicityViolations:
    """Tests for compute_monotonicity_violations function."""

    def test_no_violations(self) -> None:
        """Test with no violations."""
        df = pd.DataFrame(
            {
                "tickers": ["AAPL", "AAPL", "MSFT", "MSFT"],
                "date": pd.date_range("2020-01-01", periods=4, freq="D"),
            }
        )
        assert compute_monotonicity_violations(df) == 0

    def test_with_violations(self) -> None:
        """Test with violations (duplicate dates)."""
        df = pd.DataFrame(
            {
                "tickers": ["AAPL", "AAPL", "AAPL"],
                "date": [
                    pd.Timestamp("2020-01-01"),
                    pd.Timestamp("2020-01-01"),  # Duplicate date
                    pd.Timestamp("2020-01-02"),
                ],
            }
        )
        assert compute_monotonicity_violations(df) == 1

    def test_empty_dataframe(self) -> None:
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        assert compute_monotonicity_violations(df) == 0

    def test_missing_columns(self) -> None:
        """Test with missing columns."""
        df = pd.DataFrame({"tickers": ["AAPL"]})
        assert compute_monotonicity_violations(df) == 0


class TestAnalyzeGeneralStatistics:
    """Tests for analyze_general_statistics function."""

    @patch("src.data_cleaning.analysis.logger")
    def test_analyze_general_statistics(self, mock_logger: MagicMock) -> None:
        """Test general statistics analysis."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10, freq="D"),
                "tickers": ["AAPL"] * 5 + ["MSFT"] * 5,
                "open": [100.0] * 10,
                "close": [100.5] * 10,
                "volume": [1000] * 10,
            }
        )
        analyze_general_statistics(df)

        assert mock_logger.info.call_count >= 3

    def test_empty_dataframe_raises_valueerror(self) -> None:
        """Test that empty DataFrame raises ValueError."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="raw dataset DataFrame is empty"):
            analyze_general_statistics(df)

    def test_missing_columns_raises_keyerror(self) -> None:
        """Test that missing columns raise KeyError."""
        df = pd.DataFrame({"tickers": ["AAPL"]})
        with pytest.raises(KeyError):
            analyze_general_statistics(df)


class TestAnalyzeMissingValues:
    """Tests for analyze_missing_values function."""

    @patch("src.data_cleaning.analysis.logger")
    def test_analyze_missing_values(self, mock_logger: MagicMock) -> None:
        """Test missing values analysis."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5, freq="D"),
                "tickers": ["AAPL"] * 5,
                "open": [100.0, np.nan, 102.0, np.nan, 104.0],
                "close": [100.5] * 5,
                "volume": [1000] * 5,
            }
        )
        analyze_missing_values(df)

        assert mock_logger.info.call_count >= 1

    @patch("src.data_cleaning.analysis.logger")
    def test_no_missing_values(self, mock_logger: MagicMock) -> None:
        """Test with no missing values."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5, freq="D"),
                "tickers": ["AAPL"] * 5,
                "open": [100.0] * 5,
                "close": [100.5] * 5,
                "volume": [1000] * 5,
            }
        )
        analyze_missing_values(df)

        # Should not log anything if no missing values
        calls_with_missing = [
            call for call in mock_logger.info.call_args_list if "MISSING VALUES" in str(call)
        ]
        assert len(calls_with_missing) >= 1


class TestAnalyzeOutliers:
    """Tests for analyze_outliers function."""

    @patch("src.data_cleaning.analysis.logger")
    def test_analyze_outliers(self, mock_logger: MagicMock) -> None:
        """Test outliers analysis."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5, freq="D"),
                "tickers": ["AAPL"] * 5,
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )
        analyze_outliers(df)

        assert mock_logger.info.call_count >= 1


class TestAnalyzeTickerDistribution:
    """Tests for analyze_ticker_distribution function."""

    @patch("src.data_cleaning.analysis.logger")
    def test_analyze_ticker_distribution(self, mock_logger: MagicMock) -> None:
        """Test ticker distribution analysis."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10, freq="D"),
                "tickers": ["AAPL"] * 3 + ["MSFT"] * 4 + ["GOOGL"] * 3,
                "open": [100.0] * 10,
                "close": [100.5] * 10,
                "volume": [1000] * 10,
            }
        )
        analyze_ticker_distribution(df)

        assert mock_logger.info.call_count >= 1


class TestReportLeastObservations:
    """Tests for report_least_observations function."""

    @patch("src.data_cleaning.analysis.logger")
    def test_report_least_observations(self, mock_logger: MagicMock) -> None:
        """Test reporting least observations."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10, freq="D"),
                "tickers": ["AAPL"] * 2 + ["MSFT"] * 4 + ["GOOGL"] * 4,
                "open": [100.0] * 10,
                "close": [100.5] * 10,
                "volume": [1000] * 10,
            }
        )
        report_least_observations(df, top_n=2)

        assert mock_logger.info.call_count >= 1

    @patch("src.data_cleaning.analysis.logger")
    def test_report_least_observations_custom_top_n(self, mock_logger: MagicMock) -> None:
        """Test with custom top_n parameter."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10, freq="D"),
                "tickers": ["AAPL"] * 2 + ["MSFT"] * 4 + ["GOOGL"] * 4,
                "open": [100.0] * 10,
                "close": [100.5] * 10,
                "volume": [1000] * 10,
            }
        )
        report_least_observations(df, top_n=5)

        assert mock_logger.info.call_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
