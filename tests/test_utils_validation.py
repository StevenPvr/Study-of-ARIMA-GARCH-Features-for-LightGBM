"""Unit tests for look-ahead bias validation functions in utils.py."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


import pandas as pd
import pytest

from src.utils import log_split_dates, validate_temporal_split


class TestValidateTemporalSplit:
    """Tests for validate_temporal_split function."""

    def test_validate_temporal_split_valid_aggregated(self) -> None:
        """Test validation with valid temporal split for aggregated data."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "split": ["train"] * 80 + ["test"] * 20,
                "value": range(100),
            }
        )
        # Should not raise
        validate_temporal_split(df, function_name="test")

    def test_validate_temporal_split_valid_ticker_level(self) -> None:
        """Test validation with valid temporal split for ticker-level data."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "date": dates.tolist() + dates.tolist(),
                "ticker": ["AAPL"] * 100 + ["MSFT"] * 100,
                "split": ["train"] * 80 + ["test"] * 20 + ["train"] * 80 + ["test"] * 20,
                "value": range(200),
            }
        )
        # Should not raise
        validate_temporal_split(df, ticker_col="ticker", function_name="test")

    def test_validate_temporal_split_look_ahead_bias_aggregated(self) -> None:
        """Test validation detects look-ahead bias in aggregated data."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        # Mix train and test dates to create look-ahead bias
        splits = ["train"] * 50 + ["test"] * 30 + ["train"] * 20
        df = pd.DataFrame(
            {
                "date": dates,
                "split": splits,
                "value": range(100),
            }
        )
        with pytest.raises(ValueError, match="Look-ahead bias detected"):
            validate_temporal_split(df, function_name="test")

    def test_validate_temporal_split_look_ahead_bias_ticker_level(self) -> None:
        """Test validation detects look-ahead bias in ticker-level data."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        # Mix train and test dates for one ticker
        splits_aapl = ["train"] * 50 + ["test"] * 30 + ["train"] * 20
        splits_msft = ["train"] * 80 + ["test"] * 20
        df = pd.DataFrame(
            {
                "date": dates.tolist() + dates.tolist(),
                "ticker": ["AAPL"] * 100 + ["MSFT"] * 100,
                "split": splits_aapl + splits_msft,
                "value": range(200),
            }
        )
        with pytest.raises(ValueError, match="Look-ahead bias detected.*AAPL"):
            validate_temporal_split(df, ticker_col="ticker", function_name="test")

    def test_validate_temporal_split_missing_date_column(self) -> None:
        """Test validation raises error when date column is missing."""
        df = pd.DataFrame(
            {
                "split": ["train", "test"],
                "value": [1, 2],
            }
        )
        with pytest.raises(ValueError, match="must contain 'date' column"):
            validate_temporal_split(df, function_name="test")

    def test_validate_temporal_split_missing_split_column(self) -> None:
        """Test validation raises error when split column is missing."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "value": range(10),
            }
        )
        with pytest.raises(ValueError, match="must contain 'split' column"):
            validate_temporal_split(df, function_name="test")

    def test_validate_temporal_split_invalid_split_values(self) -> None:
        """Test validation raises error for invalid split values."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "split": ["train", "validation", "test"] * 3 + ["train"],
                "value": range(10),
            }
        )
        with pytest.raises(ValueError, match="Invalid split values"):
            validate_temporal_split(df, function_name="test")

    def test_validate_temporal_split_only_train(self) -> None:
        """Test validation handles data with only train split."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "split": ["train"] * 10,
                "value": range(10),
            }
        )
        # Should not raise (only warns)
        validate_temporal_split(df, function_name="test")

    def test_validate_temporal_split_only_test(self) -> None:
        """Test validation handles data with only test split."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "split": ["test"] * 10,
                "value": range(10),
            }
        )
        # Should not raise (only warns)
        validate_temporal_split(df, function_name="test")


class TestLogSplitDates:
    """Tests for log_split_dates function."""

    def test_log_split_dates_aggregated(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test logging for aggregated data."""
        import logging

        with caplog.at_level(logging.INFO):
            dates = pd.date_range("2020-01-01", periods=100, freq="D")
            df = pd.DataFrame(
                {
                    "date": dates,
                    "split": ["train"] * 80 + ["test"] * 20,
                    "value": range(100),
                }
            )
            log_split_dates(df, function_name="test")
            # Check that logs were generated
            assert "test" in caplog.text
            assert "Train split" in caplog.text or "Test split" in caplog.text

    def test_log_split_dates_ticker_level(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test logging for ticker-level data."""
        import logging

        with caplog.at_level(logging.INFO):
            dates = pd.date_range("2020-01-01", periods=100, freq="D")
            df = pd.DataFrame(
                {
                    "date": dates.tolist() + dates.tolist(),
                    "ticker": ["AAPL"] * 100 + ["MSFT"] * 100,
                    "split": ["train"] * 80 + ["test"] * 20 + ["train"] * 80 + ["test"] * 20,
                    "value": range(200),
                }
            )
            log_split_dates(df, ticker_col="ticker", function_name="test")
            # Check that logs were generated
            assert "test" in caplog.text
            assert "AAPL" in caplog.text or "MSFT" in caplog.text

    def test_log_split_dates_missing_columns(self) -> None:
        """Test logging handles missing columns gracefully."""
        df = pd.DataFrame(
            {
                "value": [1, 2, 3],
            }
        )
        # Should not raise
        log_split_dates(df, function_name="test")

    def test_log_split_dates_only_train(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test logging for data with only train split."""
        import logging

        with caplog.at_level(logging.INFO):
            dates = pd.date_range("2020-01-01", periods=10, freq="D")
            df = pd.DataFrame(
                {
                    "date": dates,
                    "split": ["train"] * 10,
                    "value": range(10),
                }
            )
            log_split_dates(df, function_name="test")
            assert "test" in caplog.text


if __name__ == "__main__":  # pragma: no cover - convenience runner
    pytest.main([__file__, "-q", "-x"])
