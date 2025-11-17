"""Unit tests for integrity fixing functions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest

from src.data_cleaning.integrity import apply_basic_integrity_fixes


class TestApplyBasicIntegrityFixes:
    """Tests for apply_basic_integrity_fixes function."""

    def test_empty_dataframe(self) -> None:
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result_df, counters = apply_basic_integrity_fixes(df)

        assert result_df.empty
        assert counters == {
            "duplicates_removed": 0,
            "missing_values_filled": 0,
            "missing_dates_filled": 0,
        }

    def test_no_duplicates_no_missing(self) -> None:
        """Test with clean data (no duplicates, no missing values)."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5, freq="D"),
                "tickers": ["AAPL"] * 5,
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )
        result_df, counters = apply_basic_integrity_fixes(df)

        assert len(result_df) == 5
        assert counters["duplicates_removed"] == 0
        assert counters["missing_values_filled"] == 0
        assert counters["missing_dates_filled"] == 0

    def test_duplicates_removed(self) -> None:
        """Test that duplicates are removed."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="D").tolist() * 2,
                "tickers": ["AAPL"] * 6,
                "open": [100.0] * 6,
                "close": [100.5] * 6,
                "volume": [1000] * 6,
            }
        )
        result_df, counters = apply_basic_integrity_fixes(df)

        assert len(result_df) == 3
        assert counters["duplicates_removed"] == 3
        assert counters["missing_values_filled"] == 0
        assert counters["missing_dates_filled"] == 0

    def test_missing_values_filled(self) -> None:
        """Test that missing values in critical columns are filled with zeros."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5, freq="D"),
                "tickers": ["AAPL"] * 5,
                "open": [100.0, np.nan, 102.0, np.nan, 104.0],
                "close": [100.5, 101.5, np.nan, 103.5, 104.5],
                "volume": [1000, 1100, np.nan, 1300, 1400],
            }
        )
        result_df, counters = apply_basic_integrity_fixes(df)

        assert len(result_df) == 5
        assert counters["duplicates_removed"] == 0
        assert counters["missing_values_filled"] == 4
        assert counters["missing_dates_filled"] == 0
        assert result_df["open"].isna().sum() == 0
        assert result_df["close"].isna().sum() == 0
        assert result_df["volume"].isna().sum() == 0
        assert (result_df["open"].fillna(999) == 0).any()
        assert (result_df["close"].fillna(999) == 0).any()
        assert (result_df["volume"].fillna(999) == 0).any()

    def test_multiple_tickers_sorted(self) -> None:
        """Test that multiple tickers are properly sorted."""
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"],
                "tickers": ["MSFT", "MSFT", "AAPL", "AAPL"],
                "open": [200.0, 201.0, 100.0, 101.0],
                "close": [200.5, 201.5, 100.5, 101.5],
                "volume": [2000, 2100, 1000, 1100],
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        result_df, _ = apply_basic_integrity_fixes(df)

        assert len(result_df) == 4
        assert result_df["tickers"].iloc[0] == "AAPL"
        assert result_df["tickers"].iloc[2] == "MSFT"
        assert result_df["date"].iloc[0] < result_df["date"].iloc[1]

    def test_missing_columns_handled(self) -> None:
        """Test that missing columns are handled gracefully."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="D"),
                "tickers": ["AAPL"] * 3,
            }
        )
        result_df, counters = apply_basic_integrity_fixes(df)

        assert len(result_df) == 3
        assert counters["duplicates_removed"] == 0
        assert counters["missing_values_filled"] == 0

    def test_combined_fixes(self) -> None:
        """Test that all fixes are applied together."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="D").tolist() * 2,
                "tickers": ["AAPL"] * 6,
                "open": [100.0, np.nan, 102.0, 100.0, np.nan, 102.0],
                "close": [100.5, 101.5, np.nan, 100.5, 101.5, np.nan],
                "volume": [1000, 1100, np.nan, 1000, 1100, np.nan],
            }
        )
        result_df, counters = apply_basic_integrity_fixes(df)

        assert len(result_df) == 3
        assert counters["duplicates_removed"] == 3
        assert counters["missing_values_filled"] == 3
        assert result_df["open"].isna().sum() == 0
        assert result_df["close"].isna().sum() == 0
        assert result_df["volume"].isna().sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
