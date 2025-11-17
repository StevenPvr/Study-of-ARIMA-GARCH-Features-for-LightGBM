"""Unit tests for data processing functions."""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).parent.parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest

from src.data_fetching.processing import process_ticker_data


class TestProcessTickerData:
    """Tests for process_ticker_data function."""

    def test_process_ticker_data_success(self, sample_ticker_data) -> None:
        """Test successful processing of ticker data."""
        result = process_ticker_data(sample_ticker_data, "TEST")

        assert len(result) == 5
        assert list(result.columns) == ["date", "high", "low", "close", "volume", "tickers"]
        assert all(result["tickers"] == "TEST")
        assert list(result["high"]) == [101.0, 102.0, 103.0, 104.0, 105.0]

    def test_process_ticker_data_empty_dataframe(self) -> None:
        """Test that empty DataFrame raises ValueError."""
        import pandas as pd

        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="ticker data for TEST DataFrame is empty"):
            process_ticker_data(empty_df, "TEST")

    def test_process_ticker_data_missing_columns(self) -> None:
        """Test that missing required columns raises KeyError."""
        import pandas as pd

        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        ticker_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
                # Missing Close and Volume
            },
            index=dates,
        )

        with pytest.raises(KeyError, match="Missing required columns"):
            process_ticker_data(ticker_data, "TEST")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
