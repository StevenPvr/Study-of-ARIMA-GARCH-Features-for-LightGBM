"""Unit tests for download-related functions."""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).parent.parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.data_fetching.download import download_ticker_data, get_date_range


class TestGetDateRange:
    """Tests for get_date_range function."""

    def test_get_date_range(self) -> None:
        """Test that date range returns fixed dates for reproducibility."""
        start, end = get_date_range()

        # Verify fixed dates: 2013-01-01 to 2024-12-31
        assert start == datetime(2013, 1, 1)
        assert end == datetime(2024, 12, 31)
        assert start < end


class TestDownloadTickerData:
    """Tests for download_ticker_data function."""

    @patch("src.data_fetching.download.yf.Ticker")
    def test_download_ticker_data_success(self, mock_yf_ticker: MagicMock) -> None:
        """Test successful download of ticker data."""
        import pandas as pd

        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        mock_data = pd.DataFrame(
            {
                "Date": dates,
                "Open": [100.0, 101.0, 102.0],
                "High": [101.0, 102.0, 103.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1000, 1100, 1200],
            },
        )
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_data.set_index("Date")
        mock_yf_ticker.return_value = mock_ticker_instance

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)

        result = download_ticker_data("TEST", start_date, end_date)

        assert result is not None
        assert len(result) == 3
        mock_yf_ticker.assert_called_once_with("TEST")

    def test_download_ticker_data_empty_ticker(self) -> None:
        """Test that empty ticker returns None."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)

        result = download_ticker_data("", start_date, end_date)
        assert result is None

    def test_download_ticker_data_none_ticker(self) -> None:
        """Test that None ticker returns None."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)

        result = download_ticker_data(None, start_date, end_date)  # type: ignore[arg-type]
        assert result is None

    def test_download_ticker_data_invalid_date_range(self) -> None:
        """Test that invalid date range returns None."""
        start_date = datetime(2024, 1, 10)
        end_date = datetime(2024, 1, 1)  # start > end

        result = download_ticker_data("TEST", start_date, end_date)
        assert result is None

    @patch("src.data_fetching.download._download_yfinance_data")
    def test_download_ticker_data_empty_result(self, mock_yfinance_data: MagicMock) -> None:
        """Test that empty download result returns None."""
        import pandas as pd

        mock_yfinance_data.return_value = pd.DataFrame()

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)

        result = download_ticker_data("TEST", start_date, end_date)
        assert result is None

    @patch("src.data_fetching.download._download_yfinance_data")
    def test_download_ticker_data_exception_handling(self, mock_yfinance_data: MagicMock) -> None:
        """Test that exceptions during download are handled gracefully."""
        mock_yfinance_data.side_effect = ValueError("Invalid ticker")

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)

        result = download_ticker_data("INVALID", start_date, end_date)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
