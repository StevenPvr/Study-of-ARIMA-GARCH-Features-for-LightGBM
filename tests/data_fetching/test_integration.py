"""Integration tests for data_fetching module.

These tests verify the end-to-end workflow of the data fetching pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).parent.parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_fetching.data_fetching import download_sp500_data


class TestDownloadSP500DataIntegration:
    """Integration tests for download_sp500_data function."""

    @patch("src.data_fetching.data_fetching.combine_and_save_data")
    @patch("src.data_fetching.data_fetching.process_ticker_data")
    @patch("src.data_fetching.data_fetching.download_ticker_data")
    @patch("src.data_fetching.data_fetching.load_tickers")
    def test_download_sp500_data_success(
        self,
        mock_load_tickers: MagicMock,
        mock_download: MagicMock,
        mock_process: MagicMock,
        mock_combine: MagicMock,
    ) -> None:
        """Test successful end-to-end download of S&P 500 data."""
        # Setup: Mock a small list of tickers
        mock_load_tickers.return_value = ["AAPL", "MSFT"]

        # Setup: Mock successful downloads
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        sample_data = pd.DataFrame(
            {
                "Date": dates,
                "Open": [100.0, 101.0, 102.0],
                "High": [101.0, 102.0, 103.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1000, 1100, 1200],
            }
        )
        mock_download.return_value = sample_data

        # Setup: Mock processing
        processed_data = pd.DataFrame(
            {
                "date": dates,
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [1000, 1100, 1200],
                "tickers": ["AAPL", "AAPL", "AAPL"],
            }
        )
        mock_process.return_value = processed_data

        # Execute
        download_sp500_data()

        # Verify: load_tickers was called
        mock_load_tickers.assert_called_once()

        # Verify: download was called for each ticker
        assert mock_download.call_count == 2

        # Verify: process was called for each successful download
        assert mock_process.call_count == 2

        # Verify: combine_and_save_data was called with data and no failures
        mock_combine.assert_called_once()
        call_args = mock_combine.call_args[0]
        data_list, failed_tickers = call_args
        assert len(data_list) == 2  # Both tickers succeeded
        assert len(failed_tickers) == 0  # No failures

    @patch("src.data_fetching.data_fetching.combine_and_save_data")
    @patch("src.data_fetching.data_fetching.process_ticker_data")
    @patch("src.data_fetching.data_fetching.download_ticker_data")
    @patch("src.data_fetching.data_fetching.load_tickers")
    def test_download_sp500_data_with_failures(
        self,
        mock_load_tickers: MagicMock,
        mock_download: MagicMock,
        mock_process: MagicMock,
        mock_combine: MagicMock,
    ) -> None:
        """Test handling of partial failures during download."""
        # Setup: Mock tickers
        mock_load_tickers.return_value = ["AAPL", "INVALID", "MSFT"]

        # Setup: Mock downloads with one failure
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        sample_data = pd.DataFrame(
            {
                "Date": dates,
                "Open": [100.0, 101.0, 102.0],
                "High": [101.0, 102.0, 103.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1000, 1100, 1200],
            }
        )

        def download_side_effect(ticker, start, end):
            if ticker == "INVALID":
                return None  # Simulate failure
            return sample_data

        mock_download.side_effect = download_side_effect

        # Setup: Mock processing
        processed_data = pd.DataFrame(
            {
                "date": dates,
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [1000, 1100, 1200],
                "tickers": ["AAPL", "AAPL", "AAPL"],
            }
        )
        mock_process.return_value = processed_data

        # Execute
        download_sp500_data()

        # Verify: download was attempted for all tickers
        assert mock_download.call_count == 3

        # Verify: process was only called for successful downloads
        assert mock_process.call_count == 2

        # Verify: failed ticker was tracked
        mock_combine.assert_called_once()
        call_args = mock_combine.call_args[0]
        data_list, failed_tickers = call_args
        assert len(data_list) == 2  # Two successful
        assert len(failed_tickers) == 1  # One failure
        assert "INVALID" in failed_tickers

    @patch("src.data_fetching.data_fetching.combine_and_save_data")
    @patch("src.data_fetching.data_fetching.download_ticker_data")
    @patch("src.data_fetching.data_fetching.load_tickers")
    def test_download_sp500_data_progress_logging(
        self,
        mock_load_tickers: MagicMock,
        mock_download: MagicMock,
        mock_combine: MagicMock,
    ) -> None:
        """Test that progress is logged during download."""
        # Setup: Mock a list of tickers
        mock_load_tickers.return_value = ["TICK1", "TICK2", "TICK3"]
        mock_download.return_value = None  # All fail for simplicity

        # Execute - should not raise even with all failures
        download_sp500_data()

        # Verify: all tickers were attempted
        assert mock_download.call_count == 3

        # Verify: combine was called (even with all failures tracked)
        mock_combine.assert_called_once()
        call_args = mock_combine.call_args[0]
        data_list, failed_tickers = call_args
        assert len(data_list) == 0
        assert len(failed_tickers) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
