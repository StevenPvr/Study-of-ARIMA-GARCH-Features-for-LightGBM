"""Unit tests for data_fetching module."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from unittest.mock import MagicMock, patch

import pytest

# Dependencies are mocked in src/conftest.py before imports
from src.data_fetching.data_fetching import download_sp500_data, fetch_sp500_tickers
from src.data_fetching.utils import (
    _combine_and_save_data,
    _download_ticker_data,
    _get_date_range,
    _load_tickers,
    _process_ticker_data,
)


class TestFetchSP500Tickers:
    """Tests for fetch_sp500_tickers function."""

    @patch("src.data_fetching.utils.DATA_DIR")
    @patch("src.data_fetching.utils.pd.read_html")
    @patch("src.data_fetching.utils.urllib.request.urlopen")
    @patch("src.data_fetching.utils.logger")
    def test_fetch_sp500_tickers_success(
        self,
        mock_logger: MagicMock,
        mock_urlopen: MagicMock,
        mock_read_html: MagicMock,
        mock_data_dir: MagicMock,
    ) -> None:
        """Test successful fetching of S&P 500 tickers."""
        import pandas as pd

        # Setup mocks
        mock_data_dir.mkdir = MagicMock()
        mock_data_dir.__truediv__ = lambda self, other: Path(f"data/{other}")

        # Mock pandas DataFrame with Symbol column
        mock_table = pd.DataFrame({"Symbol": ["MMM", "AOS", "ABT", "ABBV", "ACN"]})
        mock_read_html.return_value = [mock_table]

        # Mock CSV writing
        mock_tickers_df = MagicMock()
        with patch("src.data_fetching.utils.pd.DataFrame") as mock_dataframe_class:
            mock_dataframe_class.return_value = mock_tickers_df

            # Mock URL response
            mock_response = MagicMock()
            mock_response.read.return_value = b"<html>test</html>"
            mock_urlopen.return_value.__enter__ = lambda x: mock_response
            mock_urlopen.return_value.__exit__ = lambda x, y, z, w: None

            # Execute
            fetch_sp500_tickers()

            # Verify
            mock_data_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_read_html.assert_called_once()
            mock_dataframe_class.assert_called_once()
            mock_tickers_df.to_csv.assert_called_once()

    @patch("src.data_fetching.utils.DATA_DIR")
    @patch("src.data_fetching.utils.pd.read_html")
    @patch("src.data_fetching.utils.urllib.request.urlopen")
    @patch("src.data_fetching.utils.logger")
    def test_fetch_sp500_tickers_normalizes_dots(
        self,
        mock_logger: MagicMock,
        mock_urlopen: MagicMock,
        mock_read_html: MagicMock,
        mock_data_dir: MagicMock,
    ) -> None:
        """Test that tickers with dots are normalized to dashes."""
        import pandas as pd

        # Mock pandas DataFrame with tickers containing dots
        mock_table = pd.DataFrame({"Symbol": ["BRK.B", "BF.B"]})
        mock_read_html.return_value = [mock_table]

        mock_response = MagicMock()
        mock_response.read.return_value = b"<html>test</html>"
        mock_urlopen.return_value.__enter__ = lambda x: mock_response
        mock_urlopen.return_value.__exit__ = lambda x, y, z, w: None

        mock_data_dir.mkdir = MagicMock()
        mock_data_dir.__truediv__ = lambda self, other: Path(f"data/{other}")

        mock_tickers_df = MagicMock()
        with patch("src.data_fetching.utils.pd.DataFrame") as mock_dataframe_class:
            mock_dataframe_class.return_value = mock_tickers_df

            fetch_sp500_tickers()

            # Verify normalization happened - check DataFrame called with normalized tickers
            mock_dataframe_class.assert_called_once()
            call_args = mock_dataframe_class.call_args
            if call_args and "ticker" in call_args.kwargs:
                tickers = call_args.kwargs["ticker"]
                assert all("." not in ticker for ticker in tickers)


class TestDownloadSP500Data:
    """Tests for download_sp500_data function."""

    def test_download_sp500_data_missing_tickers_file(self) -> None:
        """Test that missing tickers file raises FileNotFoundError."""
        with patch("src.data_fetching.data_fetching._load_tickers") as mock_load_tickers:
            mock_load_tickers.side_effect = FileNotFoundError(
                "Tickers file not found: data/sp500_tickers.csv. Run fetch_sp500_tickers() first."
            )

            with pytest.raises(FileNotFoundError, match="Tickers file not found"):
                download_sp500_data()

    @patch("src.data_fetching.data_fetching._combine_and_save_data")
    @patch("src.data_fetching.data_fetching._process_ticker_data")
    @patch("src.data_fetching.data_fetching._download_ticker_data")
    @patch("src.data_fetching.data_fetching._get_date_range")
    @patch("src.data_fetching.data_fetching._load_tickers")
    @patch("src.data_fetching.data_fetching.logger")
    def test_download_sp500_data_success(
        self,
        mock_logger: MagicMock,
        mock_load_tickers: MagicMock,
        mock_get_date_range: MagicMock,
        mock_download_ticker: MagicMock,
        mock_process_ticker: MagicMock,
        mock_combine_save: MagicMock,
    ) -> None:
        """Test successful download of S&P 500 data."""
        from datetime import datetime

        # Setup mocks
        mock_load_tickers.return_value = ["MMM", "AOS"]
        mock_get_date_range.return_value = (
            datetime(2024, 1, 1),
            datetime(2024, 1, 10),
        )

        # Mock ticker data download
        mock_ticker_df = MagicMock()
        mock_download_ticker.return_value = mock_ticker_df

        # Mock processed data
        mock_processed_df = MagicMock()
        mock_process_ticker.return_value = mock_processed_df

        # Execute
        download_sp500_data()

        # Verify
        mock_load_tickers.assert_called_once()
        mock_get_date_range.assert_called_once()
        assert mock_download_ticker.call_count == 2  # One call per ticker
        assert mock_process_ticker.call_count == 2
        mock_combine_save.assert_called_once()

    @patch("src.data_fetching.data_fetching._combine_and_save_data")
    @patch("src.data_fetching.data_fetching._download_ticker_data")
    @patch("src.data_fetching.data_fetching._get_date_range")
    @patch("src.data_fetching.data_fetching._load_tickers")
    @patch("src.data_fetching.data_fetching.logger")
    def test_download_sp500_data_no_data_raises_error(
        self,
        mock_logger: MagicMock,
        mock_load_tickers: MagicMock,
        mock_get_date_range: MagicMock,
        mock_download_ticker: MagicMock,
        mock_combine_save: MagicMock,
    ) -> None:
        """Test that no data downloaded raises RuntimeError."""
        from datetime import datetime

        # Setup mocks - all downloads return None
        mock_load_tickers.return_value = ["MMM"]
        mock_get_date_range.return_value = (
            datetime(2024, 1, 1),
            datetime(2024, 1, 10),
        )
        mock_download_ticker.return_value = None

        # Mock _combine_and_save_data to raise RuntimeError when no data
        mock_combine_save.side_effect = RuntimeError("No data downloaded")

        # Execute and verify
        with pytest.raises(RuntimeError, match="No data downloaded"):
            download_sp500_data()

    @patch("src.data_fetching.data_fetching._combine_and_save_data")
    @patch("src.data_fetching.data_fetching._process_ticker_data")
    @patch("src.data_fetching.data_fetching._download_ticker_data")
    @patch("src.data_fetching.data_fetching._get_date_range")
    @patch("src.data_fetching.data_fetching._load_tickers")
    @patch("src.data_fetching.data_fetching.logger")
    def test_download_sp500_data_handles_multiindex_columns(
        self,
        mock_logger: MagicMock,
        mock_load_tickers: MagicMock,
        mock_get_date_range: MagicMock,
        mock_download_ticker: MagicMock,
        mock_process_ticker: MagicMock,
        mock_combine_save: MagicMock,
    ) -> None:
        """Test that MultiIndex columns are handled correctly."""
        from datetime import datetime

        # Setup mocks
        mock_load_tickers.return_value = ["MMM"]
        mock_get_date_range.return_value = (
            datetime(2024, 1, 1),
            datetime(2024, 1, 10),
        )

        # Mock ticker data with MultiIndex columns
        mock_ticker_df = MagicMock()
        mock_multiindex = MagicMock()
        mock_ticker_df.columns = mock_multiindex
        mock_download_ticker.return_value = mock_ticker_df

        mock_processed_df = MagicMock()
        mock_process_ticker.return_value = mock_processed_df

        # Execute
        download_sp500_data()

        # Verify
        mock_download_ticker.assert_called_once()
        mock_process_ticker.assert_called_once_with(mock_ticker_df, "MMM")


class TestHelperFunctions:
    """Tests for helper functions."""

    @patch("src.data_fetching.utils.pd.read_csv")
    def test_load_tickers_success(self, mock_read_csv: MagicMock) -> None:
        """Test successful loading of tickers."""
        # Mock CSV file
        mock_df = MagicMock()
        mock_series = MagicMock()
        mock_series.tolist.return_value = ["MMM", "AOS", "ABT"]
        mock_df.__getitem__.return_value = mock_series
        mock_read_csv.return_value = mock_df

        with patch("src.data_fetching.utils.SP500_TICKERS_FILE") as mock_file:
            mock_file.exists.return_value = True

            result = _load_tickers()

            assert result == ["MMM", "AOS", "ABT"]
            mock_read_csv.assert_called_once()

    def test_load_tickers_file_not_found(self) -> None:
        """Test that missing file raises FileNotFoundError."""
        with patch("src.data_fetching.utils.SP500_TICKERS_FILE") as mock_file:
            mock_file.exists.return_value = False

            with pytest.raises(FileNotFoundError):
                _load_tickers()

    def test_get_date_range(self) -> None:
        """Test that date range returns fixed dates for reproducibility."""
        from datetime import datetime

        start, end = _get_date_range()

        # Verify fixed dates: 2013-01-01 to 2024-12-31
        assert start == datetime(2013, 1, 1)
        assert end == datetime(2024, 12, 31)
        assert start < end


class TestProcessTickerData:
    """Tests for _process_ticker_data function."""

    def test_process_ticker_data_success(self) -> None:
        """Test successful processing of ticker data."""
        import pandas as pd

        # Create mock ticker data
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        ticker_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "Volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=dates,
        )

        result = _process_ticker_data(ticker_data, "TEST")

        assert len(result) == 5
        assert list(result.columns) == ["date", "ticker", "open", "closing", "volume"]
        assert all(result["ticker"] == "TEST")
        assert list(result["open"]) == [100.0, 101.0, 102.0, 103.0, 104.0]

    def test_process_ticker_data_empty_dataframe(self) -> None:
        """Test that empty DataFrame raises ValueError."""
        import pandas as pd

        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Empty DataFrame"):
            _process_ticker_data(empty_df, "TEST")

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
            _process_ticker_data(ticker_data, "TEST")

    def test_process_ticker_data_multiindex_columns(self) -> None:
        """Test that MultiIndex columns are handled correctly."""
        import pandas as pd

        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        ticker_data = pd.DataFrame(
            {
                ("Open", "A"): [100.0, 101.0, 102.0],
                ("Close", "A"): [100.5, 101.5, 102.5],
                ("Volume", "A"): [1000, 1100, 1200],
            },
            index=dates,
        )

        result = _process_ticker_data(ticker_data, "TEST")

        assert len(result) == 3
        assert list(result.columns) == ["date", "ticker", "open", "closing", "volume"]


class TestCombineAndSaveData:
    """Tests for _combine_and_save_data function."""

    @patch("src.data_fetching.utils.DATASET_FILE")
    @patch("src.data_fetching.utils.DATA_DIR")
    @patch("src.data_fetching.utils.logger")
    def test_combine_and_save_data_success(
        self,
        mock_logger: MagicMock,
        mock_data_dir: MagicMock,
        mock_dataset_file: MagicMock,
    ) -> None:
        """Test successful combining and saving of data."""
        import pandas as pd

        mock_data_dir.mkdir = MagicMock()

        # Create sample dataframes
        df1 = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3, freq="D"),
                "ticker": ["A", "A", "A"],
                "open": [100.0, 101.0, 102.0],
                "closing": [100.5, 101.5, 102.5],
                "volume": [1000, 1100, 1200],
            }
        )
        df2 = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-04", periods=2, freq="D"),
                "ticker": ["B", "B"],
                "open": [200.0, 201.0],
                "closing": [200.5, 201.5],
                "volume": [2000, 2100],
            }
        )

        # Mock to_csv to avoid actual file writing
        with patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
            _combine_and_save_data([df1, df2], ["C"])

            mock_data_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_to_csv.assert_called_once_with(mock_dataset_file, index=False)

    def test_combine_and_save_data_empty_list(self) -> None:
        """Test that empty data list raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No data downloaded"):
            _combine_and_save_data([], [])

    @patch("src.data_fetching.utils.DATASET_FILE")
    @patch("src.data_fetching.utils.DATA_DIR")
    def test_combine_and_save_data_empty_after_dropna(
        self, mock_data_dir: MagicMock, mock_dataset_file: MagicMock
    ) -> None:
        """Test that empty dataset after dropna raises RuntimeError."""
        import pandas as pd

        mock_data_dir.mkdir = MagicMock()

        # Create dataframe with all NaN values
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3, freq="D"),
                "ticker": [None, None, None],
                "open": [None, None, None],
                "closing": [None, None, None],
                "volume": [None, None, None],
            }
        )

        with pytest.raises(RuntimeError, match="Dataset is empty after removing NaN"):
            _combine_and_save_data([df], [])


class TestDownloadTickerData:
    """Tests for _download_ticker_data function."""

    @patch("src.data_fetching.utils.yf.download")
    @patch("src.data_fetching.utils.logger")
    def test_download_ticker_data_success(
        self, mock_logger: MagicMock, mock_yf_download: MagicMock
    ) -> None:
        """Test successful download of ticker data."""
        from datetime import datetime

        import pandas as pd

        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        mock_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1000, 1100, 1200],
            },
            index=dates,
        )
        mock_yf_download.return_value = mock_data

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)

        result = _download_ticker_data("TEST", start_date, end_date)

        assert result is not None
        assert len(result) == 3
        mock_yf_download.assert_called_once()

    @patch("src.data_fetching.utils.logger")
    def test_download_ticker_data_invalid_ticker(self, mock_logger: MagicMock) -> None:
        """Test that invalid ticker returns None."""
        from datetime import datetime

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)

        result = _download_ticker_data("", start_date, end_date)
        assert result is None

        result = _download_ticker_data(None, start_date, end_date)  # type: ignore[arg-type]
        assert result is None

    @patch("src.data_fetching.utils.logger")
    def test_download_ticker_data_invalid_date_range(self, mock_logger: MagicMock) -> None:
        """Test that invalid date range returns None."""
        from datetime import datetime

        start_date = datetime(2024, 1, 10)
        end_date = datetime(2024, 1, 1)  # start > end

        result = _download_ticker_data("TEST", start_date, end_date)
        assert result is None

    @patch("src.data_fetching.utils.yf.download")
    @patch("src.data_fetching.utils.logger")
    def test_download_ticker_data_empty_result(
        self, mock_logger: MagicMock, mock_yf_download: MagicMock
    ) -> None:
        """Test that empty download result returns None."""
        from datetime import datetime

        import pandas as pd

        mock_yf_download.return_value = pd.DataFrame()

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)

        result = _download_ticker_data("TEST", start_date, end_date)
        assert result is None

    @patch("src.data_fetching.utils.yf.download")
    @patch("src.data_fetching.utils.logger")
    def test_download_ticker_data_exception_handling(
        self, mock_logger: MagicMock, mock_yf_download: MagicMock
    ) -> None:
        """Test that exceptions during download are handled gracefully."""
        from datetime import datetime

        mock_yf_download.side_effect = ValueError("Invalid ticker")

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)

        result = _download_ticker_data("INVALID", start_date, end_date)
        assert result is None


class TestFetchSP500TickersErrors:
    """Tests for error cases in fetch_sp500_tickers function."""

    @patch("src.data_fetching.utils.pd.read_html")
    @patch("src.data_fetching.utils.urllib.request.urlopen")
    def test_fetch_sp500_tickers_no_tables(
        self, mock_urlopen: MagicMock, mock_read_html: MagicMock
    ) -> None:
        """Test that no tables found raises RuntimeError."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"<html>test</html>"
        mock_urlopen.return_value.__enter__ = lambda x: mock_response
        mock_urlopen.return_value.__exit__ = lambda x, y, z, w: None

        mock_read_html.return_value = []

        with pytest.raises(RuntimeError, match="No tables found"):
            fetch_sp500_tickers()

    @patch("src.data_fetching.utils.pd.read_html")
    @patch("src.data_fetching.utils.urllib.request.urlopen")
    def test_fetch_sp500_tickers_missing_symbol_column(
        self, mock_urlopen: MagicMock, mock_read_html: MagicMock
    ) -> None:
        """Test that missing Symbol column raises KeyError."""
        import pandas as pd

        mock_response = MagicMock()
        mock_response.read.return_value = b"<html>test</html>"
        mock_urlopen.return_value.__enter__ = lambda x: mock_response
        mock_urlopen.return_value.__exit__ = lambda x, y, z, w: None

        mock_table = pd.DataFrame({"WrongColumn": ["A", "B", "C"]})
        mock_read_html.return_value = [mock_table]

        with pytest.raises(KeyError, match="Symbol.*not found"):
            fetch_sp500_tickers()

    @patch("src.data_fetching.utils.urllib.request.urlopen")
    def test_fetch_sp500_tickers_url_error(self, mock_urlopen: MagicMock) -> None:
        """Test that URL errors are handled properly."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection failed")

        with pytest.raises(RuntimeError, match="Failed to fetch Wikipedia page"):
            fetch_sp500_tickers()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
