"""Unit tests for Wikipedia-related functions."""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).parent.parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data_fetching import wikipedia
from src.data_fetching.data_fetching import fetch_sp500_tickers
from src.data_fetching.wikipedia import (
    fetch_wikipedia_html,
    load_tickers,
    parse_tickers_from_table,
    save_tickers_to_csv,
)


class TestFetchSP500Tickers:
    """Tests for fetch_sp500_tickers function."""

    @patch("src.data_fetching.data_fetching.save_tickers_to_csv")
    @patch("src.data_fetching.data_fetching.parse_tickers_from_table")
    @patch("src.data_fetching.data_fetching.fetch_wikipedia_html")
    def test_fetch_sp500_tickers_success(
        self,
        mock_fetch_html: MagicMock,
        mock_parse_table: MagicMock,
        mock_save_csv: MagicMock,
    ) -> None:
        """Test successful fetching of S&P 500 tickers."""
        mock_fetch_html.return_value = b"<html>raw</html>"
        mock_parse_table.return_value = ["AAA", "BBB"]

        fetch_sp500_tickers()

        mock_fetch_html.assert_called_once_with()
        mock_parse_table.assert_called_once_with(b"<html>raw</html>")
        mock_save_csv.assert_called_once_with(["AAA", "BBB"])

    @patch("src.data_fetching.wikipedia.pd.read_html")
    def test_parse_tickers_from_table_normalizes_dots(self, mock_read_html: MagicMock) -> None:
        """Test that tickers with dots are normalized to dashes."""
        import pandas as pd

        mock_table = pd.DataFrame({"Symbol": ["BRK.B", "BF.B"]})
        mock_read_html.return_value = [mock_table]

        result = parse_tickers_from_table(b"<html></html>")

        assert result == ["BF-B", "BRK-B"]


class TestLoadTickers:
    """Tests for load_tickers function."""

    @patch("src.data_fetching.wikipedia.pd.read_csv")
    def test_load_tickers_success(self, mock_read_csv: MagicMock) -> None:
        """Test successful loading of tickers."""
        # Mock CSV file
        mock_df = MagicMock()
        mock_series = MagicMock()
        mock_series.tolist.return_value = ["MMM", "AOS", "ABT"]
        mock_df.__getitem__.return_value = mock_series
        mock_read_csv.return_value = mock_df

        with patch("src.data_fetching.wikipedia.SP500_TICKERS_FILE") as mock_file:
            mock_file.exists.return_value = True

            result = load_tickers()

            assert result == ["MMM", "AOS", "ABT"]
            mock_read_csv.assert_called_once()

    def test_load_tickers_file_not_found(self) -> None:
        """Test that missing file raises FileNotFoundError."""
        with patch("src.data_fetching.wikipedia.SP500_TICKERS_FILE") as mock_file:
            mock_file.exists.return_value = False

            with pytest.raises(FileNotFoundError):
                load_tickers()


class TestFetchSP500TickersErrors:
    """Tests for error cases in fetch_sp500_tickers function."""

    @patch("src.data_fetching.wikipedia.pd.read_html")
    def test_parse_tickers_from_table_no_tables(self, mock_read_html: MagicMock) -> None:
        """Test that no tables found raises RuntimeError."""
        mock_read_html.return_value = []

        with pytest.raises(RuntimeError, match="No tables found"):
            parse_tickers_from_table(b"<html>test</html>")

    @patch("src.data_fetching.wikipedia.pd.read_html")
    def test_parse_tickers_from_table_missing_symbol_column(
        self, mock_read_html: MagicMock
    ) -> None:
        """Test that missing Symbol column raises RuntimeError."""
        import pandas as pd

        mock_table = pd.DataFrame({"WrongColumn": ["A", "B", "C"]})
        mock_read_html.return_value = [mock_table]

        with pytest.raises(RuntimeError, match="S&P 500 constituents table not found"):
            parse_tickers_from_table(b"<html>test</html>")

    @patch("src.data_fetching.wikipedia.urllib.request.urlopen")
    def test_fetch_wikipedia_html_url_error(self, mock_urlopen: MagicMock) -> None:
        """Test that URL errors are handled properly."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection failed")

        with pytest.raises(RuntimeError, match="Failed to fetch Wikipedia page"):
            fetch_wikipedia_html()

    def test_save_tickers_to_csv_writes_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure save_tickers_to_csv writes CSV with correct content."""
        output_dir = tmp_path / "data"
        csv_file = output_dir / "tickers.csv"
        # Patch SP500_TICKERS_FILE in wikipedia module since it's imported at module level
        monkeypatch.setattr(wikipedia, "SP500_TICKERS_FILE", csv_file)

        save_tickers_to_csv(["AAA", "BBB"])

        assert csv_file.exists()
        content = csv_file.read_text(encoding="utf-8")
        assert "AAA" in content
        assert "BBB" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
