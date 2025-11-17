"""Unit tests for reporting functions."""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).parent.parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.data_fetching import reporting
from src.data_fetching.reporting import combine_and_save_data


class TestCombineAndSaveData:
    """Tests for combine_and_save_data function."""

    def test_combine_and_save_data_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test successful combining and saving of data."""
        import pandas as pd

        data_dir = tmp_path / "data"
        dataset_file = data_dir / "dataset.csv"
        fetch_report_file = data_dir / "fetch_report.json"

        mock_logger = MagicMock()
        monkeypatch.setattr(reporting, "logger", mock_logger)
        monkeypatch.setattr(reporting, "DATASET_FILE", dataset_file)
        monkeypatch.setattr(reporting, "FETCH_REPORT_FILE", fetch_report_file)

        mock_save = MagicMock()
        monkeypatch.setattr(reporting, "save_parquet_and_csv", mock_save)

        # Create sample dataframes
        df1 = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3, freq="D"),
                "tickers": ["A", "A", "A"],
                "open": [100.0, 101.0, 102.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [1000, 1100, 1200],
            }
        )
        df2 = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-04", periods=2, freq="D"),
                "tickers": ["B", "B"],
                "open": [200.0, 201.0],
                "close": [200.5, 201.5],
                "volume": [2000, 2100],
            }
        )

        # Mock file operations to avoid actual file writing
        combine_and_save_data([df1, df2], ["C"])

        assert data_dir.exists()
        mock_save.assert_called_once()
        saved_dataset, saved_path = mock_save.call_args[0]
        assert isinstance(saved_dataset, pd.DataFrame)
        assert saved_path == dataset_file.with_suffix(".parquet")
        assert fetch_report_file.exists()

    def test_combine_and_save_data_empty_list(self) -> None:
        """Test that empty data list raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No data downloaded to combine"):
            combine_and_save_data([], [])

    def test_combine_and_save_data_empty_after_dropna(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that empty dataset after dropna raises RuntimeError."""
        import pandas as pd

        data_dir = tmp_path / "data"
        dataset_file = data_dir / "dataset.csv"
        fetch_report_file = data_dir / "fetch_report.json"
        monkeypatch.setattr(reporting, "DATASET_FILE", dataset_file)
        monkeypatch.setattr(reporting, "FETCH_REPORT_FILE", fetch_report_file)

        # Create empty dataframe (simulating all rows dropped during processing)
        empty_df = pd.DataFrame()

        with pytest.raises(RuntimeError, match="Dataset is empty after removing NaN values"):
            combine_and_save_data([empty_df], [])

    def test_combine_dataframes_mixed_timezone_timestamps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that combining DataFrames with mixed timezone timestamps works correctly."""
        import pandas as pd

        from src.data_fetching.reporting import _combine_dataframes

        # Create DataFrames with different timezone settings
        tz_naive_dates = pd.date_range("2024-01-01", periods=3, freq="D")
        tz_aware_dates = pd.date_range("2024-01-04", periods=2, freq="D", tz="UTC")

        df1 = pd.DataFrame(
            {
                "date": tz_naive_dates,
                "tickers": ["A", "A", "A"],
                "close": [100.5, 101.5, 102.5],
            }
        )

        df2 = pd.DataFrame(
            {
                "date": tz_aware_dates,
                "tickers": ["B", "B"],
                "close": [200.5, 201.5],
            }
        )

        # This should not raise an error
        result = _combine_dataframes([df1, df2])

        # Verify the result is sorted correctly
        assert len(result) == 5
        assert result["date"].is_monotonic_increasing  # Should be sorted
        assert list(result["tickers"]) == ["A", "A", "A", "B", "B"]  # Sorted by date then ticker

        # Verify all dates are tz-naive after processing
        assert all(pd.isna(dt.tz) for dt in result["date"] if pd.notna(dt))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
