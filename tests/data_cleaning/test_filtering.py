"""Unit tests for filtering functions."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from unittest.mock import MagicMock, patch

import pytest

# Dependencies are mocked in conftest.py before imports
from src.data_cleaning.data_cleaning import filter_by_membership
from src.data_cleaning.filtering import (
    calculate_observations_per_ticker,
    save_filtered_dataset,
    validate_filtering_inputs,
    write_filtered_dataset,
)
from tests.data_cleaning.test_helpers import (
    create_mock_dataframe,
    setup_mock_date_conversion,
    setup_mock_file_operations,
    setup_mock_integrity_fixes,
)


class TestFilterByMembership:
    """Tests for filter_by_membership function."""

    @patch("src.data_cleaning.data_cleaning.apply_basic_integrity_fixes")
    @patch("src.data_cleaning.validation.pd.to_datetime")
    @patch("src.data_cleaning.data_cleaning.DATASET_FILTERED_FILE")
    @patch("src.data_cleaning.validation.DATASET_FILE")
    @patch("src.data_cleaning.validation.pd.read_csv")
    @patch("src.data_cleaning.data_cleaning.logger")
    def test_filter_by_membership_success(
        self,
        mock_logger: MagicMock,
        mock_read_csv: MagicMock,
        mock_dataset_file: MagicMock,
        mock_filtered_file: MagicMock,
        mock_to_datetime: MagicMock,
        mock_apply_integrity_fixes: MagicMock,
    ) -> None:
        """Test successful filtering by membership."""
        mock_dataset_file.exists.return_value = True

        mock_df = create_mock_dataframe(row_count=5000, empty=False)
        setup_mock_date_conversion(mock_df, mock_to_datetime)
        mock_df_after = setup_mock_integrity_fixes(mock_df, after_drop_count=4000)

        # Mock apply_basic_integrity_fixes to return the cleaned DataFrame and counters
        mock_apply_integrity_fixes.return_value = (
            mock_df_after,
            {
                "duplicates_removed": 10,
                "missing_values_filled": 5,
                "missing_dates_filled": 100,
            },
        )

        setup_mock_file_operations(mock_filtered_file, mock_read_csv, mock_df)

        with patch.object(pd.DataFrame, "to_csv"), patch.object(pd.DataFrame, "to_parquet"):
            filter_by_membership()

        mock_read_csv.assert_called_once_with(mock_dataset_file)
        mock_apply_integrity_fixes.assert_called_once_with(mock_df)
        mock_filtered_file.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        import csv

        mock_df_after.to_csv.assert_called_once_with(
            mock_filtered_file,
            index=False,
            sep=",",
            encoding="utf-8",
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        mock_df_after.to_parquet.assert_called_once()
        mock_logger.info.assert_called()

    @patch("src.data_cleaning.data_cleaning.apply_basic_integrity_fixes")
    @patch("src.data_cleaning.validation.pd.to_datetime")
    @patch("src.data_cleaning.data_cleaning.DATASET_FILTERED_FILE")
    @patch("src.data_cleaning.validation.DATASET_FILE")
    @patch("src.data_cleaning.validation.pd.read_csv")
    @patch("src.data_cleaning.data_cleaning.logger")
    def test_filter_by_membership_no_removals(
        self,
        mock_logger: MagicMock,
        mock_read_csv: MagicMock,
        mock_dataset_file: MagicMock,
        mock_filtered_file: MagicMock,
        mock_to_datetime: MagicMock,
        mock_apply_integrity_fixes: MagicMock,
    ) -> None:
        """Test filtering when no integrity fixes are needed."""
        mock_dataset_file.exists.return_value = True

        mock_df = create_mock_dataframe(row_count=5000, empty=False)
        setup_mock_date_conversion(mock_df, mock_to_datetime)
        mock_df_after = setup_mock_integrity_fixes(mock_df, after_drop_count=5000)

        # Mock apply_basic_integrity_fixes to return the cleaned DataFrame and counters
        mock_apply_integrity_fixes.return_value = (
            mock_df_after,
            {
                "duplicates_removed": 0,
                "missing_values_filled": 0,
                "missing_dates_filled": 0,
            },
        )

        setup_mock_file_operations(mock_filtered_file, mock_read_csv, mock_df)

        with patch.object(pd.DataFrame, "to_csv"), patch.object(pd.DataFrame, "to_parquet"):
            filter_by_membership()

        mock_read_csv.assert_called_once()
        mock_apply_integrity_fixes.assert_called_once_with(mock_df)
        mock_df_after.to_csv.assert_called_once()
        mock_df_after.to_parquet.assert_called_once()

    @patch("src.data_cleaning.data_cleaning.apply_basic_integrity_fixes")
    @patch("src.data_cleaning.validation.pd.to_datetime")
    @patch("src.data_cleaning.data_cleaning.DATASET_FILTERED_FILE")
    @patch("src.data_cleaning.validation.DATASET_FILE")
    @patch("src.data_cleaning.validation.pd.read_csv")
    @patch("src.data_cleaning.data_cleaning.logger")
    def test_filter_by_membership_empty_result(
        self,
        mock_logger: MagicMock,
        mock_read_csv: MagicMock,
        mock_dataset_file: MagicMock,
        mock_filtered_file: MagicMock,
        mock_to_datetime: MagicMock,
        mock_apply_integrity_fixes: MagicMock,
    ) -> None:
        """Test filtering when result is empty."""
        mock_dataset_file.exists.return_value = True

        mock_df = create_mock_dataframe(row_count=1000, empty=False)
        setup_mock_date_conversion(mock_df, mock_to_datetime)
        mock_df_after = setup_mock_integrity_fixes(mock_df, after_drop_count=0, empty=True)

        # Mock apply_basic_integrity_fixes to return empty DataFrame and counters
        mock_apply_integrity_fixes.return_value = (
            mock_df_after,
            {
                "duplicates_removed": 0,
                "missing_values_filled": 0,
                "missing_dates_filled": 0,
            },
        )

        setup_mock_file_operations(mock_filtered_file, mock_read_csv, mock_df)

        with patch.object(pd.DataFrame, "to_csv"), patch.object(pd.DataFrame, "to_parquet"):
            filter_by_membership()

        mock_read_csv.assert_called_once()
        mock_apply_integrity_fixes.assert_called_once_with(mock_df)
        assert mock_logger.info.call_count > 0


class TestCalculateObservationsPerTicker:
    """Tests for calculate_observations_per_ticker function."""

    def test_basic_calculation(self) -> None:
        """Test basic observation count calculation."""
        df = pd.DataFrame(
            {
                "tickers": ["AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
                "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            }
        )
        result = calculate_observations_per_ticker(df)

        assert len(result) == 2
        assert result["AAPL"] == 2
        assert result["MSFT"] == 3

    def test_empty_dataframe_raises_valueerror(self) -> None:
        """Test that empty DataFrame raises ValueError."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="raw dataset DataFrame is empty"):
            calculate_observations_per_ticker(df)

    def test_missing_tickers_column_raises_keyerror(self) -> None:
        """Test that missing tickers column raises KeyError."""
        df = pd.DataFrame({"date": ["2020-01-01"]})
        with pytest.raises(KeyError, match="Missing required column: 'tickers'"):
            calculate_observations_per_ticker(df)


class TestValidateFilteringInputs:
    """Tests for validate_filtering_inputs function."""

    def test_valid_inputs(self) -> None:
        """Test with valid inputs."""
        df = pd.DataFrame({"tickers": ["AAPL", "MSFT"]})
        valid_tickers = ["AAPL", "MSFT"]
        validate_filtering_inputs(df, valid_tickers)

    def test_empty_dataframe_raises_valueerror(self) -> None:
        """Test that empty DataFrame raises ValueError."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="raw dataset DataFrame is empty"):
            validate_filtering_inputs(df, ["AAPL"])

    def test_missing_tickers_column_raises_keyerror(self) -> None:
        """Test that missing tickers column raises KeyError."""
        df = pd.DataFrame({"date": ["2020-01-01"]})
        with pytest.raises(KeyError, match="Missing required column: 'tickers'"):
            validate_filtering_inputs(df, ["AAPL"])

    def test_empty_valid_tickers_raises_valueerror(self) -> None:
        """Test that empty valid_tickers raises ValueError."""
        df = pd.DataFrame({"tickers": ["AAPL"]})
        with pytest.raises(ValueError, match="No valid tickers to filter"):
            validate_filtering_inputs(df, [])


class TestWriteFilteredDataset:
    """Tests for write_filtered_dataset function."""

    def test_write_csv_and_parquet(self) -> None:
        """Test that both CSV and Parquet files are written."""
        df = pd.DataFrame(
            {
                "tickers": ["AAPL", "MSFT"],
                "date": pd.date_range("2020-01-01", periods=2, freq="D"),
                "open": [100.0, 200.0],
                "close": [100.5, 200.5],
                "volume": [1000, 2000],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "filtered_dataset.csv"
            result_path = write_filtered_dataset(df, output_file=output_file)

            assert result_path == output_file
            assert output_file.exists()
            assert output_file.with_suffix(".parquet").exists()

            # Verify CSV content
            loaded_df = pd.read_csv(output_file)
            assert len(loaded_df) == 2
            assert "tickers" in loaded_df.columns

            # Verify Parquet content
            loaded_parquet = pd.read_parquet(output_file.with_suffix(".parquet"))
            assert len(loaded_parquet) == 2

    def test_write_with_default_path(self) -> None:
        """Test writing with default path."""
        df = pd.DataFrame({"tickers": ["AAPL"], "date": ["2020-01-01"]})
        df["date"] = pd.to_datetime(df["date"])

        with patch("src.data_cleaning.filtering.DATASET_FILTERED_FILE") as mock_file:
            mock_file.parent.mkdir = MagicMock()
            with patch.object(pd.DataFrame, "to_csv"), patch.object(pd.DataFrame, "to_parquet"):
                write_filtered_dataset(df)

            mock_file.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_write_failure_raises_oserror(self) -> None:
        """Test that write failure raises OSError."""
        df = pd.DataFrame({"tickers": ["AAPL"]})

        # Mock to_csv to raise OSError
        with patch.object(pd.DataFrame, "to_csv", side_effect=OSError("Cannot write file")):
            with pytest.raises(OSError, match="Failed to save filtered dataset"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    invalid_path = Path(tmpdir) / "filtered.csv"
                    write_filtered_dataset(df, output_file=invalid_path)


class TestSaveFilteredDataset:
    """Tests for save_filtered_dataset function."""

    def test_save_filtered_dataset_success(self) -> None:
        """Test successful filtering and saving."""
        df = pd.DataFrame(
            {
                "tickers": ["AAPL", "MSFT", "GOOGL"],
                "date": pd.date_range("2020-01-01", periods=3, freq="D"),
                "open": [100.0, 200.0, 300.0],
                "close": [100.5, 200.5, 300.5],
                "volume": [1000, 2000, 3000],
            }
        )
        valid_tickers = ["AAPL", "MSFT"]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            csv_path = tmpdir_path / "filtered.csv"
            parquet_path = tmpdir_path / "filtered.parquet"

            with patch("src.data_cleaning.filtering.DATASET_FILTERED_FILE", csv_path):
                with patch(
                    "src.data_cleaning.filtering.DATASET_FILTERED_PARQUET_FILE", parquet_path
                ):
                    result_df = save_filtered_dataset(df, valid_tickers)

                    assert len(result_df) == 2
                    assert set(result_df["tickers"].unique()) == {"AAPL", "MSFT"}

    def test_empty_result_warning(self) -> None:
        """Test that empty result logs warning."""
        df = pd.DataFrame(
            {
                "tickers": ["AAPL", "MSFT"],
                "date": pd.date_range("2020-01-01", periods=2, freq="D"),
            }
        )
        valid_tickers = ["GOOGL"]

        with patch("src.data_cleaning.filtering.logger") as mock_logger:
            with patch("src.data_cleaning.filtering.write_filtered_dataset"):
                result_df = save_filtered_dataset(df, valid_tickers)

                assert result_df.empty
                mock_logger.warning.assert_called_once()

    def test_invalid_inputs_raise_errors(self) -> None:
        """Test that invalid inputs raise appropriate errors."""
        df = pd.DataFrame({"tickers": ["AAPL"]})

        # Empty valid_tickers
        with pytest.raises(ValueError, match="No valid tickers to filter"):
            save_filtered_dataset(df, [])

        # Empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="raw dataset DataFrame is empty"):
            save_filtered_dataset(empty_df, ["AAPL"])

        # Missing tickers column
        no_tickers_df = pd.DataFrame({"date": ["2020-01-01"]})
        with pytest.raises(KeyError, match="Missing required column: 'tickers'"):
            save_filtered_dataset(no_tickers_df, ["AAPL"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
