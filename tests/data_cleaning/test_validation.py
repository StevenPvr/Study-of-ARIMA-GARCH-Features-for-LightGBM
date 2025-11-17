"""Unit tests for validation functions."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from unittest.mock import MagicMock, patch

from src.data_cleaning.validation import (
    REQUIRED_OHLCV_COLUMNS,
    assert_not_empty,
    assert_required_columns,
    convert_and_validate_dates,
    load_dataset,
    validate_columns,
    validate_file_exists,
)


class TestAssertRequiredColumns:
    """Tests for assert_required_columns function."""

    def test_all_columns_present(self) -> None:
        """Test with all required columns present."""
        df = pd.DataFrame(
            {
                "date": ["2020-01-01"],
                "tickers": ["AAPL"],
                "open": [100.0],
                "close": [100.5],
                "volume": [1000],
            }
        )
        assert_required_columns(df, REQUIRED_OHLCV_COLUMNS)

    def test_missing_column_raises_keyerror(self) -> None:
        """Test that missing column raises KeyError."""
        df = pd.DataFrame(
            {
                "date": ["2020-01-01"],
                "tickers": ["AAPL"],
                "open": [100.0],
                "close": [100.5],
            }
        )
        with pytest.raises(KeyError, match="Missing required columns"):
            assert_required_columns(df, REQUIRED_OHLCV_COLUMNS)

    def test_multiple_missing_columns(self) -> None:
        """Test with multiple missing columns."""
        df = pd.DataFrame({"date": ["2020-01-01"]})
        with pytest.raises(KeyError, match="Missing required columns"):
            assert_required_columns(df, REQUIRED_OHLCV_COLUMNS)


class TestAssertNotEmpty:
    """Tests for assert_not_empty function."""

    def test_non_empty_dataframe(self) -> None:
        """Test with non-empty DataFrame."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        assert_not_empty(df, name="test context")

    def test_empty_dataframe_raises_valueerror(self) -> None:
        """Test that empty DataFrame raises ValueError."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="test context DataFrame is empty"):
            assert_not_empty(df, name="test context")


class TestValidateFileExists:
    """Tests for validate_file_exists function."""

    @patch("src.data_cleaning.validation._validate_file_exists")
    @patch("src.data_cleaning.validation.DATASET_FILE")
    def test_file_exists(self, mock_dataset_file: MagicMock, mock_validate: MagicMock) -> None:
        """Test when file exists."""
        mock_dataset_file.exists.return_value = True
        validate_file_exists()
        mock_validate.assert_called_once_with(mock_dataset_file)

    @patch("src.data_cleaning.validation._validate_file_exists")
    @patch("src.data_cleaning.validation.DATASET_FILE")
    def test_file_not_exists_raises_error(
        self, mock_dataset_file: MagicMock, mock_validate: MagicMock
    ) -> None:
        """Test that missing file raises FileNotFoundError."""
        mock_validate.side_effect = FileNotFoundError("File not found")
        with pytest.raises(FileNotFoundError):
            validate_file_exists()


class TestValidateColumns:
    """Tests for validate_columns function."""

    def test_valid_columns(self) -> None:
        """Test with valid columns."""
        df = pd.DataFrame(
            {
                "date": ["2020-01-01"],
                "tickers": ["AAPL"],
                "open": [100.0],
                "close": [100.5],
                "volume": [1000],
            }
        )
        validate_columns(df)

    def test_missing_column_raises_keyerror(self) -> None:
        """Test that missing column raises KeyError."""
        df = pd.DataFrame({"date": ["2020-01-01"], "tickers": ["AAPL"]})
        with pytest.raises(KeyError, match="Missing required columns"):
            validate_columns(df)


class TestConvertAndValidateDates:
    """Tests for convert_and_validate_dates function."""

    def test_valid_dates(self) -> None:
        """Test with valid date strings."""
        df = pd.DataFrame({"date": ["2020-01-01", "2020-01-02", "2020-01-03"]})
        convert_and_validate_dates(df)

        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert len(df) == 3

    def test_missing_date_column_raises_keyerror(self) -> None:
        """Test that missing date column raises KeyError."""
        df = pd.DataFrame({"tickers": ["AAPL"]})
        with pytest.raises(KeyError, match="Missing required column: 'date'"):
            convert_and_validate_dates(df)

    def test_invalid_dates_raises_valueerror(self) -> None:
        """Test that invalid dates raise ValueError."""
        import warnings

        df = pd.DataFrame({"date": ["invalid", "2020-01-02", "also-invalid"]})
        # Filter the expected warning from pandas about date parsing
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message=".*Could not infer format.*"
            )
            with pytest.raises(ValueError, match="invalid date value"):
                convert_and_validate_dates(df)

    def test_timezone_aware_dates_normalized(self) -> None:
        """Test that timezone-aware dates are normalized."""
        df = pd.DataFrame({"date": ["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"]})
        convert_and_validate_dates(df)

        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert df["date"].dt.tz is None


class TestLoadDataset:
    """Tests for load_dataset function."""

    @patch("src.data_cleaning.validation.convert_and_validate_dates")
    @patch("src.data_cleaning.validation.validate_columns")
    @patch("src.data_cleaning.validation.assert_not_empty")
    @patch("src.data_cleaning.validation.validate_file_exists")
    @patch("src.data_cleaning.validation.pd.read_csv")
    @patch("src.data_cleaning.validation.DATASET_FILE")
    def test_load_dataset_success(
        self,
        mock_dataset_file: MagicMock,
        mock_read_csv: MagicMock,
        mock_validate_file: MagicMock,
        mock_assert_not_empty: MagicMock,
        mock_validate_columns: MagicMock,
        mock_convert_dates: MagicMock,
    ) -> None:
        """Test successful dataset loading."""
        mock_df = pd.DataFrame(
            {
                "date": ["2020-01-01"],
                "tickers": ["AAPL"],
                "open": [100.0],
                "close": [100.5],
                "volume": [1000],
            }
        )
        mock_read_csv.return_value = mock_df

        result = load_dataset()

        mock_validate_file.assert_called_once()
        mock_read_csv.assert_called_once_with(mock_dataset_file)
        mock_validate_columns.assert_called_once_with(mock_df)
        mock_assert_not_empty.assert_called_once()
        mock_convert_dates.assert_called_once_with(mock_df)
        assert result is mock_df

    @patch("src.data_cleaning.validation.validate_file_exists")
    @patch("src.data_cleaning.validation.pd.read_csv")
    def test_load_dataset_file_not_found(
        self, mock_read_csv: MagicMock, mock_validate_file: MagicMock
    ) -> None:
        """Test that missing file raises FileNotFoundError."""
        mock_validate_file.side_effect = FileNotFoundError("File not found")
        with pytest.raises(FileNotFoundError):
            load_dataset()

    @patch("src.data_cleaning.validation.validate_file_exists")
    @patch("src.data_cleaning.validation.pd.read_csv")
    def test_load_dataset_missing_columns(
        self, mock_read_csv: MagicMock, mock_validate_file: MagicMock
    ) -> None:
        """Test that missing columns raise KeyError."""
        mock_df = pd.DataFrame({"date": ["2020-01-01"]})
        mock_read_csv.return_value = mock_df

        with pytest.raises(KeyError):
            load_dataset()

    @patch("src.data_cleaning.validation.convert_and_validate_dates")
    @patch("src.data_cleaning.validation.validate_columns")
    @patch("src.data_cleaning.validation.assert_not_empty")
    @patch("src.data_cleaning.validation.validate_file_exists")
    @patch("src.data_cleaning.validation.pd.read_csv")
    @patch("src.data_cleaning.validation.DATASET_FILE")
    def test_load_dataset_empty(
        self,
        mock_dataset_file: MagicMock,
        mock_read_csv: MagicMock,
        mock_validate_file: MagicMock,
        mock_assert_not_empty: MagicMock,
        mock_validate_columns: MagicMock,
        mock_convert_dates: MagicMock,
    ) -> None:
        """Test that empty dataset raises ValueError."""
        mock_df = pd.DataFrame()
        mock_read_csv.return_value = mock_df
        mock_assert_not_empty.side_effect = ValueError("raw dataset DataFrame is empty")

        with pytest.raises(ValueError, match="raw dataset DataFrame is empty"):
            load_dataset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
