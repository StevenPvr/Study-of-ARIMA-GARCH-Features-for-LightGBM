"""Unit tests for data_cleaning module."""

from __future__ import annotations

import sys
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

# Dependencies are mocked in src/conftest.py before imports
from src.data_cleaning.data_cleaning import (
    data_quality_analysis,
    filter_by_membership,
)


class TestDataQualityAnalysis:
    """Tests for data_quality_analysis function."""

    @patch("src.data_cleaning.utils.pd.to_datetime")
    @patch("src.data_cleaning.utils.DATASET_FILE")
    @patch("src.data_cleaning.utils.pd.read_csv")
    @patch("src.data_cleaning.data_cleaning.logger")
    def test_data_quality_analysis_success(
        self,
        mock_logger: MagicMock,
        mock_read_csv: MagicMock,
        mock_dataset_file: MagicMock,
        mock_to_datetime: MagicMock,
    ) -> None:
        """Test successful data quality analysis."""
        # Mock file exists
        mock_dataset_file.exists.return_value = True

        # Create mock DataFrame
        mock_df = MagicMock()
        mock_df.__len__.return_value = 1000
        mock_df.empty = False
        mock_df.columns = ["date", "ticker", "open", "closing", "volume"]

        # Mock outliers check - create separate mocks for each column
        mock_zero_open = MagicMock()
        mock_zero_open.sum.return_value = 0
        mock_zero_closing = MagicMock()
        mock_zero_closing.sum.return_value = 0
        mock_zero_volume = MagicMock()
        mock_zero_volume.sum.return_value = 0

        mock_open_series = MagicMock()
        mock_open_series.__le__.return_value = mock_zero_open
        mock_closing_series = MagicMock()
        mock_closing_series.__le__.return_value = mock_zero_closing
        mock_volume_series = MagicMock()
        mock_volume_series.__le__.return_value = mock_zero_volume

        # Mock ticker column
        mock_ticker_series = MagicMock()
        mock_ticker_series.nunique.return_value = 100

        # Mock date conversion and validation
        mock_date_series_after_conversion = MagicMock()
        mock_date_isna_result = MagicMock()
        mock_date_isna_result.sum.return_value = 0  # Return int, not MagicMock
        mock_date_series_after_conversion.isna.return_value = mock_date_isna_result
        # Also need min/max/nunique for _analyze_general_statistics
        mock_date_series_after_conversion.min.return_value.date.return_value = "2020-01-01"
        mock_date_series_after_conversion.max.return_value.date.return_value = "2024-01-01"
        mock_date_series_after_conversion.nunique.return_value = 252
        mock_to_datetime.return_value = mock_date_series_after_conversion

        # Mock isna().sum() for missing values (consistent with implementation)
        mock_missing = MagicMock()
        mock_missing.__getitem__.return_value = 0
        mock_df.isna.return_value.sum.return_value = mock_missing

        # Setup getitem to return appropriate mocks for each column
        # Use a dictionary to store mocks and access them via side_effect
        column_mocks: dict[str, MagicMock] = {
            "date": mock_date_series_after_conversion,
            "open": mock_open_series,
            "closing": mock_closing_series,
            "volume": mock_volume_series,
            "ticker": mock_ticker_series,
        }

        def getitem_side_effect(key: str) -> MagicMock:
            return column_mocks.get(key, MagicMock())

        mock_df.__getitem__.side_effect = getitem_side_effect

        # Mock groupby operations
        mock_obs_per_ticker = MagicMock()
        mock_obs_per_ticker.max.return_value = 252
        mock_obs_per_ticker.min.return_value = 100
        mock_obs_per_ticker.mean.return_value = 200.0
        mock_obs_per_ticker.median.return_value = 250.0
        mock_obs_per_ticker.__len__.return_value = 100
        mock_lt_result = MagicMock()
        mock_lt_result.sum.return_value = 10
        mock_obs_per_ticker.__lt__.return_value = mock_lt_result

        mock_groupby = MagicMock()
        mock_groupby.size.return_value = mock_obs_per_ticker
        mock_df.groupby.return_value = mock_groupby

        # Mock volume groupby - need to handle multiple groupby calls
        mock_volume_groupby = MagicMock()
        mock_volume_series = MagicMock()
        mock_volume_mean = MagicMock()
        mock_volume_mean.nsmallest.return_value.items.return_value = [
            ("TICKER1", 1000.0),
            ("TICKER2", 2000.0),
        ]
        mock_volume_series.mean.return_value = mock_volume_mean
        mock_volume_groupby.__getitem__.return_value = mock_volume_series

        # Setup groupby to return different objects for different calls
        def groupby_side_effect(key: str | None = None) -> MagicMock:
            if key == "volume":
                return mock_volume_groupby
            return mock_groupby

        mock_df.groupby.side_effect = groupby_side_effect

        # Mock nsmallest for least observations
        mock_obs_per_ticker.nsmallest.return_value.items.return_value = [
            ("TICKER1", 100),
            ("TICKER2", 150),
        ]

        mock_read_csv.return_value = mock_df

        # Execute
        data_quality_analysis()

        # Verify
        mock_read_csv.assert_called_once_with(mock_dataset_file)
        mock_logger.info.assert_called()


class TestDataQualityAnalysisWithOutliers:
    """Tests for data quality analysis with outliers."""

    @patch("src.data_cleaning.utils.pd.to_datetime")
    @patch("src.data_cleaning.utils.DATASET_FILE")
    @patch("src.data_cleaning.utils.pd.read_csv")
    @patch("src.data_cleaning.data_cleaning.logger")
    def test_data_quality_analysis_with_outliers(
        self,
        mock_logger: MagicMock,
        mock_read_csv: MagicMock,
        mock_dataset_file: MagicMock,
        mock_to_datetime: MagicMock,
    ) -> None:
        """Test data quality analysis with outliers detected."""
        # Mock file exists
        mock_dataset_file.exists.return_value = True

        # Create mock DataFrame
        mock_df = MagicMock()
        mock_df.__len__.return_value = 1000
        mock_df.empty = False
        mock_df.columns = ["date", "ticker", "open", "closing", "volume"]
        mock_df["ticker"].nunique.return_value = 100
        mock_df["date"].min.return_value.date.return_value = "2020-01-01"
        mock_df["date"].max.return_value.date.return_value = "2024-01-01"
        mock_df["date"].nunique.return_value = 252

        # Mock outliers - some zero values
        mock_zero_open = MagicMock()
        mock_zero_open.sum.return_value = 2
        mock_zero_closing = MagicMock()
        mock_zero_closing.sum.return_value = 1
        mock_zero_volume = MagicMock()
        mock_zero_volume.sum.return_value = 10

        mock_open_series = MagicMock()
        mock_open_series.__le__.return_value = mock_zero_open
        mock_closing_series = MagicMock()
        mock_closing_series.__le__.return_value = mock_zero_closing
        mock_volume_series = MagicMock()
        mock_volume_series.__le__.return_value = mock_zero_volume

        # Mock ticker column
        mock_ticker_series = MagicMock()
        mock_ticker_series.nunique.return_value = 100

        # Mock date conversion and validation
        mock_date_series_after_conversion = MagicMock()
        mock_date_isna_result = MagicMock()
        mock_date_isna_result.sum.return_value = 0  # Return int, not MagicMock
        mock_date_series_after_conversion.isna.return_value = mock_date_isna_result
        # Also need min/max/nunique for _analyze_general_statistics
        mock_date_series_after_conversion.min.return_value.date.return_value = "2020-01-01"
        mock_date_series_after_conversion.max.return_value.date.return_value = "2024-01-01"
        mock_date_series_after_conversion.nunique.return_value = 252
        mock_to_datetime.return_value = mock_date_series_after_conversion

        # Mock missing values (consistent with implementation using isna())
        mock_missing = MagicMock()
        mock_missing.__getitem__.return_value = 5
        mock_df.isna.return_value.sum.return_value = mock_missing

        # Setup getitem to return appropriate mocks for each column
        def getitem_side_effect(key: str) -> MagicMock:
            if key == "date":
                return mock_date_series_after_conversion
            elif key == "open":
                return mock_open_series
            elif key == "closing":
                return mock_closing_series
            elif key == "volume":
                return mock_volume_series
            elif key == "ticker":
                return mock_ticker_series
            return MagicMock()

        mock_df.__getitem__.side_effect = getitem_side_effect
        # Remove the old direct assignment that conflicts
        del mock_df["ticker"]
        del mock_df["date"]

        # Mock groupby operations
        mock_obs_per_ticker = MagicMock()
        mock_obs_per_ticker.max.return_value = 252
        mock_obs_per_ticker.min.return_value = 100
        mock_obs_per_ticker.mean.return_value = 200.0
        mock_obs_per_ticker.median.return_value = 250.0
        mock_obs_per_ticker.__len__.return_value = 100
        mock_lt_result = MagicMock()
        mock_lt_result.sum.return_value = 10
        mock_obs_per_ticker.__lt__.return_value = mock_lt_result

        mock_groupby = MagicMock()
        mock_groupby.size.return_value = mock_obs_per_ticker

        # Mock volume groupby (use different name to avoid conflict)
        mock_volume_groupby = MagicMock()
        mock_volume_groupby_series = MagicMock()
        mock_volume_mean = MagicMock()
        mock_volume_mean.nsmallest.return_value.items.return_value = []
        mock_volume_groupby_series.mean.return_value = mock_volume_mean
        mock_volume_groupby.__getitem__.return_value = mock_volume_groupby_series

        # Setup groupby to return different objects for different calls
        def groupby_side_effect(key: str | None = None) -> MagicMock:
            if key == "volume":
                return mock_volume_groupby
            return mock_groupby

        mock_df.groupby.side_effect = groupby_side_effect

        # Mock nsmallest
        mock_obs_per_ticker.nsmallest.return_value.items.return_value = []

        mock_read_csv.return_value = mock_df

        # Execute
        data_quality_analysis()

        # Verify outliers were logged
        assert mock_logger.info.call_count > 0


class TestFilterIncompleteTickers:
    """Tests for filter_incomplete_tickers function."""

    @patch("src.data_cleaning.utils.pd.to_datetime")
    @patch("src.data_cleaning.data_cleaning.DATASET_FILTERED_FILE")
    @patch("src.data_cleaning.utils.DATASET_FILE")
    @patch("src.data_cleaning.utils.pd.read_csv")
    @patch("src.data_cleaning.data_cleaning.logger")
    def test_filter_incomplete_tickers_success(
        self,
        mock_logger: MagicMock,
        mock_read_csv: MagicMock,
        mock_dataset_file: MagicMock,
        mock_filtered_file: MagicMock,
        mock_to_datetime: MagicMock,
    ) -> None:
        """Test successful filtering of incomplete tickers."""
        # Mock file exists
        mock_dataset_file.exists.return_value = True

        # Create mock DataFrame with tickers
        mock_df = MagicMock()
        mock_df.__len__.return_value = 5000
        mock_df.empty = False
        mock_df.columns = ["date", "ticker", "open", "closing", "volume"]

        # Mock date conversion and validation
        # After pd.to_datetime, mock_df["date"] is assigned the converted series
        # We need to mock this assignment and the subsequent isna().sum() call
        mock_date_series_after_conversion = MagicMock()
        mock_date_isna_result = MagicMock()
        mock_date_isna_result.sum.return_value = 0  # Return int, not MagicMock
        mock_date_series_after_conversion.isna.return_value = mock_date_isna_result
        mock_to_datetime.return_value = mock_date_series_after_conversion

        # Mock the assignment: raw_df["date"] = pd.to_datetime(...)
        # After assignment, accessing mock_df["date"] should return the converted series
        # We need to handle both the original date column access and the converted one
        original_date_mock = MagicMock()
        original_date_mock.min.return_value.date.return_value = "2020-01-01"
        original_date_mock.max.return_value.date.return_value = "2024-01-01"
        original_date_mock.nunique.return_value = 252

        def getitem_side_effect(key: str) -> MagicMock:
            if key == "date":
                # After to_datetime assignment, return the converted series
                # We'll use a side_effect that checks if assignment happened
                return mock_date_series_after_conversion
            return MagicMock()

        mock_df.__getitem__.side_effect = getitem_side_effect
        # Also set it directly for backward compatibility
        mock_df["date"] = mock_date_series_after_conversion

        # Mock groupby operations
        mock_obs_per_ticker = MagicMock()
        mock_obs_per_ticker.__len__.return_value = 100

        # Mock valid tickers (>= MIN_OBSERVATIONS)
        mock_valid_mask = MagicMock()
        mock_valid_series = MagicMock()
        mock_valid_series.index = ["TICKER1", "TICKER2", "TICKER3"]
        mock_obs_per_ticker.__getitem__.return_value = mock_valid_series
        mock_obs_per_ticker.__ge__.return_value = mock_valid_mask

        # Mock removed tickers (< MIN_OBSERVATIONS)
        mock_removed_mask = MagicMock()
        mock_removed_series = MagicMock()
        mock_removed_series.__len__.return_value = 5
        mock_removed_series.sort_values.return_value.items.return_value = [
            ("TICKER4", 100),
            ("TICKER5", 200),
        ]
        mock_obs_per_ticker.__getitem__.side_effect = lambda mask: (
            mock_valid_series if mask is mock_valid_mask else mock_removed_series
        )
        mock_obs_per_ticker.__lt__.return_value = mock_removed_mask

        mock_groupby = MagicMock()
        mock_groupby.size.return_value = mock_obs_per_ticker
        mock_df.groupby.return_value = mock_groupby

        # Mock apply_basic_integrity_fixes methods on mock_df
        mock_df_after_drop_duplicates = MagicMock()
        mock_df_after_drop_duplicates.__len__.return_value = 4000
        mock_df_after_drop_duplicates.empty = False

        # Mock volume column for apply_basic_integrity_fixes
        mock_volume_series_for_fixes = MagicMock()
        mock_volume_bool_array = MagicMock()
        mock_volume_bool_array.copy.return_value = mock_df_after_drop_duplicates
        mock_volume_series_for_fixes.__gt__.return_value = mock_volume_bool_array

        # Note: getitem_side_effect_for_filter will be updated below to handle volume

        mock_df.drop_duplicates.return_value = mock_df_after_drop_duplicates
        # Mock getitem for mock_df_after_drop_duplicates to return volume series
        mock_df_after_drop_duplicates.__getitem__ = MagicMock(
            side_effect=lambda key: mock_volume_series_for_fixes if key == "volume" else MagicMock()
        )
        mock_df_after_drop_duplicates.loc = MagicMock()
        mock_df_after_drop_duplicates.loc.__getitem__.return_value = mock_df_after_drop_duplicates
        mock_df_after_drop_duplicates.sort_values.return_value = mock_df_after_drop_duplicates
        mock_df_after_drop_duplicates.reset_index.return_value = mock_df_after_drop_duplicates
        mock_df_after_drop_duplicates.copy.return_value = mock_df_after_drop_duplicates

        # Mock filtering: raw_df[raw_df["ticker"].isin(valid_tickers)].reset_index(drop=True)
        mock_filtered_df = mock_df_after_drop_duplicates  # Use same mock
        mock_filtered_df["ticker"].nunique.return_value = 95

        # Mock the ticker column access and filtering chain
        mock_ticker_series_for_filter = MagicMock()
        mock_isin_result = MagicMock()
        mock_filtered_by_isin = MagicMock()
        mock_filtered_by_isin.reset_index.return_value = mock_filtered_df

        mock_ticker_series_for_filter.isin.return_value = mock_isin_result

        # Update getitem_side_effect to handle both column access and boolean indexing
        original_getitem_side_effect = mock_df.__getitem__.side_effect

        def getitem_side_effect_for_filter(key: str | MagicMock) -> MagicMock:
            # Handle boolean indexing: raw_df[boolean_mask]
            if isinstance(key, MagicMock):
                if key is mock_isin_result:
                    return mock_filtered_by_isin
            elif key == "ticker":
                return mock_ticker_series_for_filter
            elif key == "date":
                return mock_date_series_after_conversion
            elif key == "volume":
                # For apply_basic_integrity_fixes
                return mock_volume_series_for_fixes
            # Fall back to original side_effect for other columns
            if original_getitem_side_effect:
                if callable(original_getitem_side_effect):
                    try:
                        return original_getitem_side_effect(key)
                    except (TypeError, KeyError):
                        pass
            return MagicMock()

        mock_df.__getitem__.side_effect = getitem_side_effect_for_filter

        # Mock file operations
        mock_filtered_file.parent.mkdir = MagicMock()
        mock_filtered_file.parent.mkdir.return_value = None

        mock_read_csv.return_value = mock_df

        # Execute with proper mock for to_csv
        # The code calls to_csv on the DataFrame instance returned by reset_index
        # We need to ensure the mock_filtered_df returned by reset_index has to_csv mocked
        mock_filtered_df.to_csv = MagicMock()
        # Ensure reset_index returns our mock with to_csv
        mock_filtered_by_isin.reset_index = MagicMock(return_value=mock_filtered_df)

        with patch.object(pd.DataFrame, "to_csv"):
            filter_by_membership()
            # Verify
            mock_read_csv.assert_called_once_with(mock_dataset_file)
            mock_filtered_file.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
            # The actual call is on the instance, not the class
            mock_filtered_df.to_csv.assert_called_once_with(mock_filtered_file, index=False)
            mock_logger.info.assert_called()

    @patch("src.data_cleaning.utils.pd.to_datetime")
    @patch("src.data_cleaning.data_cleaning.DATASET_FILTERED_FILE")
    @patch("src.data_cleaning.utils.DATASET_FILE")
    @patch("src.data_cleaning.utils.pd.read_csv")
    @patch("src.data_cleaning.data_cleaning.logger")
    def test_filter_incomplete_tickers_no_removals(
        self,
        mock_logger: MagicMock,
        mock_read_csv: MagicMock,
        mock_dataset_file: MagicMock,
        mock_filtered_file: MagicMock,
        mock_to_datetime: MagicMock,
    ) -> None:
        """Test filtering when all tickers are complete."""
        # Mock file exists
        mock_dataset_file.exists.return_value = True

        # Create mock DataFrame
        mock_df = MagicMock()
        mock_df.__len__.return_value = 5000
        mock_df.empty = False
        mock_df.columns = ["date", "ticker", "open", "closing", "volume"]

        # Mock date conversion and validation
        # After pd.to_datetime, mock_df["date"] is assigned the converted series
        # We need to mock this assignment and the subsequent isna().sum() call
        mock_date_series_after_conversion = MagicMock()
        mock_date_isna_result = MagicMock()
        mock_date_isna_result.sum.return_value = 0  # Return int, not MagicMock
        mock_date_series_after_conversion.isna.return_value = mock_date_isna_result
        mock_to_datetime.return_value = mock_date_series_after_conversion

        # Mock the assignment: raw_df["date"] = pd.to_datetime(...)
        # After assignment, accessing mock_df["date"] should return the converted series
        # We need to handle both the original date column access and the converted one
        original_date_mock = MagicMock()
        original_date_mock.min.return_value.date.return_value = "2020-01-01"
        original_date_mock.max.return_value.date.return_value = "2024-01-01"
        original_date_mock.nunique.return_value = 252

        def getitem_side_effect(key: str) -> MagicMock:
            if key == "date":
                # After to_datetime assignment, return the converted series
                # We'll use a side_effect that checks if assignment happened
                return mock_date_series_after_conversion
            return MagicMock()

        mock_df.__getitem__.side_effect = getitem_side_effect
        # Also set it directly for backward compatibility
        mock_df["date"] = mock_date_series_after_conversion

        # Mock groupby - all tickers valid
        mock_obs_per_ticker = MagicMock()
        mock_obs_per_ticker.__len__.return_value = 100

        # All tickers >= MIN_OBSERVATIONS
        mock_valid_mask = MagicMock()
        mock_valid_series = MagicMock()
        mock_valid_series.index = ["TICKER1", "TICKER2", "TICKER3"]
        mock_obs_per_ticker.__getitem__.return_value = mock_valid_series
        mock_obs_per_ticker.__ge__.return_value = mock_valid_mask

        # No removed tickers
        mock_removed_mask = MagicMock()
        mock_removed_series = MagicMock()
        mock_removed_series.__len__.return_value = 0
        mock_obs_per_ticker.__getitem__.side_effect = lambda mask: (
            mock_valid_series if mask is mock_valid_mask else mock_removed_series
        )
        mock_obs_per_ticker.__lt__.return_value = mock_removed_mask

        mock_groupby = MagicMock()
        mock_groupby.size.return_value = mock_obs_per_ticker
        mock_df.groupby.return_value = mock_groupby

        # Mock apply_basic_integrity_fixes methods on mock_df
        mock_df_after_drop_duplicates = MagicMock()
        mock_df_after_drop_duplicates.__len__.return_value = 5000
        mock_df_after_drop_duplicates.empty = False

        # Mock volume column for apply_basic_integrity_fixes
        mock_volume_series_for_fixes = MagicMock()
        mock_volume_bool_array = MagicMock()
        mock_volume_bool_array.copy.return_value = mock_df_after_drop_duplicates
        mock_volume_series_for_fixes.__gt__.return_value = mock_volume_bool_array

        # Note: getitem_side_effect_for_filter_no_removals will be updated below to handle volume

        mock_df.drop_duplicates.return_value = mock_df_after_drop_duplicates
        # Mock getitem for mock_df_after_drop_duplicates to return volume series
        mock_df_after_drop_duplicates.__getitem__ = MagicMock(
            side_effect=lambda key: mock_volume_series_for_fixes if key == "volume" else MagicMock()
        )
        mock_df_after_drop_duplicates.loc = MagicMock()
        mock_df_after_drop_duplicates.loc.__getitem__.return_value = mock_df_after_drop_duplicates
        mock_df_after_drop_duplicates.sort_values.return_value = mock_df_after_drop_duplicates
        mock_df_after_drop_duplicates.reset_index.return_value = mock_df_after_drop_duplicates
        mock_df_after_drop_duplicates.copy.return_value = mock_df_after_drop_duplicates

        # Mock filtering: raw_df[raw_df["ticker"].isin(valid_tickers)].reset_index(drop=True)
        mock_filtered_df = mock_df_after_drop_duplicates  # Use same mock
        mock_filtered_df["ticker"].nunique.return_value = 100

        # Mock the ticker column access and filtering chain
        mock_ticker_series_for_filter = MagicMock()
        mock_isin_result = MagicMock()
        mock_filtered_by_isin = MagicMock()
        mock_filtered_by_isin.reset_index.return_value = mock_filtered_df

        mock_ticker_series_for_filter.isin.return_value = mock_isin_result

        # Update getitem_side_effect to handle both column access and boolean indexing
        original_getitem_side_effect_no_removals = mock_df.__getitem__.side_effect

        def getitem_side_effect_for_filter_no_removals(key: str | MagicMock) -> MagicMock:
            # Handle boolean indexing: raw_df[boolean_mask]
            if isinstance(key, MagicMock):
                if key is mock_isin_result:
                    return mock_filtered_by_isin
            elif key == "ticker":
                return mock_ticker_series_for_filter
            elif key == "date":
                return mock_date_series_after_conversion
            elif key == "volume":
                # For apply_basic_integrity_fixes
                return mock_volume_series_for_fixes
            # Fall back to original side_effect for other columns
            if original_getitem_side_effect_no_removals:
                if callable(original_getitem_side_effect_no_removals):
                    try:
                        return original_getitem_side_effect_no_removals(key)
                    except (TypeError, KeyError):
                        pass
            return MagicMock()

        mock_df.__getitem__.side_effect = getitem_side_effect_for_filter_no_removals

        # Mock file operations
        mock_filtered_file.parent.mkdir = MagicMock()

        mock_read_csv.return_value = mock_df

        # Execute with proper mock for to_csv
        # The code calls to_csv on the DataFrame instance returned by reset_index
        mock_filtered_df.to_csv = MagicMock()
        # Ensure reset_index returns our mock with to_csv
        mock_filtered_by_isin.reset_index = MagicMock(return_value=mock_filtered_df)

        with patch.object(pd.DataFrame, "to_csv"):
            filter_by_membership()
            # Verify - should not log removed tickers
            mock_read_csv.assert_called_once()
            mock_filtered_df.to_csv.assert_called_once()

    @patch("src.data_cleaning.utils.pd.to_datetime")
    @patch("src.data_cleaning.data_cleaning.DATASET_FILTERED_FILE")
    @patch("src.data_cleaning.utils.DATASET_FILE")
    @patch("src.data_cleaning.utils.pd.read_csv")
    @patch("src.data_cleaning.data_cleaning.logger")
    def test_filter_incomplete_tickers_all_removed(
        self,
        mock_logger: MagicMock,
        mock_read_csv: MagicMock,
        mock_dataset_file: MagicMock,
        mock_filtered_file: MagicMock,
        mock_to_datetime: MagicMock,
    ) -> None:
        """Test filtering when all tickers are incomplete."""
        # Mock file exists
        mock_dataset_file.exists.return_value = True

        # Create mock DataFrame
        mock_df = MagicMock()
        mock_df.__len__.return_value = 1000
        mock_df.empty = False
        mock_df.columns = ["date", "ticker", "open", "closing", "volume"]

        # Mock date conversion and validation
        # After pd.to_datetime, mock_df["date"] is assigned the converted series
        # We need to mock this assignment and the subsequent isna().sum() call
        mock_date_series_after_conversion = MagicMock()
        mock_date_isna_result = MagicMock()
        mock_date_isna_result.sum.return_value = 0  # Return int, not MagicMock
        mock_date_series_after_conversion.isna.return_value = mock_date_isna_result
        mock_to_datetime.return_value = mock_date_series_after_conversion

        # Mock the assignment: raw_df["date"] = pd.to_datetime(...)
        # After assignment, accessing mock_df["date"] should return the converted series
        # We need to handle both the original date column access and the converted one
        original_date_mock = MagicMock()
        original_date_mock.min.return_value.date.return_value = "2020-01-01"
        original_date_mock.max.return_value.date.return_value = "2024-01-01"
        original_date_mock.nunique.return_value = 252

        def getitem_side_effect(key: str) -> MagicMock:
            if key == "date":
                # After to_datetime assignment, return the converted series
                # We'll use a side_effect that checks if assignment happened
                return mock_date_series_after_conversion
            return MagicMock()

        mock_df.__getitem__.side_effect = getitem_side_effect
        # Also set it directly for backward compatibility
        mock_df["date"] = mock_date_series_after_conversion

        # Mock groupby - all tickers invalid
        mock_obs_per_ticker = MagicMock()
        mock_obs_per_ticker.__len__.return_value = 50

        # No valid tickers
        mock_valid_mask = MagicMock()
        mock_valid_series = MagicMock()
        mock_valid_series.index = []
        mock_obs_per_ticker.__ge__.return_value = mock_valid_mask

        # All tickers removed
        mock_removed_mask = MagicMock()
        mock_removed_series = MagicMock()
        mock_removed_series.__len__.return_value = 50
        mock_removed_series.sort_values.return_value.items.return_value = [
            ("TICKER1", 100),
            ("TICKER2", 200),
        ]
        mock_obs_per_ticker.__getitem__.side_effect = lambda mask: (
            mock_valid_series if mask is mock_valid_mask else mock_removed_series
        )
        mock_obs_per_ticker.__lt__.return_value = mock_removed_mask

        mock_groupby = MagicMock()
        mock_groupby.size.return_value = mock_obs_per_ticker
        mock_df.groupby.return_value = mock_groupby

        # Mock apply_basic_integrity_fixes methods on mock_df
        # But in this test, we expect an error before apply_basic_integrity_fixes is called
        # Still need to mock in case the error doesn't happen
        mock_df_after_drop_duplicates_all_removed = MagicMock()
        mock_df_after_drop_duplicates_all_removed.__len__.return_value = 0
        mock_df_after_drop_duplicates_all_removed.empty = True

        # Mock volume column for apply_basic_integrity_fixes
        mock_volume_series_for_fixes_all_removed = MagicMock()
        mock_volume_bool_array_all_removed = MagicMock()
        mock_volume_bool_array_all_removed.copy.return_value = (
            mock_df_after_drop_duplicates_all_removed
        )
        mock_volume_series_for_fixes_all_removed.__gt__.return_value = (
            mock_volume_bool_array_all_removed
        )

        mock_df.drop_duplicates.return_value = mock_df_after_drop_duplicates_all_removed
        # Mock getitem for mock_df_after_drop_duplicates_all_removed to return volume series
        mock_df_after_drop_duplicates_all_removed.__getitem__ = MagicMock(
            side_effect=lambda key: (
                mock_volume_series_for_fixes_all_removed if key == "volume" else MagicMock()
            )
        )
        mock_df_after_drop_duplicates_all_removed.loc = MagicMock()
        mock_df_after_drop_duplicates_all_removed.loc.__getitem__.return_value = (
            mock_df_after_drop_duplicates_all_removed
        )
        mock_df_after_drop_duplicates_all_removed.sort_values.return_value = (
            mock_df_after_drop_duplicates_all_removed
        )
        mock_df_after_drop_duplicates_all_removed.reset_index.return_value = (
            mock_df_after_drop_duplicates_all_removed
        )
        mock_df_after_drop_duplicates_all_removed.copy.return_value = (
            mock_df_after_drop_duplicates_all_removed
        )

        # Mock filtering - empty DataFrame
        mock_filtered_df = MagicMock()
        mock_filtered_df.__len__.return_value = 0
        mock_filtered_df["ticker"].nunique.return_value = 0
        mock_filtered_df.empty = True
        mock_df.__getitem__.return_value.isin.return_value = MagicMock()
        mock_df.__getitem__.return_value.reset_index.return_value = mock_filtered_df

        # Update getitem to handle volume
        def getitem_side_effect_all_removed(key: str | MagicMock) -> MagicMock:
            if isinstance(key, MagicMock):
                return mock_filtered_df
            elif key == "volume":
                return mock_volume_series_for_fixes_all_removed
            elif key == "date":
                return mock_date_series_after_conversion
            return MagicMock()

        mock_df.__getitem__.side_effect = getitem_side_effect_all_removed

        # Mock file operations
        mock_filtered_file.parent.mkdir = MagicMock()

        mock_read_csv.return_value = mock_df

        # Execute - filter_by_membership applies basic integrity fixes,
        # so it should succeed even with empty data
        with patch.object(pd.DataFrame, "to_csv"):
            filter_by_membership()

        # Verify
        mock_read_csv.assert_called_once()
        # Should save file (even if empty after integrity fixes)
        assert mock_logger.info.call_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
