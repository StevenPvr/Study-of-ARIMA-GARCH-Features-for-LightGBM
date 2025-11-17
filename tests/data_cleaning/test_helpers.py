"""Test helper functions and fixtures for data_cleaning tests."""

from __future__ import annotations

from unittest.mock import MagicMock


def create_mock_date_series() -> MagicMock:
    """Create a mock date series for testing.

    Returns:
        Mock date series with required methods.
    """
    mock_date_series = MagicMock()
    mock_date_isna_result = MagicMock()
    mock_date_isna_result.sum.return_value = 0
    mock_date_series.isna.return_value = mock_date_isna_result
    mock_date_series.min.return_value.date.return_value = "2020-01-01"
    mock_date_series.max.return_value.date.return_value = "2024-01-01"
    mock_date_series.nunique.return_value = 252

    # Setup .dt accessor for timezone handling
    # Chain the operations so each returns the series itself
    mock_dt_accessor = MagicMock()
    mock_dt_accessor.tz = None  # Mock dates are timezone-naive
    # Each dt operation returns the series itself to allow chaining
    mock_dt_accessor.tz_convert.return_value = mock_date_series
    mock_dt_accessor.tz_localize.return_value = mock_date_series
    mock_dt_accessor.normalize.return_value = mock_date_series
    mock_date_series.dt = mock_dt_accessor

    # Also support direct method calls for fallback case
    mock_date_series.tz_localize.return_value = mock_date_series
    mock_date_series.normalize.return_value = mock_date_series

    return mock_date_series


def create_mock_dataframe(
    row_count: int = 5000,
    empty: bool = False,
    columns: list[str] | None = None,
) -> MagicMock:
    """Create a mock DataFrame for testing.

    Args:
        row_count: Number of rows in the DataFrame.
        empty: Whether the DataFrame is empty.
        columns: List of column names.

    Returns:
        Mock DataFrame with required methods.
    """
    if columns is None:
        columns = ["date", "tickers", "open", "close", "volume"]

    mock_df = MagicMock()
    mock_df.__len__.return_value = row_count
    mock_df.empty = empty
    mock_df.columns = columns
    return mock_df


def setup_mock_date_conversion(
    mock_df: MagicMock,
    mock_to_datetime: MagicMock,
) -> MagicMock:
    """Setup mock date conversion for a DataFrame.

    Args:
        mock_df: Mock DataFrame to setup.
        mock_to_datetime: Mock pd.to_datetime function.

    Returns:
        Mock date series after conversion.
    """
    mock_date_series = create_mock_date_series()
    mock_to_datetime.return_value = mock_date_series

    def getitem_side_effect(key: str) -> MagicMock:
        if key == "date":
            return mock_date_series
        return MagicMock()

    mock_df.__getitem__.side_effect = getitem_side_effect
    mock_df["date"] = mock_date_series
    return mock_date_series


def setup_mock_integrity_fixes(
    mock_df: MagicMock,
    after_drop_count: int = 4000,
    empty: bool = False,
    _tickers_with_missing_removed: int = 0,
    _tickers_with_invalid_volume_removed: int = 0,
) -> MagicMock:
    """Setup mock for apply_basic_integrity_fixes operations.

    Args:
        mock_df: Mock DataFrame to setup.
        after_drop_count: Row count after dropping duplicates and removing tickers.
        empty: Whether the DataFrame is empty after fixes.
        _tickers_with_missing_removed: Number of tickers removed due to missing data.
        _tickers_with_invalid_volume_removed: Number of tickers removed due to invalid volume.

    Returns:
        Mock DataFrame after integrity fixes.
    """
    # Final mock DataFrame after all fixes
    mock_df_after = MagicMock()
    mock_df_after.__len__.return_value = after_drop_count
    mock_df_after.empty = empty
    mock_df_after.columns = ["date", "ticker", "open", "closing", "volume"]

    # Setup final operations
    mock_df_after.sort_values.return_value = mock_df_after
    mock_df_after.reset_index.return_value = mock_df_after
    mock_df_after.copy.return_value = mock_df_after

    return mock_df_after


def setup_mock_file_operations(
    mock_filtered_file: MagicMock,
    mock_read_csv: MagicMock,
    mock_df: MagicMock,
) -> None:
    """Setup mock file operations.

    Args:
        mock_filtered_file: Mock filtered file path.
        mock_read_csv: Mock pd.read_csv function.
        mock_df: Mock DataFrame to return from read_csv.
    """
    mock_filtered_file.parent.mkdir = MagicMock()
    mock_filtered_file.parent.mkdir.return_value = None
    mock_read_csv.return_value = mock_df
