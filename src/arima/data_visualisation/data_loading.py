"""Data loading utilities for visualization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import filter_by_date_range, get_logger, load_dataframe, validate_required_columns

logger = get_logger(__name__)
_YEAR_MIN = 1900
_YEAR_MAX = 2100


def _load_csv_dataframe(data_file: str) -> pd.DataFrame:
    """Load CSV file and return DataFrame with date index.

    Delegates to src.utils.load_dataframe() for consistency.

    Args:
        data_file: Path to CSV file.

    Returns:
        DataFrame with date index.

    Raises:
        FileNotFoundError: If data_file does not exist.
    """
    data_path = Path(data_file)
    df = load_dataframe(
        data_path,
        date_columns=["date"],
        validate_not_empty=False,  # Let caller handle empty validation
    )
    return df.set_index("date")


def _validate_column_exists(dataframe: pd.DataFrame, column: str, data_file: str) -> None:
    """Validate that a column exists in the dataframe.

    Delegates to src.utils.validate_required_columns() for consistency.

    Args:
        dataframe: DataFrame to check.
        column: Column name to validate.
        data_file: Path to data file (for error message).

    Raises:
        ValueError: If column is missing or data is empty.
    """
    validate_required_columns(dataframe, [column], df_name=f"Data file {data_file}")

    series = dataframe[column].dropna()
    if series.empty:
        msg = f"No valid data found in {data_file}"
        raise ValueError(msg)


def load_and_validate_data(
    data_file: str,
    required_column: str,
) -> pd.DataFrame:
    """Load and validate CSV data file.

    Args:
        data_file: Path to CSV file.
        required_column: Required column name.

    Returns:
        DataFrame with date index.

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If required column is missing or data is empty.
    """
    dataframe = _load_csv_dataframe(data_file)
    _validate_column_exists(dataframe, required_column, data_file)
    return dataframe


def _validate_year(year: int) -> None:
    """Validate year is within reasonable range.

    Args:
        year: Calendar year to validate.

    Raises:
        ValueError: If year is out of range.
    """
    if not (_YEAR_MIN <= year <= _YEAR_MAX):
        msg = f"Invalid year: {year}"
        raise ValueError(msg)


def _filter_dataframe_by_year(dataframe: pd.DataFrame, year: int) -> pd.DataFrame:
    """Filter dataframe to a specific calendar year.

    Delegates to src.utils.filter_by_date_range() for consistency.

    Args:
        dataframe: DataFrame with date index.
        year: Calendar year to filter.

    Returns:
        DataFrame filtered for the year.

    Raises:
        ValueError: If no data available for the year.
    """
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    return filter_by_date_range(
        dataframe,
        start_date=start,
        end_date=end,
        raise_if_empty=True,
    )


def _load_dataframe_for_year(data_file: str, year: int) -> pd.DataFrame:
    """Load and validate dataframe for a given year.

    Args:
        data_file: Path to CSV file.
        year: Calendar year to filter.

    Returns:
        DataFrame with date index filtered for the year.

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If no data available for the year.
    """
    logger.info("Loading data for seasonal component (%s)", year)
    dataframe = _load_csv_dataframe(data_file)
    return _filter_dataframe_by_year(dataframe, year)


def load_series_for_year(*, year: int, data_file: str, column: str) -> pd.Series:
    """Load a column as a pandas Series filtered to a given calendar year.

    Args:
        year: Calendar year to filter.
        data_file: Path to CSV file.
        column: Column name to extract.

    Returns:
        Filtered Series for the specified year.

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If year is invalid, column is missing, or no data for the year.
    """
    _validate_year(year)
    df_year = _load_dataframe_for_year(data_file, year)

    if column not in df_year.columns:
        msg = f"Column '{column}' not found in {data_file}"
        raise ValueError(msg)

    series = df_year[column]
    if not isinstance(series, pd.Series):
        msg = f"Column '{column}' did not return a Series"
        raise ValueError(msg)

    return series.dropna()


def load_residuals(
    predictions_file: str,
    *,
    column: str = "residual",
    dropna: bool = True,
) -> pd.Series:
    """Load residuals from predictions file.

    Args:
        predictions_file: Path to predictions CSV file.
        column: Column name for residuals (default: "residual").
        dropna: Whether to drop NaN values.

    Returns:
        Residuals series.

    Raises:
        FileNotFoundError: If predictions_file does not exist.
        ValueError: If column is missing or no valid data.
    """
    from pathlib import Path

    data_path = Path(predictions_file)
    if not data_path.exists():
        msg = f"Predictions file not found: {predictions_file}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(data_path)

    if column not in df.columns:
        msg = f"Column '{column}' not found in predictions file"
        raise ValueError(msg)

    residuals = df[column]

    if dropna:
        residuals = residuals.dropna()

    if residuals.empty:
        msg = f"No valid data found in column '{column}'"
        raise ValueError(msg)

    return residuals


def load_predictions_with_columns(
    predictions_file: str,
    required_columns: list[str],
    *,
    index_col: str = "date",
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Load and validate predictions CSV with required columns.

    Args:
        predictions_file: Path to predictions CSV file.
        required_columns: List of required column names.
        index_col: Column to use as index (default: "date").
        parse_dates: Whether to parse dates.

    Returns:
        DataFrame with required columns.

    Raises:
        FileNotFoundError: If predictions_file does not exist.
        ValueError: If required columns are missing or no valid data.
    """
    from pathlib import Path

    data_path = Path(predictions_file)
    if not data_path.exists():
        msg = f"Predictions file not found: {predictions_file}"
        raise FileNotFoundError(msg)

    # Load data
    if parse_dates and index_col:
        df = pd.read_csv(data_path, parse_dates=[index_col], index_col=index_col)
    else:
        df = pd.read_csv(data_path)

    # Validate columns
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    # Drop rows with NaN in required columns
    df_clean = df[required_columns].dropna()

    if df_clean.empty:
        msg = "No valid data found in predictions file"
        raise ValueError(msg)

    return df_clean


def load_predictions_with_split(
    predictions_file: str,
    train_test_split_date: str,
    *,
    value_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load predictions and split into train/test dataframes.

    Args:
        predictions_file: Path to predictions CSV file.
        train_test_split_date: Date string marking train/test boundary.
        value_columns: Columns to include. If None, includes all columns.

    Returns:
        Tuple of (train_df, test_df).

    Raises:
        FileNotFoundError: If predictions_file does not exist.
        ValueError: If data is invalid or split date is out of range.
    """
    from pathlib import Path

    data_path = Path(predictions_file)
    if not data_path.exists():
        msg = f"Predictions file not found: {predictions_file}"
        raise FileNotFoundError(msg)

    # Load data with date index
    df = pd.read_csv(data_path, parse_dates=["date"], index_col="date")

    if value_columns is not None:
        missing = [col for col in value_columns if col not in df.columns]
        if missing:
            msg = f"Missing required columns: {missing}"
            raise ValueError(msg)
        df = df[value_columns]

    df = df.dropna()

    if df.empty:
        msg = "No valid data found in predictions file"
        raise ValueError(msg)

    # Split data
    split_date = pd.to_datetime(train_test_split_date)
    train_df = df[df.index < split_date]
    test_df = df[df.index >= split_date]

    return train_df, test_df
