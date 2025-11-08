"""Utility functions for data fetching module."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime
from io import StringIO
from urllib.parse import urlparse

import pandas as pd
import yfinance as yf

from src.constants import (
    DATA_DIR,
    DATA_FETCH_END_DATE,
    DATA_FETCH_START_DATE,
    DATASET_FILE,
    FETCH_REPORT_FILE,
    SP500_TICKERS_FILE,
    SP500_WIKI_URL,
)
from src.utils import get_logger

logger = get_logger(__name__)


def _fetch_wikipedia_html() -> bytes:
    """Fetch HTML content from Wikipedia S&P 500 page.

    Returns:
        HTML content as bytes.

    Raises:
        ValueError: If URL scheme is not HTTPS.
        RuntimeError: If unable to fetch the page.
    """
    # Validate URL scheme (enforce HTTPS)
    parsed = urlparse(str(SP500_WIKI_URL))
    if parsed.scheme.lower() != "https":
        raise ValueError("SP500_WIKI_URL must use HTTPS")

    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    req = urllib.request.Request(str(SP500_WIKI_URL), headers=headers)

    try:
        # URL validated for HTTPS; set timeout to avoid hangs
        with urllib.request.urlopen(req, timeout=15) as response:  # nosec B310
            return response.read()
    except urllib.error.URLError as e:
        msg = f"Failed to fetch Wikipedia page: {e}"
        raise RuntimeError(msg) from e


def _read_html_tables(html_content: bytes) -> list[pd.DataFrame]:
    """Read HTML tables from content.

    Args:
        html_content: HTML content as bytes.

    Returns:
        List of DataFrames parsed from HTML.

    Raises:
        RuntimeError: If unable to parse tables.
    """
    try:
        html_string = html_content.decode("utf-8")
        tables = pd.read_html(StringIO(html_string))
    except pd.errors.EmptyDataError as e:
        msg = "Wikipedia page returned empty data"
        raise RuntimeError(msg) from e

    if not tables:
        msg = "No tables found in Wikipedia page"
        raise RuntimeError(msg)

    return tables


def _extract_tickers_from_table(sp500_table: pd.DataFrame) -> list[str]:
    """Extract and validate tickers from S&P 500 table.

    Args:
        sp500_table: DataFrame containing S&P 500 data.

    Returns:
        List of ticker symbols.

    Raises:
        KeyError: If 'Symbol' column is not found.
        RuntimeError: If no tickers found.
    """
    if "Symbol" not in sp500_table.columns:
        msg = f"'Symbol' column not found. Available columns: {list(sp500_table.columns)}"
        raise KeyError(msg)

    tickers = sp500_table["Symbol"].tolist()
    if not tickers:
        msg = "No tickers found in 'Symbol' column"
        raise RuntimeError(msg)

    return tickers


def _normalize_tickers(tickers: list[str]) -> list[str]:
    """Normalize ticker symbols by replacing dots with dashes.

    Args:
        tickers: List of ticker symbols.

    Returns:
        List of normalized ticker symbols.
    """
    return [ticker.replace(".", "-") for ticker in tickers]


def _parse_tickers_from_table(html_content: bytes) -> list[str]:
    """Parse tickers from Wikipedia HTML table.

    Args:
        html_content: HTML content from Wikipedia page.

    Returns:
        List of normalized ticker symbols.

    Raises:
        RuntimeError: If unable to parse tables or extract tickers.
        KeyError: If 'Symbol' column is not found in the parsed table.
    """
    tables = _read_html_tables(html_content)
    sp500_table = tables[0]
    tickers = _extract_tickers_from_table(sp500_table)
    tickers = _normalize_tickers(tickers)

    logger.info(f"Retrieved {len(tickers)} tickers from Wikipedia")
    logger.info(f"Sample tickers: {', '.join(tickers[:5])}")

    return tickers


def _save_tickers_to_csv(tickers: list[str]) -> None:
    """Save tickers to CSV file.

    Args:
        tickers: List of ticker symbols to save.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tickers_df = pd.DataFrame({"ticker": tickers})
    tickers_df.to_csv(SP500_TICKERS_FILE, index=False)

    logger.info(f"Saved tickers to {SP500_TICKERS_FILE}")


def _load_tickers() -> list[str]:
    """Load tickers from CSV file.

    Returns:
        List of ticker symbols.

    Raises:
        FileNotFoundError: If tickers file does not exist.
    """
    if not SP500_TICKERS_FILE.exists():
        msg = f"Tickers file not found: {SP500_TICKERS_FILE}. " "Run fetch_sp500_tickers() first."
        raise FileNotFoundError(msg)

    tickers_df = pd.read_csv(SP500_TICKERS_FILE)
    return tickers_df["ticker"].tolist()


def _get_date_range() -> tuple[datetime, datetime]:
    """Get fixed date range for historical data download.

    Returns a fixed date range (2013-01-01 to 2024-12-31) to ensure
    perfect reproducibility of the dataset.

    Returns:
        Tuple of (start_date, end_date).
    """
    return DATA_FETCH_START_DATE, DATA_FETCH_END_DATE


def _validate_ticker_input(ticker: str) -> bool:
    """Validate ticker symbol input.

    Args:
        ticker: Ticker symbol to validate.

    Returns:
        True if valid, False otherwise.
    """
    if not ticker or not isinstance(ticker, str):
        logger.warning(f"Invalid ticker symbol: {ticker}")
        return False
    return True


def _validate_date_range(start_date: datetime, end_date: datetime, ticker: str) -> bool:
    """Validate date range for data download.

    Args:
        start_date: Start date.
        end_date: End date.
        ticker: Ticker symbol for logging.

    Returns:
        True if valid, False otherwise.
    """
    if start_date >= end_date:
        logger.warning(f"Invalid date range for {ticker}: start_date >= end_date")
        return False
    return True


def _handle_download_error(ticker: str, error: Exception) -> None:
    """Handle download errors with appropriate logging.

    Args:
        ticker: Ticker symbol.
        error: Exception that occurred during download.
    """
    if isinstance(error, (ValueError, KeyError)):
        logger.warning(f"Invalid ticker or data format for {ticker}: {error}")
    elif isinstance(error, (ConnectionError, TimeoutError)):
        logger.warning(f"Network error downloading {ticker}: {error}")
    else:
        logger.warning(f"Unexpected error downloading {ticker}: {error}")


def _is_valid_ticker_data(ticker_data: pd.DataFrame | None) -> bool:
    """Check if ticker data is valid and non-empty.

    Args:
        ticker_data: DataFrame to validate.

    Returns:
        True if data is valid and non-empty, False otherwise.
    """
    return ticker_data is not None and not ticker_data.empty


def _download_yfinance_data(
    ticker: str, start_date: datetime, end_date: datetime
) -> pd.DataFrame | None:
    """Download data from yfinance API.

    Args:
        ticker: Ticker symbol.
        start_date: Start date.
        end_date: End date.

    Returns:
        DataFrame with ticker data or None if download fails.
    """
    try:
        ticker_data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )
        if _is_valid_ticker_data(ticker_data):
            return ticker_data
        return None
    except Exception as e:
        _handle_download_error(ticker, e)
        return None


def _download_ticker_data(
    ticker: str, start_date: datetime, end_date: datetime
) -> pd.DataFrame | None:
    """Download historical data for a single ticker.

    Args:
        ticker: Ticker symbol.
        start_date: Start date for data download.
        end_date: End date for data download.

    Returns:
        DataFrame with ticker data or None if download fails.
    """
    if not _validate_ticker_input(ticker):
        return None

    if not _validate_date_range(start_date, end_date, ticker):
        return None

    return _download_yfinance_data(ticker, start_date, end_date)


def _validate_ticker_data_not_empty(ticker_data: pd.DataFrame, ticker: str) -> None:
    """Validate that ticker data is not empty.

    Args:
        ticker_data: DataFrame to validate.
        ticker: Ticker symbol for error message.

    Raises:
        ValueError: If DataFrame is empty.
    """
    if ticker_data.empty:
        msg = f"Empty DataFrame provided for ticker {ticker}"
        raise ValueError(msg)


def _normalize_dataframe_columns(ticker_data: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame columns by handling MultiIndex.

    Args:
        ticker_data: DataFrame with potentially MultiIndex columns.

    Returns:
        DataFrame with normalized columns.
    """
    if isinstance(ticker_data.columns, pd.MultiIndex):
        ticker_data.columns = ticker_data.columns.get_level_values(0)
    return ticker_data


def _get_missing_columns(
    ticker_data: pd.DataFrame, required_columns: list[str]
) -> list[str]:
    """Get list of missing required columns.

    Args:
        ticker_data: DataFrame to check.
        required_columns: List of required column names.

    Returns:
        List of missing column names.
    """
    return [col for col in required_columns if col not in ticker_data.columns]


def _validate_required_columns(ticker_data: pd.DataFrame, ticker: str) -> None:
    """Validate that required columns exist in ticker data.

    Args:
        ticker_data: DataFrame to validate.
        ticker: Ticker symbol for error message.

    Raises:
        KeyError: If required columns are missing.
    """
    required_columns = ["Open", "Close", "Volume"]
    missing_columns = _get_missing_columns(ticker_data, required_columns)
    if missing_columns:
        msg = f"Missing required columns for {ticker}: {missing_columns}"
        raise KeyError(msg)


def _create_processed_dataframe(ticker_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Create processed DataFrame with standardized columns.

    Args:
        ticker_data: Raw DataFrame from yfinance.
        ticker: Ticker symbol.

    Returns:
        Processed DataFrame with columns: date, ticker, open, closing, volume.
    """
    return pd.DataFrame(
        {
            "date": ticker_data.index,
            "ticker": ticker,
            "open": ticker_data["Open"],
            "closing": ticker_data["Close"],
            "volume": ticker_data["Volume"],
        }
    )


def _process_ticker_data(ticker_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Process downloaded ticker data into standardized format.

    Args:
        ticker_data: Raw DataFrame from yfinance.
        ticker: Ticker symbol.

    Returns:
        Processed DataFrame with columns: date, ticker, open, closing, volume.

    Raises:
        KeyError: If required columns are missing from ticker_data.
        ValueError: If ticker_data is empty or invalid.
    """
    _validate_ticker_data_not_empty(ticker_data, ticker)
    ticker_data = _normalize_dataframe_columns(ticker_data)
    _validate_required_columns(ticker_data, ticker)
    return _create_processed_dataframe(ticker_data, ticker)


def _is_data_list_empty(data_list: list[pd.DataFrame]) -> bool:
    """Check if data list is empty.

    Args:
        data_list: List to check.

    Returns:
        True if empty, False otherwise.
    """
    return not data_list


def _are_all_dataframes(data_list: list[pd.DataFrame]) -> bool:
    """Check if all items in list are DataFrames.

    Args:
        data_list: List to check.

    Returns:
        True if all items are DataFrames, False otherwise.
    """
    return all(isinstance(df, pd.DataFrame) for df in data_list)


def _validate_data_list(data_list: list[pd.DataFrame]) -> None:
    """Validate that data_list is non-empty and contains only DataFrames.

    Args:
        data_list: List of DataFrames to validate.

    Raises:
        RuntimeError: If data_list is empty.
        ValueError: If data_list contains non-DataFrame items.
    """
    if _is_data_list_empty(data_list):
        msg = "No data downloaded. Check tickers and network connection."
        raise RuntimeError(msg)

    if not _are_all_dataframes(data_list):
        msg = "All items in data_list must be pandas DataFrames"
        raise ValueError(msg)


def _concatenate_dataframes(data_list: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate list of DataFrames into a single DataFrame.

    Args:
        data_list: List of DataFrames to concatenate.

    Returns:
        Concatenated DataFrame.

    Raises:
        ValueError: If concatenation fails.
    """
    try:
        return pd.concat(data_list, ignore_index=True)
    except (ValueError, TypeError) as e:
        msg = f"Failed to concatenate dataframes: {e}"
        raise ValueError(msg) from e


def _validate_non_empty_dataset(dataset: pd.DataFrame, initial_count: int) -> None:
    """Validate that dataset is not empty after cleaning.

    Args:
        dataset: DataFrame to validate.
        initial_count: Number of rows before cleaning.

    Raises:
        RuntimeError: If dataset is empty.
    """
    if dataset.empty:
        msg = (
            f"Dataset is empty after removing NaN values "
            f"(had {initial_count} rows before dropna)"
        )
        raise RuntimeError(msg)


def _combine_dataframes(data_list: list[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    """Combine and clean dataframes.

    Args:
        data_list: List of DataFrames with ticker data.

    Returns:
        Tuple of (cleaned dataset, initial row count).

    Raises:
        RuntimeError: If no data was downloaded or dataset is empty after processing.
        ValueError: If data_list contains invalid DataFrames.
    """
    _validate_data_list(data_list)
    dataset = _concatenate_dataframes(data_list)
    initial_count = len(dataset)
    dataset = dataset.dropna()
    _validate_non_empty_dataset(dataset, initial_count)
    return dataset, initial_count


def _extract_dates_from_dataset(dataset: pd.DataFrame) -> pd.Series | None:
    """Extract and convert date column to datetime series.

    Args:
        dataset: DataFrame with date column.

    Returns:
        Series of datetime objects or None if extraction fails.
    """
    if "date" not in dataset.columns:
        logger.warning("'date' column not found in dataset")
        return None

    dates = pd.to_datetime(dataset["date"], errors="coerce")
    if dates.isna().all():
        return None

    return dates


def _format_date_range(dates: pd.Series) -> tuple[str, str]:
    """Format date range as strings.

    Args:
        dates: Series of datetime objects.

    Returns:
        Tuple of (start_date, end_date) as formatted strings.
    """
    realized_start = dates.min().strftime("%Y-%m-%d")
    realized_end = dates.max().strftime("%Y-%m-%d")
    return realized_start, realized_end


def _get_realized_date_range(dataset: pd.DataFrame) -> tuple[str | None, str | None]:
    """Extract realized date range from dataset.

    Args:
        dataset: DataFrame with date column.

    Returns:
        Tuple of (realized_start, realized_end) as strings or None.
    """
    try:
        dates = _extract_dates_from_dataset(dataset)
        if dates is None:
            return None, None
        return _format_date_range(dates)
    except (KeyError, ValueError, AttributeError) as e:
        logger.warning(f"Failed to extract date range: {e}")
        return None, None


def _build_fetch_report(
    data_list: list[pd.DataFrame],
    failed_tickers: list[str],
    initial_count: int,
    final_count: int,
    realized_start: str | None,
    realized_end: str | None,
) -> dict[str, int | str | list[str] | None]:
    """Build fetch report dictionary.

    Args:
        data_list: List of successfully downloaded DataFrames.
        failed_tickers: List of tickers that failed to download.
        initial_count: Number of rows before dropna.
        final_count: Number of rows after dropna.
        realized_start: Realized start date or None.
        realized_end: Realized end date or None.

    Returns:
        Dictionary with fetch report data.
    """
    return {
        "tickers_succeeded": int(len(data_list)),
        "tickers_failed": int(len(failed_tickers)),
        "failed_tickers_sample": failed_tickers[:20],
        "rows_before_dropna": int(initial_count),
        "rows_after_dropna": int(final_count),
        "requested_start": DATA_FETCH_START_DATE.strftime("%Y-%m-%d"),
        "requested_end": DATA_FETCH_END_DATE.strftime("%Y-%m-%d"),
        "realized_start": realized_start,
        "realized_end": realized_end,
        "dataset_path": str(DATASET_FILE),
    }


def _save_fetch_report(fetch_report: dict[str, int | str | list[str] | None]) -> None:
    """Save fetch report to JSON file.

    Args:
        fetch_report: Dictionary with fetch report data.
    """
    try:
        FETCH_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(FETCH_REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(fetch_report, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.warning(f"Failed to write fetch report JSON: {e}")


def _log_download_summary(
    data_list: list[pd.DataFrame],
    failed_tickers: list[str],
    initial_count: int,
    final_count: int,
) -> None:
    """Log download summary information.

    Args:
        data_list: List of successfully downloaded DataFrames.
        failed_tickers: List of tickers that failed to download.
        initial_count: Number of rows before dropna.
        final_count: Number of rows after dropna.
    """
    logger.info("Download completed:")
    logger.info(f"  {len(data_list)} tickers succeeded")
    logger.info(f"  {len(failed_tickers)} tickers failed")
    logger.info(f"  {initial_count} total observations before dropna")
    logger.info(f"  {final_count} total observations after dropna")
    logger.info(f"  Saved to {DATASET_FILE}")
    logger.info(f"  Fetch report: {FETCH_REPORT_FILE}")

    if failed_tickers:
        logger.warning(f"Failed tickers: {', '.join(failed_tickers[:10])}")
        if len(failed_tickers) > 10:
            logger.warning(f"  ... and {len(failed_tickers) - 10} more")


def _combine_and_save_data(data_list: list[pd.DataFrame], failed_tickers: list[str]) -> None:
    """Combine all ticker data and save to CSV.

    Args:
        data_list: List of DataFrames with ticker data.
        failed_tickers: List of tickers that failed to download.

    Raises:
        RuntimeError: If no data was downloaded or dataset is empty after processing.
    """
    dataset, initial_count = _combine_dataframes(data_list)

    # Persist dataset
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DATASET_FILE, index=False)

    # Build and persist fetch report JSON for reproducibility/traceability
    realized_start, realized_end = _get_realized_date_range(dataset)
    fetch_report = _build_fetch_report(
        data_list,
        failed_tickers,
        initial_count,
        len(dataset),
        realized_start,
        realized_end,
    )
    _save_fetch_report(fetch_report)
    _log_download_summary(data_list, failed_tickers, initial_count, len(dataset))
