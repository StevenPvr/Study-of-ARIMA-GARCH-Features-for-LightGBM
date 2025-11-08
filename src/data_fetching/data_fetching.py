"""Data fetching functions for S&P 500 tickers and historical data."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data_fetching.utils import (
    _combine_and_save_data,
    _download_ticker_data,
    _fetch_wikipedia_html,
    _get_date_range,
    _load_tickers,
    _parse_tickers_from_table,
    _process_ticker_data,
    _save_tickers_to_csv,
)
from src.utils import get_logger

logger = get_logger(__name__)


def fetch_sp500_tickers() -> None:
    """Fetch S&P 500 tickers from Wikipedia and save to CSV.

    Retrieves the list of S&P 500 company tickers from Wikipedia,
    normalizes them (replaces '.' with '-'), and saves to a CSV file.

    Raises:
        ValueError: If URL scheme is not HTTPS.
        RuntimeError: If unable to fetch or parse tickers from Wikipedia.
        KeyError: If 'Symbol' column is not found in the parsed table.
    """
    logger.info("Fetching S&P 500 tickers from Wikipedia")

    html_content = _fetch_wikipedia_html()
    tickers = _parse_tickers_from_table(html_content)
    _save_tickers_to_csv(tickers)


def _process_single_ticker(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    data_list: list[pd.DataFrame],
    failed_tickers: list[str],
) -> None:
    """Process a single ticker and update data lists.

    Args:
        ticker: Ticker symbol to process.
        start_date: Start date for data download.
        end_date: End date for data download.
        data_list: List to append successful dataframes.
        failed_tickers: List to append failed tickers.
    """
    ticker_data = _download_ticker_data(ticker, start_date, end_date)

    if ticker_data is None:
        failed_tickers.append(ticker)
        logger.warning(f"Empty data for ticker: {ticker}")
        return

    try:
        processed_data = _process_ticker_data(ticker_data, ticker)
        data_list.append(processed_data)
    except (KeyError, ValueError) as e:
        failed_tickers.append(ticker)
        logger.warning(f"Failed to process data for {ticker}: {e}")


def download_sp500_data() -> None:
    """Download historical data for all S&P 500 tickers via yfinance.

    Downloads historical price data (open, close, volume) for all tickers
    using adjusted prices to avoid splits/dividend jumps. Saves to CSV.

    Note: Uses a fixed date range (2013-01-01 to 2024-12-31) to ensure
    perfect reproducibility. The dataset is not intended to be updated.
    """
    logger.info("Starting download of S&P 500 historical data")

    tickers = _load_tickers()
    start_date, end_date = _get_date_range()

    logger.info(f"Downloading data for {len(tickers)} tickers")
    logger.info(f"Period: {start_date.date()} → {end_date.date()}")

    data_list: list[pd.DataFrame] = []
    failed_tickers: list[str] = []

    for i, ticker in enumerate(tickers):
        _process_single_ticker(ticker, start_date, end_date, data_list, failed_tickers)

        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i + 1}/{len(tickers)} tickers downloaded")

    _combine_and_save_data(data_list, failed_tickers)
