"""High level orchestration for S&P 500 ticker and price downloads."""

from __future__ import annotations

import pandas as pd

from src.constants import DATA_FETCH_LOG_PROGRESS_INTERVAL
from src.data_fetching.download import download_ticker_data, get_date_range
from src.data_fetching.processing import process_ticker_data
from src.data_fetching.reporting import combine_and_save_data
from src.data_fetching.wikipedia import (
    fetch_wikipedia_html,
    load_tickers,
    parse_tickers_from_table,
    save_tickers_to_csv,
)
from src.utils import get_logger

logger = get_logger(__name__)


def fetch_sp500_tickers() -> None:
    """Fetch S&P 500 tickers from Wikipedia and persist them to CSV.

    The low-level HTTP, HTML parsing and normalization logic is delegated to
    the wikipedia.py submodule. This function only orchestrates the workflow.
    """
    logger.info("Fetching S&P 500 tickers from Wikipedia")

    html_content = fetch_wikipedia_html()
    tickers: list[str] = parse_tickers_from_table(html_content)
    save_tickers_to_csv(tickers)

    logger.info("Retrieved %d tickers from Wikipedia", len(tickers))


def download_sp500_data() -> None:
    """Download and persist historical S&P 500 data via yfinance.

    All validation, IO details and edge cases live in the dedicated
    submodules (download.py, processing.py, reporting.py). This orchestration
    layer remains intentionally thin to ease auditing and testing.
    """
    tickers = load_tickers()
    start_date, end_date = get_date_range()

    logger.info(
        "Downloading historical data for %d tickers between %s and %s",
        len(tickers),
        start_date.date(),
        end_date.date(),
    )

    data_list: list[pd.DataFrame] = []
    failed_tickers: list[str] = []

    total = len(tickers)
    for i, ticker in enumerate(tickers, start=1):
        ticker_data = download_ticker_data(ticker, start_date, end_date)

        if ticker_data is None:
            failed_tickers.append(ticker)
            logger.warning("No data returned for ticker %s", ticker)
        else:
            processed_data = process_ticker_data(ticker_data, ticker)
            data_list.append(processed_data)

        if i % DATA_FETCH_LOG_PROGRESS_INTERVAL == 0 or i == total:
            logger.info("Progress: %d/%d tickers processed", i, total)

    combine_and_save_data(data_list, failed_tickers)
