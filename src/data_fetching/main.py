"""CLI entry point for data_fetching module."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data_fetching.data_fetching import download_sp500_data, fetch_sp500_tickers
from src.utils import get_logger


def main() -> None:
    """Main CLI function to fetch S&P 500 data."""

    logger = get_logger(__name__)

    try:
        logger.info("Starting data fetching process")
        fetch_sp500_tickers()
        download_sp500_data()
        logger.info("Data fetching completed successfully")
    except (RuntimeError, ValueError, KeyError, FileNotFoundError) as e:
        logger.error(f"Data fetching failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during data fetching: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
