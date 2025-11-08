"""CLI entry point for data_cleaning module."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data_cleaning.data_cleaning import data_quality_analysis, filter_by_membership
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Main CLI function to clean S&P 500 data.

    Executes data quality analysis and applies integrity fixes.
    Exits with code 1 on error, 0 on success.
    """
    try:
        data_quality_analysis()
        filter_by_membership()
        logger.info("Data cleaning completed successfully")
    except (FileNotFoundError, KeyError, ValueError, OSError) as e:
        logger.error(f"Data cleaning failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error during data cleaning: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
