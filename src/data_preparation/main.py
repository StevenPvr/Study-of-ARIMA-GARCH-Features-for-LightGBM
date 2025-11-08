"""CLI for data preparation module."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config_logging import setup_logging
from src.data_preparation.data_preparation import load_train_test_data, split_train_test
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Main CLI function for data preparation."""
    setup_logging()

    split_train_test()

    train_series, test_series = load_train_test_data()
    logger.info(f"Data preparation complete: train={len(train_series)}, test={len(test_series)}")


if __name__ == "__main__":
    main()
