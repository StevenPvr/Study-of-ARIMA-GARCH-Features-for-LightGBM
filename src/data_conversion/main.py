"""CLI entry point for data_conversion module.

Supports two modes:
- default: no-look-ahead aggregation with trailing-window weights
- --static: legacy static weights over full period (susceptible to look-ahead)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.constants import LIQUIDITY_WEIGHTS_WINDOW_DEFAULT
from src.data_conversion.data_conversion import (
    compute_weighted_log_returns,
    compute_weighted_log_returns_no_lookahead,
)


def main() -> None:
    """Main CLI function to convert S&P 500 data to weighted log returns."""
    parser = argparse.ArgumentParser(description="S&P500 data conversion")
    parser.add_argument(
        "--static",
        action="store_true",
        help="Use legacy static weights (may include look-ahead)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Trailing window (days) for no-look-ahead weights (default: from constants)",
    )
    args = parser.parse_args()

    if args.static:
        compute_weighted_log_returns()
    else:
        window = args.window if args.window is not None else LIQUIDITY_WEIGHTS_WINDOW_DEFAULT
        compute_weighted_log_returns_no_lookahead(window=window)


if __name__ == "__main__":
    main()
