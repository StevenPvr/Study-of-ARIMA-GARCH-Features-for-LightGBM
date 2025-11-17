"""CLI entry point for data_conversion module.

Supports two modes:
- default: no-look-ahead aggregation with trailing-window weights
- --static: legacy static weights over full period (susceptible to look-ahead)
"""

from __future__ import annotations

import argparse

from src.utils import setup_project_path

# Ensure project root
setup_project_path()

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
        "--input",
        type=str,
        default=None,
        help="Explicit input file path (.parquet or .csv). No fallback is used.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Explicit output file path for aggregated series.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Trailing window (days) for no-look-ahead weights (default: from constants)",
    )
    args = parser.parse_args()

    if args.static:
        compute_weighted_log_returns(
            input_file=args.input,
            returns_output_file=args.output,
        )
    else:
        window = args.window if args.window is not None else LIQUIDITY_WEIGHTS_WINDOW_DEFAULT
        compute_weighted_log_returns_no_lookahead(
            input_file=args.input,
            returns_output_file=args.output,
            window=window,
        )


if __name__ == "__main__":
    main()
