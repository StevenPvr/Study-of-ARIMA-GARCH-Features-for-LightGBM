"""CLI entry to run volatility backtest and save outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.benchmark.bench_volatility import run_vol_backtest, save_vol_backtest_outputs
from src.constants import VOL_EWMA_LAMBDA_DEFAULT, VOL_ROLLING_WINDOW_DEFAULT
from src.garch.structure_garch.detection import load_garch_dataset
from src.utils import get_logger

logger = get_logger(__name__)


def _run_backtest(args: argparse.Namespace) -> None:
    """Execute volatility backtest with parsed arguments."""
    logger.info(
        "Vol backtest: ewma=%.3f, window=%d, refit_every=%d",
        args.ewma_lambda,
        args.window,
        args.refit_every,
    )
    data_frame = load_garch_dataset()  # uses constants path
    forecasts, metrics = run_vol_backtest(
        data_frame,
        ewma_lambda=float(args.ewma_lambda),
        rolling_window=int(args.window),
        refit_every=int(args.refit_every),
    )
    save_vol_backtest_outputs(forecasts, metrics)
    logger.info("Volatility backtest completed successfully")


def main() -> None:
    """CLI entry point for volatility backtest."""
    parser = argparse.ArgumentParser(description="Volatility backtest: ARIMA-GARCH vs baselines")
    parser.add_argument("--ewma-lambda", type=float, default=VOL_EWMA_LAMBDA_DEFAULT)
    parser.add_argument("--window", type=int, default=VOL_ROLLING_WINDOW_DEFAULT)
    parser.add_argument(
        "--refit-every",
        type=int,
        default=20,
        help="GARCH refit frequency (in test observations). Defaults to 20.",
    )
    args = parser.parse_args()

    try:
        _run_backtest(args)
    except FileNotFoundError as e:
        logger.error("Data file not found: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Invalid parameter: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during volatility backtest: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
