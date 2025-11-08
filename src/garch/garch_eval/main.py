"""CLI to compute GARCH variance forecasts, VaR, and intervals."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json

from src.constants import (
    GARCH_ESTIMATION_FILE,
    GARCH_EVAL_DEFAULT_ALPHAS,
    GARCH_EVAL_DEFAULT_HORIZON,
    GARCH_EVAL_DEFAULT_LEVEL,
)
from src.garch.garch_eval.eval import _choose_best_from_estimation, forecast_from_artifacts
from src.garch.garch_eval.metrics import compute_classic_metrics_from_artifacts, save_metrics_json
from src.garch.garch_eval.plotting import generate_eval_plots_from_artifacts
from src.utils import get_logger

logger = get_logger(__name__)


def _parse_alphas(alphas_str: str) -> list[float]:
    """Parse comma-separated alpha values.

    Args:
        alphas_str: Comma-separated string of alpha values.

    Returns:
        List of parsed alpha values.
    """
    return [float(a) for a in str(alphas_str).split(",") if a]


def _load_best_model() -> tuple[dict[str, float], str, str, float | None, float | None]:
    """Load best model from estimation file.

    Returns:
        Tuple of (params, name, dist, nu, lambda_skew).
    """
    with GARCH_ESTIMATION_FILE.open() as f:
        est = json.load(f)
    params, name, nu, lambda_skew = _choose_best_from_estimation(est)
    if "skewt" in name:
        dist = "skewt"
    else:
        dist = "normal"
    return params, name, dist, nu, lambda_skew


def _compute_and_save_metrics(
    params: dict[str, float],
    name: str,
    dist: str,
    nu: float | None,
    lambda_skew: float | None,
    alphas: list[float],
    apply_mz_calibration: bool = True,
) -> None:
    """Compute and save GARCH evaluation metrics and plots.

    Args:
        params: Model parameters.
        name: Model name.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter (for Skew-t).
        alphas: VaR alpha levels.
        apply_mz_calibration: Whether to apply MZ calibration to variances.
    """
    metrics = compute_classic_metrics_from_artifacts(
        params=params,
        model_name=name,
        dist=dist,
        nu=nu,
        lambda_skew=lambda_skew,
        alphas=alphas,
        apply_mz_calibration=apply_mz_calibration,
    )
    save_metrics_json(metrics)
    generate_eval_plots_from_artifacts(
        params=params, model_name=name, dist=dist, nu=nu, lambda_skew=lambda_skew, alphas=alphas
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GARCH forecasts and metrics from artifacts")
    parser.add_argument(
        "--horizon",
        type=int,
        default=GARCH_EVAL_DEFAULT_HORIZON,
        help=f"Forecast horizon H (default: {GARCH_EVAL_DEFAULT_HORIZON})",
    )
    parser.add_argument(
        "--level",
        type=float,
        default=GARCH_EVAL_DEFAULT_LEVEL,
        help=f"PI level (default: {GARCH_EVAL_DEFAULT_LEVEL})",
    )
    default_alphas_str = ",".join(str(a) for a in GARCH_EVAL_DEFAULT_ALPHAS)
    parser.add_argument(
        "--alphas",
        type=str,
        default=default_alphas_str,
        help=f"Comma-separated VaR alphas (default: {default_alphas_str})",
    )
    args = parser.parse_args()

    logger.info("GARCH evaluation: horizon=%d, level=%.3f", args.horizon, args.level)
    # Keep MZ calibration for diagnostics only; never for scoring by default
    forecast_from_artifacts(
        horizon=int(args.horizon),
        level=float(args.level),
        use_mz_calibration=False,
    )

    # Always compute and save metrics automatically
    try:
        alphas = _parse_alphas(args.alphas)
        params, name, dist, nu, lambda_skew = _load_best_model()
        _compute_and_save_metrics(params, name, dist, nu, lambda_skew, alphas, apply_mz_calibration=True)
    except Exception:
        logger.exception("Failed to compute/save GARCH metrics")


if __name__ == "__main__":
    main()
