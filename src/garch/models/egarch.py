"""EGARCH model wrapper for backtesting imports.

This exposes a stable, minimal API to run rolling EGARCH(1,1) forecasts
without forcing benchmark code to depend on internal module layouts.

Under the hood, it delegates to `src.garch.rolling_garch.rolling` which
implements the state-of-the-art methodology: distribution selection,
online variance recursion, periodic refits, and VaR/MZ calibration.
"""

from __future__ import annotations

from typing import Any, Tuple

import pandas as pd

from src.garch.rolling_garch.rolling import (
    run_from_artifacts as _rolling_run_from_artifacts,
    run_rolling_egarch as _run_rolling_egarch,
)
from src.utils import get_logger

logger = get_logger(__name__)


def run_from_artifacts(
    *,
    refit_every: int,
    window: str,
    window_size: int,
    dist_preference: str,
    keep_nu_between_refits: bool,
    var_alphas: list[float] | None,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """Run rolling EGARCH forecasts loading dataset/artifacts from constants.

    This is the canonical entry used by benchmarks/CLIs to obtain a test-aligned
    one-step-ahead variance path and its evaluation metrics.

    Args:
        refit_every: Refit frequency (in test observations).
        window: 'expanding' or 'rolling' window mode.
        window_size: Window size for 'rolling' mode.
        dist_preference: 'auto' | 'normal' | 'student' | 'skewt'.
        keep_nu_between_refits: Keep Student-t nu fixed between refits.
        var_alphas: VaR alpha levels.

    Returns:
        (forecasts_df, metrics_dict)
    """
    logger.info(
        "EGARCH(run_from_artifacts): refit_every=%d window=%s size=%d dist=%s keep_nu=%s alphas=%s",
        refit_every,
        window,
        window_size,
        dist_preference,
        keep_nu_between_refits,
        var_alphas,
    )
    return _rolling_run_from_artifacts(
        refit_every=refit_every,
        window=window,
        window_size=window_size,
        dist_preference=dist_preference,
        keep_nu_between_refits=keep_nu_between_refits,
        var_alphas=var_alphas,
    )


def run_from_df(
    df: pd.DataFrame,
    *,
    refit_every: int,
    window: str,
    window_size: int,
    dist_preference: str,
    keep_nu_between_refits: bool,
    var_alphas: list[float] | None,
    calibrate_mz: bool = False,
    calibrate_var: bool = False,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """Run rolling EGARCH forecasts directly from a provided dataset DataFrame.

    Convenience for pipelines/tests that already hold the prepared dataset in
    memory. Mirrors the behavior of the artifact-driven variant.

    Args:
        df: Dataset with 'date', 'split' and returns/residual columns.
        refit_every: Refit frequency (in test observations).
        window: 'expanding' or 'rolling' window mode.
        window_size: Window size for 'rolling' mode.
        dist_preference: 'auto' | 'normal' | 'student' | 'skewt'.
        keep_nu_between_refits: Keep Student-t nu fixed between refits.
        var_alphas: VaR alpha levels.
        calibrate_mz: MZ calibration (diagnostic only; off by default).
        calibrate_var: Empirical VaR quantile calibration (off by default).

    Returns:
        (forecasts_df, metrics_dict)
    """
    logger.info(
        "EGARCH(run_from_df): n=%d refit_every=%d window=%s size=%d dist=%s keep_nu=%s alphas=%s",
        len(df),
        refit_every,
        window,
        window_size,
        dist_preference,
        keep_nu_between_refits,
        var_alphas,
    )
    return _run_rolling_egarch(
        df,
        refit_every=refit_every,
        window=window,
        window_size=window_size,
        dist_preference=dist_preference,
        keep_nu_between_refits=keep_nu_between_refits,
        var_alphas=var_alphas,
        calibrate_mz=calibrate_mz,
        calibrate_var=calibrate_var,
    )


__all__ = [
    "run_from_artifacts",
    "run_from_df",
]
