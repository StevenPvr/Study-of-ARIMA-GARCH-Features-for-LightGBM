"""EGARCH wrapper facade delegating to internal implementation.

This thin module enables imports like `import src.models.egarch as eg` while
keeping the real implementation under `src.garch.models.egarch`.
"""

from __future__ import annotations

from typing import Any, Tuple

import pandas as pd

from src.garch.models.egarch import logger  # reuse same logger for consistency
# Expose internals for test monkeypatching compatibility and local delegation
from src.garch.rolling_garch.rolling import (  # type: ignore F401
    run_from_artifacts as _rolling_run_from_artifacts,
    run_rolling_egarch as _run_rolling_egarch,
)

def run_from_artifacts(
    *,
    refit_every: int,
    window: str,
    window_size: int,
    dist_preference: str,
    keep_nu_between_refits: bool,
    var_alphas: list[float] | None,
):
    """Delegate to rolling implementation (kept patchable for tests)."""
    logger.info(
        "EGARCH(models.facade.run_from_artifacts): refit_every=%d window=%s size=%d dist=%s alphas=%s",
        refit_every,
        window,
        window_size,
        dist_preference,
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
    df: "pd.DataFrame",
    *,
    refit_every: int,
    window: str,
    window_size: int,
    dist_preference: str,
    keep_nu_between_refits: bool,
    var_alphas: list[float] | None,
    calibrate_mz: bool = False,
    calibrate_var: bool = False,
):
    """Delegate to rolling implementation (kept patchable for tests)."""
    logger.info(
        "EGARCH(models.facade.run_from_df): n=%d refit_every=%d window=%s size=%d dist=%s alphas=%s",
        len(df),
        refit_every,
        window,
        window_size,
        dist_preference,
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
    "_rolling_run_from_artifacts",
    "_run_rolling_egarch",
]
