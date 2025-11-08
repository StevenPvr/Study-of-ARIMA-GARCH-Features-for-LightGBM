"""GARCH training utilities.

This module assumes parameters have been pre-estimated and saved in
``results/garch_estimation.json``. It:
- loads the best distribution (Normal or Student) from this file
- computes the variance trajectory and standardized residuals
- saves artifacts (joblib + metadata) and variance outputs
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.constants import GARCH_ESTIMATION_FILE
from src.garch.training_garch.utils import (
    _build_variance_and_std_full,
    _compute_std_resid_diagnostics,
    _fits_from_preestimation,
    _fit_single_egarch,
    _prepare_residuals_from_dataset,
    _save_garch_artifacts,
)
from src.utils import get_logger

logger = get_logger(__name__)


def fit_egarch_candidates(resid: np.ndarray) -> dict[str, dict[str, float]]:
    """Fit EGARCH candidates (Normal vs Student-t vs Skew-t).

    Args:
        resid: Residuals array.

    Returns:
        Dictionary with fit results for each candidate distribution.

    Raises:
        ValueError: If residuals are empty or invalid.
    """
    if resid.size == 0:
        msg = "Residuals array is empty"
        raise ValueError(msg)

    e = np.asarray(resid, dtype=float)
    e = e[np.isfinite(e)]
    n = int(e.size)

    if n == 0:
        msg = "No finite residuals found"
        raise ValueError(msg)

    out: dict[str, dict[str, float]] = {}
    normal_fit = _fit_single_egarch(e, "normal", n)
    if normal_fit:
        out["egarch_normal"] = normal_fit

    skewt_fit = _fit_single_egarch(e, "skewt", n)
    if skewt_fit:
        out["egarch_skewt"] = skewt_fit

    return out


def choose_best_fit(fits: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]]:
    """Select converged EGARCH model with minimal AIC; prefer Skew-t on ties.

    Accepts keys either as {"egarch_normal","egarch_skewt"} or legacy
    {"normal","skewt"}. Returns the key in the same naming style
    as provided for the winning candidate.

    Args:
        fits: Dictionary of candidate fits with their parameters.

    Returns:
        Tuple of (best_distribution_key, best_parameters).

    Raises:
        ValueError: If no fits provided or no converged candidates.
    """
    if not fits:
        msg = "No candidate fits provided"
        raise ValueError(msg)

    # Normalize keys to canonical labels but track original keys
    norm_map = {
        "normal": "egarch_normal",
        "skewt": "egarch_skewt",
    }
    normalized: list[tuple[str, dict[str, float], str]] = []
    for original_key, v in fits.items():
        key = norm_map.get(original_key, original_key)
        if bool(v.get("converged", True)):
            normalized.append((key, v, original_key))
    if not normalized:
        msg = "No converged EGARCH candidates"
        raise ValueError(msg)

    order = {"egarch_skewt": 0, "egarch_normal": 1}
    normalized.sort(key=lambda kv: (float(kv[1].get("aic", float("inf"))), order.get(kv[0], 9)))
    best_norm_key, best_val, best_original_key = normalized[0]
    # Return in the original naming style used for that candidate
    return best_original_key, best_val


def attach_outputs_to_dataframe(
    df: pd.DataFrame, sigma2: np.ndarray, z: np.ndarray
) -> pd.DataFrame:
    """Attach variance and standardized residuals to merged split DataFrame.

    Assumes df sorted by date and matches length of sigma2/z.

    Args:
        df: DataFrame with date column.
        sigma2: Variance array.
        z: Standardized residuals array.

    Returns:
        DataFrame with added columns: sigma2_garch, sigma_garch, std_resid_garch.

    Raises:
        ValueError: If array lengths don't match DataFrame length.
    """
    if len(df) != len(sigma2) or len(sigma2) != len(z):
        msg = f"Length mismatch: df={len(df)}, sigma2={len(sigma2)}, z={len(z)}"
        raise ValueError(msg)

    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)
    out["sigma2_garch"] = sigma2
    out["sigma_garch"] = np.sqrt(sigma2)
    out["std_resid_garch"] = z
    return out


def train_egarch_from_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Train EGARCH(1,1) model from garch dataset and save artifacts.

    Steps:
    - extract residuals (train+test)
    - load pre-estimated parameters (assumed best per diagnostics)
    - compute variance and standardized residuals on full period
    - save model (joblib + metadata) and variance CSV

    Args:
        df: DataFrame with date, split, and residual columns.

    Returns:
        Dictionary with distribution, parameters, outputs shape, and diagnostics.

    Raises:
        FileNotFoundError: If pre-estimation file is missing.
        ValueError: If dataset is invalid or residuals cannot be prepared.
    """
    resid_all = _prepare_residuals_from_dataset(df)

    fits = _fits_from_preestimation(n=int(resid_all.size))
    if not fits:
        msg = (
            f"Pre-estimated params not found at {GARCH_ESTIMATION_FILE}. "
            "Run the estimation step before training."
        )
        raise FileNotFoundError(msg)

    best_dist, chosen_params = choose_best_fit(fits)
    dist_public = {
        "egarch_normal": "normal",
        "egarch_skewt": "skewt",
    }.get(best_dist, best_dist)

    df_sorted = df.sort_values("date").reset_index(drop=True)
    sigma2, z = _build_variance_and_std_full(resid_all, chosen_params)
    outputs = attach_outputs_to_dataframe(df_sorted, sigma2, z)
    diag = _compute_std_resid_diagnostics(z)

    _save_garch_artifacts(dist_public, chosen_params, fits, int(resid_all.size), outputs)

    return {
        "dist": dist_public,
        "params": chosen_params,
        "outputs_shape": outputs.shape,
        "std_resid_diagnostics": diag,
    }
