"""Utility functions for GARCH training.

This module contains helper functions used by the training module.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.constants import (
    GARCH_ESTIMATION_FILE,
    GARCH_MODEL_FILE,
    GARCH_MODEL_METADATA_FILE,
    GARCH_VARIANCE_OUTPUTS_FILE,
)
from src.garch.garch_params.estimation import egarch11_variance
from src.garch.structure_garch.utils import prepare_residuals
from src.utils import get_logger

logger = get_logger(__name__)


def _compute_aic(loglik: float, k: int, n: int) -> float:
    """Compute Akaike Information Criterion (lower is better)."""
    return 2.0 * k - 2.0 * float(loglik)


def _get_k_params(dist: str) -> int:
    """Get number of parameters for a distribution.

    Args:
        dist: Distribution name ("normal", "student", or "skewt").

    Returns:
        Number of parameters.
    """
    if dist == "skewt":
        return 6
    if dist == "student":
        return 5
    return 4


def _add_skewt_params(result: dict[str, float], eg_result: dict[str, Any]) -> None:
    """Add skewt-specific parameters to result dictionary.

    Args:
        result: Result dictionary to update.
        eg_result: Estimation result dictionary.
    """
    if "nu" in eg_result:
        result["nu"] = float(eg_result["nu"])  # type: ignore[assignment]
    if "lambda" in eg_result:
        result["lambda"] = float(eg_result["lambda"])  # type: ignore[assignment]


def _fit_single_egarch(e: np.ndarray, dist: str, n: int) -> dict[str, float] | None:
    """Fit a single EGARCH model with given distribution.

    Args:
        e: Cleaned residuals array.
        dist: Distribution name ("normal", "student", or "skewt").
        n: Number of observations.

    Returns:
        Dictionary with fit parameters or None if fit failed.
    """
    from src.garch.garch_params.estimation import estimate_egarch_mle

    try:
        eg_result = estimate_egarch_mle(e, dist=dist)
        k_params = _get_k_params(dist)
        result: dict[str, float] = {
            "omega": float(eg_result["omega"]),
            "alpha": float(eg_result["alpha"]),
            "gamma": float(eg_result["gamma"]),
            "beta": float(eg_result["beta"]),
            "loglik": float(eg_result["loglik"]),
            "converged": bool(eg_result.get("converged", True)),
            "aic": _compute_aic(float(eg_result["loglik"]), k=k_params, n=n),
        }
        if dist == "skewt":
            _add_skewt_params(result, eg_result)
        return result
    except Exception as ex:
        logger.debug("EGARCH(%s) fit failed; continuing: %s", dist, ex)
        return None


def _build_fit_entry(info: dict[str, Any], k_params: int, n: int) -> dict[str, float]:
    """Build a fit entry dictionary from estimation info.

    Args:
        info: Estimation info dictionary.
        k_params: Number of parameters for AIC calculation.
        n: Number of observations.

    Returns:
        Fit entry dictionary with parameters and metrics.
    """
    entry: dict[str, float] = {
        "omega": float(info["omega"]),
        "alpha": float(info["alpha"]),
        "gamma": float(info.get("gamma", 0.0)),
        "beta": float(info["beta"]),
        "loglik": float(info["loglik"]),
        "converged": bool(info.get("converged", True)),
        "aic": _compute_aic(float(info["loglik"]), k=k_params, n=n),
    }
    if "nu" in info:
        entry["nu"] = float(info["nu"])  # type: ignore[assignment]
    if "lambda" in info:
        entry["lambda"] = float(info["lambda"])  # type: ignore[assignment]
    return entry


def _extract_estimation_info(
    est: dict[str, Any], key_canonical: str, key_legacy: str
) -> dict[str, Any] | None:
    """Extract estimation info using canonical or legacy key.

    Args:
        est: Estimation dictionary.
        key_canonical: Canonical key name.
        key_legacy: Legacy key name.

    Returns:
        Estimation info dictionary or None if not found.
    """
    if key_canonical in est:
        return est[key_canonical]
    if key_legacy in est:
        return est[key_legacy]
    return None


def _load_estimation_file() -> dict[str, Any] | None:
    """Load estimation file if it exists.

    Returns:
        Estimation dictionary or None if file doesn't exist or is unreadable.
    """
    if not GARCH_ESTIMATION_FILE.exists():
        return None
    try:
        with GARCH_ESTIMATION_FILE.open() as f:
            return json.load(f)
    except Exception:
        logger.debug("Pre-estimation file not available or unreadable; skipping cache")
        return None


def _process_distribution_fit(
    est: dict[str, Any], dist_name: str, k_params: int, n: int
) -> dict[str, float] | None:
    """Process a single distribution fit from estimation data.

    Args:
        est: Estimation dictionary.
        dist_name: Distribution name ("egarch_normal" or "egarch_skewt").
        k_params: Number of parameters for AIC calculation.
        n: Number of observations.

    Returns:
        Fit entry dictionary or None if not found.
    """
    legacy_key = dist_name.replace("egarch_", "")
    info = _extract_estimation_info(est, dist_name, legacy_key)
    if info:
        return _build_fit_entry(info, k_params=k_params, n=n)
    return None


def _fits_from_preestimation(n: int) -> dict[str, dict[str, float]] | None:
    """Build a fits dict from prior estimation results if available.

    Uses results in GARCH_ESTIMATION_FILE to avoid re-fitting when
    parameters have already been estimated elsewhere. Computes AICs
    with k=4 (normal) and k=5 (student).
    """
    est = _load_estimation_file()
    if not est:
        return None

    out: dict[str, dict[str, float]] = {}

    normal_fit = _process_distribution_fit(est, "egarch_normal", k_params=4, n=n)
    if normal_fit:
        out["egarch_normal"] = normal_fit

    skewt_fit = _process_distribution_fit(est, "egarch_skewt", k_params=6, n=n)
    if skewt_fit:
        out["egarch_skewt"] = skewt_fit

    return out or None


def _validate_egarch_params(params: dict[str, Any]) -> None:
    """Validate that required EGARCH parameters are present.

    Args:
        params: Parameters dictionary.

    Raises:
        ValueError: If required parameters are missing.
    """
    required = {"omega", "alpha", "beta"}
    missing = required - set(params.keys())
    if missing:
        msg = f"Missing required EGARCH parameters: {', '.join(sorted(missing))}"
        raise ValueError(msg)


def _extract_nu_param(params: dict[str, Any]) -> float | None:
    """Extract and convert nu parameter if present.

    Args:
        params: Parameters dictionary.

    Returns:
        Nu parameter as float or None.
    """
    nu_val = params.get("nu")
    if isinstance(nu_val, (int, float, np.floating)):
        return float(nu_val)
    return None


def _infer_distribution(params: dict[str, Any], nu_arg: float | None, lambda_skew: float | None) -> str:
    """Infer distribution name from parameters.

    Args:
        params: Parameters dictionary.
        nu_arg: Nu parameter value or None.
        lambda_skew: Lambda skew parameter value or None.

    Returns:
        Distribution name ("normal", "student", or "skewt").
    """
    dist_key = str(params.get("dist", "")).lower()
    if dist_key in {"normal", "student", "skewt"}:
        return dist_key
    if lambda_skew is not None:
        return "skewt"
    if nu_arg is not None:
        return "student"
    return "normal"


def _build_variance_and_std_full(
    resid_all: np.ndarray, params: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute variance path and standardized residuals over full dataset order.

    Args:
        resid_all: Full residuals array.
        params: EGARCH parameters dictionary.

    Returns:
        Tuple of (variance_array, standardized_residuals_array).

    Raises:
        ValueError: If required parameters are missing.
    """
    if resid_all.size == 0:
        msg = "Residuals array is empty"
        raise ValueError(msg)

    _validate_egarch_params(params)

    e_all = np.asarray(resid_all, dtype=float)
    nu_arg = _extract_nu_param(params)
    lambda_arg = params.get("lambda")
    lambda_skew = float(lambda_arg) if lambda_arg is not None else None

    dist_name = _infer_distribution(params, nu_arg, lambda_skew)

    sigma2 = egarch11_variance(
        e_all,
        float(params["omega"]),
        float(params["alpha"]),
        float(params.get("gamma", 0.0)),
        float(params["beta"]),
        dist=dist_name,
        nu=nu_arg,
        lambda_skew=lambda_skew,
    )
    z = e_all / np.sqrt(sigma2)
    return sigma2, z


def _compute_std_resid_diagnostics(z: np.ndarray) -> dict[str, float | int]:
    """Compute basic diagnostics on standardized residuals.

    Returns mean, variance, std, and tail counts for |z|>2 and |z|>3.

    Args:
        z: Standardized residuals array.

    Returns:
        Dictionary with diagnostics metrics.
    """
    if z.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "var": float("nan"),
            "std": float("nan"),
            "abs_gt_2": 0,
            "abs_gt_3": 0,
        }

    zf = np.asarray(z, dtype=float)
    zf = zf[np.isfinite(zf)]
    n = int(zf.size)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "var": float("nan"),
            "std": float("nan"),
            "abs_gt_2": 0,
            "abs_gt_3": 0,
        }
    mean = float(np.mean(zf))
    var = float(np.var(zf))
    std = float(np.sqrt(var))
    abs_gt_2 = int(np.sum(np.abs(zf) > 2.0))
    abs_gt_3 = int(np.sum(np.abs(zf) > 3.0))
    return {
        "n": n,
        "mean": mean,
        "var": var,
        "std": std,
        "abs_gt_2": abs_gt_2,
        "abs_gt_3": abs_gt_3,
    }


def _prepare_residuals_from_dataset(df: pd.DataFrame) -> np.ndarray:
    """Extract and clean residuals from dataset.

    Args:
        df: DataFrame with date and split columns.

    Returns:
        Cleaned residuals array.

    Raises:
        ValueError: If residuals cannot be prepared.
    """
    if df.empty:
        msg = "DataFrame is empty"
        raise ValueError(msg)
    if "date" not in df.columns:
        msg = "DataFrame missing 'date' column"
        raise ValueError(msg)

    df_sorted = df.sort_values("date").reset_index(drop=True)
    resid_all = prepare_residuals(df_sorted, use_test_only=False)
    resid_all = resid_all[np.isfinite(resid_all)]

    if resid_all.size == 0:
        msg = "No valid residuals found after cleaning"
        raise ValueError(msg)

    return resid_all


def _save_garch_artifacts(
    dist_public: str,
    chosen_params: dict[str, float],
    fits: dict[str, dict[str, float]],
    n_total: int,
    outputs: pd.DataFrame,
) -> None:
    """Save GARCH model artifacts to disk.

    Args:
        dist_public: Distribution name (student/normal).
        chosen_params: Selected model parameters.
        fits: All candidate fits.
        n_total: Total number of observations.
        outputs: DataFrame with variance outputs.
    """
    GARCH_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"dist": dist_public, "params": chosen_params}, GARCH_MODEL_FILE)

    meta = {
        "dist": dist_public,
        "params": chosen_params,
        "fits": fits,
        "n_total": n_total,
        "source": str(GARCH_ESTIMATION_FILE),
        "std_resid_diagnostics": _compute_std_resid_diagnostics(
            np.asarray(outputs["std_resid_garch"].values)
        ),
    }
    with GARCH_MODEL_METADATA_FILE.open("w") as f:
        json.dump(meta, f, indent=2)

    GARCH_VARIANCE_OUTPUTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    outputs.to_csv(GARCH_VARIANCE_OUTPUTS_FILE, index=False)

    logger.info("Saved GARCH model: %s", GARCH_MODEL_FILE)
    logger.info("Saved GARCH metadata: %s", GARCH_MODEL_METADATA_FILE)
    logger.info("Saved variance outputs: %s", GARCH_VARIANCE_OUTPUTS_FILE)

