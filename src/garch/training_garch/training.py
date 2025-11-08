"""GARCH training utilities.

This module assumes parameters have been pre-estimated and saved in
``results/garch_estimation.json``. It:
- loads the best distribution (Normal or Student) from this file
- computes the variance trajectory and standardized residuals
- saves artifacts (joblib + metadata) and variance outputs
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
from src.garch.structure_garch.detection import prepare_residuals
from src.utils import get_logger

logger = get_logger(__name__)


def _compute_aic(loglik: float, k: int, n: int) -> float:
    """Compute Akaike Information Criterion (lower is better)."""
    return 2.0 * k - 2.0 * float(loglik)


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
        k_params = 6 if dist == "skewt" else (5 if dist == "student" else 4)
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
            if "nu" in eg_result:
                result["nu"] = float(eg_result["nu"])  # type: ignore[assignment]
            if "lambda" in eg_result:
                result["lambda"] = float(eg_result["lambda"])  # type: ignore[assignment]
        return result
    except Exception as ex:
        logger.debug("EGARCH(%s) fit failed; continuing: %s", dist, ex)
        return None


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

    # Infer distribution robustly when not explicitly present in params
    # Prefer explicit key; otherwise infer from presence of nu/lambda
    dist_key = str(params.get("dist", "")).lower()
    if dist_key in {"normal", "student", "skewt"}:
        dist_name = dist_key
    elif lambda_skew is not None:
        dist_name = "skewt"
    elif nu_arg is not None:
        dist_name = "student"
    else:
        dist_name = "normal"

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
