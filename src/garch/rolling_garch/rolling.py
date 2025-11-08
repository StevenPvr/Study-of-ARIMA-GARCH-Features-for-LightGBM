"""Rolling EGARCH(1,1) one-step-ahead backtesting.

Implements a clean walk-forward (h=1) variance forecasting loop on the
`arima_residual_return` column only (no other column is used for modeling),
as requested. The code avoids data leakage by refitting using information
available strictly up to time t when forecasting t+1.

Key features:
- Expanding or rolling window with periodic full refits
- Normal/Student innovations support (skew-t can be added later)
- Optional VaR levels computed from the chosen residuals distribution
- Minimal but robust implementation to integrate with existing tests

Notes:
- Parameter estimation is intentionally abstracted via `_fit_initial_params`.
  Tests monkeypatch this function to provide pre-estimated parameters from the
  offline calibration step. Optimized MLE parameters are required - no fallback
  to prevent biased results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple, cast

import json
import math
import numpy as np
import pandas as pd

from src.constants import (
    GARCH_DATASET_FILE,
    GARCH_ESTIMATION_FILE,
    GARCH_FIT_MIN_SIZE,
    GARCH_FIT_MIN_SIZE_KURTOSIS,
    GARCH_LOG_VAR_MAX,
    GARCH_MIN_INIT_VAR,
    GARCH_REFIT_EVERY_DEFAULT,
    GARCH_REFIT_WINDOW_DEFAULT,
    GARCH_REFIT_WINDOW_SIZE_DEFAULT,
    GARCH_ROLLING_EVAL_FILE,
    GARCH_ROLLING_FORECASTS_FILE,
    GARCH_STUDENT_NU_INIT,
    GARCH_STUDENT_NU_MIN,
    RESULTS_DIR,
)
from src.garch.garch_eval.metrics import qlike_loss
from src.garch.garch_params.estimation import egarch11_variance
from src.utils import get_logger

logger = get_logger(__name__)


# --------------------- Data structures ---------------------


@dataclass
class EgarchParams:
    """EGARCH(1,1) parameters container.

    Attributes
    ----------
    omega : float
        Intercept term in log-variance recursion.
    alpha : float
        Coefficient for |z_{t-1}| - kappa.
    beta : float
        Persistence on log-variance.
    gamma : float
        Leverage term multiplying z_{t-1}.
    nu : float | None
        Degrees of freedom for Student innovations; None for Normal.
    dist : str
        Distribution name: 'normal' or 'student'.
    model : str
        Model tag: 'egarch' (kept for compatibility).
    """

    omega: float
    alpha: float
    beta: float
    gamma: float
    nu: float | None
    dist: str
    model: str = "egarch"


# Backward-compat alias used by some tests
GarchParams = EgarchParams


# --------------------- Helpers ---------------------


def _egarch_kappa(dist: str, nu: float | None) -> float:
    """Return E[|Z|] for standardized innovations.

    Uses Normal constant by default; for Student, uses closed form with
    Gamma functions when SciPy is available. Falls back to a safe Normal
    constant if not.

    Parameters
    ----------
    dist : str
        'normal' or 'student'.
    nu : float | None
        Degrees of freedom (Student only).

    Returns
    -------
    float
        Kappa constant.
    """
    d = dist.lower()
    if d == "student" and nu is not None and nu > GARCH_STUDENT_NU_MIN:
        try:
            from scipy.special import gammaln  # type: ignore

            nu_threshold = nu - 2.0
            ln_num = 0.5 * math.log(max(nu_threshold, GARCH_MIN_INIT_VAR)) + float(
                gammaln(0.5 * (nu - 1.0))
            )
            ln_den = 0.5 * math.log(math.pi) + float(gammaln(0.5 * nu))
            return float(math.exp(ln_num - ln_den))
        except Exception as ex:  # pragma: no cover - SciPy might be absent in CI
            logger.debug("Falling back to Normal kappa (SciPy unavailable): %s", ex)
    # Normal default: sqrt(2/pi)
    return float(math.sqrt(2.0 / math.pi))


def _one_step_update(e_last: float, s2_last: float, p: EgarchParams) -> float:
    """Compute one-step-ahead variance via EGARCH recursion.

    Parameters
    ----------
    e_last : float
        Last observed residual (t).
    s2_last : float
        Last conditional variance (for t).
    p : EgarchParams
        EGARCH parameters.

    Returns
    -------
    float
        Forecast variance for t+1.
    """
    s2_prev = float(max(s2_last, GARCH_MIN_INIT_VAR))
    z_prev = float(e_last) / math.sqrt(s2_prev)
    kappa = _egarch_kappa(p.dist, p.nu)
    ln_next = (
        p.omega + p.beta * math.log(s2_prev) + p.alpha * (abs(z_prev) - kappa) + p.gamma * z_prev
    )
    # Clip for numerical stability (exp(709) ~ 8.2e307 in IEEE 754)
    ln_next = min(ln_next, GARCH_LOG_VAR_MAX)
    s2_next = float(math.exp(ln_next))
    return max(s2_next, GARCH_MIN_INIT_VAR)


def _compute_kurtosis(resid: np.ndarray) -> float:
    """Compute excess kurtosis of residuals.

    Parameters
    ----------
    resid : np.ndarray
        Residual values.

    Returns
    -------
    float
        Excess kurtosis (0 for normal distribution).
    """
    if resid.size == 0:
        return 0.0
    mean_resid = np.mean(resid)
    m2 = float(np.mean((resid - mean_resid) ** 2))
    if m2 == 0.0:
        return 0.0
    m4 = float(np.mean((resid - mean_resid) ** 4))
    return (m4 / (m2**2)) - 3.0


def _select_distribution(resid: np.ndarray, dist_preference: str) -> tuple[str, float | None]:
    """Select distribution based on preference and data characteristics.

    Parameters
    ----------
    resid : np.ndarray
        Training residuals.
    dist_preference : str
        'auto' | 'normal' | 'student'.

    Returns
    -------
    tuple[str, float | None]
        Distribution name and degrees of freedom (if Student).
    """
    if dist_preference.lower() == "normal":
        return "normal", None
    if dist_preference.lower() == "student":
        return "student", GARCH_STUDENT_NU_INIT
    # Auto mode: check kurtosis
    if resid.size < GARCH_FIT_MIN_SIZE_KURTOSIS:
        return "normal", None
    kurt = _compute_kurtosis(resid)
    if kurt > 1.0:
        return "student", GARCH_STUDENT_NU_INIT
    return "normal", None


def _load_optimized_params(dist_preference: str) -> dict[str, float | None] | None:
    """Load optimized EGARCH parameters from MLE estimation file.

    Parameters
    ----------
    dist_preference : str
        Distribution preference ('auto', 'normal', 'student').

    Returns
    -------
    dict[str, float | None] | None
        Optimized parameters dict or None if not available.
        Note: 'nu' can be None for normal distribution.
    """
    if not GARCH_ESTIMATION_FILE.exists():
        return None

    try:
        with GARCH_ESTIMATION_FILE.open() as f:
            est_data = json.load(f)

        # Select best model based on distribution preference
        if dist_preference.lower() == "student" and "egarch_student" in est_data:
            params = est_data["egarch_student"]
        elif dist_preference.lower() == "normal" and "egarch_normal" in est_data:
            params = est_data["egarch_normal"]
        elif "egarch_student" in est_data:
            # Auto: prefer student if available
            params = est_data["egarch_student"]
        elif "egarch_normal" in est_data:
            params = est_data["egarch_normal"]
        else:
            return None

        if params and params.get("converged", False):
            nu_val: float | None = None
            if "nu" in params:
                nu_val = float(params["nu"])
            omega_val = params.get("omega")
            alpha_val = params.get("alpha")
            beta_val = params.get("beta")
            gamma_val = params.get("gamma", 0.0)
            if omega_val is None or alpha_val is None or beta_val is None:
                return None
            return {
                "omega": float(omega_val),
                "alpha": float(alpha_val),
                "beta": float(beta_val),
                "gamma": float(gamma_val),
                "nu": nu_val,
            }
    except Exception as ex:
        logger.debug("Failed to load optimized parameters: %s", ex)
    return None


def _fit_initial_params(resid_train: np.ndarray, *, dist_preference: str = "auto") -> EgarchParams:
    """Return initial EGARCH parameters for a given training slice.

    Requires optimized MLE parameters to be available. Raises error if not found
    to prevent incorrect initialization that could bias results.

    Parameters
    ----------
    resid_train : np.ndarray
        Training residuals (finite values).
    dist_preference : str
        'auto' | 'normal' | 'student'. When 'auto', selects 'student'
        if residual kurtosis suggests heavy tails.

    Returns
    -------
    EgarchParams
        Parameter set for EGARCH(1,1).

    Raises
    ------
    FileNotFoundError
        If GARCH_ESTIMATION_FILE does not exist.
    ValueError
        If optimized parameters are not available or invalid.
    """
    ee = np.asarray(resid_train, dtype=float)
    ee = ee[np.isfinite(ee)]

    if ee.size == 0:
        msg = "Cannot fit parameters: empty residual array"
        raise ValueError(msg)

    # Load optimized parameters - required, no fallback
    optimized = _load_optimized_params(dist_preference)
    if not optimized:
        if not GARCH_ESTIMATION_FILE.exists():
            msg = (
                f"GARCH estimation file not found: {GARCH_ESTIMATION_FILE}. "
                "Run GARCH parameter estimation first."
            )
            raise FileNotFoundError(msg)
        msg = (
            f"No valid optimized parameters found in {GARCH_ESTIMATION_FILE}. "
            "Ensure parameters are estimated and converged."
        )
        raise ValueError(msg)

    # Use optimized MLE parameters directly - they are already calibrated
    # for the scale of ARIMA residuals. Omega in EGARCH is in log-space
    # and is already properly scaled from the MLE estimation.
    nu_val = optimized.get("nu")
    dist = "student" if nu_val is not None else "normal"
    # Extract values - omega, alpha, beta, gamma are guaranteed to be float
    # (not None) by _load_optimized_params validation
    omega_val = optimized["omega"]
    alpha_val = optimized["alpha"]
    beta_val = optimized["beta"]
    gamma_val = optimized.get("gamma", 0.0)
    # Type assertions for type checker
    assert omega_val is not None
    assert alpha_val is not None
    assert beta_val is not None
    assert gamma_val is not None

    return EgarchParams(
        omega=float(omega_val),
        alpha=float(alpha_val),
        beta=float(beta_val),
        gamma=float(gamma_val),
        nu=nu_val,
        dist=dist,
    )


def _load_dataset() -> pd.DataFrame:
    """Load the ARIMA residual dataset from constants path.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least 'date', 'split', 'arima_residual_return'.
    """
    if not GARCH_DATASET_FILE.exists():
        msg = f"GARCH dataset not found: {GARCH_DATASET_FILE}"
        raise FileNotFoundError(msg)
    df = pd.read_csv(GARCH_DATASET_FILE, parse_dates=["date"])  # type: ignore[arg-type]
    required = {"date", "split", "arima_residual_return"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    return df


def _prepare_series(
    df: pd.DataFrame, include_train_forecasts: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare residual series and aligned forecast indices.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with 'date', 'split', 'arima_residual_return'.
    include_train_forecasts : bool
        If True, include train observations in forecast positions (after min window).
        If False, only test observations are forecasted (default behavior).

    Returns
    -------
    tuple
        (resid_f, dates_f, split_f, pos_forecast, pos_train_end)
        - resid_f: filtered residuals
        - dates_f: filtered dates
        - split_f: filtered split labels ('train' or 'test')
        - pos_forecast: positions to forecast
        - pos_train_end: last train position
    """
    data = df.sort_values("date").reset_index(drop=True)
    dates_series = pd.to_datetime(data["date"])
    dates_all = np.array(dates_series, dtype="datetime64[ns]")
    resid_series = pd.to_numeric(data["arima_residual_return"], errors="coerce")
    resid_all = np.asarray(resid_series, dtype=np.float64)
    split_all = data["split"].astype(str).to_numpy()

    valid_mask = np.isfinite(resid_all)
    resid_f = resid_all[valid_mask]
    dates_f = dates_all[valid_mask]
    split_f = split_all[valid_mask]

    idx_all = np.arange(data.shape[0])
    idx_valid = idx_all[valid_mask]

    # Map original index -> position in filtered arrays
    pos_in_valid = np.full(data.shape[0], -1, dtype=int)
    pos_in_valid[idx_valid] = np.arange(idx_valid.size)

    # Determine positions to forecast
    if include_train_forecasts:
        # Forecast all valid positions after minimum training window
        train_mask = split_all == "train"
        idx_train = idx_all[train_mask]
        idx_train_valid = np.intersect1d(idx_valid, idx_train, assume_unique=False)

        test_mask = split_all == "test"
        idx_test = idx_all[test_mask]
        idx_test_valid = np.intersect1d(idx_valid, idx_test, assume_unique=False)

        # Start forecasting after minimum train size
        if idx_train_valid.size >= GARCH_FIT_MIN_SIZE:
            # Get train positions after minimum window
            idx_train_forecast = idx_train_valid[GARCH_FIT_MIN_SIZE:]
            idx_forecast = np.concatenate([idx_train_forecast, idx_test_valid])
        else:
            # Only forecast test if not enough train data
            idx_forecast = idx_test_valid

        pos_forecast = pos_in_valid[idx_forecast]
        pos_forecast.sort()

        # Last train position
        pos_train_end = np.array([-1], dtype=int)
        if idx_train_valid.size > 0:
            pos_train_end = np.array([pos_in_valid[idx_train_valid.max()]], dtype=int)
    else:
        # Original behavior: only forecast test set
        test_mask = split_all == "test"
        idx_test = idx_all[test_mask]
        idx_test_valid = np.intersect1d(idx_valid, idx_test, assume_unique=False)

        if idx_test_valid.size == 0:
            return resid_f, dates_f, split_f, np.array([], dtype=int), np.array([], dtype=int)

        pos_forecast = pos_in_valid[idx_test_valid]
        pos_forecast.sort()

        # Last train position
        train_mask = split_all == "train"
        idx_train = idx_all[train_mask]
        idx_train_valid = np.intersect1d(idx_valid, idx_train, assume_unique=False)
        pos_train_end = np.array([-1], dtype=int)
        if idx_train_valid.size > 0:
            pos_train_end = np.array([pos_in_valid[idx_train_valid.max()]], dtype=int)

    return resid_f, dates_f, split_f, pos_forecast, pos_train_end


def _ensure_min_train_size(pos: int) -> bool:
    """Return True if we have enough data to estimate/forecast at position `pos`."""
    return int(pos) >= 2  # need at least two points to derive a valid s2_last and e_last


def _append_insufficient_data_point(
    pos: int,
    resid_f: np.ndarray,
    dates_f: np.ndarray,
    split_f: np.ndarray,
    s2_fore: list[float],
    e_test: list[float],
    dates_test: list[pd.Timestamp],
    split_test: list[str],
) -> None:
    """Append data point when insufficient training data.

    Parameters
    ----------
    pos : int
        Current position.
    resid_f : np.ndarray
        Filtered residuals.
    dates_f : np.ndarray
        Filtered dates.
    split_f : np.ndarray
        Filtered split labels.
    s2_fore : list[float]
        Forecast variances list.
    e_test : list[float]
        Test residuals list.
    dates_test : list[pd.Timestamp]
        Test dates list.
    split_test : list[str]
        Split labels list.
    """
    s2_fore.append(float("nan"))
    e_test.append(float(resid_f[pos]))
    dt_val = pd.to_datetime(dates_f[pos].item())
    dates_test.append(cast(pd.Timestamp, pd.Timestamp(dt_val)))
    split_test.append(str(split_f[pos]))


def _process_forecast_step(
    i: int,
    pos: int,
    resid_f: np.ndarray,
    window: str,
    window_size: int,
    dist_preference: str,
    refit_every: int,
    refit_count: int,
    curr_params: EgarchParams | None,
    keep_nu_between_refits: bool,
    last_nu: float | None,
    s2_fore: list[float],
    e_test: list[float],
    dates_test: list[pd.Timestamp],
    split_test: list[str],
    dates_f: np.ndarray,
    split_f: np.ndarray,
) -> tuple[int, EgarchParams, float | None]:
    """Process a single forecast step in the rolling loop.

    Parameters
    ----------
    i : int
        Current iteration index.
    pos : int
        Current position in filtered array.
    resid_f : np.ndarray
        Filtered residuals array.
    window : str
        Window type.
    window_size : int
        Window size.
    dist_preference : str
        Distribution preference.
    refit_every : int
        Refit frequency.
    refit_count : int
        Current refit count.
    curr_params : EgarchParams | None
        Current parameters.
    keep_nu_between_refits : bool
        Whether to keep nu between refits.
    last_nu : float | None
        Last nu value.
    s2_fore : list[float]
        Forecast variances list.
    e_test : list[float]
        Test residuals list.
    dates_test : list[pd.Timestamp]
        Test dates list.
    split_test : list[str]
        Split labels list.
    dates_f : np.ndarray
        Filtered dates array.
    split_f : np.ndarray
        Filtered split labels array.

    Returns
    -------
    tuple[int, EgarchParams, float | None]
        Updated refit count, parameters, and nu value.
    """
    new_refit_count = refit_count
    if _should_refit(i, refit_every):
        new_refit_count += 1

    new_params, new_nu = _refit_params_if_needed(
        i,
        pos,
        resid_f,
        window,
        window_size,
        dist_preference,
        refit_every,
        curr_params,
        keep_nu_between_refits,
        last_nu,
    )

    assert new_params is not None
    s2_next = _compute_forecast_variance(pos, resid_f, window, window_size, new_params)
    _append_forecast_point(
        pos, resid_f, dates_f, split_f, s2_next, s2_fore, e_test, dates_test, split_test
    )

    return new_refit_count, new_params, new_nu


def _append_forecast_point(
    pos: int,
    resid_f: np.ndarray,
    dates_f: np.ndarray,
    split_f: np.ndarray,
    s2_next: float,
    s2_fore: list[float],
    e_test: list[float],
    dates_test: list[pd.Timestamp],
    split_test: list[str],
) -> None:
    """Append forecast point to result lists.

    Parameters
    ----------
    pos : int
        Current position.
    resid_f : np.ndarray
        Filtered residuals.
    dates_f : np.ndarray
        Filtered dates.
    split_f : np.ndarray
        Filtered split labels.
    s2_next : float
        Forecast variance.
    s2_fore : list[float]
        Forecast variances list.
    e_test : list[float]
        Test residuals list.
    dates_test : list[pd.Timestamp]
        Test dates list.
    split_test : list[str]
        Split labels list.
    """
    s2_fore.append(s2_next)
    e_test.append(float(resid_f[int(pos)]))
    dt_val = pd.to_datetime(dates_f[int(pos)].item())
    dates_test.append(cast(pd.Timestamp, pd.Timestamp(dt_val)))
    split_test.append(str(split_f[int(pos)]))


def _compute_window_start(pos: int, window: str, window_size: int) -> int:
    """Compute window start position for training slice.

    Parameters
    ----------
    pos : int
        Current position in filtered array.
    window : str
        'expanding' or 'rolling'.
    window_size : int
        Window size for 'rolling' mode.

    Returns
    -------
    int
        Start position for training slice.
    """
    if window.lower() == "rolling":
        return max(0, int(pos) - int(window_size))
    return 0


def _should_refit(i: int, refit_every: int) -> bool:
    """Check if parameters should be refitted.

    Parameters
    ----------
    i : int
        Current iteration index.
    refit_every : int
        Refit frequency.

    Returns
    -------
    bool
        True if refit is needed.
    """
    return i == 0 or (i % int(refit_every) == 0)


def _apply_nu_constraint(
    params: EgarchParams, keep_nu: bool, last_nu: float | None
) -> EgarchParams:
    """Apply nu constraint if needed.

    Parameters
    ----------
    params : EgarchParams
        Parameters to potentially modify.
    keep_nu : bool
        Whether to keep nu from previous refit.
    last_nu : float | None
        Last nu value.

    Returns
    -------
    EgarchParams
        Parameters with nu constraint applied.
    """
    if keep_nu and params.dist == "student" and last_nu is not None:
        params.nu = float(last_nu)
    return params


def _refit_params_if_needed(
    i: int,
    pos: int,
    resid_f: np.ndarray,
    window: str,
    window_size: int,
    dist_preference: str,
    refit_every: int,
    curr_params: EgarchParams | None,
    keep_nu_between_refits: bool,
    last_nu: float | None,
) -> tuple[EgarchParams, float | None]:
    """Refit parameters if needed based on refit frequency.

    Parameters
    ----------
    i : int
        Current iteration index.
    pos : int
        Current position in filtered array.
    resid_f : np.ndarray
        Filtered residuals array.
    window : str
        'expanding' or 'rolling'.
    window_size : int
        Window size for 'rolling' mode.
    dist_preference : str
        Distribution preference.
    refit_every : int
        Refit frequency.
    curr_params : EgarchParams | None
        Current parameters (may be None).
    keep_nu_between_refits : bool
        Whether to keep nu between refits.
    last_nu : float | None
        Last nu value.

    Returns
    -------
    tuple[EgarchParams, float | None]
        Updated parameters and nu value.
    """
    if not _should_refit(i, refit_every):
        assert curr_params is not None
        return curr_params, last_nu

    start = _compute_window_start(pos, window, window_size)
    train_slice = resid_f[start : int(pos)]
    new_params = _fit_initial_params(train_slice, dist_preference=dist_preference)
    new_params = _apply_nu_constraint(new_params, keep_nu_between_refits, last_nu)
    return new_params, new_params.nu


def _compute_forecast_variance(
    pos: int,
    resid_f: np.ndarray,
    window: str,
    window_size: int,
    curr_params: EgarchParams,
) -> float:
    """Compute one-step-ahead variance forecast.

    Parameters
    ----------
    pos : int
        Current position in filtered array.
    resid_f : np.ndarray
        Filtered residuals array.
    window : str
        'expanding' or 'rolling'.
    window_size : int
        Window size for 'rolling' mode.
    curr_params : EgarchParams
        Current EGARCH parameters.

    Returns
    -------
    float
        Forecast variance for next step.
    """
    start = _compute_window_start(pos, window, window_size)
    hist = resid_f[start : int(pos)]
    s2_hist = egarch11_variance(
        hist,
        curr_params.omega,
        curr_params.alpha,
        curr_params.gamma,
        curr_params.beta,
        dist=curr_params.dist,
        nu=curr_params.nu,
    )
    if not (np.all(np.isfinite(s2_hist)) and s2_hist.size >= 1):
        msg = (
            f"Failed to compute variance history for position {pos}. "
            f"Invalid variance values detected. This may indicate numerical issues "
            f"or invalid parameters."
        )
        raise ValueError(msg)
    s2_last = float(s2_hist[-1])
    e_last = float(resid_f[int(pos) - 1])
    return _one_step_update(e_last, s2_last, curr_params)


def _build_forecasts_dataframe(
    dates_test: list[pd.Timestamp],
    e_test: list[float],
    s2_fore: list[float],
    split_test: list[str],
) -> pd.DataFrame:
    """Build forecasts DataFrame from lists.

    Parameters
    ----------
    dates_test : list[pd.Timestamp]
        Forecast dates.
    e_test : list[float]
        Realized residuals.
    s2_fore : list[float]
        Forecast variances.
    split_test : list[str]
        Split labels ('train' or 'test').

    Returns
    -------
    pd.DataFrame
        Forecasts DataFrame with columns: date, e, sigma2_forecast, split.
    """
    return pd.DataFrame(
        {
            "date": np.asarray(dates_test),
            "e": np.asarray(e_test, dtype=float),
            "sigma2_forecast": np.asarray(s2_fore, dtype=float),
            "split": np.asarray(split_test, dtype=str),
        }
    )


def _compute_metrics(
    forecasts: pd.DataFrame,
    refit_every: int,
    window: str,
    window_size: int,
    refit_count: int,
    curr_params: EgarchParams | None,
) -> dict[str, Any]:
    """Compute metrics dictionary from forecasts and parameters.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Forecasts DataFrame.
    refit_every : int
        Refit frequency.
    window : str
        Window type.
    window_size : int
        Window size.
    refit_count : int
        Number of refits.
    curr_params : EgarchParams | None
        Current parameters.

    Returns
    -------
    dict[str, Any]
        Metrics dictionary.
    """
    ql = qlike_loss(forecasts["e"].to_numpy(), forecasts["sigma2_forecast"].to_numpy())
    return {
        "n_test": int(forecasts.shape[0]),
        "refit_every": int(refit_every),
        "window": str(window),
        "window_size": int(window_size),
        "refit_count": int(refit_count),
        "dist": curr_params.dist if curr_params else "normal",
        "nu": float(curr_params.nu) if (curr_params and curr_params.nu is not None) else None,
        "qlike": float(ql) if np.isfinite(ql) else float("nan"),
    }


def _is_student_distribution(params: EgarchParams | None) -> bool:
    """Check if parameters indicate Student distribution.

    Parameters
    ----------
    params : EgarchParams | None
        Parameters to check.

    Returns
    -------
    bool
        True if Student distribution should be used.
    """
    if not params:
        return False
    if params.dist != "student":
        return False
    if not params.nu:
        return False
    return params.nu > GARCH_STUDENT_NU_MIN


def _compute_var_quantile(alpha: float, params: EgarchParams | None) -> float:
    """Compute VaR quantile for given alpha and distribution.

    Parameters
    ----------
    alpha : float
        VaR alpha level.
    params : EgarchParams | None
        Current parameters.

    Returns
    -------
    float
        Quantile value.
    """
    from scipy.stats import norm, t  # type: ignore

    if _is_student_distribution(params):
        assert params is not None and params.nu is not None
        return float(t.ppf(float(alpha), df=float(params.nu)))
    return float(norm.ppf(float(alpha)))


def _add_var_columns(
    forecasts: pd.DataFrame,
    var_alphas: list[float],
    curr_params: EgarchParams | None,
) -> None:
    """Add VaR columns to forecasts DataFrame.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Forecasts DataFrame (modified in place).
    var_alphas : list[float]
        VaR alpha levels.
    curr_params : EgarchParams | None
        Current parameters.
    """
    try:
        s = np.sqrt(forecasts["sigma2_forecast"].to_numpy())
        for a in var_alphas:
            q = _compute_var_quantile(a, curr_params)
            forecasts[f"var_{a}"] = q * s
    except Exception as ex:  # pragma: no cover - SciPy always present in project
        logger.debug("Skipping VaR series (SciPy unavailable): %s", ex)


def run_rolling_egarch(
    df: pd.DataFrame,
    *,
    refit_every: int = GARCH_REFIT_EVERY_DEFAULT,
    window: str = GARCH_REFIT_WINDOW_DEFAULT,
    window_size: int = GARCH_REFIT_WINDOW_SIZE_DEFAULT,
    dist_preference: str = "auto",
    keep_nu_between_refits: bool = True,
    var_alphas: list[float] | None = None,
    calibrate_mz: bool = False,  # reserved for future use
    calibrate_var: bool = False,  # reserved for future use
    include_train_forecasts: bool = False,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """Run walk-forward EGARCH(1,1) backtest on ARIMA residuals.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing 'date', 'split', and 'arima_residual_return'.
    refit_every : int
        Refit frequency (in test observations).
    window : str
        'expanding' or 'rolling'.
    window_size : int
        Window size for 'rolling' mode.
    dist_preference : str
        'auto' | 'normal' | 'student'.
    keep_nu_between_refits : bool
        If True, keep nu from previous refit when dist='student'.
    var_alphas : list[float] | None
        Optional VaR levels (left-tail) to compute.
    calibrate_mz : bool
        Reserved (not applied here to keep VaR coherent with variance path).
    calibrate_var : bool
        Reserved (empirical quantile calibration for VaR).

    include_train_forecasts : bool
        If True, include train set in rolling forecasts (default: False).
        Useful for generating ML features on full dataset.

    Returns
    -------
    Tuple[pd.DataFrame, dict[str, Any]]
        Forecasts dataframe and metrics dictionary.
    """
    resid_f, dates_f, split_f, pos_forecast, pos_train_end = _prepare_series(
        df, include_train_forecasts=include_train_forecasts
    )
    if pos_forecast.size == 0:
        logger.warning("No forecast observations with valid residuals; returning empty outputs.")
        empty_df = pd.DataFrame(
            {
                "date": pd.Series([], dtype="datetime64[ns]"),
                "e": pd.Series([], dtype=float),
                "sigma2_forecast": pd.Series([], dtype=float),
                "split": pd.Series([], dtype=str),
            }
        )
        return empty_df, {"n_test": 0}

    s2_fore: list[float] = []
    e_test: list[float] = []
    dates_test: list[pd.Timestamp] = []
    split_test: list[str] = []
    refit_count = 0
    curr_params: EgarchParams | None = None
    last_nu: float | None = None

    for i, pos in enumerate(pos_forecast):
        if not _ensure_min_train_size(int(pos)):
            _append_insufficient_data_point(
                pos, resid_f, dates_f, split_f, s2_fore, e_test, dates_test, split_test
            )
            continue

        refit_count, curr_params, last_nu = _process_forecast_step(
            i,
            int(pos),
            resid_f,
            window,
            window_size,
            dist_preference,
            refit_every,
            refit_count,
            curr_params,
            keep_nu_between_refits,
            last_nu,
            s2_fore,
            e_test,
            dates_test,
            split_test,
            dates_f,
            split_f,
        )

    forecasts = _build_forecasts_dataframe(dates_test, e_test, s2_fore, split_test)
    metrics = _compute_metrics(
        forecasts, refit_every, window, window_size, refit_count, curr_params
    )

    if var_alphas:
        _add_var_columns(forecasts, var_alphas, curr_params)

    return forecasts, metrics


def run_from_artifacts(
    *,
    refit_every: int = GARCH_REFIT_EVERY_DEFAULT,
    window: str = GARCH_REFIT_WINDOW_DEFAULT,
    window_size: int = GARCH_REFIT_WINDOW_SIZE_DEFAULT,
    dist_preference: str = "auto",
    keep_nu_between_refits: bool = True,
    var_alphas: list[float] | None = None,
    include_train_forecasts: bool = False,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """Load dataset from artifacts and run rolling EGARCH backtest.

    Parameters
    ----------
    include_train_forecasts : bool
        If True, include train set forecasts (default: False for backtest).

    Returns
    -------
    Tuple[pd.DataFrame, dict[str, Any]]
        Forecasts and metrics.
    """
    df = _load_dataset()
    return run_rolling_egarch(
        df,
        refit_every=refit_every,
        window=window,
        window_size=window_size,
        dist_preference=dist_preference,
        keep_nu_between_refits=keep_nu_between_refits,
        var_alphas=var_alphas,
        include_train_forecasts=include_train_forecasts,
    )


def save_rolling_outputs(forecasts: pd.DataFrame, metrics: dict[str, Any]) -> None:
    """Persist forecasts and metrics to constants paths.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Forecast rows: columns include 'date', 'e', 'sigma2_forecast', 'split'.
    metrics : dict[str, Any]
        Summary metrics dictionary.
    """
    GARCH_ROLLING_FORECASTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(GARCH_ROLLING_FORECASTS_FILE, index=False)
    GARCH_ROLLING_EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with GARCH_ROLLING_EVAL_FILE.open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(
        "Saved rolling EGARCH outputs: %s, %s",
        GARCH_ROLLING_FORECASTS_FILE,
        GARCH_ROLLING_EVAL_FILE,
    )


def build_ml_dataset(
    *,
    refit_every: int = GARCH_REFIT_EVERY_DEFAULT,
    window: str = GARCH_REFIT_WINDOW_DEFAULT,
    window_size: int = GARCH_REFIT_WINDOW_SIZE_DEFAULT,
    dist_preference: str = "auto",
    keep_nu_between_refits: bool = True,
) -> pd.DataFrame:
    """Build ML-ready dataset with GARCH volatility features.

    Generates rolling GARCH forecasts on full dataset (train+test) and merges
    with original data to create features for machine learning models.

    Parameters
    ----------
    refit_every : int
        Refit frequency (default: 20).
    window : str
        Window type: 'expanding' or 'rolling'.
    window_size : int
        Window size for rolling mode.
    dist_preference : str
        Distribution preference: 'auto', 'normal', or 'student'.
    keep_nu_between_refits : bool
        Whether to keep nu parameter between refits.

    Returns
    -------
    pd.DataFrame
        ML-ready dataset with columns:
        - date: Date
        - weighted_log_return: Log return (target variable)
        - arima_residual_return: ARIMA residuals
        - sigma2_forecast: GARCH variance forecast
        - sigma_forecast: GARCH volatility forecast (sqrt of variance)
        - split: 'train' or 'test'

    Side Effects
    ------------
    Saves garch_variance.csv in results/rolling/ with columns:
        - date, weighted_closing, weighted_open, log_weighted_return, sigma2_garch, split

    Notes
    -----
    - Forecasts on train start after GARCH_FIT_MIN_SIZE observations
    - All forecasts use only data up to t-1 (no look-ahead bias)
    - Refit occurs every `refit_every` observations
    """
    # Load base dataset
    df_base = _load_dataset()

    # Generate rolling forecasts on train+test
    forecasts, _metrics = run_rolling_egarch(
        df_base,
        refit_every=refit_every,
        window=window,
        window_size=window_size,
        dist_preference=dist_preference,
        keep_nu_between_refits=keep_nu_between_refits,
        var_alphas=None,
        include_train_forecasts=True,
    )

    # Merge forecasts with base data
    # forecasts has: date, e, sigma2_forecast, split
    # df_base has: date, weighted_log_return, weighted_return, arima_residual_return, split
    # Use weighted_log_return instead of weighted_return (log returns are better for financial models)
    df_ml = pd.merge(
        df_base[["date", "weighted_log_return", "arima_residual_return", "split"]],
        forecasts[["date", "sigma2_forecast"]],
        on="date",
        how="left",
    )

    # Add derived features
    df_ml["sigma_forecast"] = np.sqrt(df_ml["sigma2_forecast"])

    # Sort by date
    df_ml = df_ml.sort_values("date").reset_index(drop=True)

    # Save garch_variance dataset in rolling/ directory with only essential columns
    # Keep only: date, weighted_closing, weighted_open, log_weighted_return, sigma2_garch, split
    rolling_dir = RESULTS_DIR / "rolling"
    rolling_dir.mkdir(parents=True, exist_ok=True)

    # Merge with base data to get weighted_closing, weighted_open and split
    df_garch_variance = pd.merge(
        df_base[["date", "weighted_closing", "weighted_open", "split"]],
        forecasts[["date", "sigma2_forecast"]],
        on="date",
        how="left",
    )

    # Calculate log of weighted returns from prices: log(weighted_closing / weighted_open)
    # This is equivalent to log(weighted_closing) - log(weighted_open)
    df_garch_variance["log_weighted_return"] = np.log(
        df_garch_variance["weighted_closing"] / df_garch_variance["weighted_open"]
    )

    # Rename sigma2_forecast to sigma2_garch
    df_garch_variance = df_garch_variance.rename(columns={"sigma2_forecast": "sigma2_garch"})

    # Reorder columns: date, weighted_closing, weighted_open, log_weighted_return, sigma2_garch, split
    df_garch_variance = cast(
        pd.DataFrame,
        df_garch_variance[
            ["date", "weighted_closing", "weighted_open", "log_weighted_return", "sigma2_garch", "split"]
        ],
    )

    # Sort by date
    df_garch_variance = df_garch_variance.sort_values(by="date").reset_index(drop=True)

    # Save to rolling/garch_variance.csv
    garch_variance_file = rolling_dir / "garch_variance.csv"
    df_garch_variance.to_csv(garch_variance_file, index=False)
    logger.info(
        "Saved garch_variance dataset: %s (%d rows)", garch_variance_file, len(df_garch_variance)
    )

    logger.info(
        "Built ML dataset: %d rows (%d train, %d test)",
        len(df_ml),
        (df_ml["split"] == "train").sum(),
        (df_ml["split"] == "test").sum(),
    )

    return df_ml


__all__ = [
    "EgarchParams",
    "GarchParams",
    "build_ml_dataset",
    "run_from_artifacts",
    "run_rolling_egarch",
    "save_rolling_outputs",
]
