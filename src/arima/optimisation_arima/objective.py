"""Optuna objectives for ARIMA hyper-parameter search."""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd

from src.constants import (
    ARIMA_D_MAX,
    ARIMA_D_MIN,
    ARIMA_P_MAX,
    ARIMA_P_MIN,
    ARIMA_Q_MAX,
    ARIMA_Q_MIN,
    ARIMA_REFIT_EVERY_OPTIONS,
    ARIMA_TREND_ALLOWED_VALUES,
)
from src.utils import get_logger

from .model_evaluation import ArimaParams, evaluate_param_combination

logger = get_logger(__name__)


def _suggest_params(trial: optuna.Trial) -> ArimaParams:
    """Suggest ARIMA parameters for an Optuna trial.

    Optimizes all hyperparameters including refit_every frequency, which controls
    how often the model is retrained during rolling forecast. Lower values (e.g., 10)
    mean more frequent refits (better adaptation but higher cost), while higher values
    (e.g., 50) mean less frequent refits (faster but potentially less adaptive).

    Args:
        trial: Optuna trial object.

    Returns:
        ArimaParams object with suggested parameters including trend and refit_every.
    """
    p = trial.suggest_int("p", ARIMA_P_MIN, ARIMA_P_MAX)
    d = trial.suggest_int("d", ARIMA_D_MIN, ARIMA_D_MAX)
    q = trial.suggest_int("q", ARIMA_Q_MIN, ARIMA_Q_MAX)
    trend = str(trial.suggest_categorical("trend", ARIMA_TREND_ALLOWED_VALUES))

    # Optimize refit frequency (trade-off between adaptation and computational cost)
    refit_every = trial.suggest_categorical("refit_every", ARIMA_REFIT_EVERY_OPTIONS)

    return ArimaParams(p, d, q, trend, refit_every)


def _store_evaluation_results(trial: optuna.Trial, res: dict, score: float) -> None:
    """Store evaluation results in trial user attributes.

    Args:
        trial: Optuna trial object.
        res: Evaluation results dictionary.
        score: Final score (AIC or BIC) used for optimization.
    """
    if "error" not in res:
        trial.set_user_attr("aic", res.get("aic", float("inf")))
        trial.set_user_attr("bic", res.get("bic", float("inf")))
    else:
        trial.set_user_attr("error", res.get("error", "Unknown error"))


def _log_trial_parameters(trial: optuna.Trial, params: ArimaParams) -> None:
    """Log the parameters suggested for this trial.

    Args:
        trial: Optuna trial object.
        params: ARIMA parameters for this trial.
    """
    logger.info(
        "Trial %d | ARIMA(%d,%d,%d), trend=%s, refit_every=%d",
        trial.number,
        params.p,
        params.d,
        params.q,
        params.trend,
        params.refit_every,
    )


def _handle_evaluation_error(trial: optuna.Trial, res: dict) -> float | None:
    """Handle evaluation errors and return infinity if evaluation failed.

    Args:
        trial: Optuna trial object.
        res: Evaluation results dictionary.

    Returns:
        Infinity if error occurred, None otherwise.
    """
    if "error" in res:
        logger.warning("Trial %d | evaluation error: %s", trial.number, res["error"])
        return float("inf")
    return None


def _extract_criterion_value(res: dict, criterion: str) -> float | None:
    """Extract and validate the criterion value from evaluation results.

    Args:
        res: Evaluation results dictionary.
        criterion: Criterion name ("aic" or "bic").

    Returns:
        Valid criterion value or None if invalid.
    """
    criterion_value = res.get(criterion, float("inf"))
    if not isinstance(criterion_value, (int, float)) or not np.isfinite(criterion_value):
        return None
    return float(criterion_value)


def _log_final_score(
    trial: optuna.Trial,
    criterion: str,
    score: float,
) -> None:
    """Log the final score for this trial.

    Args:
        trial: Optuna trial object.
        criterion: Criterion name ("aic" or "bic").
        score: Final score (AIC or BIC).
    """
    logger.info(
        "Trial %d | %s=%.4f",
        trial.number,
        criterion.upper(),
        score,
    )


def _evaluate_and_store(
    trial: optuna.Trial,
    train_series: pd.Series,
    criterion: str,
    backtest_cfg: dict[str, int] | None = None,
    stats_callback: object | None = None,
) -> float:
    """Generic objective for AIC/BIC optimization with walk-forward CV.

    Uses AIC/BIC alone for optimization (theoretically optimal for forecasting).
    Walk-forward CV is performed to test model robustness, but only AIC/BIC
    is used for optimization (validation metrics are stored for information only).

    Returns +inf only when evaluation clearly fails (non-numeric/invalid criterion).

    Args:
        trial: Optuna trial object.
        train_series: Training time series data.
        criterion: Criterion to optimize ("aic" or "bic").
        backtest_cfg: Optional backtest configuration for walk-forward CV.
        stats_callback: Optional callback (unused, kept for backward compatibility).

    Returns:
        AIC or BIC value (lower is better).
    """
    params = _suggest_params(trial)
    _log_trial_parameters(trial, params)

    res = evaluate_param_combination(train_series, params, backtest_cfg=backtest_cfg)

    # Handle evaluation errors
    error_score = _handle_evaluation_error(trial, res)
    if error_score is not None:
        return error_score

    # Extract and validate criterion value
    criterion_value = _extract_criterion_value(res, criterion)
    if criterion_value is None:
        return float("inf")

    # Use AIC/BIC alone for optimization (theoretically optimal for forecasting)
    score = criterion_value
    _log_final_score(trial, criterion, score)
    # Store the score that was actually optimized (AIC/BIC)
    _store_evaluation_results(trial, res, score)
    return float(score)


def objective_aic(
    trial: optuna.Trial,
    train_series: pd.Series,
    backtest_cfg: dict[str, int] | None = None,
    stats_callback: object | None = None,
) -> float:
    """Optuna objective function minimizing AIC with walk-forward CV.

    AIC (Akaike Information Criterion) is used as the sole optimization criterion because:
    1. AIC minimizes one-step-ahead prediction error (asymptotically optimal for forecasting)
    2. BIC minimizes description length (better for model selection, not forecasting)
    3. For financial time series forecasting, AIC is theoretically superior

    Walk-forward CV is performed to test model robustness across different time periods,
    but only AIC is used for optimization (validation metrics are stored for information only).

    Reference: Burnham & Anderson (2004), "Multimodel Inference: Understanding AIC and BIC
    in Model Selection"

    Args:
        trial: Optuna trial object.
        train_series: Training time series data.
        backtest_cfg: Optional backtest configuration for walk-forward CV.
        stats_callback: Optional callback (unused, kept for backward compatibility).

    Returns:
        AIC value for the suggested parameters.
        Returns infinity if evaluation fails.
    """
    return _evaluate_and_store(trial, train_series, "aic", backtest_cfg, stats_callback)
