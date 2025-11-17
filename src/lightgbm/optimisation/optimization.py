"""LightGBM hyperparameter optimization using Optuna with time series cross-validation."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import lightgbm as lgb
from src.constants import (
    DEFAULT_RANDOM_STATE,
    LIGHTGBM_OPTIMIZATION_N_SPLITS,
    LIGHTGBM_OPTIMIZATION_N_TRIALS,
)
from src.lightgbm.optimisation.objective import _calculate_rmse, _lightgbm_cv_objective
from src.lightgbm.optimisation.validation import validate_optimization_data
from src.utils import get_logger

logger = get_logger(__name__)


def _create_optuna_study(study_name: str) -> optuna.Study:
    """Create Optuna study with MedianPruner.

    Args:
        study_name: Name for the Optuna study.

    Returns:
        Created Optuna study.
    """
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # Wait for 5 trials before starting to prune
        n_warmup_steps=2,  # Wait for 2 folds before pruning
        interval_steps=1,  # Check at each fold
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=DEFAULT_RANDOM_STATE),
        pruner=pruner,
    )
    return study


def _create_trial_callback(study_name: str) -> Any:
    """Create callback function to log trial results.

    Args:
        study_name: Name of the study for logging.

    Returns:
        Callback function for Optuna study.optimize.
    """

    def log_trial_with_model_name(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Log trial results with model name prefix."""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            logger.info(
                f"[{study_name}] Trial {trial.number}: value={trial.value:.8f} | "
                f"best={study.best_value:.8f}"
            )
        elif trial.state == optuna.trial.TrialState.PRUNED:
            logger.info(f"[{study_name}] Trial {trial.number}: PRUNED")

    return log_trial_with_model_name


def _prepare_fold_data(
    X_np: np.ndarray,
    y_np: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    feature_names: list[str] | None,
) -> tuple[
    pd.DataFrame | np.ndarray, pd.Series | np.ndarray, pd.DataFrame | np.ndarray, np.ndarray
]:
    """Prepare fold data for evaluation.

    Args:
        X_np: Full features array.
        y_np: Full target array.
        train_idx: Training indices.
        val_idx: Validation indices.
        feature_names: Optional list of feature names.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val).
    """
    X_train_fold_np = X_np[train_idx]
    y_train_fold = y_np[train_idx]
    X_val_fold_np = X_np[val_idx]
    y_val_fold = y_np[val_idx]

    if feature_names is not None:
        X_train_fold = pd.DataFrame(
            X_train_fold_np, columns=pd.Index(cast(list[str], feature_names))
        )
        X_val_fold = pd.DataFrame(X_val_fold_np, columns=pd.Index(cast(list[str], feature_names)))
    else:
        X_train_fold = X_train_fold_np
        X_val_fold = X_val_fold_np

    return X_train_fold, y_train_fold, X_val_fold, y_val_fold


def _evaluate_single_fold(
    best_params: dict[str, Any],
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Evaluate best parameters on a single CV fold.

    Args:
        best_params: Best hyperparameters.
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.

    Returns:
        RMSE for the fold.
    """
    model_params = best_params.copy()
    model = lgb.LGBMRegressor(**model_params, random_state=DEFAULT_RANDOM_STATE, verbosity=-1)
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    return _calculate_rmse(y_val, y_val_pred)


def _evaluate_best_params(
    best_params: dict[str, Any],
    X_np: np.ndarray,
    y_np: np.ndarray,
    tscv: TimeSeriesSplit,
    feature_names: list[str] | None,
) -> tuple[list[float], int]:
    """Evaluate best parameters on all CV folds.

    Args:
        best_params: Best hyperparameters from optimization.
        X_np: Full features array.
        y_np: Full target array.
        tscv: TimeSeriesSplit object.
        feature_names: Optional list of feature names.

    Returns:
        Tuple of (fold_rmses list, best_fold index).
    """
    fold_rmses: list[float] = []
    for train_idx, val_idx in tscv.split(X_np):
        X_train, y_train, X_val, y_val = _prepare_fold_data(
            X_np, y_np, train_idx, val_idx, feature_names
        )
        fold_rmse = _evaluate_single_fold(best_params, X_train, y_train, X_val, y_val)
        fold_rmses.append(fold_rmse)

    best_fold = int(np.argmin(fold_rmses))
    return fold_rmses, best_fold


def _prepare_optimization_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare data for optimization by validating and converting to numpy.

    Args:
        X: Features DataFrame.
        y: Target Series.

    Returns:
        Tuple of (X_np, y_np, feature_names).
    """
    validate_optimization_data(X, y)
    X_np = cast(np.ndarray, X.values)
    y_np = cast(np.ndarray, y.values)
    feature_names = list(X.columns)
    return X_np, y_np, feature_names


def _run_optuna_optimization(
    study: optuna.Study,
    X_np: np.ndarray,
    y_np: np.ndarray,
    tscv: TimeSeriesSplit,
    feature_names: list[str],
    n_trials: int,
    callback: Any,
) -> tuple[dict[str, Any], float]:
    """Run Optuna optimization study.

    Args:
        study: Optuna study object.
        X_np: Features array.
        y_np: Target array.
        tscv: TimeSeriesSplit object.
        feature_names: List of feature names.
        n_trials: Number of trials.
        callback: Callback function.

    Returns:
        Tuple of (best_params, best_rmse_cv).
    """
    logger.info("Running Optuna optimization with walk-forward CV...")
    study.optimize(
        lambda trial: _lightgbm_cv_objective(trial, X_np, y_np, tscv, feature_names),
        n_trials=n_trials,
        callbacks=[callback],
        show_progress_bar=True,
    )
    best_params = study.best_params.copy()
    best_rmse_cv = study.best_value
    return best_params, best_rmse_cv


def _build_optimization_results(
    best_params: dict[str, Any],
    best_rmse_cv: float,
    fold_rmses: list[float],
    best_fold: int,
    n_trials: int,
    study_name: str,
) -> dict[str, Any]:
    """Build results dictionary from optimization outputs.

    Args:
        best_params: Best hyperparameters.
        best_rmse_cv: Best CV RMSE score.
        fold_rmses: List of fold RMSEs.
        best_fold: Index of best fold.
        n_trials: Number of trials.
        study_name: Study name.

    Returns:
        Results dictionary.
    """
    return {
        "best_params": best_params,
        "best_rmse_cv": best_rmse_cv,
        "n_trials": n_trials,
        "study_name": study_name,
        "fold_rmses": fold_rmses,
        "best_fold": best_fold,
    }


def _log_optimization_completion(
    best_rmse_cv: float,
    fold_rmses: list[float],
    best_fold: int,
    best_params: dict[str, Any],
) -> None:
    """Log optimization completion information.

    Args:
        best_rmse_cv: Best CV RMSE score.
        fold_rmses: List of fold RMSEs.
        best_fold: Index of best fold.
        best_params: Best hyperparameters.
    """
    logger.info(f"Optimization complete - Best RMSE (CV): {best_rmse_cv:.6f} (log-volatility)")
    logger.info(f"Fold RMSEs: {fold_rmses}, Best fold: {best_fold}")
    logger.info(f"Best parameters: {best_params}")


def _log_optimization_start_info(study_name: str, n_trials: int) -> None:
    """Log optimization start information.

    Args:
        study_name: Name of the study.
        n_trials: Number of trials.
    """
    logger.info(f"Starting LightGBM optimization: {study_name}")
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"Time series CV splits: {LIGHTGBM_OPTIMIZATION_N_SPLITS}")


def _run_optimization_and_evaluate(
    X_np: np.ndarray,
    y_np: np.ndarray,
    feature_names: list[str],
    study_name: str,
    n_trials: int,
) -> tuple[dict[str, Any], float, list[float], int]:
    """Run optimization and evaluate best params.

    Args:
        X_np: Features array.
        y_np: Target array.
        feature_names: List of feature names.
        study_name: Name of the study.
        n_trials: Number of trials.

    Returns:
        Tuple of (best_params, best_rmse_cv, fold_rmses, best_fold).
    """
    tscv = TimeSeriesSplit(n_splits=LIGHTGBM_OPTIMIZATION_N_SPLITS)
    study = _create_optuna_study(study_name)
    callback = _create_trial_callback(study_name)

    best_params, best_rmse_cv = _run_optuna_optimization(
        study, X_np, y_np, tscv, feature_names, n_trials, callback
    )
    fold_rmses, best_fold = _evaluate_best_params(best_params, X_np, y_np, tscv, feature_names)

    return best_params, best_rmse_cv, fold_rmses, best_fold


def optimize_lightgbm_with_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    study_name: str,
    n_trials: int = LIGHTGBM_OPTIMIZATION_N_TRIALS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Optimize LightGBM hyperparameters using Optuna with walk-forward CV.

    Args:
        X: Features DataFrame.
        y: Target Series.
        study_name: Name for the Optuna study.
        n_trials: Number of optimization trials.

    Returns:
        Tuple of (results dict, best params dict).
    """
    _log_optimization_start_info(study_name, n_trials)
    X_np, y_np, feature_names = _prepare_optimization_data(X, y)

    best_params, best_rmse_cv, fold_rmses, best_fold = _run_optimization_and_evaluate(
        X_np, y_np, feature_names, study_name, n_trials
    )

    _log_optimization_completion(best_rmse_cv, fold_rmses, best_fold, best_params)
    results = _build_optimization_results(
        best_params, best_rmse_cv, fold_rmses, best_fold, n_trials, study_name
    )

    return results, best_params


def optimize_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    study_name: str,
    n_trials: int = LIGHTGBM_OPTIMIZATION_N_TRIALS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Optimize LightGBM hyperparameters using Optuna with walk-forward CV.

    Args:
        X: Features DataFrame.
        y: Target Series.
        study_name: Name for the Optuna study.
        n_trials: Number of optimization trials.

    Returns:
        Tuple of (best parameters dict, best params dict for compatibility).
    """
    results, best_params = optimize_lightgbm_with_optuna(X, y, study_name, n_trials)

    logger.info(f"Optimization complete: {study_name}")
    logger.info(f"Best RMSE (CV): {results['best_rmse_cv']:.6f}")
    logger.info(f"Best parameters: {results['best_params']}")

    return results, best_params
