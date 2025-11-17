"""Optuna objective functions for LightGBM optimization."""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit

import lightgbm as lgb
from src.constants import DEFAULT_RANDOM_STATE

logger = optuna.logging.get_logger(__name__)


def _calculate_rmse(y_true: Any, y_pred: Any) -> float:
    """Calculate RMSE from predictions."""
    y_true_np = np.asarray(y_true).flatten()
    y_pred_np = np.asarray(y_pred).flatten()
    return float(np.sqrt(np.mean((y_true_np - y_pred_np) ** 2)))


def _suggest_hyperparameters(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest LightGBM hyperparameters for Optuna trial.

    Args:
        trial: Optuna trial object.

    Returns:
        Dictionary of LightGBM hyperparameters.
    """
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        # Model capacity
        "num_leaves": trial.suggest_int("num_leaves", 31, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        # Speed / implicit regularization
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        # Subsampling columns / rows
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),  # 0 = off possible
        # Leaf size control
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        # Explicit regularization
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": DEFAULT_RANDOM_STATE,
        "verbosity": -1,
        "force_col_wise": True,
    }


def _create_lightgbm_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str] | None = None,
) -> tuple[lgb.Dataset, lgb.Dataset]:
    """Create LightGBM Dataset objects for training and validation.

    Args:
        X_train: Training features array.
        y_train: Training target array.
        X_val: Validation features array.
        y_val: Validation target array.
        feature_names: Optional list of feature names.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    if feature_names is not None:
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=feature_names,
            free_raw_data=False,
        )
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            feature_name=feature_names,
            reference=train_data,
            free_raw_data=False,
        )
    else:
        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)

    return train_data, val_data


def _train_fold_model(
    params: dict[str, Any],
    train_data: lgb.Dataset,
    val_data: lgb.Dataset,
) -> lgb.Booster:
    """Train LightGBM model for a single CV fold.

    Args:
        params: LightGBM hyperparameters.
        train_data: Training dataset.
        val_data: Validation dataset.

    Returns:
        Trained LightGBM booster model.
    """
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def _evaluate_fold(
    model: lgb.Booster,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Evaluate model on validation fold and return RMSE.

    Args:
        model: Trained LightGBM model.
        X_val: Validation features.
        y_val: Validation target.

    Returns:
        RMSE score for the fold.
    """
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    return _calculate_rmse(y_val, y_pred)


def _evaluate_single_cv_fold(
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str] | None,
) -> float:
    """Evaluate a single CV fold.

    Args:
        params: LightGBM hyperparameters.
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        feature_names: Optional feature names.

    Returns:
        RMSE for the fold.
    """
    train_data, val_data = _create_lightgbm_datasets(X_train, y_train, X_val, y_val, feature_names)
    model = _train_fold_model(params, train_data, val_data)
    return _evaluate_fold(model, X_val, y_val)


def _report_trial_progress(trial: optuna.Trial, fold_rmses: list[float], fold_idx: int) -> None:
    """Report trial progress for pruning.

    Args:
        trial: Optuna trial object.
        fold_rmses: List of fold RMSEs so far.
        fold_idx: Current fold index.

    Raises:
        optuna.TrialPruned: If trial should be pruned.
    """
    intermediate_score = float(np.mean(fold_rmses))
    trial.report(intermediate_score, fold_idx)
    if trial.should_prune():
        raise optuna.TrialPruned()


def _lightgbm_cv_objective(
    trial: optuna.Trial,
    X_np: np.ndarray,
    y_np: np.ndarray,
    tscv: TimeSeriesSplit,
    feature_names: list[str] | None = None,
) -> float:
    """Optuna objective with cross-validation across all folds (RMSE).

    Args:
        trial: Optuna trial object.
        X_np: Full features array.
        y_np: Full target array.
        tscv: TimeSeriesSplit object.
        feature_names: Optional list of feature names to pass to LightGBM.

    Returns:
        Mean RMSE across all CV folds.
    """
    params = _suggest_hyperparameters(trial)
    fold_rmses: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_np)):
        X_train = X_np[train_idx]
        y_train = y_np[train_idx]
        X_val = X_np[val_idx]
        y_val = y_np[val_idx]

        fold_rmse = _evaluate_single_cv_fold(params, X_train, y_train, X_val, y_val, feature_names)
        fold_rmses.append(fold_rmse)
        _report_trial_progress(trial, fold_rmses, fold_idx)

    return float(np.mean(fold_rmses))
