"""Model training utilities for LightGBM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import lightgbm as lgb
from src.constants import DEFAULT_RANDOM_STATE, LIGHTGBM_MODELS_DIR
from src.utils import ensure_output_dir, get_logger, save_json_pretty

logger = get_logger(__name__)


def _ensure_lgbm_params(params: dict[str, Any]) -> dict[str, Any]:
    """Ensure required LightGBM parameters are set.

    Args:
        params: Input parameters dictionary.

    Returns:
        Parameters dictionary with required fields set.
    """
    lgbm_params = params.copy()
    defaults = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "random_state": DEFAULT_RANDOM_STATE,
        "verbosity": -1,
        "force_col_wise": True,
    }

    for key, default_value in defaults.items():
        if key not in lgbm_params:
            lgbm_params[key] = default_value

    return lgbm_params


def _compute_train_rmse(
    model: lgb.LGBMRegressor, X_train: pd.DataFrame, y_train: pd.Series
) -> float:
    """Compute RMSE on training set.

    Args:
        model: Trained LightGBM model.
        X_train: Training features.
        y_train: Training target.

    Returns:
        RMSE value.
    """
    y_train_pred = model.predict(X_train)
    y_train_np = np.asarray(y_train.to_numpy(), dtype=float)
    return float(np.sqrt(np.mean((y_train_np - y_train_pred) ** 2)))


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any],
) -> tuple[lgb.LGBMRegressor, dict[str, Any]]:
    """Train LightGBM model on training set only.

    Args:
        X_train: Training features.
        y_train: Training target.
        params: LightGBM hyperparameters.

    Returns:
        Tuple of (trained model, training info dictionary).

    Raises:
        ValueError: If training data is empty or invalid.
    """
    if X_train.empty or y_train.empty:
        raise ValueError("Training data cannot be empty")

    logger.info("Training LightGBM model")
    logger.info(f"Train size: {len(X_train)}")
    logger.info(f"Hyperparameters: {params}")

    # Ensure required LightGBM parameters
    lgbm_params = _ensure_lgbm_params(params)

    # Create and train model
    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(X_train, y_train)

    # Calculate RMSE on train set
    train_rmse = _compute_train_rmse(model, X_train, y_train)

    info = {
        "train_rmse": train_rmse,
        "train_size": int(len(X_train)),
        "n_features": int(X_train.shape[1]),
    }

    logger.info(f"Training complete - Train RMSE: {train_rmse:.6f} (log-volatility)")

    return model, info


def save_model(
    model: lgb.LGBMRegressor,
    info: dict[str, Any],
    params: dict[str, Any],
    model_name: str,
    output_dir: Path = LIGHTGBM_MODELS_DIR,
) -> tuple[Path, Path]:
    """Save trained model and metadata.

    Args:
        model: Trained LightGBM model.
        info: Training information (RMSE, size, features).
        params: Model hyperparameters.
        model_name: Name for the model (e.g., 'lightgbm_complete').
        output_dir: Directory to save model files.

    Returns:
        Tuple of (model_path, metadata_path).
    """
    # Save model as joblib
    model_path = output_dir / f"{model_name}.joblib"
    ensure_output_dir(model_path)
    joblib.dump(model, model_path)

    # Save metadata as JSON
    metadata = {
        "model_name": model_name,
        "params": params,
        "train_info": info,
        "random_state": DEFAULT_RANDOM_STATE,
    }
    metadata_path = output_dir / f"{model_name}_metadata.json"
    save_json_pretty(metadata, metadata_path)

    logger.info(f"Saved model: {model_path}")
    logger.info(f"Saved metadata: {metadata_path}")

    return model_path, metadata_path


def _run_single_training(
    dataset_path: Path,
    model_name: str,
    best_params: dict[str, Any],
    models_dir: Path = LIGHTGBM_MODELS_DIR,
) -> tuple[str, dict[str, Any]]:
    """Train a single LightGBM model (helper for parallelization).

    Args:
        dataset_path: Path to the dataset CSV file.
        model_name: Name for the model.
        best_params: Optimized hyperparameters.
        models_dir: Directory to save models.

    Returns:
        Tuple of (model_name, results_dict).
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"TRAINING: {model_name}")
    logger.info(f"{'=' * 70}")

    # Import here to avoid circular dependency
    from src.lightgbm.training.data_loading import load_dataset

    # Load train data only
    X_train, y_train = load_dataset(dataset_path, split="train")

    # Train model
    model, info = train_lightgbm(X_train, y_train, best_params)

    # Save model
    model_path, metadata_path = save_model(model, info, best_params, model_name, models_dir)

    results = {
        "model_name": model_name,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "train_info": info,
        "params": best_params,
    }

    logger.info(f"✓ Completed training: {model_name}")
    return model_name, results
