"""Target shuffle test for data leakage detection.

This test trains a model with shuffled target to detect potential data leakage.
If the model achieves high R² with shuffled target, it indicates leakage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import lightgbm as lgb
from src.constants import (
    DEFAULT_RANDOM_STATE,
    LIGHTGBM_DATASET_COMPLETE_FILE,
    LIGHTGBM_EVAL_RESULTS_FILE,
    LIGHTGBM_OPTIMIZATION_RESULTS_FILE,
)
from src.utils import get_logger
from src.utils.transforms import extract_features_and_target

logger = get_logger(__name__)


def _load_optimization_params(model_name: str = "lightgbm_dataset_complete") -> dict[str, Any]:
    """Load best hyperparameters from optimization results.

    Args:
        model_name: Name of the model configuration.

    Returns:
        Dictionary of best hyperparameters.

    Raises:
        FileNotFoundError: If optimization results file doesn't exist.
        KeyError: If model_name not found in results.
    """
    if not LIGHTGBM_OPTIMIZATION_RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Optimization results not found: {LIGHTGBM_OPTIMIZATION_RESULTS_FILE}"
        )

    with open(LIGHTGBM_OPTIMIZATION_RESULTS_FILE) as f:
        results = json.load(f)

    if model_name not in results:
        raise KeyError(f"Model '{model_name}' not found in optimization results")

    best_params = results[model_name]["best_params"]
    logger.info(f"Loaded hyperparameters for {model_name}")
    return best_params


def _load_dataset(
    dataset_path: Path = LIGHTGBM_DATASET_COMPLETE_FILE,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load and split dataset into train and test.

    Args:
        dataset_path: Path to dataset file.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test).
    """
    if dataset_path.suffix == ".parquet":
        df = pd.read_parquet(dataset_path)
    elif dataset_path.suffix == ".csv":
        df = pd.read_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {dataset_path.suffix}")

    # Split by 'split' column
    df_train = cast(pd.DataFrame, df[df["split"] == "train"].copy())
    df_test = cast(pd.DataFrame, df[df["split"] == "test"].copy())

    # Remove metadata columns before extracting features
    from src.utils.transforms import remove_metadata_columns

    df_train_clean = remove_metadata_columns(df_train)
    df_test_clean = remove_metadata_columns(df_test)

    # Extract features and target
    X_train, y_train = extract_features_and_target(df_train_clean)
    X_test, y_test = extract_features_and_target(df_test_clean)

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    logger.info(f"Features: {len(X_train.columns)}")

    return X_train, y_train, X_test, y_test


def _shuffle_target(
    y_train: pd.Series,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.Series:
    """Shuffle target while preserving the same index.

    Args:
        y_train: Training target.
        random_state: Random seed for reproducibility.

    Returns:
        Shuffled target with original index.
    """
    y_shuffled = y_train.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    # Preserve original index for alignment
    y_shuffled.index = y_train.index
    logger.info(f"Shuffled {len(y_shuffled)} target values (random_state={random_state})")
    return y_shuffled


def _train_model_with_shuffled_target(
    X_train: pd.DataFrame,
    y_train_shuffled: pd.Series,
    best_params: dict[str, Any],
    random_state: int = DEFAULT_RANDOM_STATE,
) -> lgb.LGBMRegressor:
    """Train LightGBM model with shuffled target.

    Args:
        X_train: Training features.
        y_train_shuffled: Shuffled training target.
        best_params: Best hyperparameters from optimization.
        random_state: Random seed for reproducibility.

    Returns:
        Trained LightGBM model.
    """
    logger.info("Training LightGBM with shuffled target...")

    # Reset index for alignment
    X_train_reset = X_train.reset_index(drop=True)
    y_train_reset = y_train_shuffled.reset_index(drop=True)

    model = lgb.LGBMRegressor(
        **best_params,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(X_train_reset, y_train_reset)
    logger.info("Training completed")

    return model


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics.

    Args:
        y_true: True target values.
        y_pred: Predicted values.

    Returns:
        Dictionary with MAE, MSE, RMSE, R² metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def _load_original_metrics() -> dict[str, float]:
    """Load original model metrics for comparison.

    Returns:
        Dictionary with original test metrics.

    Raises:
        FileNotFoundError: If evaluation results file doesn't exist.
        KeyError: If lightgbm_complete not found in results.
    """
    if not LIGHTGBM_EVAL_RESULTS_FILE.exists():
        raise FileNotFoundError(f"Evaluation results not found: {LIGHTGBM_EVAL_RESULTS_FILE}")

    with open(LIGHTGBM_EVAL_RESULTS_FILE) as f:
        results = json.load(f)

    if "lightgbm_complete" not in results:
        raise KeyError("lightgbm_complete not found in evaluation results")

    original_metrics = results["lightgbm_complete"]["test_metrics"]
    logger.info("Loaded original model metrics for comparison")

    return original_metrics


def run_target_shuffle_test(
    dataset_path: Path = LIGHTGBM_DATASET_COMPLETE_FILE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Run target shuffle test to detect data leakage.

    This is the most powerful leakage test:
    - Keep the same features X
    - Shuffle the target y_train randomly
    - Re-train the model with same hyperparameters
    - Evaluate on original y_test

    Expected result if NO leakage:
    - R² ≈ 0 or negative
    - MAE/MSE/RMSE much worse than original model

    Red flag if leakage:
    - R² >> 0.1 with shuffled target
    - Metrics close to original model

    Args:
        dataset_path: Path to dataset file (default: complete dataset).
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with test results including:
        - shuffled_metrics: Metrics with shuffled target
        - original_metrics: Original model metrics
        - delta_metrics: Difference between original and shuffled
        - leakage_detected: Boolean flag
        - interpretation: Text interpretation
    """
    logger.info("=" * 70)
    logger.info("DATA LEAKAGE TEST: Target Shuffle")
    logger.info("=" * 70)

    # Load hyperparameters
    best_params = _load_optimization_params("lightgbm_dataset_complete")

    # Load dataset
    X_train, y_train, X_test, y_test = _load_dataset(dataset_path)

    # Shuffle target
    y_train_shuffled = _shuffle_target(y_train, random_state=random_state)

    # Train model with shuffled target
    model_shuffled = _train_model_with_shuffled_target(
        X_train,
        y_train_shuffled,
        best_params,
        random_state=random_state,
    )

    # Predict on test set (with ORIGINAL y_test)
    y_pred_shuffled = np.asarray(model_shuffled.predict(X_test))

    # Compute metrics with shuffled model
    shuffled_metrics = _compute_metrics(y_test, y_pred_shuffled)

    logger.info("Shuffled Model Metrics:")
    logger.info(f"  MAE:  {shuffled_metrics['mae']:.6f}")
    logger.info(f"  MSE:  {shuffled_metrics['mse']:.6f}")
    logger.info(f"  RMSE: {shuffled_metrics['rmse']:.6f}")
    logger.info(f"  R²:   {shuffled_metrics['r2']:.6f}")

    # Load original model metrics
    original_metrics = _load_original_metrics()

    logger.info("\nOriginal Model Metrics (for comparison):")
    logger.info(f"  MAE:  {original_metrics['mae']:.6f}")
    logger.info(f"  MSE:  {original_metrics['mse']:.6f}")
    logger.info(f"  RMSE: {original_metrics['rmse']:.6f}")
    logger.info(f"  R²:   {original_metrics['r2']:.6f}")

    # Compute delta metrics
    delta_metrics = {
        "delta_mae": shuffled_metrics["mae"] - original_metrics["mae"],
        "delta_mse": shuffled_metrics["mse"] - original_metrics["mse"],
        "delta_rmse": shuffled_metrics["rmse"] - original_metrics["rmse"],
        "delta_r2": shuffled_metrics["r2"] - original_metrics["r2"],
    }

    # Detect leakage
    r2_threshold = 0.1
    leakage_detected = shuffled_metrics["r2"] > r2_threshold

    logger.info("\n" + "=" * 70)
    logger.info("LEAKAGE DETECTION RESULT")
    logger.info("=" * 70)

    if leakage_detected:
        logger.warning(
            f"⚠️  POTENTIAL DATA LEAKAGE DETECTED! R² = {shuffled_metrics['r2']:.4f} > "
            f"{r2_threshold}"
        )
        logger.warning("The model achieves significant predictive power even with shuffled target.")
        logger.warning("This suggests that target information may be leaking into features.")
        interpretation = f"LEAKAGE DETECTED: R²={shuffled_metrics['r2']:.4f} > {r2_threshold}"
    else:
        logger.info(f"✓ No leakage detected. R² = {shuffled_metrics['r2']:.4f} ≤ {r2_threshold}")
        logger.info("The model performs poorly with shuffled target, as expected.")
        logger.info("Features do not contain leaked target information.")
        interpretation = f"NO LEAKAGE: R²={shuffled_metrics['r2']:.4f} ≤ {r2_threshold}"

    return {
        "test_name": "target_shuffle",
        "shuffled_metrics": shuffled_metrics,
        "original_metrics": original_metrics,
        "delta_metrics": delta_metrics,
        "leakage_detected": leakage_detected,
        "r2_threshold": r2_threshold,
        "interpretation": interpretation,
        "random_state": random_state,
        "test_size": len(X_test),
        "train_size": len(X_train),
        "n_features": len(X_train.columns),
    }
