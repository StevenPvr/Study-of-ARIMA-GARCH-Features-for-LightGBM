"""Random Forest training module using optimized hyperparameters."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.constants import DATA_DIR, DEFAULT_RANDOM_STATE, RESULTS_DIR
from src.utils import get_logger

logger = get_logger(__name__)

# Constants for training
RF_DATASET_COMPLETE = DATA_DIR / "rf_dataset_complete.csv"
RF_DATASET_WITHOUT_INSIGHTS = DATA_DIR / "rf_dataset_without_insights.csv"
RF_RESULTS_DIR = RESULTS_DIR / "random_forest"
RF_MODELS_DIR = RF_RESULTS_DIR / "models"
RF_OPTIMIZATION_RESULTS_FILE = RF_RESULTS_DIR / "optimization_results.json"
RF_TRAINING_RESULTS_FILE = RF_RESULTS_DIR / "training_results.json"


def _read_dataset_file(dataset_path: Path) -> pd.DataFrame:
    """Read dataset CSV file.

    Args:
        dataset_path: Path to the dataset CSV file.

    Returns:
        DataFrame with loaded data.

    Raises:
        FileNotFoundError: If dataset file does not exist.
        ValueError: If dataset is empty.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    logger.info(f"Loading dataset from {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Dataset is empty: {dataset_path}") from e

    if df.empty:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    return df


def _filter_by_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Filter dataset by split column if present.

    Args:
        df: Input DataFrame.
        split: Split to filter ('train' or 'test').

    Returns:
        Filtered DataFrame.

    Raises:
        ValueError: If no data found for split.
    """
    if "split" not in df.columns:
        return df

    df_filtered = cast(pd.DataFrame, df[df["split"] == split].copy())
    logger.info(f"Filtered to {split} split: {len(df_filtered)} rows")
    if df_filtered.empty:
        raise ValueError(f"No data found for split '{split}'")

    return df_filtered


def _remove_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove date and split columns from DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with metadata columns removed.
    """
    columns_to_drop = [col for col in ["date", "split"] if col in df.columns]
    if columns_to_drop:
        return df.drop(columns=columns_to_drop)
    return df


def _extract_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract features and target from dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (features DataFrame, target Series).

    Raises:
        ValueError: If target column is missing.
    """
    if "weighted_log_return" not in df.columns:
        raise ValueError("Dataset must contain 'weighted_log_return' column")

    X = cast(pd.DataFrame, df.drop(columns=["weighted_log_return"]))
    y = cast(pd.Series, df["weighted_log_return"].copy())
    return X, y


def load_dataset(dataset_path: Path, split: str = "train") -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset and split into features and target.

    Args:
        dataset_path: Path to the dataset CSV file.
        split: Split to load ('train' or 'test'). Default is 'train'.

    Returns:
        Tuple of (features DataFrame, target Series).

    Raises:
        FileNotFoundError: If dataset file does not exist.
        ValueError: If dataset is empty or missing required columns.
    """
    df = _read_dataset_file(dataset_path)
    df = _filter_by_split(df, split)
    df = _remove_metadata_columns(df)
    X, y = _extract_features_and_target(df)

    logger.info(f"Loaded dataset: {X.shape[0]} rows, {X.shape[1]} features")
    return X, y


def load_optimization_results(
    results_path: Path = RF_OPTIMIZATION_RESULTS_FILE,
) -> dict[str, Any]:
    """Load optimization results from JSON file.

    Args:
        results_path: Path to optimization results JSON file.

    Returns:
        Dictionary with optimization results for both datasets.

    Raises:
        FileNotFoundError: If results file does not exist.
        ValueError: If results file is invalid or missing required keys.
    """
    if not results_path.exists():
        raise FileNotFoundError(
            f"Optimization results not found: {results_path}. " "Run optimization first."
        )

    logger.info(f"Loading optimization results from {results_path}")
    with open(results_path, "r") as f:
        results = json.load(f)

    # Validate required keys
    required_keys = ["rf_dataset_complete", "rf_dataset_without_insights"]
    for key in required_keys:
        if key not in results:
            raise ValueError(f"Missing required key in optimization results: {key}")
        if "best_params" not in results[key]:
            raise ValueError(f"Missing 'best_params' in {key}")

    logger.info("Optimization results loaded successfully")
    return results


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any],
) -> tuple[RandomForestRegressor, dict[str, Any]]:
    """Train Random Forest model on training set only.

    Args:
        X_train: Training features.
        y_train: Training target.
        params: Random Forest hyperparameters.

    Returns:
        Tuple of (trained model, training info dictionary).

    Raises:
        ValueError: If training data is empty or invalid.
    """
    if X_train.empty or y_train.empty:
        raise ValueError("Training data cannot be empty")

    logger.info("Training Random Forest model")
    logger.info(f"Train size: {len(X_train)}")
    logger.info(f"Hyperparameters: {params}")

    # Create and train model
    model = RandomForestRegressor(
        **params,
        random_state=DEFAULT_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Calculate log loss on train set (for consistency with optimization)
    y_train_pred = model.predict(X_train)
    squared_errors = (y_train - y_train_pred) ** 2
    train_log_loss = float(np.mean(np.log(1.0 + squared_errors)))

    info = {
        "train_log_loss": train_log_loss,
        "train_size": int(len(X_train)),
        "n_features": int(X_train.shape[1]),
    }

    logger.info(f"Training complete - Train log loss: {train_log_loss:.6f}")

    return model, info


def save_model(
    model: RandomForestRegressor,
    info: dict[str, Any],
    params: dict[str, Any],
    model_name: str,
    output_dir: Path = RF_MODELS_DIR,
) -> tuple[Path, Path]:
    """Save trained model and metadata.

    Args:
        model: Trained Random Forest model.
        info: Training information (log loss, size, features).
        params: Model hyperparameters.
        model_name: Name for the model (e.g., 'rf_complete').
        output_dir: Directory to save model files.

    Returns:
        Tuple of (model_path, metadata_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model as joblib
    model_path = output_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)

    # Save metadata as JSON
    metadata = {
        "model_name": model_name,
        "params": params,
        "train_info": info,
        "random_state": DEFAULT_RANDOM_STATE,
    }
    metadata_path = output_dir / f"{model_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved model: {model_path}")
    logger.info(f"Saved metadata: {metadata_path}")

    return model_path, metadata_path


def _run_single_training(
    dataset_path: Path,
    model_name: str,
    best_params: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Train a single Random Forest model (helper for parallelization).

    Args:
        dataset_path: Path to the dataset CSV file.
        model_name: Name for the model.
        best_params: Optimized hyperparameters.

    Returns:
        Tuple of (model_name, results_dict).
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"TRAINING: {model_name}")
    logger.info(f"{'=' * 70}")

    # Load train data only
    X_train, y_train = load_dataset(dataset_path, split="train")

    # Train model
    model, info = train_random_forest(X_train, y_train, best_params)

    # Save model
    model_path, metadata_path = save_model(model, info, best_params, model_name)

    results = {
        "model_name": model_name,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "train_info": info,
        "params": best_params,
    }

    logger.info(f"✓ Completed training: {model_name}")
    return model_name, results


def _prepare_training_tasks(
    opt_results: dict[str, Any],
) -> list[tuple[Path, str, dict[str, Any]]]:
    """Prepare training tasks from optimization results.

    Args:
        opt_results: Optimization results dictionary.

    Returns:
        List of (dataset_path, model_name, params) tuples.
    """
    return [
        (
            RF_DATASET_COMPLETE,
            "rf_complete",
            opt_results["rf_dataset_complete"]["best_params"],
        ),
        (
            RF_DATASET_WITHOUT_INSIGHTS,
            "rf_without_insights",
            opt_results["rf_dataset_without_insights"]["best_params"],
        ),
    ]


def _run_parallel_training(
    tasks: list[tuple[Path, str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Run training tasks in parallel.

    Args:
        tasks: List of (dataset_path, model_name, params) tuples.

    Returns:
        Dictionary mapping model names to results.

    Raises:
        Exception: If any training task fails.
    """
    results_dict = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_name = {
            executor.submit(_run_single_training, path, name, params): name
            for path, name, params in tasks
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                model_name, results = future.result()
                results_dict[model_name] = results
                logger.info(f"✓ Training completed: {model_name}")
            except Exception as e:
                logger.error(f"✗ Training failed for {name}: {e}")
                raise

    return results_dict


def _save_training_results(
    results_dict: dict[str, dict[str, Any]],
) -> None:
    """Save training results to JSON file.

    Args:
        results_dict: Dictionary mapping model names to results.
    """
    training_results = {
        "rf_complete": results_dict["rf_complete"],
        "rf_without_insights": results_dict["rf_without_insights"],
    }

    RF_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RF_TRAINING_RESULTS_FILE, "w") as f:
        json.dump(training_results, f, indent=2)

    logger.info(f"Training results saved to {RF_TRAINING_RESULTS_FILE}")


def _log_training_summary(results_dict: dict[str, dict[str, Any]]) -> None:
    """Log training summary to console.

    Args:
        results_dict: Dictionary mapping model names to results.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)

    for name, res in results_dict.items():
        info = res["train_info"]
        logger.info(f"\n{name}:")
        logger.info(f"  Train log loss: {info['train_log_loss']:.6f}")
        logger.info(f"  Train size: {info['train_size']}")
        logger.info(f"  N features: {info['n_features']}")
        logger.info(f"  Model saved: {res['model_path']}")

    logger.info("\n" + "=" * 70)


def run_training(
    optimization_results_path: Path = RF_OPTIMIZATION_RESULTS_FILE,
) -> dict[str, dict[str, Any]]:
    """Train Random Forest models for both datasets in parallel on train split only.

    Args:
        optimization_results_path: Path to optimization results JSON.

    Returns:
        Dictionary with training results for both models.

    Raises:
        FileNotFoundError: If optimization results or datasets are missing.
        ValueError: If optimization results are invalid.
    """
    logger.info("=" * 70)
    logger.info("Random Forest Training (Parallel) - Train Split Only")
    logger.info("=" * 70)

    opt_results = load_optimization_results(optimization_results_path)
    tasks = _prepare_training_tasks(opt_results)
    results_dict = _run_parallel_training(tasks)
    _save_training_results(results_dict)
    _log_training_summary(results_dict)

    return {
        "rf_complete": results_dict["rf_complete"],
        "rf_without_insights": results_dict["rf_without_insights"],
    }
