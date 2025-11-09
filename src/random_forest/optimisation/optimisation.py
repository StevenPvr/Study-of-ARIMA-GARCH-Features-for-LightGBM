"""Random Forest hyperparameter optimization using Optuna with walk-forward cross-validation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import json
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from src.constants import (
    DEFAULT_RANDOM_STATE,
    RF_DATASET_COMPLETE_FILE,
    RF_DATASET_WITHOUT_INSIGHTS_FILE,
    RF_DATASET_SIGMA2_ONLY_FILE,
    RF_DATASET_RSI14_ONLY_FILE,
    RF_DATASET_TECHNICAL_INDICATORS_FILE,
    RF_OPTIMIZATION_N_SPLITS,
    RF_OPTIMIZATION_N_TRIALS,
    RF_OPTIMIZATION_RESULTS_FILE,
)
from src.utils import get_logger
from src.random_forest.data_preparation.utils import (
    ensure_sigma2_only_dataset,
    ensure_rsi14_only_dataset,
    ensure_technical_indicators_dataset,
)

logger = get_logger(__name__)


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


def _filter_train_split(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataset to train split if split column exists.

    Args:
        df: Input DataFrame.

    Returns:
        Filtered DataFrame.

    Raises:
        ValueError: If no train data found.
    """
    if "split" not in df.columns:
        return df

    df_filtered = cast(pd.DataFrame, df[df["split"] == "train"].copy())
    logger.info(f"Filtered to train split: {len(df_filtered)} rows")

    if df_filtered.empty:
        raise ValueError("No data found for split 'train'")

    return df_filtered


def _remove_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove date and split columns (not used as features).

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


def load_dataset(dataset_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset and split into features and target.

    Args:
        dataset_path: Path to the dataset CSV file.

    Returns:
        Tuple of (features DataFrame, target Series).

    Raises:
        FileNotFoundError: If dataset file does not exist.
        ValueError: If dataset is empty or missing required columns.
    """
    df = _read_dataset_file(dataset_path)
    df = _filter_train_split(df)
    df = _remove_metadata_columns(df)
    X, y = _extract_features_and_target(df)

    logger.info(f"Loaded dataset: {X.shape[0]} rows, {X.shape[1]} features")
    return X, y


def walk_forward_cv_score(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any],
) -> float:
    """Evaluate model using walk-forward cross-validation with pruning.

    Args:
        trial: Optuna trial object for pruning.
        X: Features DataFrame.
        y: Target Series.
        params: Random Forest hyperparameters.

    Returns:
        Log loss (loss to minimize).
    """
    tscv = TimeSeriesSplit(n_splits=RF_OPTIMIZATION_N_SPLITS)
    loss_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Create and train model
        model = RandomForestRegressor(
            **params,
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Predict on validation set
        y_pred = model.predict(X_val)

        # Calculate log loss directly from errors (negative log-likelihood for regression)
        # Using log(1 + squared_error) to avoid log(0) and penalize errors logarithmically
        squared_errors = (y_val - y_pred) ** 2
        log_loss = np.mean(np.log(1.0 + squared_errors))
        loss_scores.append(log_loss)

        # Report intermediate value for pruning
        trial.report(log_loss, fold_idx)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(loss_scores))


def objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
) -> float:
    """Optuna objective function to minimize log loss.

    Args:
        trial: Optuna trial object.
        X: Features DataFrame.
        y: Target Series.

    Returns:
        Log loss (loss to minimize).
    """
    # Suggest hyperparameters (balanced for ~3000 rows with ~11 features)
    # Based on academic recommendations for small datasets (2024)
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True]),
    }

    # Evaluate with walk-forward CV (with pruning)
    loss_mean = walk_forward_cv_score(trial, X, y, params)

    # Return loss to minimize
    return loss_mean


def _create_optuna_study(study_name: str) -> optuna.Study:
    """Create Optuna study with pruning configuration.

    Args:
        study_name: Name for the Optuna study.

    Returns:
        Optuna study object.
    """
    return optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=DEFAULT_RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2),
    )


def _collect_best_results(study: optuna.Study, study_name: str) -> dict[str, Any]:
    """Collect best trial results from Optuna study.

    Args:
        study: Optuna study object.
        study_name: Name of the study.

    Returns:
        Dictionary with best parameters and metrics.
    """
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value if best_trial.value is not None else float("inf")

    return {
        "best_params": best_params,
        "best_loss": float(best_value),
        "n_trials": len(study.trials),
        "study_name": study_name,
    }


def optimize_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    study_name: str,
    n_trials: int = RF_OPTIMIZATION_N_TRIALS,
) -> tuple[dict[str, Any], optuna.Study]:
    """Optimize Random Forest hyperparameters using Optuna.

    Args:
        X: Features DataFrame.
        y: Target Series.
        study_name: Name for the Optuna study.
        n_trials: Number of optimization trials.

    Returns:
        Tuple of (best parameters dict, Optuna study object).
    """
    logger.info(f"Starting optimization: {study_name}")
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"Walk-forward CV splits: {RF_OPTIMIZATION_N_SPLITS}")

    study = _create_optuna_study(study_name)

    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_results = _collect_best_results(study, study_name)

    logger.info(f"Optimization complete: {study_name}")
    logger.info(f"Best log loss: {best_results['best_loss']:.6f}")
    logger.info(f"Best parameters: {best_results['best_params']}")

    return best_results, study


def save_optimization_results(
    results_complete: dict[str, Any],
    results_without_insights: dict[str, Any],
    output_path: Path = RF_OPTIMIZATION_RESULTS_FILE,
    *,
    results_sigma2_only: dict[str, Any] | None = None,
    results_rsi14_only: dict[str, Any] | None = None,
    results_technical: dict[str, Any] | None = None,
) -> None:
    """Save optimization results to JSON file.

    Args:
        results_complete: Results for complete dataset.
        results_without_insights: Results for dataset without insights.
        output_path: Path to save results JSON file.
        results_sigma2_only: Optional results for sigma2-only dataset.
        results_rsi14_only: Optional results for rsi14-only dataset.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "rf_dataset_complete": results_complete,
        "rf_dataset_without_insights": results_without_insights,
    }
    if results_sigma2_only is not None:
        results["rf_dataset_sigma2_only"] = results_sigma2_only
    if results_rsi14_only is not None:
        results["rf_dataset_rsi14_only"] = results_rsi14_only
    if results_technical is not None:
        results["rf_dataset_technical_indicators"] = results_technical

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Optimization results saved to {output_path}")


def _run_single_optimization(
    dataset_path: Path,
    study_name: str,
    n_trials: int,
) -> tuple[str, dict[str, Any], optuna.Study]:
    """Run optimization for a single dataset (helper for parallelization).

    Args:
        dataset_path: Path to the dataset CSV file.
        study_name: Name for the Optuna study.
        n_trials: Number of Optuna trials.

    Returns:
        Tuple of (dataset_name, results_dict, study_object).
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"OPTIMIZATION: {dataset_path.name}")
    logger.info(f"{'=' * 70}")
    X, y = load_dataset(dataset_path)
    results, study = optimize_random_forest(
        X,
        y,
        study_name=study_name,
        n_trials=n_trials,
    )
    return study_name, results, study


def _run_parallel_optimizations(
    tasks: list[tuple[Path, str, int]],
) -> dict[str, dict[str, Any]]:
    """Run multiple optimizations in parallel.

    Args:
        tasks: List of (dataset_path, study_name, n_trials) tuples.

    Returns:
        Dictionary mapping study names to results.

    Raises:
        Exception: If any optimization fails.
    """
    results_dict = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_name = {
            executor.submit(_run_single_optimization, path, name, trials): name
            for path, name, trials in tasks
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                study_name, results, _ = future.result()
                results_dict[study_name] = results
                logger.info(f"✓ Completed optimization: {study_name}")
            except Exception as e:
                logger.error(f"✗ Optimization failed for {name}: {e}")
                raise

    return results_dict


def _log_optimization_summary(
    results_complete: dict[str, Any],
    results_without: dict[str, Any],
    results_sigma2_only: dict[str, Any] | None = None,
    results_rsi14_only: dict[str, Any] | None = None,
    results_technical: dict[str, Any] | None = None,
) -> None:
    """Log optimization summary results.

    Args:
        results_complete: Results for complete dataset.
        results_without: Results for dataset without insights.
        results_sigma2_only: Optional results for sigma2-only dataset.
        results_rsi14_only: Optional results for rsi14-only dataset.
    """
    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("=" * 70)
    logger.info("\nComplete Dataset:")
    logger.info(f"  Best log loss: {results_complete['best_loss']:.6f}")
    logger.info(f"  Best params: {results_complete['best_params']}")

    logger.info("\nWithout Insights Dataset:")
    logger.info(f"  Best log loss: {results_without['best_loss']:.6f}")
    logger.info(f"  Best params: {results_without['best_params']}")

    if results_sigma2_only is not None:
        logger.info("\nSigma2-Only Dataset:")
        logger.info(f"  Best log loss: {results_sigma2_only['best_loss']:.6f}")
        logger.info(f"  Best params: {results_sigma2_only['best_params']}")

    if results_rsi14_only is not None:
        logger.info("\nRSI14-Only Dataset:")
        logger.info(f"  Best log loss: {results_rsi14_only['best_loss']:.6f}")
        logger.info(f"  Best params: {results_rsi14_only['best_params']}")

    if results_technical is not None:
        logger.info("\nTechnical Indicators Dataset:")
        logger.info(f"  Best log loss: {results_technical['best_loss']:.6f}")
        logger.info(f"  Best params: {results_technical['best_params']}")

    logger.info("\n" + "=" * 70)


def run_optimization(
    n_trials: int = RF_OPTIMIZATION_N_TRIALS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run hyperparameter optimization for all datasets in parallel.

    Args:
        n_trials: Number of Optuna trials per dataset.

    Returns:
        Tuple of (results_complete, results_without_insights).
    """
    logger.info("=" * 70)
    logger.info("Random Forest Hyperparameter Optimization (Parallel)")
    logger.info("=" * 70)
    logger.info(f"Running {n_trials} trials per dataset in parallel")

    # Ensure sigma2-only and rsi14-only datasets exist (best possible: include lags)
    ensure_sigma2_only_dataset(include_lags=True)
    ensure_rsi14_only_dataset(include_lags=True)
    ensure_technical_indicators_dataset(include_lags=True)

    tasks = [
        (RF_DATASET_COMPLETE_FILE, "rf_complete", n_trials),
        (RF_DATASET_WITHOUT_INSIGHTS_FILE, "rf_without_insights", n_trials),
        (RF_DATASET_SIGMA2_ONLY_FILE, "rf_sigma2_only", n_trials),
        (RF_DATASET_RSI14_ONLY_FILE, "rf_rsi14_only", n_trials),
        (RF_DATASET_TECHNICAL_INDICATORS_FILE, "rf_technical_indicators", n_trials),
    ]

    results_dict = _run_parallel_optimizations(tasks)

    results_complete = results_dict["rf_complete"]
    results_without = results_dict["rf_without_insights"]
    results_sigma2 = results_dict.get("rf_sigma2_only")
    results_rsi14 = results_dict.get("rf_rsi14_only")
    results_technical = results_dict.get("rf_technical_indicators")

    save_optimization_results(
        results_complete,
        results_without,
        results_sigma2_only=results_sigma2,
        results_rsi14_only=results_rsi14,
        results_technical=results_technical,
    )
    _log_optimization_summary(
        results_complete,
        results_without,
        results_sigma2,
        results_rsi14,
        results_technical,
    )

    return results_complete, results_without
