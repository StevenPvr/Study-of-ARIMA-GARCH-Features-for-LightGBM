"""End-to-end ARIMA optimization (Optuna/grid), diagnostics and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd

from src.constants import (
    ARIMA_OPTIMIZATION_N_SPLITS,
    ARIMA_OPTIMIZATION_N_TRIALS,
    ARIMA_OPTIMIZATION_TEST_SIZE,
    ARIMA_OPTIMIZATION_TOP_K_RESULTS,
    DEFAULT_RANDOM_STATE,
)
from src.path import ARIMA_BEST_MODELS_FILE, ARIMA_OPTIMIZATION_RESULTS_FILE
from src.utils import get_logger, get_parquet_path, load_parquet_file, validate_required_columns

from .objective import objective_aic
from .results_processing import pick_best, save_results, to_dataframe
from .validation import compute_test_size_from_ratio, validate_backtest_config, validate_series

logger = get_logger(__name__)


def _extract_and_sort_train_data(
    df: pd.DataFrame, value_col: str, date_col: Optional[str]
) -> pd.Series:
    """Extract training data from DataFrame and optionally sort by date.

    Args:
        df: DataFrame with 'split' column.
        value_col: Name of the numeric column to extract.
        date_col: Optional name of date column for sorting.

    Returns:
        Training series sorted by date if date_col provided.

    Raises:
        ValueError: If no train data found.
    """
    train_df = df[df["split"] == "train"].copy()
    if train_df.empty:
        msg = "No 'train' data found in split column"
        raise ValueError(msg)
    if date_col and date_col in train_df.columns:
        train_df[date_col] = pd.to_datetime(train_df[date_col], utc=False)
        date_col_str: str = date_col
        train_df = train_df.sort_values(by=[date_col_str])  # type: ignore[arg-type]
    y = pd.Series(train_df[value_col].astype(float)).reset_index(drop=True)
    validate_series("train_series", y)
    return y


def load_train_data(
    csv_path: Path,
    value_col: str,
    date_col: Optional[str] = None,
) -> pd.Series:
    """Load training data from Parquet file with split column.

    Loads a Parquet file and returns only the training split.
    The file must contain a 'split' column with 'train' and 'test' values.
    Only 'train' data is returned for optimization (no test data leakage).

    Args:
        csv_path: Path to data file (Parquet file will be loaded from this path
            with .parquet extension).
        value_col: Name of the numeric column to model.
        date_col: Optional name of date column for sorting (if None, no sorting).

    Returns:
        Training series (only 'train' split from the data).

    Raises:
        FileNotFoundError: If Parquet file does not exist.
        ValueError: If path exists but is not a file, required columns are missing,
            or no train data found.
    """
    parquet_path = get_parquet_path(csv_path)
    if not parquet_path.exists():
        msg = f"Parquet file not found: {parquet_path}"
        raise FileNotFoundError(msg)
    if not parquet_path.is_file():
        msg = f"Path is not a file: {parquet_path}"
        raise ValueError(msg)

    logger.info(f"Loading training dataset from {parquet_path}")
    df = load_parquet_file(parquet_path)
    if df is None or df.empty:
        msg = f"Failed to load data from {parquet_path}"
        raise ValueError(msg)
    validate_required_columns(df, [value_col, "split"], df_name="Data file")
    return _extract_and_sort_train_data(df, value_col, date_col)


def _prepare_optuna(seed: int = DEFAULT_RANDOM_STATE) -> optuna.Study:
    """Create and configure an Optuna study for minimization.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Configured Optuna study with TPE sampler.
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    return optuna.create_study(direction="minimize", sampler=sampler)


def _extract_results_from_trials(
    study: optuna.Study, top_k: int = ARIMA_OPTIMIZATION_TOP_K_RESULTS
) -> List[Dict]:
    """Extract complete evaluation results from top Optuna trials.

    Results are already stored in trial user_attrs during optimization,
    so no re-evaluation is needed.

    Args:
        study: Optuna study with completed trials.
        top_k: Number of top trials to extract (default: ARIMA_OPTIMIZATION_TOP_K_RESULTS).

    Returns:
        List of evaluation result dictionaries from top trials.
    """
    trials = sorted(
        study.trials, key=lambda tr: tr.value if tr.value is not None else float("inf")
    )[: min(top_k, len(study.trials))]
    results: List[Dict] = []
    for tr in trials:
        p = tr.params
        params_dict = {
            "p": int(p.get("p", 0)),
            "d": int(p.get("d", 0)),
            "q": int(p.get("q", 0)),
            "P": int(p.get("P", 0)),
            "D": int(p.get("D", 0)),
            "Q": int(p.get("Q", 0)),
            "s": int(p.get("s", 1)),
            "trend": str(p.get("trend", "n")),
            "refit_every": int(p["refit_every"]),
        }
        # Extract results from trial user_attrs (stored during optimization)
        if "error" in tr.user_attrs:
            results.append({"params": params_dict, "error": tr.user_attrs["error"]})
        else:
            result_dict: Dict[str, Any] = {
                "params": params_dict,
                "aic": tr.user_attrs.get("aic", float("inf")),
                "bic": tr.user_attrs.get("bic", float("inf")),
            }
            results.append(result_dict)
    return results


def _run_optuna(
    train: pd.Series,
    n_trials: int,
    criterion: str = "aic",
    seed: int = DEFAULT_RANDOM_STATE,
    backtest_cfg: Optional[Dict[str, int]] = None,
) -> List[Dict]:
    """Run Optuna optimization and extract results from trials.

    Performs hyperparameter search using Optuna with walk-forward CV.
    Complete evaluation results (AIC, BIC) are stored in trial user_attrs
    during optimization. Walk-forward CV is performed to test robustness
    but only AIC is used for optimization.

    Args:
        train: Training time series data.
        n_trials: Number of Optuna trials to run.
        criterion: Optimization criterion ("aic" or "bic", default: "aic").
        seed: Random seed for reproducibility.
        backtest_cfg: Backtest configuration for walk-forward CV.

    Returns:
        List of evaluation result dictionaries for top candidates.
    """
    study = _prepare_optuna(seed)
    # Only AIC is supported - optimal for forecasting applications
    if criterion.lower() != "aic":
        raise ValueError(
            f"Only 'aic' criterion is supported (got '{criterion}'). "
            "BIC was removed as AIC is theoretically optimal for forecasting. "
            "AIC minimizes one-step-ahead prediction error, while BIC minimizes description length."
        )
    obj = objective_aic
    logger.debug("Using AIC objective function for optimization")

    logger.info(f"Starting {criterion.upper()} optimization with {n_trials} trials...")

    try:
        study.optimize(
            lambda t: obj(t, train, backtest_cfg=backtest_cfg, stats_callback=None),
            n_trials=n_trials,
        )
        logger.info(
            f"{criterion.upper()} optimization completed - " f"Best value: {study.best_value:.4f}"
        )

    except Exception as e:
        logger.error(f"Optuna optimization failed for {criterion}: {e}")
        raise
    logger.info(f"Extracting results from top trials for {criterion} optimization")
    results = _extract_results_from_trials(study)
    if not results:
        logger.warning(f"No valid results found for {criterion} optimization")
        return []
    logger.info(f"Extracted {len(results)} top candidates for {criterion}")
    return results


def _validate_optimization_inputs(
    train_series: pd.Series,
    test_series: Optional[pd.Series],
    backtest_cfg: Optional[Dict[str, int]],
    search: str,
) -> None:
    """Validate inputs for ARIMA optimization.

    Args:
        train_series: Training time series data.
        test_series: Optional test series.
        backtest_cfg: Optional backtest configuration dict for walk-forward CV.
        search: Search method (must be "optuna").

    Raises:
        ValueError: If validation fails.
    """
    validate_series("train_series", train_series)
    if test_series is not None:
        validate_series("test_series", test_series)
    if search.lower() != "optuna":
        raise ValueError("Only 'optuna' is supported for parameter 'search'.")


def _run_aic_optimization(
    train_series: pd.Series,
    n_trials: int,
    random_state: int,
    backtest_cfg: Optional[Dict[str, int]],
) -> List[Dict]:
    """Run AIC optimization with walk-forward CV.

    AIC (Akaike Information Criterion) is used exclusively for ARIMA optimization
    because it is theoretically optimal for minimizing one-step-ahead prediction error,
    which is the primary objective in financial forecasting.

    Walk-forward CV is performed to test model robustness across different time periods,
    but only AIC is used for optimization (not validation metrics).

    BIC (Bayesian Information Criterion) was removed because:
    - BIC minimizes description length, not prediction error
    - BIC penalizes model complexity more heavily than AIC
    - For forecasting (vs model identification), AIC is asymptotically optimal

    Args:
        train_series: Training time series data.
        n_trials: Number of Optuna trials.
        random_state: Random seed for reproducibility.
        backtest_cfg: Backtest configuration for walk-forward CV.

    Returns:
        List of evaluation results from AIC optimization.
    """
    logger.info("Starting AIC optimization (%d trials)", n_trials)

    # Run AIC optimization
    logger.info("Running AIC optimization...")
    results_aic = _run_optuna(
        train_series,
        n_trials=n_trials,
        criterion="aic",
        seed=random_state,
        backtest_cfg=backtest_cfg,
    )
    logger.info("AIC optimization completed")

    return results_aic


def _select_and_log_best_model(df: pd.DataFrame) -> Dict:
    """Select best model and log results.

    Args:
        df: DataFrame containing optimization results.

    Returns:
        Best model dictionary.
    """
    # Select best model based on AIC (theoretically optimal for forecasting)
    best_aic, _ = pick_best(df)  # pick_best returns (best_aic, best_bic)
    # We only use best_aic; best_bic is None/ignored

    logger.info(
        "Best model: params=%s | AIC=%.6f | BIC=%.6f (BIC for reference only)",
        best_aic.get("params"),
        float(best_aic.get("aic", float("nan"))),
        float(best_aic.get("bic", float("nan"))),  # BIC computed but not optimized
    )
    return best_aic


def _save_optimization_results(
    df: pd.DataFrame,
    best_model: Dict,
    out_dir: Optional[Path],
) -> None:
    """Save optimization results to disk.

    Args:
        df: DataFrame containing optimization results.
        best_model: Best model dictionary.
        out_dir: Optional output directory for saving results.
            If None, no saving is performed.
    """
    if out_dir is None:
        return  # Don't save results when out_dir is explicitly None (for testing)

    save_results(
        df,
        best_model,
        None,  # best_bic set to None (no longer optimized)
        out_dir,
        best_models_file=ARIMA_BEST_MODELS_FILE,
        results_file=ARIMA_OPTIMIZATION_RESULTS_FILE,
    )


def _compute_backtest_configuration(
    backtest_n_splits: Optional[int],
    backtest_test_size: Optional[int],
    backtest_refit_every: Optional[int],
    train_series: pd.Series,
) -> Dict[str, int]:
    """Compute backtest configuration with defaults.

    Args:
        backtest_n_splits: Number of temporal backtest splits
            (default: ARIMA_OPTIMIZATION_N_SPLITS).
        backtest_test_size: Absolute size of each test window (default: computed from ratio).
        backtest_refit_every: Refit frequency (required).
        train_series: Training time series data for size calculations.

    Returns:
        Configured backtest parameters dictionary.
    """
    n_splits = backtest_n_splits if backtest_n_splits is not None else ARIMA_OPTIMIZATION_N_SPLITS
    if backtest_refit_every is None:
        raise ValueError("backtest_refit_every must be provided explicitly")
    refit_every = backtest_refit_every

    if backtest_test_size is None:
        test_size = compute_test_size_from_ratio(
            train_len=len(train_series),
            test_size_ratio=ARIMA_OPTIMIZATION_TEST_SIZE,
            n_splits=n_splits,
        )
    else:
        test_size = int(backtest_test_size)

    backtest_cfg = {
        "n_splits": n_splits,
        "test_size": test_size,
        "refit_every": refit_every,
    }

    # Validate configuration
    validate_backtest_config(n_splits, test_size, refit_every, len(train_series))

    return backtest_cfg


def optimize_arima_models(
    train_series: pd.Series,
    test_series: Optional[pd.Series] = None,
    *,
    search: str = "optuna",  # kept for backward compat; only "optuna" is supported
    n_trials: int = ARIMA_OPTIMIZATION_N_TRIALS,
    n_jobs: int | None = None,
    backtest_cfg: Optional[Dict[str, int]] = None,
    backtest_n_splits: Optional[int] = None,
    backtest_test_size: Optional[int] = None,
    backtest_refit_every: Optional[int] = None,
    out_dir: Optional[Path] = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict, Optional[Dict]]:
    """Main entry point for ARIMA model optimization using Optuna with walk-forward CV.

    Performs hyperparameter search using Optuna with walk-forward cross-validation.
    The walk-forward CV tests model robustness across different time periods, but
    optimization uses AIC alone (theoretically optimal for forecasting).

    **Important**: This function runs AIC optimization only (BIC removed for forecasting).
    Uses AIC alone for optimization (theoretically optimal for forecasting).

    Args:
        train_series: Training time series data.
        test_series: Optional test series (currently unused, reserved for future use).
        search: Only "optuna" is supported. Other values raise ValueError.
        n_trials: Number of Optuna trials (default: 100).
        n_jobs: Deprecated parameter, kept for backward compatibility (ignored).
        backtest_cfg: Optional backtest configuration dict. If None, computed from other parameters.
        backtest_n_splits: Number of temporal backtest splits
            (default: ARIMA_OPTIMIZATION_N_SPLITS).
        backtest_test_size: Absolute size of each test window (default: computed from ratio).
        backtest_refit_every: Refit frequency (required).
        out_dir: Optional output directory for saving results.
        random_state: Random seed for reproducibility (default: 42).

    Returns:
        Tuple of (results_dataframe, best_model_dict, None).
        The third element is None since BIC optimization was removed.

    Raises:
        ValueError: If search method is not "optuna" or if backtest parameters are invalid.
    """
    # Set random seed for reproducibility (numpy, Optuna handles its own seed)
    np.random.seed(random_state)

    # Handle deprecated parameters
    if n_jobs not in (None, 1):
        logger.warning("Parameter n_jobs=%s is deprecated and ignored.", n_jobs)

    # Compute backtest configuration
    if backtest_cfg is None:
        backtest_cfg = _compute_backtest_configuration(
            backtest_n_splits, backtest_test_size, backtest_refit_every, train_series
        )
    else:
        # Validate backtest_cfg if provided directly
        validate_backtest_config(
            backtest_cfg.get("n_splits", 0),
            backtest_cfg.get("test_size", 0),
            backtest_cfg.get("refit_every", 0),
            len(train_series),
        )

    # Validate all inputs
    _validate_optimization_inputs(train_series, test_series, backtest_cfg, search)

    # Run AIC optimization with walk-forward CV (BIC removed - AIC is optimal for forecasting)
    results = _run_aic_optimization(train_series, n_trials, random_state, backtest_cfg)
    logger.info("Collected %d AIC candidates", len(results))

    # Process results
    df = to_dataframe(results)
    best_model = _select_and_log_best_model(df)
    _save_optimization_results(df, best_model, out_dir)

    # Return format: (df, best_model, None) since BIC optimization was removed
    return df, best_model, None
