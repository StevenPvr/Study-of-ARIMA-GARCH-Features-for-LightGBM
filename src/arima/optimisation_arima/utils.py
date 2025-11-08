"""Utility functions for SARIMA model optimization."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Any

import numpy as np

import pandas as pd

from src.arima.models.sarima_model import fit_sarima_model
from src.constants import (
    SARIMA_BEST_MODELS_FILE,
    SARIMA_OPTIMIZATION_RESULTS_FILE,
    SARIMA_LJUNGBOX_LAGS_DEFAULT,
)
from src.utils import get_logger

logger = get_logger(__name__)


def _validate_split_data_columns(split_data: pd.DataFrame) -> None:
    """
    Validate that split data contains required columns.

    Args:
        split_data: DataFrame to validate

    Raises:
        ValueError: If required columns are missing
    """
    required_cols = ["date", "split", "weighted_log_return"]
    missing_cols = [col for col in required_cols if col not in split_data.columns]
    if missing_cols:
        msg = f"Missing required columns: {missing_cols}"
        raise ValueError(msg)


def _validate_split_values(split_data: pd.DataFrame) -> None:
    """
    Validate that split data contains valid split values.

    Args:
        split_data: DataFrame to validate

    Raises:
        ValueError: If split values are invalid
    """
    valid_splits = {"train", "test"}
    actual_splits = set(split_data["split"].unique())
    if not valid_splits.issubset(actual_splits):
        msg = f"Invalid split values. Expected {valid_splits}, got {actual_splits}"
        raise ValueError(msg)


def _validate_split_sets_not_empty(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """
    Validate that train and test sets are not empty.

    Args:
        train_data: Training data DataFrame
        test_data: Test data DataFrame

    Raises:
        ValueError: If either set is empty
    """
    if len(train_data) == 0:
        msg = "Train set is empty"
        raise ValueError(msg)
    if len(test_data) == 0:
        msg = "Test set is empty"
        raise ValueError(msg)


def _validate_train_series_not_empty(train_series: pd.Series) -> None:
    """
    Validate that training series is not empty.

    Args:
        train_series: Training time series data

    Raises:
        ValueError: If series is empty
    """
    if len(train_series) == 0:
        msg = "train_series cannot be empty"
        raise ValueError(msg)


def _validate_seasonal_period(s: int) -> None:
    """
    Validate that seasonal period is valid.

    Args:
        s: Seasonal period

    Raises:
        ValueError: If seasonal period is invalid
    """
    if s < 1:
        msg = f"Seasonal period s must be >= 1, got {s}"
        raise ValueError(msg)


def _validate_parameter_values(
    p: int,
    d: int,
    q: int,
    P: int,
    D: int,
    Q: int,
    s: int,
) -> None:
    """
    Validate that all SARIMA parameter values are valid.

    Args:
        p: AR order
        d: Differencing order
        q: MA order
        P: Seasonal AR order
        D: Seasonal differencing order
        Q: Seasonal MA order
        s: Seasonal period

    Raises:
        ValueError: If any parameter is invalid
    """
    params = [p, d, q, P, D, Q]
    if any(param < 0 for param in params):
        msg = f"Invalid parameters: p={p}, d={d}, q={q}, P={P}, D={D}, Q={Q}, s={s}"
        raise ValueError(msg)
    _validate_seasonal_period(s)


def _validate_sarima_parameters(
    train_series: pd.Series,
    p: int,
    d: int,
    q: int,
    P: int,
    D: int,
    Q: int,
    s: int,
) -> None:
    """
    Validate SARIMA model parameters.

    Args:
        train_series: Training time series data
        p: AR order
        d: Differencing order
        q: MA order
        P: Seasonal AR order
        D: Seasonal differencing order
        Q: Seasonal MA order
        s: Seasonal period

    Raises:
        ValueError: If parameters are invalid
    """
    _validate_train_series_not_empty(train_series)
    _validate_parameter_values(p, d, q, P, D, Q, s)


def _build_model_result_dict(
    p: int,
    d: int,
    q: int,
    P: int,
    D: int,
    Q: int,
    s: int,
    aic_value: float,
    bic_value: float,
) -> dict[str, Any]:
    """
    Build result dictionary for a SARIMA model.

    Args:
        p: AR order
        d: Differencing order
        q: MA order
        P: Seasonal AR order
        D: Seasonal differencing order
        Q: Seasonal MA order
        s: Seasonal period
        aic_value: AIC value
        bic_value: BIC value

    Returns:
        Dictionary with model results
    """
    return {
        "p": p,
        "d": d,
        "q": q,
        "P": P,
        "D": D,
        "Q": Q,
        "s": s,
        "aic": aic_value,
        "bic": bic_value,
        "params": f"SARIMA({p},{d},{q})({P},{D},{Q})[{s}]",
    }


def _test_sarima_model(
    train_series: pd.Series,
    p: int,
    d: int,
    q: int,
    P: int = 0,
    D: int = 0,
    Q: int = 0,
    s: int = 12,
) -> dict[str, Any] | None:
    """
    Test a single SARIMA model configuration.

    Args:
        train_series: Training time series data
        p: AR order
        d: Differencing order
        q: MA order
        P: Seasonal AR order
        D: Seasonal differencing order
        Q: Seasonal MA order
        s: Seasonal period

    Returns:
        Dictionary with model results or None if model failed

    Raises:
        ValueError: If parameters are invalid
    """
    _validate_sarima_parameters(train_series, p, d, q, P, D, Q, s)

    try:
        fitted_model = fit_sarima_model(
            train_series, p=p, d=d, q=q, P=P, D=D, Q=Q, s=s, verbose=False
        )

        # Extract AIC and BIC with type safety using getattr to avoid type checker issues
        aic_value = float(getattr(fitted_model, "aic", float("inf")))
        bic_value = float(getattr(fitted_model, "bic", float("inf")))

        # Ljung–Box whiteness criterion on in-sample residuals
        resid = getattr(fitted_model, "resid", None)
        if resid is None:
            raise RuntimeError("Fitted model has no residuals for whiteness test")
        from src.arima.evaluation_arima.evaluation_arima import ljung_box_on_residuals

        lb = ljung_box_on_residuals(np.asarray(resid, dtype=float), lags=SARIMA_LJUNGBOX_LAGS_DEFAULT)
        lb_p_last: float = float(lb["p_value"][-1]) if lb.get("p_value") else float("nan")
        lb_reject_5pct: bool = bool(lb.get("reject_5pct", False))

        result = _build_model_result_dict(p, d, q, P, D, Q, s, aic_value, bic_value)
        result["lb_pvalue_last"] = lb_p_last
        result["lb_reject_5pct"] = lb_reject_5pct
        result["lb_lags"] = lb.get("lags", [])
        return result
    except Exception as e:
        logger.debug(f"Model SARIMA({p},{d},{q})({P},{D},{Q})[{s}] failed: {e}")
        return None


def _test_single_model_wrapper(
    train_series_data: tuple[Any, ...],
    p: int,
    d: int,
    q: int,
    P: int,
    D: int,
    Q: int,
    s: int,
) -> dict[str, Any] | None:
    """
    Wrapper function for testing a single model (for multiprocessing).

    Args:
        train_series_data: Tuple containing (values, index) to reconstruct Series
        p: AR order
        d: Differencing order
        q: MA order
        P: Seasonal AR order
        D: Seasonal differencing order
        Q: Seasonal MA order
        s: Seasonal period

    Returns:
        Dictionary with model results or None if model failed
    """
    # Reconstruct Series from pickled data
    values, index = train_series_data
    train_series = pd.Series(values, index=pd.DatetimeIndex(index))
    return _test_sarima_model(train_series, p, d, q, P, D, Q, s)


def _evaluate_models_sequential(
    train_series: pd.Series,
    param_combinations: list[tuple[int, int, int, int, int, int, int]],
) -> list[dict[str, Any]]:
    """
    Evaluate SARIMA models sequentially.

    Args:
        train_series: Training time series data
        param_combinations: List of (p, d, q, P, D, Q, s) tuples to test

    Returns:
        List of successful model results
    """
    sequential_results: list[dict[str, Any]] = []
    for idx, (p, d, q, P, D, Q, s) in enumerate(param_combinations, 1):
        model_result = _test_sarima_model(train_series, p, d, q, P, D, Q, s)
        if model_result is not None:
            sequential_results.append(model_result)
        if idx % 20 == 0:
            logger.info(f"Progress: {idx}/{len(param_combinations)} models tested")
    return sequential_results


def _collect_parallel_results(
    executor: ProcessPoolExecutor,
    future_to_params: dict[Any, tuple[int, int, int, int, int, int, int]],
    total_combinations: int,
) -> list[dict[str, Any]]:
    """
    Collect results from parallel execution.

    Args:
        executor: ProcessPoolExecutor instance
        future_to_params: Dictionary mapping futures to parameter tuples
        total_combinations: Total number of combinations

    Returns:
        List of successful model results
    """
    parallel_results: list[dict[str, Any]] = []
    completed = 0

    for future in as_completed(future_to_params):
        completed += 1
        try:
            model_result = future.result()
            if model_result is not None:
                parallel_results.append(model_result)
        except Exception as e:
            params = future_to_params.get(future, "unknown")
            logger.debug(f"Model SARIMA{params} failed: {e}")

        if completed % 20 == 0:
            logger.info(f"Progress: {completed}/{total_combinations} models tested")

    return parallel_results


def _evaluate_models_parallel(
    train_series: pd.Series,
    param_combinations: list[tuple[int, int, int, int, int, int, int]],
    n_jobs: int,
) -> list[dict[str, Any]]:
    """
    Evaluate SARIMA models in parallel.

    Args:
        train_series: Training time series data
        param_combinations: List of (p, d, q, P, D, Q, s) tuples to test
        n_jobs: Number of parallel workers

    Returns:
        List of successful model results
    """
    # Prepare Series data for pickling (needed for multiprocessing)
    train_series_data = (train_series.values.tolist(), train_series.index.tolist())

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all tasks
        future_to_params = {
            executor.submit(
                _test_single_model_wrapper,
                train_series_data,
                p,
                d,
                q,
                P,
                D,
                Q,
                s,
            ): (p, d, q, P, D, Q, s)
            for p, d, q, P, D, Q, s in param_combinations
        }

        return _collect_parallel_results(executor, future_to_params, len(param_combinations))


def _evaluate_sarima_models(
    train_series: pd.Series,
    param_combinations: list[tuple[int, int, int, int, int, int, int]],
    n_jobs: int | None = None,
) -> list[dict[str, Any]]:
    """
    Evaluate all SARIMA model combinations with optional parallelization.

    Args:
        train_series: Training time series data
        param_combinations: List of (p, d, q, P, D, Q, s) tuples to test
        n_jobs: Number of parallel jobs. If None, uses cpu_count(). If 1, sequential.

    Returns:
        List of successful model results
    """
    if n_jobs == 1 or len(param_combinations) < 10:
        return _evaluate_models_sequential(train_series, param_combinations)

    # Parallel execution
    effective_n_jobs = n_jobs if n_jobs is not None else max(1, cpu_count() - 1)
    return _evaluate_models_parallel(train_series, param_combinations, effective_n_jobs)


def _build_best_model_dict(model_row: pd.Series) -> dict[str, Any]:
    """
    Build dictionary for best model from DataFrame row.

    Args:
        model_row: DataFrame row with model information

    Returns:
        Dictionary with model parameters and metrics

    Raises:
        ValueError: If required SARIMA parameters are missing
    """
    required_cols = ["p", "d", "q", "P", "D", "Q", "s", "aic", "bic"]
    missing_cols = [col for col in required_cols if col not in model_row.index]
    if missing_cols:
        msg = (
            f"Missing required SARIMA parameters in results: {missing_cols}. "
            "SARIMA requires all seasonal parameters (P, D, Q, s). "
            "If data has no seasonality, SARIMA is not appropriate."
        )
        raise ValueError(msg)

    p = int(model_row["p"])
    d = int(model_row["d"])
    q = int(model_row["q"])
    P = int(model_row["P"])
    D = int(model_row["D"])
    Q = int(model_row["Q"])
    s = int(model_row["s"])
    aic_value = float(model_row["aic"])
    bic_value = float(model_row["bic"])

    return _build_model_result_dict(p, d, q, P, D, Q, s, aic_value, bic_value)


def _select_best_models(
    results_df: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Select best AIC and BIC models from results DataFrame.

    Args:
        results_df: DataFrame with all optimization results

    Returns:
        Tuple of (best_aic_dict, best_bic_dict)
    """
    # Prefer whiteness (non-rejection) first, then AIC/BIC
    if "lb_reject_5pct" in results_df.columns:
        sort_aic = results_df.sort_values(["lb_reject_5pct", "aic"]).reset_index(drop=True)
        sort_bic = results_df.sort_values(["lb_reject_5pct", "bic"]).reset_index(drop=True)
    else:
        sort_aic = results_df.sort_values("aic").reset_index(drop=True)
        sort_bic = results_df.sort_values("bic").reset_index(drop=True)

    best_aic_dict = _build_best_model_dict(sort_aic.iloc[0])
    best_bic_dict = _build_best_model_dict(sort_bic.iloc[0])
    return best_aic_dict, best_bic_dict


def _prepare_optimization_results(
    results: list[dict[str, Any]], total_combinations: int
) -> pd.DataFrame:
    """
    Prepare and validate optimization results DataFrame.

    Args:
        results: List of successful model results
        total_combinations: Total number of combinations tested

    Returns:
        DataFrame with sorted results

    Raises:
        RuntimeError: If no models converged
    """
    logger.info(f"Optimization complete: {len(results)}/{total_combinations} converged")

    results_df = pd.DataFrame(results)
    if results_df.empty:
        msg = "No SARIMA models converged. Adjust grid or check series."
        raise RuntimeError(msg)

    return results_df.sort_values("aic").reset_index(drop=True)


def _log_optimization_start(
    param_combinations: list[tuple[int, int, int, int, int, int, int]],
    p_range: range,
    d_range: range,
    q_range: range,
    P_range: range,
    D_range: range,
    Q_range: range,
    s: int,
    effective_n_jobs: int,
) -> None:
    """Log optimization start information."""
    logger.info(f"Starting SARIMA optimization: {len(param_combinations)} combinations")
    logger.info(f"Using {effective_n_jobs} parallel workers")
    logger.info(
        f"p_range: {list(p_range)}, d_range: {list(d_range)}, q_range: {list(q_range)}, "
        f"P_range: {list(P_range)}, D_range: {list(D_range)}, Q_range: {list(Q_range)}, s: {s}"
    )


def _save_optimization_results(
    results_df: pd.DataFrame,
    best_aic: dict[str, Any],
    best_bic: dict[str, Any],
    train_series: pd.Series,
    test_series: pd.Series,
) -> None:
    """
    Save optimization results to files.

    Args:
        results_df: DataFrame with all optimization results
        best_aic: Best AIC model dictionary
        best_bic: Best BIC model dictionary
        train_series: Training series (for metadata)
        test_series: Test series (for metadata)
    """
    results_df.to_csv(SARIMA_OPTIMIZATION_RESULTS_FILE, index=False)
    logger.info(f"Saved optimization results: {SARIMA_OPTIMIZATION_RESULTS_FILE}")

    best_models = {
        "best_aic": best_aic,
        "best_bic": best_bic,
        "total_models_tested": len(results_df),
        "models_converged": len(results_df),
        "train_size": len(train_series),
        "test_size": len(test_series),
    }

    SARIMA_BEST_MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SARIMA_BEST_MODELS_FILE.open("w") as f:
        json.dump(best_models, f, indent=2)

    logger.info(f"Saved best models: {SARIMA_BEST_MODELS_FILE}")
