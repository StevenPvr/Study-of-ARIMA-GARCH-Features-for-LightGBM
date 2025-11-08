"""SARIMA model evaluation module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.constants import (
    SARIMA_BACKTEST_N_SPLITS_DEFAULT,
    SARIMA_BACKTEST_TEST_SIZE_DEFAULT,
    LJUNGBOX_RESIDUALS_SARIMA_FILE,
    LJUNGBOX_SIGNIFICANCE_LEVEL,
    RESULTS_DIR,
    ROLLING_PREDICTIONS_SARIMA_FILE,
    ROLLING_VALIDATION_METRICS_SARIMA_FILE,
    SARIMA_LJUNGBOX_LAGS_DEFAULT,
    SARIMA_REFIT_EVERY_DEFAULT,
    SARIMA_RESIDUALS_LJUNGBOX_PLOT,
)
from src.utils import get_logger

from .utils import (
    _add_ljungbox_summary_to_plot,
    _add_point_to_series,
    _compute_ljungbox_fallback,
    _create_acf_figure,
    _ensure_datetime_index,
    _ensure_supported_forecast_index,
    _extract_forecast_value,
    _extract_ljungbox_from_dataframe,
    _format_dates_from_index,
    _plot_acf_on_axis,
    _predict_single_step,
    _prepare_lags_list,
    _validate_series_not_empty,
    _validate_rolling_forecast_inputs,
)

logger = get_logger(__name__)


def rolling_forecast(
    train_series: pd.Series,
    test_series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    refit_every: int = SARIMA_REFIT_EVERY_DEFAULT,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform rolling forecast using SARIMA model with periodic refits.

    Refits the model every `refit_every` observations, reusing the model between refits.
    Forecasts remain at 1-day horizon (steps=1) for daily predictions.

    Args:
        train_series: Initial training series
        test_series: Test series to forecast
        order: SARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, s)
        refit_every: Refit frequency (in test observations). Defaults to 20.
        verbose: Whether to log progress

    Returns:
        Tuple of (predictions, actuals) arrays

    Raises:
        ValueError: If input series are empty or order is invalid
        RuntimeError: If any forecasting step fails (no silent fallbacks)
    """
    _validate_rolling_forecast_inputs(train_series, test_series, order, seasonal_order)

    if refit_every < 1:
        raise ValueError(f"refit_every must be >= 1, got {refit_every}")

    predictions: list[float] = []
    actuals: list[float] = []
    current_train = _ensure_datetime_index(train_series.copy())
    fitted_model: Any | None = None
    last_refit_len: int | None = None

    for i, (date, actual) in enumerate(test_series.items()):
        # Determine if we should refit the model
        should_refit = i == 0 or (i % refit_every == 0)

        if should_refit:
            # Full refit: pass None to force refit
            pred, fitted_model = _predict_single_step(
                current_train, order, seasonal_order, fitted_model=None
            )
            # Store length of data seen by the refit
            last_refit_len = len(_ensure_supported_forecast_index(current_train))
            if verbose:
                logger.info(f"Model refit at step {i + 1}/{len(test_series)}")
        else:
            # Reuse existing model: append only new observations since last refit
            if fitted_model is not None and last_refit_len is not None:
                supported_train = _ensure_supported_forecast_index(current_train)
                # Guard against unexpected shorter series
                last_refit_len = min(last_refit_len, len(supported_train))
                # Get only the new data since last refit
                new_data = supported_train.iloc[last_refit_len:]
                if len(new_data) > 0:
                    try:
                        # Append new observations without refitting parameters
                        updated_model = fitted_model.append(new_data, refit=False)
                        fc = updated_model.forecast(steps=1)
                        pred = _extract_forecast_value(fc)
                        fitted_model = updated_model
                        last_refit_len = len(supported_train)
                    except (ValueError, RuntimeError, AttributeError, TypeError):
                        # If append fails, fall back to full refit
                        logger.debug("Model append failed, falling back to full refit")
                        pred, fitted_model = _predict_single_step(
                            current_train, order, seasonal_order, fitted_model=None
                        )
                        last_refit_len = len(_ensure_supported_forecast_index(current_train))
                else:
                    # No new data, use existing model
                    fc = fitted_model.forecast(steps=1)
                    pred = _extract_forecast_value(fc)
            else:
                # Fallback: full refit if no model available
                pred, fitted_model = _predict_single_step(
                    current_train, order, seasonal_order, fitted_model=None
                )
                last_refit_len = len(_ensure_supported_forecast_index(current_train))

        if pred is None:
            msg = f"Forecast failed at step {i + 1}/{len(test_series)}"
            raise RuntimeError(msg)
        predictions.append(pred)
        current_train = _add_point_to_series(current_train, float(actual), date)
        actuals.append(float(actual))

        if verbose and (i + 1) % 10 == 0:
            logger.info(f"Rolling forecast: {i + 1}/{len(test_series)} completed")

    return np.array(predictions), np.array(actuals)


def walk_forward_backtest(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    *,
    n_splits: int = SARIMA_BACKTEST_N_SPLITS_DEFAULT,
    test_size: int = SARIMA_BACKTEST_TEST_SIZE_DEFAULT,
    max_train_size: int | None = None,
    refit_every: int = SARIMA_REFIT_EVERY_DEFAULT,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run walk-forward backtesting using rolling forecasts on multiple splits."""

    _validate_series_not_empty(series, "series")
    if n_splits < 1 or test_size < 1:
        raise ValueError(
            f"n_splits and test_size must be >=1, got n_splits={n_splits}, test_size={test_size}"
        )

    ordered_series = _ensure_datetime_index(series.sort_index())
    splitter = TimeSeriesSplit(
        n_splits=int(n_splits),
        test_size=int(test_size),
        max_train_size=None if max_train_size is None else int(max_train_size),
    )

    rows: list[dict[str, Any]] = []
    metric_history: dict[str, list[float]] = {}
    split_idx = 0

    try:
        for split_idx, (train_idx, val_idx) in enumerate(
            splitter.split(np.arange(len(ordered_series))), start=1
        ):
            train_slice = ordered_series.iloc[train_idx]
            val_slice = ordered_series.iloc[val_idx]
            preds, actuals = rolling_forecast(
                train_slice,
                val_slice,
                order,
                seasonal_order,
                refit_every=refit_every,
                verbose=False,
            )
            metrics = calculate_metrics(preds, actuals)
            for name, value in metrics.items():
                metric_history.setdefault(name.lower(), []).append(float(value))
            rows.append(
                {
                    "split": split_idx,
                    "train_start": train_slice.index[0],
                    "train_end": train_slice.index[-1],
                    "validation_start": val_slice.index[0],
                    "validation_end": val_slice.index[-1],
                    **metrics,
                }
            )
    except ValueError as exc:
        msg = (
            "Cannot create walk-forward splits with "
            f"n_splits={n_splits}, test_size={test_size}, series_length={len(ordered_series)}"
        )
        raise ValueError(msg) from exc

    if split_idx == 0:
        raise ValueError("No walk-forward splits were generated")

    split_metrics = pd.DataFrame(rows)
    summary: dict[str, float] = {}
    for name, values in metric_history.items():
        arr = np.asarray(values, dtype=float)
        summary[f"{name}_mean"] = float(np.mean(arr))
        summary[f"{name}_std"] = float(np.std(arr, ddof=0))

    logger.debug(
        "Completed walk-forward backtest with %s splits (test_size=%s)",
        len(split_metrics),
        test_size,
    )
    return split_metrics, summary


def calculate_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict[str, float]:
    """
    Calculate evaluation metrics from predictions and actuals.

    Args:
        predictions: Array of predictions
        actuals: Array of actual values

    Returns:
        Dictionary with MSE, RMSE, and MAE metrics

    Raises:
        ValueError: If inputs are empty or have mismatched lengths
        RuntimeError: If all predictions are NaN
    """
    if len(predictions) == 0 or len(actuals) == 0:
        raise ValueError("Predictions and actuals must be non-empty")
    if len(predictions) != len(actuals):
        raise ValueError(f"Length mismatch: predictions={len(predictions)}, actuals={len(actuals)}")

    valid_mask = ~np.isnan(predictions)
    if not valid_mask.any():
        msg = "All predictions are NaN"
        raise RuntimeError(msg)

    predictions_clean = predictions[valid_mask]
    actuals_clean = actuals[valid_mask]

    mse = mean_squared_error(actuals_clean, predictions_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals_clean, predictions_clean)

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
    }


def evaluate_model(
    train_series: pd.Series,
    test_series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    model_info: dict[str, Any] | None = None,
    refit_every: int = SARIMA_REFIT_EVERY_DEFAULT,
) -> dict[str, Any]:
    """
    Evaluate SARIMA model with rolling forecast validation.

    Args:
        train_series: Training time series
        test_series: Test time series
        order: SARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, s)
        model_info: Optional model information dictionary
        refit_every: Refit frequency (in test observations). Defaults to 20.

    Returns:
        Dictionary with evaluation results including metrics, predictions, and actuals
    """
    if model_info is None:
        model_info = {"params": f"SARIMA{order}{seasonal_order}"}

    logger.info(f"Evaluating model: {model_info.get('params', f'SARIMA{order}{seasonal_order}')}")
    logger.info(f"Refit frequency: every {refit_every} observations")

    predictions, actuals = rolling_forecast(
        train_series, test_series, order, seasonal_order, refit_every=refit_every, verbose=True
    )

    metrics = calculate_metrics(predictions, actuals)

    logger.info(f"Test metrics - RMSE: {metrics['RMSE']:.6f}, MAE: {metrics['MAE']:.6f}")

    dates_list = _format_dates_from_index(test_series, len(predictions))

    return {
        "model": model_info.get("params", f"SARIMA{order}{seasonal_order}"),
        "order": order,
        "seasonal_order": seasonal_order,
        "metrics": metrics,
        "predictions": predictions.tolist(),
        "actuals": actuals.tolist(),
        "dates": dates_list,
    }


def compute_residuals(actuals: Iterable[float], predictions: Iterable[float]) -> np.ndarray:
    """Compute residuals array as actuals - predictions.

    Args:
        actuals: Sequence of actual values.
        predictions: Sequence of predicted values.

    Returns:
        Residuals as numpy array.
    """
    a = np.asarray(list(actuals), dtype=float)
    p = np.asarray(list(predictions), dtype=float)
    n = min(a.size, p.size)
    return a[:n] - p[:n]


def ljung_box_on_residuals(
    residuals: np.ndarray, lags: int | Iterable[int] = SARIMA_LJUNGBOX_LAGS_DEFAULT
) -> dict[str, Any]:
    """Run Ljung–Box test on SARIMA residuals to assess whiteness.

    Computes Q-statistics and p-values at specified lags. Also flags
    rejection at 5% on the largest lag.

    Args:
        residuals: Residuals array (actual - predicted).
        lags: Single lag or an iterable of lags.

    Returns:
        Dict with lags, q_stat, p_value, reject_5pct (bool), and n.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox  # type: ignore

    res = np.asarray(residuals, dtype=float)
    res = res[np.isfinite(res)]
    lags_list = _prepare_lags_list(lags)

    if res.size == 0 or not lags_list:
        return {"lags": [], "q_stat": [], "p_value": [], "reject_5pct": False, "n": 0}

    df = acorr_ljungbox(res, lags=lags_list, return_df=True)
    q_list, p_list = _extract_ljungbox_from_dataframe(df, lags_list)

    reject = bool(p_list[-1] < LJUNGBOX_SIGNIFICANCE_LEVEL)
    return {
        "lags": lags_list,
        "q_stat": q_list,
        "p_value": p_list,
        "reject_5pct": reject,
        "n": int(res.size),
    }


def save_evaluation_results(results: dict[str, Any]) -> None:
    """
    Save evaluation results to files.

    Args:
        results: Dictionary with evaluation results including 'predictions', 'actuals'
    """
    # Save predictions without dates
    df_dict = {
        "prediction": results["predictions"],
        "actual": results["actuals"],
    }
    if "dates" in results:
        df_dict = {"date": results["dates"], **df_dict}
    predictions_df = pd.DataFrame(df_dict)
    predictions_df.to_csv(ROLLING_PREDICTIONS_SARIMA_FILE, index=False)
    logger.info(f"Saved predictions: {ROLLING_PREDICTIONS_SARIMA_FILE}")

    # Save metrics
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if "seasonal_order" not in results:
        raise ValueError("Results dictionary must contain 'seasonal_order' key")

    with Path(ROLLING_VALIDATION_METRICS_SARIMA_FILE).open("w") as f:
        json.dump(
            {
                "model": results["model"],
                "order": results["order"],
                "seasonal_order": results["seasonal_order"],
                "metrics": results["metrics"],
            },
            f,
            indent=2,
        )
    logger.info(f"Saved metrics: {ROLLING_VALIDATION_METRICS_SARIMA_FILE}")


def save_ljung_box_results(report: dict[str, Any]) -> None:
    """Save Ljung–Box report as JSON to the configured path.

    Args:
        report: Dictionary returned by ljung_box_on_residuals.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with Path(LJUNGBOX_RESIDUALS_SARIMA_FILE).open("w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved Ljung–Box report: {LJUNGBOX_RESIDUALS_SARIMA_FILE}")


def plot_residuals_acf_with_ljungbox(
    residuals: np.ndarray,
    *,
    lags: int = SARIMA_LJUNGBOX_LAGS_DEFAULT,
    out_path: Path | None = None,
) -> Path:
    """Plot ACF of SARIMA residuals with 95% bounds and Ljung–Box summary.

    Args:
        residuals: Residuals array.
        lags: Number of lags for ACF.
        out_path: Optional output path. Defaults to project constant.

    Returns:
        Path to the saved plot.
    """
    res = np.asarray(residuals, dtype=float)
    res = res[np.isfinite(res)]

    if out_path is None:
        out_path = SARIMA_RESIDUALS_LJUNGBOX_PLOT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, canvas, ax = _create_acf_figure()
    _plot_acf_on_axis(res, lags, ax)
    lb_result = ljung_box_on_residuals(res, lags=int(lags))
    _add_ljungbox_summary_to_plot(res, lags, ax, ljung_box_result=lb_result)

    canvas.print_png(str(out_path))
    logger.info(f"Saved residuals ACF + Ljung–Box plot: {out_path}")
    return out_path
