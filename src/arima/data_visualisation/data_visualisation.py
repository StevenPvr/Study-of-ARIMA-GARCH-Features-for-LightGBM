"""Data visualization functions for the S&P 500 Forecasting project."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.constants import SARIMA_DATA_VISU_PLOTS_DIR
from src.utils import get_logger

from .utils import (
    _decompose_seasonal_component,
    _format_date_axis,
    _infer_seasonal_period,
    _load_and_validate_data,
    _load_series_for_year,
    _maybe_resample,
    _plot_forecast_lines,
    _plot_predictions_vs_actuals,
    _plot_residuals_histogram,
    _plot_residuals_qq,
    _plot_residuals_timeseries,
    _plot_seasonal_component_only,
    _plot_seasonal_daily_long_period,
    _save_plot,
    _validate_forecast_inputs,
    _validate_minimum_periods,
)

logger = get_logger(__name__)


def plot_weighted_series(
    data_file: str,
    output_file: str,
) -> None:
    """Plot the time series of weighted log-returns.

    For 10 years of daily data, the series is resampled to weekly frequency
    for better readability while preserving the overall trend.

    Args:
        data_file: Path to weighted log-returns CSV file.
        output_file: Path to save the plot.

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If required column is missing or data is empty.
    """
    logger.info("Loading weighted log-returns data")
    aggregated_returns = _load_and_validate_data(data_file, "weighted_log_return")
    weighted_series = aggregated_returns["weighted_log_return"].dropna()

    # Resample to weekly frequency for better readability with 10 years of daily data
    # This reduces ~2500 daily points to ~520 weekly points while preserving trends
    weighted_series_resampled = weighted_series.resample("W").mean()

    # Create larger figure for better readability
    _, ax = plt.subplots(figsize=(18, 6))
    ax.plot(weighted_series_resampled.index, weighted_series_resampled, linewidth=0.6, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Rendements logarithmiques pondérés du portefeuille (10 ans)", fontsize=14)
    ax.set_ylabel("Log-return", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.grid(alpha=0.3, linestyle="--")

    _format_date_axis(ax)

    plt.tight_layout()

    SARIMA_DATA_VISU_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {output_file}")
    plt.close()


def plot_acf_pacf(
    data_file: str,
    output_file: str,
    lags: int = 30,
) -> None:
    """Plot autocorrelation (ACF) and partial autocorrelation (PACF) functions.

    The lag 0 is excluded from the plot as it always has a correlation of 1
    (comparing a series with itself).

    Args:
        data_file: Path to weighted log-returns CSV file.
        output_file: Path to save the plot.
        lags: Number of lags to display (excluding lag 0).

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If lags is invalid or required column is missing.
    """
    if lags <= 0:
        msg = f"lags must be positive, got {lags}"
        raise ValueError(msg)

    logger.info("Loading weighted log-returns data for ACF/PACF")
    aggregated_returns = _load_and_validate_data(data_file, "weighted_log_return")
    weighted_series = aggregated_returns["weighted_log_return"].dropna()

    if len(weighted_series) < lags:
        logger.warning(
            f"Series length ({len(weighted_series)}) is less than requested lags ({lags}). "
            f"Using {len(weighted_series) - 1} lags instead."
        )
        lags = len(weighted_series) - 1

    _, axes = plt.subplots(1, 2, figsize=(14, 4))

    plot_acf(weighted_series, lags=lags, ax=axes[0], zero=False)
    axes[0].set_title("Fonction d'autocorrélation (ACF)")

    plot_pacf(weighted_series, lags=lags, ax=axes[1], method="ywm", zero=False)
    axes[1].set_title("Fonction d'autocorrélation partielle (PACF)")

    plt.tight_layout()

    SARIMA_DATA_VISU_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {output_file}")
    plt.close()


def plot_seasonality_for_year(
    year: int,
    *,
    data_file: str,
    output_file: str,
    period: int | None = None,
    model: str = "additive",
    column: str = "weighted_log_return",
    resample_to: str | None = "B",
) -> None:
    """Plot ONLY the seasonal component for a given calendar year.

    The goal is readability: focus on one year and a regular time grid.
    """
    data_path = str(data_file)
    out_path = str(output_file)

    base_series = _load_series_for_year(year=year, data_file=data_path, column=column)
    series = _maybe_resample(base_series, resample_to=resample_to)
    eff_period = _infer_seasonal_period(resample_to=resample_to, override=period)
    seasonal = _decompose_seasonal_component(series, model=model, period=eff_period)
    _plot_seasonal_component_only(
        seasonal,
        title=f"Seasonal component - {year} (model={model}, period={eff_period})",
        full_period=False,
    )
    _save_plot(out_path)


def _validate_seasonal_params(model: str, period: int | None) -> None:
    """Validate seasonal decomposition parameters.

    Args:
        model: Decomposition model ('additive' or 'multiplicative').
        period: Seasonal period override.

    Raises:
        ValueError: If model or period is invalid.
    """
    if model not in ("additive", "multiplicative"):
        msg = f"model must be 'additive' or 'multiplicative', got {model}"
        raise ValueError(msg)

    if period is not None and period <= 0:
        msg = f"period must be positive, got {period}"
        raise ValueError(msg)


def plot_seasonality_full_period(
    *,
    data_file: str,
    output_file: str,
    period: int | None = None,
    model: str = "additive",
    column: str = "weighted_log_return",
) -> None:
    """Plot the seasonal component for the full period (10 years) with weekly resampling.

    The series is resampled to weekly frequency for better readability while preserving
    the overall seasonal patterns.

    Args:
        data_file: Path to weighted log-returns CSV file.
        output_file: Path to save the plot.
        period: Seasonal period override. If None, inferred from weekly frequency (52).
        model: Decomposition model ('additive' or 'multiplicative').
        column: Column name to use from the data file.

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If model is invalid, period is invalid, or column is missing.
    """
    _validate_seasonal_params(model, period)

    logger.info("Loading data for seasonal component (full period)")
    dataframe = _load_and_validate_data(data_file, column)
    base_series = dataframe[column].dropna()

    # Resample to weekly frequency for better readability with 10 years of daily data
    # This reduces ~2500 daily points to ~520 weekly points while preserving trends
    series = base_series.resample("W").mean().dropna()

    # Infer seasonal period for weekly data (52 weeks per year)
    eff_period = period if period is not None else 52

    seasonal = _decompose_seasonal_component(series, model=model, period=eff_period)
    _plot_seasonal_component_only(
        seasonal,
        title=f"Seasonal component - Full period (model={model}, period={eff_period})",
        full_period=True,
    )
    _save_plot(output_file)


def plot_seasonality_daily(
    *,
    data_file: str,
    output_file: str,
    period: int = 5,
    model: str = "additive",
    column: str = "weighted_log_return",
    years: int = 1,
) -> None:
    """Plot seasonal component for daily data (weekly seasonality - 5 business days).

    Analyzes weekly patterns in daily returns (Monday effect, Friday effect, etc.).
    Uses only the last N years of data for better readability (default: 1 year).

    Args:
        data_file: Path to weighted log-returns CSV file.
        output_file: Path to save the plot.
        period: Seasonal period in days. Default 5 for weekly (business days).
        model: Decomposition model ('additive' or 'multiplicative').
        column: Column name to use from the data file.
        years: Number of recent years to use for analysis. Default 1.

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If model is invalid, period is invalid, or insufficient data.
    """
    _validate_seasonal_params(model, period)

    logger.info(
        "Loading daily data for weekly seasonal component (period=%d, last %d years)", period, years
    )
    dataframe = _load_and_validate_data(data_file, column)
    series = dataframe[column].dropna()
    if not isinstance(series, pd.Series):
        msg = f"Column '{column}' did not return a Series"
        raise ValueError(msg)

    # Filter to last N years for better readability
    if len(series) > 0:
        max_date = series.index.max()
        # Series index is DatetimeIndex, so max() returns Timestamp
        end_date = pd.Timestamp(max_date)  # type: ignore[arg-type]
        start_date = end_date - pd.DateOffset(years=years)
        series = series.loc[start_date:end_date]
        logger.info(
            "Filtered to %d observations from %s to %s",
            len(series),
            start_date.date(),
            end_date.date(),
        )

    _validate_minimum_periods(series, period, min_periods=2)

    seasonal = _decompose_seasonal_component(series, model=model, period=period)
    _plot_seasonal_daily_long_period(
        seasonal,
        title=f"Seasonal component - Daily data (last {years} year{'s' if years > 1 else ''}, weekly pattern, period={period} days, model={model})",
    )
    _save_plot(output_file)


def plot_seasonality_monthly(
    *,
    data_file: str,
    output_file: str,
    period: int = 12,
    model: str = "additive",
    column: str = "weighted_log_return",
) -> None:
    """Plot seasonal component for monthly data (annual seasonality - 12 months).

    Analyzes monthly patterns in returns (January effect, month-end effect, etc.).

    Args:
        data_file: Path to weighted log-returns CSV file.
        output_file: Path to save the plot.
        period: Seasonal period in months. Default 12 for annual seasonality.
        model: Decomposition model ('additive' or 'multiplicative').
        column: Column name to use from the data file.

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If model is invalid, period is invalid, or insufficient data.
    """
    _validate_seasonal_params(model, period)

    logger.info("Loading data for monthly seasonal component (period=%d)", period)
    dataframe = _load_and_validate_data(data_file, column)
    base_series = dataframe[column].dropna()

    # Resample to monthly frequency
    series = base_series.resample("ME").mean().dropna()

    _validate_minimum_periods(series, period, min_periods=2)

    seasonal = _decompose_seasonal_component(series, model=model, period=period)
    _plot_seasonal_component_only(
        seasonal,
        title=f"Seasonal component - Monthly data (annual pattern, period={period} months, model={model})",
        full_period=True,
    )
    _save_plot(output_file)


def plot_rolling_forecast_sarima_000(
    test_series: pd.Series,
    actuals: np.ndarray,
    predictions: np.ndarray,
    sarima_order: tuple[int, int, int],
    metrics: dict[str, float],
    output_file: str,
) -> None:
    """Visualize SARIMA rolling forecast predictions vs actual values.

    Args:
        test_series: Test set time series with date index.
        actuals: Actual values array.
        predictions: Predicted values array.
        sarima_order: SARIMA order tuple (p, d, q).
        metrics: Dictionary with 'RMSE' and 'MAE' keys.
        output_file: Path to save the plot.

    Raises:
        ValueError: If arrays have mismatched lengths or required metrics are missing.
    """
    _validate_forecast_inputs(test_series, actuals, predictions, sarima_order, metrics)

    test_dates = test_series.index
    _, ax = plt.subplots(figsize=(16, 6))

    _plot_forecast_lines(ax, test_dates, actuals, predictions, sarima_order, metrics)
    plt.tight_layout()
    _save_plot(output_file)


def analyze_residuals_sarima_000(
    test_series: pd.Series,
    actuals: np.ndarray,
    predictions: np.ndarray,
    sarima_order: tuple[int, int, int],
    output_file: str,
) -> np.ndarray:
    """Analyze and visualize SARIMA model residuals.

    Creates a 2x2 subplot with:
    - Time series of residuals
    - Histogram of residuals
    - Q-Q plot for normality
    - Predictions vs actuals scatter plot

    Args:
        test_series: Test set time series with date index.
        actuals: Actual values array.
        predictions: Predicted values array.
        sarima_order: SARIMA order tuple (p, d, q).
        output_file: Path to save the plot.

    Returns:
        Array of residuals (actuals - predictions).

    Raises:
        ValueError: If arrays have mismatched lengths or sarima_order is invalid.
    """
    _validate_forecast_inputs(test_series, actuals, predictions, sarima_order)

    residuals = actuals - predictions
    test_dates = test_series.index

    _, axes = plt.subplots(2, 2, figsize=(16, 10))
    _plot_residuals_timeseries(axes[0, 0], test_dates, residuals)
    _plot_residuals_histogram(axes[0, 1], residuals)
    _plot_residuals_qq(axes[1, 0], residuals)
    _plot_predictions_vs_actuals(axes[1, 1], actuals, predictions)

    plt.suptitle(
        f"Analyse des résidus - SARIMA{sarima_order}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    _save_plot(output_file)

    return residuals
