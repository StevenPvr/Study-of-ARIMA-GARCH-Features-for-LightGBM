"""Utility functions for data visualization operations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy import stats as scipy_stats
from statsmodels.tsa.seasonal import seasonal_decompose

from src.constants import SARIMA_DATA_VISU_PLOTS_DIR
from src.utils import get_logger

logger = get_logger(__name__)


def _load_csv_dataframe(data_file: str) -> pd.DataFrame:
    """Load CSV file and return DataFrame with date index.

    Args:
        data_file: Path to CSV file.

    Returns:
        DataFrame with date index.

    Raises:
        FileNotFoundError: If data_file does not exist.
    """
    data_path = Path(data_file)
    if not data_path.exists():
        msg = f"Data file not found: {data_file}"
        raise FileNotFoundError(msg)

    try:
        return pd.read_csv(data_file, parse_dates=["date"]).set_index("date")
    except Exception as e:
        logger.error(f"Failed to read data file {data_file}: {e}")
        raise


def _validate_column_exists(dataframe: pd.DataFrame, column: str, data_file: str) -> None:
    """Validate that a column exists in the dataframe.

    Args:
        dataframe: DataFrame to check.
        column: Column name to validate.
        data_file: Path to data file (for error message).

    Raises:
        ValueError: If column is missing or data is empty.
    """
    if column not in dataframe.columns:
        msg = f"Column '{column}' not found in {data_file}"
        raise ValueError(msg)

    series = dataframe[column].dropna()
    if series.empty:
        msg = f"No valid data found in {data_file}"
        raise ValueError(msg)


def _load_and_validate_data(
    data_file: str,
    required_column: str,
) -> pd.DataFrame:
    """Load and validate CSV data file.

    Args:
        data_file: Path to CSV file.
        required_column: Required column name.

    Returns:
        DataFrame with date index.

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If required column is missing or data is empty.
    """
    dataframe = _load_csv_dataframe(data_file)
    _validate_column_exists(dataframe, required_column, data_file)
    return dataframe


def _format_date_axis(ax: Axes) -> None:
    """Format x-axis dates for better readability.

    Args:
        ax: Matplotlib axes to format.
    """
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


def _validate_array_lengths(
    actuals: np.ndarray,
    predictions: np.ndarray,
    test_series: pd.Series,
) -> None:
    """Validate that arrays have matching lengths.

    Args:
        actuals: Actual values array.
        predictions: Predicted values array.
        test_series: Test set time series.

    Raises:
        ValueError: If lengths don't match.
    """
    if len(actuals) != len(predictions):
        msg = (
            f"actuals and predictions must have same length, "
            f"got {len(actuals)} and {len(predictions)}"
        )
        raise ValueError(msg)

    if len(actuals) != len(test_series):
        msg = (
            f"actuals length ({len(actuals)}) must match "
            f"test_series length ({len(test_series)})"
        )
        raise ValueError(msg)


def _validate_sarima_order(sarima_order: tuple[int, int, int]) -> None:
    """Validate SARIMA order tuple.

    Args:
        sarima_order: SARIMA order tuple (p, d, q).

    Raises:
        ValueError: If order is invalid.
    """
    if len(sarima_order) != 3 or any(x < 0 for x in sarima_order):
        msg = f"sarima_order must be tuple of 3 non-negative integers, got {sarima_order}"
        raise ValueError(msg)


def _validate_metrics(metrics: dict[str, float]) -> None:
    """Validate metrics dictionary contains required keys.

    Args:
        metrics: Dictionary with 'RMSE' and 'MAE' keys.

    Raises:
        ValueError: If required metrics are missing.
    """
    required_metrics = {"RMSE", "MAE"}
    if not required_metrics.issubset(metrics.keys()):
        missing = required_metrics - set(metrics.keys())
        msg = f"metrics must contain {required_metrics}, missing {missing}"
        raise ValueError(msg)


def _validate_forecast_inputs(
    test_series: pd.Series,
    actuals: np.ndarray,
    predictions: np.ndarray,
    sarima_order: tuple[int, int, int],
    metrics: dict[str, float] | None = None,
) -> None:
    """Validate inputs for forecast plotting functions.

    Args:
        test_series: Test set time series with date index.
        actuals: Actual values array.
        predictions: Predicted values array.
        sarima_order: SARIMA order tuple (p, d, q).
        metrics: Optional dictionary with 'RMSE' and 'MAE' keys.

    Raises:
        ValueError: If inputs are invalid.
    """
    _validate_array_lengths(actuals, predictions, test_series)
    _validate_sarima_order(sarima_order)
    if metrics is not None:
        _validate_metrics(metrics)


def _plot_forecast_lines(
    ax: Axes,
    test_dates: pd.Index,
    actuals: np.ndarray,
    predictions: np.ndarray,
    sarima_order: tuple[int, int, int],
    metrics: dict[str, float],
) -> None:
    """Plot forecast lines on the given axes.

    Args:
        ax: Matplotlib axes to plot on.
        test_dates: Date index for the time series.
        actuals: Actual values array.
        predictions: Predicted values array.
        sarima_order: SARIMA order tuple (p, d, q).
        metrics: Dictionary with 'RMSE' and 'MAE' keys.
    """
    ax.plot(
        test_dates,
        actuals,
        label="Valeurs réelles",
        linewidth=1.5,
        alpha=0.8,
    )
    ax.plot(
        test_dates,
        predictions,
        label="Prédictions (rolling)",
        linewidth=1.2,
        alpha=0.7,
    )
    ax.axhline(0, linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_title(
        (
            f"Prévision Rolling - SARIMA{sarima_order}\n"
            f'RMSE={metrics["RMSE"]:.6f} | MAE={metrics["MAE"]:.6f}'
        ),
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Rendement logarithmique pondéré", fontsize=10)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--")


def _save_plot(output_file: str) -> None:
    """Save the current plot to file.

    Args:
        output_file: Path to save the plot.
    """
    SARIMA_DATA_VISU_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {output_file}")
    plt.close()


def _validate_year(year: int) -> None:
    """Validate year is within reasonable range.

    Args:
        year: Calendar year to validate.

    Raises:
        ValueError: If year is out of range.
    """
    if not (1900 <= year <= 2100):
        msg = f"Invalid year: {year}"
        raise ValueError(msg)


def _filter_dataframe_by_year(dataframe: pd.DataFrame, year: int) -> pd.DataFrame:
    """Filter dataframe to a specific calendar year.

    Args:
        dataframe: DataFrame with date index.
        year: Calendar year to filter.

    Returns:
        DataFrame filtered for the year.

    Raises:
        ValueError: If no data available for the year.
    """
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    df_year = dataframe.loc[start:end]
    if df_year.empty:
        msg = f"No data available for year {year}"
        raise ValueError(msg)
    return df_year


def _load_dataframe_for_year(data_file: str, year: int) -> pd.DataFrame:
    """Load and validate dataframe for a given year.

    Args:
        data_file: Path to CSV file.
        year: Calendar year to filter.

    Returns:
        DataFrame with date index filtered for the year.

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If no data available for the year.
    """
    logger.info("Loading data for seasonal component (%s)", year)
    dataframe = _load_csv_dataframe(data_file)
    return _filter_dataframe_by_year(dataframe, year)


def _load_series_for_year(*, year: int, data_file: str, column: str) -> pd.Series:
    """Load a column as a pandas Series filtered to a given calendar year.

    Args:
        year: Calendar year to filter.
        data_file: Path to CSV file.
        column: Column name to extract.

    Returns:
        Filtered Series for the specified year.

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If year is invalid, column is missing, or no data for the year.
    """
    _validate_year(year)
    df_year = _load_dataframe_for_year(data_file, year)

    if column not in df_year.columns:
        msg = f"Column '{column}' not found in {data_file}"
        raise ValueError(msg)

    series = df_year[column]
    if not isinstance(series, pd.Series):
        msg = f"Column '{column}' did not return a Series"
        raise ValueError(msg)

    return series.dropna()


def _maybe_resample(series: pd.Series, resample_to: str | None) -> pd.Series:
    """Resample series to a regular grid if requested, returning a clean Series."""
    if resample_to is None or series.empty:
        return series
    return series.resample(resample_to).mean().dropna()


def _infer_seasonal_period(resample_to: str | None, override: int | None) -> int:
    """Infer a reasonable seasonal period from the target frequency unless overridden."""
    if override is not None:
        return int(override)
    default_map = {"B": 5, "D": 7, "M": 12}
    key = (resample_to or "").upper()
    return int(default_map.get(key, 12))


def _decompose_seasonal_component(series: pd.Series, *, model: str, period: int) -> pd.Series:
    """Run seasonal decomposition and return the seasonal component as a Series."""
    result = seasonal_decompose(series, model=model, period=int(period))
    return result.seasonal


def _plot_seasonal_component_only(
    seasonal: pd.Series, *, title: str, full_period: bool = False
) -> None:
    """Plot a single seasonal component line with basic styling.

    Args:
        seasonal: Seasonal component series to plot.
        title: Plot title.
        full_period: If True, format for full period (10 years) with weekly resampling.
    """
    if full_period:
        fig, ax = plt.subplots(figsize=(18, 6))
        ax.plot(seasonal.index, seasonal, linewidth=0.6, alpha=0.8)
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Seasonal", fontsize=12)
        ax.grid(alpha=0.3, linestyle="--")
        # Format x-axis dates for better readability
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    else:
        fig, ax = plt.subplots(figsize=(14, 4))
        x = seasonal.index.to_numpy(dtype="datetime64[ns]")
        y = np.asarray(seasonal.values, dtype=float)
        ax.plot(x, y, linewidth=1.0)
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Seasonal")
    plt.tight_layout()


def _plot_seasonal_daily_long_period(seasonal: pd.Series, *, title: str) -> None:
    """Plot seasonal component for daily data over period (typically 1 year).

    Uses a standard figure size optimized for 1 year of daily data.

    Args:
        seasonal: Seasonal component series to plot.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(seasonal.index, seasonal, linewidth=0.6, alpha=0.7)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Seasonal Component", fontsize=12)
    ax.grid(alpha=0.3, linestyle="--")
    # Format x-axis for 1 year period
    ax.xaxis.set_major_locator(mdates.MonthLocator((1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()


def _validate_minimum_periods(series: pd.Series, period: int, min_periods: int = 2) -> None:
    """Validate that series has enough data for seasonal decomposition.

    Args:
        series: Time series to validate.
        period: Seasonal period.
        min_periods: Minimum number of complete periods required.

    Raises:
        ValueError: If series is too short for the requested period.
    """
    if len(series) < period * min_periods:
        msg = (
            f"Series length ({len(series)}) is insufficient for period {period}. "
            f"Need at least {period * min_periods} observations (got {len(series)})."
        )
        raise ValueError(msg)


def _plot_residuals_timeseries(
    ax: Axes,
    test_dates: pd.Index,
    residuals: np.ndarray,
) -> None:
    """Plot time series of residuals on the given axes.

    Args:
        ax: Matplotlib axes to plot on.
        test_dates: Date index for the time series.
        residuals: Residual values array.
    """
    ax.plot(test_dates, residuals, linewidth=0.8, alpha=0.7)
    ax.axhline(0, linewidth=1, linestyle="--")
    ax.fill_between(test_dates, 0, residuals, alpha=0.3)
    ax.set_title("1️⃣ Résidus dans le temps", fontsize=11, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Résidu (Réel - Prédit)", fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")


def _plot_residuals_histogram(
    ax: Axes,
    residuals: np.ndarray,
) -> None:
    """Plot histogram of residuals on the given axes.

    Args:
        ax: Matplotlib axes to plot on.
        residuals: Residual values array.
    """
    ax.hist(residuals, bins=30, alpha=0.7, edgecolor="black")
    ax.axvline(0, linewidth=2, linestyle="--", label="Moyenne théorique (0)")
    mean_residual = float(np.mean(residuals))
    ax.axvline(
        mean_residual,
        linewidth=2,
        linestyle="-",
        label=f"Moyenne réelle: {mean_residual:.6f}",
    )
    ax.set_title("2️⃣ Distribution des résidus", fontsize=11, fontweight="bold")
    ax.set_xlabel("Résidu", fontsize=10)
    ax.set_ylabel("Fréquence", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")


def _plot_residuals_qq(
    ax: Axes,
    residuals: np.ndarray,
) -> None:
    """Plot Q-Q plot for residuals normality on the given axes.

    Args:
        ax: Matplotlib axes to plot on.
        residuals: Residual values array.
    """
    scipy_stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("3️⃣ Q-Q Plot (normalité des résidus)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Quantiles théoriques (loi normale)", fontsize=10)
    ax.set_ylabel("Quantiles observés", fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")


def _plot_predictions_vs_actuals(
    ax: Axes,
    actuals: np.ndarray,
    predictions: np.ndarray,
) -> None:
    """Plot predictions vs actuals scatter plot on the given axes.

    Args:
        ax: Matplotlib axes to plot on.
        actuals: Actual values array.
        predictions: Predicted values array.
    """
    ax.scatter(actuals, predictions, alpha=0.5, s=20, edgecolors="black", linewidth=0.3)
    min_val = actuals.min()
    max_val = actuals.max()
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Ligne parfaite (y=x)",
    )
    ax.set_title("4️⃣ Prédictions vs Valeurs réelles", fontsize=11, fontweight="bold")
    ax.set_xlabel("Valeurs réelles", fontsize=10)
    ax.set_ylabel("Prédictions", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
