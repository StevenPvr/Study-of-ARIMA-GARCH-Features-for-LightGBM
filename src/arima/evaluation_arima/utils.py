"""Utility functions for SARIMA model evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from src.constants import (
    DATE_FORMAT_DEFAULT,
    PLACEHOLDER_DATE_PREFIX,
    PLOT_FIGURE_SIZE_DEFAULT,
)
from src.utils import get_logger

logger = get_logger(__name__)


def _extract_forecast_value(fc: Any) -> float:
    """
    Extract forecast value from forecast object.

    Args:
        fc: Forecast object from SARIMA model

    Returns:
        Forecast value as float
    """
    try:
        return float(np.asarray(fc)[0])
    except (ValueError, TypeError, IndexError, AttributeError):
        if hasattr(fc, "iloc"):
            return float(fc.iloc[0])
        return float(next(iter(fc)))


def _predict_single_step(
    current_train: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    fitted_model: Any | None = None,
) -> tuple[float | None, Any]:
    """
    Predict the next value using a SARIMA model fit on a supported index.

    To avoid statsmodels FutureWarning about unsupported indices during
    forecasting, the model is fit on a RangeIndex copy of the data. The
    numeric forecast is identical; only the internal prediction index differs.

    Args:
        current_train: Current training series.
        order: SARIMA order (p, d, q).
        seasonal_order: Seasonal order (P, D, Q, s).
        fitted_model: Optional pre-fitted model. If provided and data hasn't changed
            significantly, will use apply() to update with new data. If None, will fit new model.

    Returns:
        Tuple of (prediction value, fitted_model).
    """
    from src.arima.models.sarima_model import fit_sarima_model

    # Preserve date index externally, but fit on supported index
    _ = _ensure_datetime_index(current_train)
    train_for_model = _ensure_supported_forecast_index(current_train)

    # If we have a fitted model, try to append new data (faster than refit)
    if fitted_model is not None:
        try:
            # Append new observations to existing model without refitting parameters
            # This is more efficient than full refit when data continues the series
            updated_model = fitted_model.append(train_for_model, refit=False)
            fc = updated_model.forecast(steps=1)
            return _extract_forecast_value(fc), updated_model
        except (ValueError, RuntimeError, AttributeError, TypeError):
            # If append fails (e.g., index mismatch), fall back to full refit
            logger.debug("Model append failed, falling back to full refit")

    # Full refit (either no model provided or apply failed)
    fitted_model = fit_sarima_model(
        train_for_model, order=order, seasonal_order=seasonal_order, verbose=False
    )
    fc = fitted_model.forecast(steps=1)
    return _extract_forecast_value(fc), fitted_model


def _validate_series_not_empty(series: pd.Series, name: str) -> None:
    """Validate that a series is not empty.

    Args:
        series: Series to validate
        name: Name of the series for error message

    Raises:
        ValueError: If series is empty
    """
    if len(series) == 0:
        raise ValueError(f"{name} must be non-empty")


def _validate_order_non_negative(order: tuple[int, int, int]) -> None:
    """Validate that SARIMA order values are non-negative.

    Args:
        order: SARIMA order (p, d, q)

    Raises:
        ValueError: If order contains negative values
    """
    if any(x < 0 for x in order):
        raise ValueError(f"SARIMA order must be non-negative: {order}")


def _validate_seasonal_order_valid(seasonal_order: tuple[int, int, int, int]) -> None:
    """Validate that seasonal order values are valid.

    Args:
        seasonal_order: Seasonal order (P, D, Q, s)

    Raises:
        ValueError: If seasonal order is invalid
    """
    if any(x < 0 for x in seasonal_order[:3]) or seasonal_order[3] <= 0:
        raise ValueError(f"Seasonal order invalid: {seasonal_order}")


def _validate_rolling_forecast_inputs(
    train_series: pd.Series,
    test_series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> None:
    """Validate inputs for rolling forecast.

    Args:
        train_series: Initial training series
        test_series: Test series to forecast
        order: SARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, s)

    Raises:
        ValueError: If inputs are invalid
    """
    _validate_series_not_empty(train_series, "train_series")
    _validate_series_not_empty(test_series, "test_series")
    _validate_order_non_negative(order)
    _validate_seasonal_order_valid(seasonal_order)


def _ensure_datetime_index(series: pd.Series) -> pd.Series:
    """Ensure series has a DatetimeIndex (no freq enforcement).

    We keep date information for downstream consumers (plots, saved CSVs),
    but do not enforce a frequency here because trading calendars are often
    irregular. Forecasting will use a supported index variant separately.

    Args:
        series: Input series

    Returns:
        Series with DatetimeIndex
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        series = series.copy()
        series.index = pd.to_datetime(series.index)
    return series


def _ensure_supported_forecast_index(series: pd.Series) -> pd.Series:
    """Return a copy with an index supported by statsmodels forecasting.

    Why: statsmodels raises a FutureWarning when no supported index is
    available during prediction. A simple and robust fix is to fit the
    SARIMA model on a RangeIndex copy so prediction index construction
    is always supported, without affecting forecast values.

    Args:
        series: Input series (any index).

    Returns:
        Series copy indexed by RangeIndex(0, len(series)).
    """
    return pd.Series(series.values.copy(), index=pd.RangeIndex(len(series)))


def _add_point_to_series(series: pd.Series, value: float, date: Any) -> pd.Series:
    """Add a point to a series while maintaining DatetimeIndex.

    Args:
        series: Current series
        value: Value to add
        date: Date for the new point

    Returns:
        Series with added point and DatetimeIndex
    """
    series = _ensure_datetime_index(series)
    # Convert date to Timestamp - handle both single values and index-like objects
    if isinstance(date, pd.Timestamp):
        date_ts = date
    else:
        # Use pd.to_datetime with a list to handle type checking
        date_ts = pd.to_datetime([date])[0]

    new_point = pd.Series([value], index=pd.DatetimeIndex([date_ts]))
    result = pd.concat([series, new_point])
    return _ensure_datetime_index(result)


def _format_datetime_index(index: pd.DatetimeIndex) -> list[str]:
    """Format DatetimeIndex to date strings.

    Args:
        index: DatetimeIndex to format

    Returns:
        List of formatted date strings
    """
    return [d.strftime(DATE_FORMAT_DEFAULT) for d in index]


def _format_non_datetime_index(index: pd.Index, predictions_length: int) -> list[str]:
    """Format non-datetime index to date strings.

    Args:
        index: Index to format
        predictions_length: Length of predictions array

    Returns:
        List of formatted date strings
    """
    try:
        dates = pd.to_datetime(index)
        return [d.strftime(DATE_FORMAT_DEFAULT) for d in dates]
    except (ValueError, TypeError):
        return [str(d) for d in index]


def _format_dates_from_index(test_series: pd.Series, predictions_length: int) -> list[str]:
    """Format dates from test series index.

    Args:
        test_series: Test series with index to format
        predictions_length: Length of predictions array

    Returns:
        List of formatted date strings
    """
    if len(test_series.index) == 0:
        return [f"{PLACEHOLDER_DATE_PREFIX}{i}" for i in range(predictions_length)]

    if isinstance(test_series.index, pd.DatetimeIndex):
        return _format_datetime_index(test_series.index)

    return _format_non_datetime_index(test_series.index, predictions_length)


def _prepare_lags_list(lags: int | Iterable[int]) -> list[int]:
    """Prepare list of lags for Ljung-Box test.

    Args:
        lags: Single lag or an iterable of lags

    Returns:
        Sorted list of positive integer lags
    """
    if isinstance(lags, int):
        return list(range(1, int(lags) + 1))
    return sorted(int(x) for x in lags if int(x) > 0)


def _get_series_from_dataframe(df: Any, key: str) -> Any:
    """Get series from DataFrame if available.

    Args:
        df: DataFrame object
        key: Key to retrieve

    Returns:
        Series if available, None otherwise
    """
    if hasattr(df, "get"):
        return df.get(key)
    return None


def _extract_ljungbox_from_dataframe(
    df: Any, lags_list: list[int]
) -> tuple[list[float], list[float]]:
    """Extract Q-statistics and p-values from DataFrame result.

    Args:
        df: DataFrame from acorr_ljungbox with return_df=True
        lags_list: List of lags used

    Returns:
        Tuple of (q_list, p_list)
    """

    q_series = _get_series_from_dataframe(df, "lb_stat")
    p_series = _get_series_from_dataframe(df, "lb_pvalue")

    if q_series is not None and p_series is not None:
        q_list = [float(x) for x in np.ravel(q_series.values)]
        p_list = [float(x) for x in np.ravel(p_series.values)]
        return q_list, p_list

    # Unexpected API shape -> raise to stop pipeline
    raise ValueError("Unexpected acorr_ljungbox(return_df=True) result format")


def _compute_ljungbox_fallback(
    res: np.ndarray, lags_list: list[int]
) -> tuple[list[float], list[float]]:
    """Compute Ljung-Box using fallback method (return_df=False).

    Args:
        res: Cleaned residuals array
        lags_list: List of lags to use

    Returns:
        Tuple of (q_list, p_list)
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox  # type: ignore

    q, p = acorr_ljungbox(res, lags=lags_list, return_df=False)
    q_list = [float(x) for x in np.ravel(q)]
    p_list = [float(x) for x in np.ravel(p)]
    return q_list, p_list


def _create_acf_figure() -> tuple[Any, Any, Any]:
    """Create matplotlib figure and axis for ACF plot.

    Returns:
        Tuple of (fig, canvas, ax)
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # type: ignore
    from matplotlib.figure import Figure  # type: ignore

    fig = Figure(figsize=PLOT_FIGURE_SIZE_DEFAULT, constrained_layout=True)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)
    return fig, canvas, ax


def _plot_acf_on_axis(res: np.ndarray, lags: int, ax: Any) -> None:
    """Plot ACF on given axis.

    Args:
        res: Cleaned residuals array
        lags: Number of lags for ACF
        ax: Matplotlib axis
    """
    from statsmodels.graphics.tsaplots import plot_acf  # type: ignore

    if res.size > 0:
        plot_acf(res, lags=int(lags), ax=ax)
    ax.set_title("SARIMA Residuals ACF with Ljung–Box Test")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")


def _add_ljungbox_summary_to_plot(
    res: np.ndarray, lags: int, ax: Any, ljung_box_result: dict[str, Any] | None = None
) -> None:
    """Add Ljung–Box test summary text to plot.

    Args:
        res: Cleaned residuals array
        lags: Number of lags used
        ax: Matplotlib axis
        ljung_box_result: Optional pre-computed Ljung-Box result dict. If None, will be computed.
    """
    try:
        if ljung_box_result is None:
            from src.arima.evaluation_arima.evaluation_arima import ljung_box_on_residuals

            lb = ljung_box_on_residuals(res, lags=int(lags))
        else:
            lb = ljung_box_result

        if lb["lags"]:
            txt = f"Q({lb['lags'][-1]})={lb['q_stat'][-1]:.3f}, p={lb['p_value'][-1]:.3g}"
            ax.text(
                0.02,
                0.95,
                txt,
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox={"boxstyle": "round", "facecolor": "#f0f0f0", "alpha": 0.8},
                fontsize=9,
            )
    except (ValueError, RuntimeError, AttributeError) as e:
        logger.debug(f"Could not add Ljung–Box summary to plot: {e}")
