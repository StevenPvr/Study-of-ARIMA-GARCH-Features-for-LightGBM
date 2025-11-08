"""SARIMA model creation and fitting functions."""

from __future__ import annotations

import warnings
from typing import Any, cast

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.utils import get_logger

logger = get_logger(__name__)

# Type alias for fitted SARIMA model (SARIMAXResults)
FittedSARIMAModel = Any  # statsmodels.tsa.statespace.sarimax.SARIMAXResults


def _extract_parameters_from_tuples(
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> tuple[int, int, int, int, int, int, int]:
    """
    Extract individual parameters from tuple format.

    Args:
        order: SARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, s)

    Returns:
        Tuple of (p, d, q, P, D, Q, s)
    """
    p_val, d_val, q_val = order
    P_val, D_val, Q_val, s_val = seasonal_order
    return p_val, d_val, q_val, P_val, D_val, Q_val, s_val


def _extract_parameters_from_individuals(
    p: int,
    d: int,
    q: int,
    P: int,
    D: int,
    Q: int,
    s: int,
) -> tuple[int, int, int, int, int, int, int]:
    """
    Extract parameters from individual arguments.

    Args:
        p: AR order
        d: Differencing order
        q: MA order
        P: Seasonal AR order
        D: Seasonal differencing order
        Q: Seasonal MA order
        s: Seasonal period

    Returns:
        Tuple of (p, d, q, P, D, Q, s)
    """
    return p, d, q, P, D, Q, s


def _determine_sarima_parameters(
    order: tuple[int, int, int] | None,
    seasonal_order: tuple[int, int, int, int] | None,
    p: int | None,
    d: int | None,
    q: int | None,
    P: int | None,
    D: int | None,
    Q: int | None,
    s: int | None,
) -> tuple[int, int, int, int, int, int, int]:
    """
    Determine SARIMA parameters from either tuple or individual format.

    Args:
        order: SARIMA order (p, d, q) - alternative to p, d, q
        seasonal_order: Seasonal order (P, D, Q, s) - alternative to P, D, Q, s
        p: AR order
        d: Differencing order
        q: MA order
        P: Seasonal AR order
        D: Seasonal differencing order
        Q: Seasonal MA order
        s: Seasonal period

    Returns:
        Tuple of (p, d, q, P, D, Q, s)

    Raises:
        ValueError: If parameters are invalid or missing
    """
    if order is not None and seasonal_order is not None:
        return _extract_parameters_from_tuples(order, seasonal_order)
    if all(param is not None for param in [p, d, q, P, D, Q, s]):
        # Type checker: we've verified all params are not None
        return _extract_parameters_from_individuals(
            cast(int, p),
            cast(int, d),
            cast(int, q),
            cast(int, P),
            cast(int, D),
            cast(int, Q),
            cast(int, s),
        )
    msg = (
        "Must provide either (order, seasonal_order) tuples "
        "or all individual parameters (p, d, q, P, D, Q, s)"
    )
    raise ValueError(msg)


def _create_and_fit_model(
    train_series: pd.Series,
    p_val: int,
    d_val: int,
    q_val: int,
    P_val: int,
    D_val: int,
    Q_val: int,
    s_val: int,
    verbose: bool,
) -> FittedSARIMAModel:
    """
    Create and fit SARIMA model with given parameters.

    Args:
        train_series: Training time series data
        p_val: AR order
        d_val: Differencing order
        q_val: MA order
        P_val: Seasonal AR order
        D_val: Seasonal differencing order
        Q_val: Seasonal MA order
        s_val: Seasonal period
        verbose: Whether to show fitting progress

    Returns:
        Fitted SARIMA model

    Raises:
        RuntimeError: If model fitting fails
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            model = SARIMAX(
                train_series,
                order=(p_val, d_val, q_val),
                seasonal_order=(P_val, D_val, Q_val, s_val),
            )
            fitted_model = model.fit(verbose=verbose)
            # Use getattr for type safety (aic may not be recognized by type checker)
            aic_value = getattr(fitted_model, "aic", None)
            if aic_value is not None:
                logger.debug(f"Model fitted successfully - AIC: {aic_value:.2f}")
            return fitted_model
        except Exception as e:
            msg = (
                f"Failed to fit SARIMA({p_val},{d_val},{q_val})"
                f"({P_val},{D_val},{Q_val})[{s_val}] model: {e}"
            )
            logger.error(msg)
            raise RuntimeError(msg) from e


def fit_sarima_model(
    train_series: pd.Series,
    order: tuple[int, int, int] | None = None,
    seasonal_order: tuple[int, int, int, int] | None = None,
    p: int | None = None,
    d: int | None = None,
    q: int | None = None,
    P: int | None = None,
    D: int | None = None,
    Q: int | None = None,
    s: int | None = None,
    verbose: bool = False,
) -> FittedSARIMAModel:
    """
    Fit a SARIMA model with given parameters.

    Can accept parameters either as tuples (order, seasonal_order) or as individual
    parameters (p, d, q, P, D, Q, s).

    Args:
        train_series: Training time series data
        order: SARIMA order (p, d, q) - alternative to p, d, q
        seasonal_order: Seasonal order (P, D, Q, s) - alternative to P, D, Q, s
        p: AR order
        d: Differencing order
        q: MA order
        P: Seasonal AR order
        D: Seasonal differencing order
        Q: Seasonal MA order
        s: Seasonal period
        verbose: Whether to show fitting progress

    Returns:
        Fitted SARIMA model (SARIMAXResults)

    Raises:
        ValueError: If parameters are invalid or missing
        RuntimeError: If model fitting fails
    """
    p_val, d_val, q_val, P_val, D_val, Q_val, s_val = _determine_sarima_parameters(
        order, seasonal_order, p, d, q, P, D, Q, s
    )

    logger.debug(
        f"Fitting SARIMA({p_val},{d_val},{q_val})({P_val},{D_val},{Q_val})[{s_val}] "
        f"on {len(train_series)} observations"
    )

    return _create_and_fit_model(
        train_series, p_val, d_val, q_val, P_val, D_val, Q_val, s_val, verbose
    )

