from __future__ import annotations

__doc__ = """Validation helpers for ARIMA optimization."""

import pandas as pd

from src.constants import ARIMA_BACKTEST_MIN_TRAIN_MARGIN
from src.utils import validate_required_columns


def validate_split_data_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Validate that a DataFrame contains all required columns.

    Delegates to src.utils.validate_required_columns() for consistency.

    Args:
        df: DataFrame to validate.
        required: List of column names that must be present.

    Raises:
        ValueError: If any required columns are missing.
    """
    validate_required_columns(df, required, df_name="Split data")


def validate_series(name: str, s: pd.Series) -> None:
    """Validate that a pandas Series meets requirements for ARIMA modeling.

    Args:
        name: Name of the series (used in error messages).
        s: Series to validate.

    Raises:
        ValueError: If series is empty or contains NaN values.
        TypeError: If series is not numeric.
    """
    if s.empty:
        raise ValueError(f"{name} is empty.")
    if not pd.api.types.is_numeric_dtype(s):
        raise TypeError(f"{name} must be numeric.")
    if s.isna().any():
        raise ValueError(f"{name} contains NaNs; fill or drop them before calling.")






def validate_arima_params(p: int, d: int, q: int) -> None:
    """Validate ARIMA parameter values.

    Args:
        p: AR order (non-negative integer).
        d: Differencing order (non-negative integer).
        q: MA order (non-negative integer).

    Raises:
        ValueError: If any parameter is negative.
    """
    param_names = ("p", "d", "q")
    param_values = (p, d, q)

    for name, value in zip(param_names, param_values, strict=False):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Parameter {name} must be a non-negative integer.")


def compute_test_size_from_ratio(train_len: int, test_size_ratio: float, n_splits: int) -> int:
    """Compute absolute test_size from ratio for walk-forward cross-validation.

    Calculates the test window size as a percentage of the total training data,
    divided equally across n_splits. For example, with 1000 observations, 5 splits,
    and a 20% ratio, each split will use 40 observations (1000 * 0.2 / 5 = 40).

    Args:
        train_len: Length of training series.
        test_size_ratio: Ratio of training data to use for validation (0 < ratio < 1).
        n_splits: Number of time splits for backtesting.

    Returns:
        Absolute test_size (number of observations per split).

    Raises:
        ValueError: If test_size_ratio is not in (0, 1) or if computed test_size < 1.
    """
    if not 0 < test_size_ratio < 1:
        msg = f"test_size_ratio must be in (0, 1), got {test_size_ratio}"
        raise ValueError(msg)
    if n_splits < 1:
        msg = f"n_splits must be >= 1, got {n_splits}"
        raise ValueError(msg)

    # Compute absolute test_size: total_test_observations / n_splits
    total_test_size = int(train_len * test_size_ratio)
    test_size = total_test_size // n_splits

    if test_size < 1:
        msg = (
            f"Computed test_size={test_size} is too small. "
            f"With train_len={train_len}, test_size_ratio={test_size_ratio}, "
            f"and n_splits={n_splits}, need larger dataset or smaller n_splits."
        )
        raise ValueError(msg)

    return test_size


def validate_backtest_config(
    n_splits: int, test_size: int, refit_every: int, train_len: int
) -> None:
    """Validate walk-forward backtest configuration parameters.

    Args:
        n_splits: Number of time splits for backtesting (must be >= 1).
        test_size: Size of each test set (must be >= 1).
        refit_every: Frequency of model refitting (must be >= 1).
        train_len: Length of training series.

    Raises:
        ValueError: If any parameter is invalid or series is too short for backtest.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1.")
    if test_size < 1:
        raise ValueError("test_size must be >= 1.")
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1.")
    min_margin = max(ARIMA_BACKTEST_MIN_TRAIN_MARGIN, test_size)
    min_len = n_splits * test_size + min_margin
    if train_len < min_len:
        raise ValueError(f"Series too short for backtest: need >= {min_len}, got {train_len}.")
