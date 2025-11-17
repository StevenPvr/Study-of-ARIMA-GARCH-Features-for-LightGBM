"""Evaluation helpers for ARIMA (keeps external `src.*` imports intact)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.constants import DEFAULT_PLACEHOLDER_DATE  # type: ignore
from src.utils import get_logger  # type: ignore

logger = get_logger(__name__)


def detect_value_column(df: pd.DataFrame) -> str:
    """Detect the value column name from DataFrame.

    Args:
        df: DataFrame with data columns.

    Returns:
        Column name for target values.

    Raises:
        ValueError: If no suitable column found.
    """
    candidates = [
        c for c in df.columns if c.lower() in {"y", "target", "log_return", "weighted_log_return"}
    ]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if "return" in c.lower() and c not in {"date", "split"}:
            return c
    msg = (
        "Could not identify the returns column "
        "(expected one of: y, target, log_return, weighted_log_return)."
    )
    raise ValueError(msg)


def is_refit_step(step_index: int, refit_every: int) -> bool:
    """Return True if model should be refit at this step (0-based)."""
    if refit_every <= 0:
        return step_index == 0
    return step_index == 0 or (step_index % refit_every == 0)


def add_point_to_series(current_train: pd.Series, value: float, date: Any) -> pd.Series:
    """Append a single (date, value) to a Series."""
    if not isinstance(current_train, pd.Series):
        raise TypeError("current_train must be a pandas Series")
    s = current_train.copy()
    new_point = pd.Series([float(value)], index=[date], name=s.name)
    return pd.concat([s, new_point])


def to_date_index(s: pd.Series, start: str = DEFAULT_PLACEHOLDER_DATE) -> pd.Series:
    """Ensure Series has a datetime index; if not, fabricate a daily index."""
    if isinstance(s.index, pd.DatetimeIndex):
        return s
    idx = pd.date_range(pd.Timestamp(start), periods=len(s), freq="D")
    s2 = s.copy()
    s2.index = idx
    return s2
