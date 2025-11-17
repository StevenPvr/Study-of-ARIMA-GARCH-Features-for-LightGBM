"""LightGBM model for return prediction."""

from __future__ import annotations

import lightgbm as lgb
from src.constants import DEFAULT_RANDOM_STATE


def create_model() -> lgb.LGBMRegressor:
    """Create a LightGBM Regressor model with default parameters.

    Returns:
        LGBMRegressor instance with default parameters.
    """
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        random_state=DEFAULT_RANDOM_STATE,
        verbosity=-1,
    )
