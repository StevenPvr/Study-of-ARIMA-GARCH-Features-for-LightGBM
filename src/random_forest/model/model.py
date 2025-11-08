"""Random Forest model for return prediction."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from src.constants import DEFAULT_RANDOM_STATE


def create_model() -> RandomForestRegressor:
    """Create a Random Forest Regressor model with default parameters.

    Returns:
        RandomForestRegressor instance with default parameters.
    """
    return RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE)
