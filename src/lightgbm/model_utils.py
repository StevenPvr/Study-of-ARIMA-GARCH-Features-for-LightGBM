"""Common utilities for LightGBM model management."""

from __future__ import annotations

from pathlib import Path

from src.constants import (
    LIGHTGBM_DATASET_INSIGHTS_ONLY_FILE,
    LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE,
    LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE,
    LIGHTGBM_DATASET_TECHNICAL_ONLY_NO_TARGET_LAGS_FILE,
    LIGHTGBM_DATASET_TECHNICAL_PLUS_INSIGHTS_NO_TARGET_LAGS_FILE,
    LIGHTGBM_MODELS_DIR,
)


def get_optional_model_configs() -> list[tuple[Path, Path, str]]:
    """Get list of optional model configurations (included if files exist).

    Returns:
        List of tuples (dataset_path, model_path, model_name).
    """
    return [
        (
            LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE,
            LIGHTGBM_MODELS_DIR / "lightgbm_sigma_plus_base.joblib",
            "lightgbm_sigma_plus_base",
        ),
        (
            LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE,
            LIGHTGBM_MODELS_DIR / "lightgbm_log_volatility_only.joblib",
            "lightgbm_log_volatility_only",
        ),
        (
            LIGHTGBM_DATASET_INSIGHTS_ONLY_FILE,
            LIGHTGBM_MODELS_DIR / "lightgbm_insights_only.joblib",
            "lightgbm_insights_only",
        ),
        (
            LIGHTGBM_DATASET_TECHNICAL_ONLY_NO_TARGET_LAGS_FILE.with_suffix(".parquet"),
            LIGHTGBM_MODELS_DIR / "lightgbm_technical_only.joblib",
            "lightgbm_technical_only",
        ),
        (
            LIGHTGBM_DATASET_TECHNICAL_PLUS_INSIGHTS_NO_TARGET_LAGS_FILE.with_suffix(".parquet"),
            LIGHTGBM_MODELS_DIR / "lightgbm_technical_plus_insights.joblib",
            "lightgbm_technical_plus_insights",
        ),
    ]


def filter_existing_models(
    model_configs: list[tuple[Path, Path, str]],
) -> list[tuple[Path, Path, str]]:
    """Filter model configurations to only include those where both dataset and model files exist.

    Args:
        model_configs: List of (dataset_path, model_path, model_name) tuples.

    Returns:
        Filtered list of existing model configurations.
    """
    return [
        (dataset_path, model_path, model_name)
        for dataset_path, model_path, model_name in model_configs
        if dataset_path.exists() and model_path.exists()
    ]


__all__ = ["get_optional_model_configs", "filter_existing_models"]
