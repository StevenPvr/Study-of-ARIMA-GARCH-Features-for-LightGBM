"""Utility functions for LightGBM data preparation.

This module re-exports functions from specialized modules to maintain backward compatibility.
"""

from __future__ import annotations

# For backward compatibility, also export internal functions that might be used elsewhere
from src.lightgbm.data_preparation.columns import (
    get_non_observable_columns_to_drop,
    get_technical_indicator_columns,
    remove_missing_values,
)

# Re-export all public functions from specialized modules
from src.lightgbm.data_preparation.data_loading import (
    ensure_ticker_id_column,
    load_ticker_data_with_fallback,
)
from src.lightgbm.data_preparation.dataset_builders import (
    create_dataset_log_volatility_only,
    create_dataset_technical_indicators,
    ensure_log_volatility_only_dataset,
    ensure_technical_indicators_dataset,
)
from src.lightgbm.data_preparation.dataset_creation import prepare_datasets

# create_dataset_without_sigma2 is defined in dataset_variants
from src.lightgbm.data_preparation.dataset_variants import (
    create_dataset_log_volatility_only_from_clean,
    create_dataset_sigma_plus_base_from_clean,
    create_dataset_without_insights,
    create_dataset_without_insights_from_file,
    create_dataset_without_sigma2,
    ensure_dataset_without_insights,
    ensure_sigma_plus_base_dataset,
)
from src.lightgbm.data_preparation.features import add_lag_features, get_base_columns
from src.lightgbm.data_preparation.target_creation import (
    create_target_columns,
    get_target_column_name,
    normalize_column_names,
)
from src.lightgbm.data_preparation.ticker_processing import (
    add_indicators_to_ticker_parquet,
    process_all_ticker_parquets,
)

__all__ = [
    # Data loading
    "load_ticker_data_with_fallback",
    "ensure_ticker_id_column",
    # Lag features
    "add_lag_features",
    "get_base_columns",
    # Target creation
    "create_target_columns",
    "get_target_column_name",
    "normalize_column_names",
    # Dataset creation
    "prepare_datasets",
    "create_dataset_without_sigma2",
    "create_dataset_log_volatility_only",
    "create_dataset_technical_indicators",
    "ensure_sigma_plus_base_dataset",
    "ensure_log_volatility_only_dataset",
    "ensure_technical_indicators_dataset",
    # Ticker processing
    "add_indicators_to_ticker_parquet",
    "process_all_ticker_parquets",
    # Column selection (for internal use)
    "get_non_observable_columns_to_drop",
    "get_technical_indicator_columns",
    "remove_missing_values",
    # Dataset variants (for internal use)
    "create_dataset_log_volatility_only_from_clean",
    "create_dataset_sigma_plus_base_from_clean",
    "create_dataset_without_insights",
    "create_dataset_without_insights_from_file",
    "ensure_dataset_without_insights",
]
