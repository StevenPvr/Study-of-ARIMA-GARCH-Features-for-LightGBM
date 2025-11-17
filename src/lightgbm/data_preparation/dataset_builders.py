"""Individual dataset builder functions (thin wrappers).

These builders reuse the consolidated pipeline in `dataset_creation`
to avoid duplication: normalize → features+lags → clean → select columns → save.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.constants import (
    LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE,
    LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE,
    LIGHTGBM_DATASET_TECHNICAL_INDICATORS_FILE,
)
from src.lightgbm.data_preparation.columns import (
    get_technical_indicator_columns,
    select_log_volatility_only_columns,
    select_sigma_plus_base_columns,
    select_technical_indicators_dataset_columns,
)
from src.lightgbm.data_preparation.dataset_creation import (
    _add_features_and_lags,
    _prepare_clean_base_df,
    prepare_datasets,
)
from src.lightgbm.data_preparation.dataset_pipeline import (
    drop_rows_with_missing_features,
    prepare_base_dataframe,
)
from src.lightgbm.data_preparation.features import get_base_columns
from src.utils import get_logger, save_parquet_and_csv

logger = get_logger(__name__)


def _build_clean_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Run the unified feature+lags pipeline and cleaning steps."""
    df_with_features = _add_features_and_lags(df)
    df_clean = _prepare_clean_base_df(df_with_features)
    return df_clean


def create_dataset_sigma_plus_base(
    df: pd.DataFrame | None = None,
    output_path: Path | None = None,
    *,
    include_lags: bool = True,
) -> pd.DataFrame:
    """Create dataset combining GARCH insights with base features (log_volatility).

    Args:
        df: Ticker data DataFrame. If None, loads from ticker-level data.
        output_path: Path to save dataset. If None, uses LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE.
        include_lags: If True, include lag features.

    Returns:
        Dataset with sigma insights + base features.
    """
    df_prepared = prepare_base_dataframe(df, prefer_insights=False)
    df_clean = _build_clean_feature_frame(df_prepared)

    keep_cols = select_sigma_plus_base_columns(df_clean, include_lags)
    keep_cols = [c for c in keep_cols if c in df_clean.columns]
    df_selected = drop_rows_with_missing_features(df_clean, keep_cols)

    if output_path is None:
        output_path = LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE.with_suffix(".parquet")

    if "ticker" in df_selected.columns and "tickers" not in df_selected.columns:
        df_selected = df_selected.rename(columns={"ticker": "tickers"})
    save_parquet_and_csv(df_selected, output_path)
    logger.info(
        "Saved sigma-plus-base dataset: %s (%d rows, %d columns)",
        output_path,
        len(df_selected),
        len(df_selected.columns),
    )

    return df_selected


def create_dataset_log_volatility_only(
    df: pd.DataFrame | None = None,
    output_path: Path | None = None,
    *,
    include_lags: bool = True,
) -> pd.DataFrame:
    """Create dataset with only base features (log_volatility) and its lags.

    Args:
        df: Ticker data DataFrame. If None, loads from ticker-level data.
        output_path: Path to save dataset. If None, uses LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE.
        include_lags: If True, include lag features.

    Returns:
        Dataset with only base features.
    """
    df_prepared = prepare_base_dataframe(df, prefer_insights=False)
    df_clean = _build_clean_feature_frame(df_prepared)

    keep_cols = select_log_volatility_only_columns(df_clean, include_lags)
    keep_cols = [c for c in keep_cols if c in df_clean.columns]
    df_base = drop_rows_with_missing_features(df_clean, keep_cols)

    if output_path is None:
        output_path = LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE.with_suffix(".parquet")

    if "ticker" in df_base.columns and "tickers" not in df_base.columns:
        df_base = df_base.rename(columns={"ticker": "tickers"})
    save_parquet_and_csv(df_base, output_path)
    logger.info(
        "Saved base features dataset: %s (%d rows, %d columns)",
        output_path,
        len(df_base),
        len(df_base.columns),
    )

    return df_base


def create_dataset_technical_indicators(
    df: pd.DataFrame | None = None,
    output_path: Path | None = None,
    *,
    include_lags: bool = True,
) -> pd.DataFrame:
    """Create dataset with technical indicators and their lags.

    Args:
        df: Ticker data DataFrame. If None, loads from ticker-level data.
        output_path: Path to save dataset. If None, uses LIGHTGBM_DATASET_TECHNICAL_INDICATORS_FILE.
        include_lags: If True, include lag features.

    Returns:
        Dataset with technical indicators.
    """
    df_prepared = prepare_base_dataframe(df, prefer_insights=False)
    df_clean = _build_clean_feature_frame(df_prepared)

    keep_cols = select_technical_indicators_dataset_columns(df_clean, include_lags)
    keep_cols = list(dict.fromkeys(keep_cols))
    df_selected = drop_rows_with_missing_features(df_clean, keep_cols)

    if output_path is None:
        output_path = LIGHTGBM_DATASET_TECHNICAL_INDICATORS_FILE.with_suffix(".parquet")

    if "ticker" in df_selected.columns and "tickers" not in df_selected.columns:
        df_selected = df_selected.rename(columns={"ticker": "tickers"})
    save_parquet_and_csv(df_selected, output_path)
    logger.info(
        "Saved technical-indicators dataset: %s (%d rows, %d columns)",
        output_path,
        len(df_selected),
        len(df_selected.columns),
    )

    return df_selected


def ensure_log_volatility_only_dataset(*, include_lags: bool = True) -> Path:
    """Ensure the log-volatility-only dataset exists; create it if missing.

    If the dataset doesn't exist, calls prepare_datasets() to create all datasets
    at once, ensuring they all have the same number of rows.

    Args:
        include_lags: If True, include lag features when available.

    Returns:
        Path to the log-volatility-only dataset.
    """
    dataset_path = LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE.with_suffix(".parquet")

    # Check if complete dataset exists - if not, create all datasets at once
    from src.constants import LIGHTGBM_DATASET_COMPLETE_FILE

    complete_path = LIGHTGBM_DATASET_COMPLETE_FILE.with_suffix(".parquet")
    if not complete_path.exists():
        logger.info(
            "Complete dataset not found. Creating all datasets at once to ensure consistency."
        )
        prepare_datasets()
        return dataset_path

    if dataset_path.exists():
        if include_lags:
            df_existing = pd.read_parquet(dataset_path, engine="pyarrow")
            expected_cols = get_base_columns(include_lags=True)
            missing_cols = [col for col in expected_cols if col not in df_existing.columns]
            if missing_cols:
                logger.info(
                    "Log-volatility-only dataset missing columns %s. Recreating all datasets.",
                    missing_cols,
                )
                prepare_datasets()
        return dataset_path

    # Dataset doesn't exist - create all datasets at once for consistency
    logger.info("Log-volatility-only dataset not found. Creating all datasets at once.")
    prepare_datasets()
    return dataset_path


def ensure_technical_indicators_dataset(*, include_lags: bool = True) -> Path:
    """Ensure the technical indicators dataset exists; create it if missing.

    If the dataset doesn't exist, calls prepare_datasets() to create all datasets
    at once, ensuring they all have the same number of rows.

    Args:
        include_lags: If True, include lag features when available.

    Returns:
        Path to the technical indicators dataset.
    """
    dataset_path = LIGHTGBM_DATASET_TECHNICAL_INDICATORS_FILE.with_suffix(".parquet")

    # Check if complete dataset exists - if not, create all datasets at once
    from src.constants import LIGHTGBM_DATASET_COMPLETE_FILE

    complete_path = LIGHTGBM_DATASET_COMPLETE_FILE.with_suffix(".parquet")
    if not complete_path.exists():
        logger.info(
            "Complete dataset not found. Creating all datasets at once to ensure consistency."
        )
        prepare_datasets()
        return dataset_path

    if dataset_path.exists():
        if include_lags:
            df_existing = pd.read_parquet(dataset_path, engine="pyarrow")
            expected_cols = get_technical_indicator_columns(include_lags=True)
            missing_cols = [col for col in expected_cols if col not in df_existing.columns]
            if missing_cols:
                logger.info(
                    "Technical indicators dataset missing columns %s. Recreating all datasets.",
                    missing_cols,
                )
                prepare_datasets()
        return dataset_path

    # Dataset doesn't exist - create all datasets at once for consistency
    logger.info("Technical indicators dataset not found. Creating all datasets at once.")
    prepare_datasets()
    return dataset_path
