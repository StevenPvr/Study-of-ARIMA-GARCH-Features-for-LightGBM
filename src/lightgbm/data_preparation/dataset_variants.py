"""Dataset variant creation functions (sigma-plus-base, log-volatility-only, etc.)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd


def _select_and_filter_columns(
    df_clean: pd.DataFrame, column_selectors: list[list[str]]
) -> pd.DataFrame:
    """Select and filter columns from a clean DataFrame using multiple selectors.

    Args:
        df_clean: Already prepared and cleaned DataFrame.
        column_selectors: List of column selector functions results.

    Returns:
        DataFrame with selected columns only.
    """
    keep_cols = []
    for selector_result in column_selectors:
        keep_cols.extend(selector_result)

    # Remove duplicates and select columns
    keep_cols = list(dict.fromkeys(keep_cols))
    keep_cols = [c for c in keep_cols if c in df_clean.columns]

    return cast(pd.DataFrame, df_clean[keep_cols].copy())


from src.constants import (
    LIGHTGBM_ARIMA_GARCH_INSIGHT_COLUMNS,
    LIGHTGBM_DATASET_COMPLETE_FILE,
    LIGHTGBM_DATASET_INSIGHTS_ONLY_FILE,
    LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE,
    LIGHTGBM_LAG_WINDOWS,
)
from src.lightgbm.data_preparation.columns import (
    select_base_feature_columns,
    select_close_column,
    select_garch_insight_columns,
    select_metadata_columns,
    select_target_columns,
    select_technical_indicator_columns,
)
from src.lightgbm.data_preparation.target_creation import get_target_column_name
from src.utils import get_logger

logger = get_logger(__name__)


def create_dataset_sigma_plus_base_from_clean(
    df_clean: pd.DataFrame, *, include_lags: bool = True
) -> pd.DataFrame:
    """Create sigma-plus-base dataset from already prepared and cleaned DataFrame.

    Selects only GARCH insights (sigma2, sigma, std_resid, arima_residual) +
    base features (log_volatility).
    All datasets created from the same df_clean will have the same number of rows.

    Args:
        df_clean: Already prepared and cleaned DataFrame with all columns and no missing values.
        include_lags: If True, include lag features.

    Returns:
        Dataset with only sigma insights + base features (log_volatility).
    """
    return _select_and_filter_columns(
        df_clean,
        [
            select_metadata_columns(df_clean),
            select_target_columns(df_clean, include_lags),
            select_garch_insight_columns(df_clean, include_lags),
            select_base_feature_columns(df_clean, include_lags),
        ],
    )


def create_dataset_log_volatility_only_from_clean(
    df_clean: pd.DataFrame, *, include_lags: bool = True
) -> pd.DataFrame:
    """Create base features only dataset from already prepared and cleaned DataFrame.

    Selects only base features (log_volatility, no insights, no other technical indicators).
    All datasets created from the same df_clean will have the same number of rows.

    Args:
        df_clean: Already prepared and cleaned DataFrame with all columns and no missing values.
        include_lags: If True, include lag features.

    Returns:
        Dataset with only base features (log_volatility).
    """
    return _select_and_filter_columns(
        df_clean,
        [
            select_metadata_columns(df_clean),
            # No target lags for log-volatility-only
            select_target_columns(df_clean, include_lags=False),
            select_base_feature_columns(df_clean, include_lags),
        ],
    )


def create_dataset_without_insights(df: pd.DataFrame) -> pd.DataFrame:
    """Create dataset without ARIMA-GARCH insight columns.

    Keeps ALL technical indicators but removes GARCH insights.

    Args:
        df: Complete dataset with all columns.

    Returns:
        Dataset without ARIMA-GARCH insight columns but WITH all technical indicators.
    """
    from src.lightgbm.data_preparation.columns import get_insight_columns_to_drop

    logger.info("Creating dataset without ARIMA-GARCH insights (keeping all technical indicators)")
    columns_to_drop = get_insight_columns_to_drop(df)
    if columns_to_drop:
        logger.info(f"Dropping {len(columns_to_drop)} insight columns: {columns_to_drop[:5]}...")
    return df.drop(columns=columns_to_drop).copy()


def create_dataset_without_insights_from_file(
    input_file: Path | None = None,
    output_file: Path | None = None,
) -> pd.DataFrame:
    """Create dataset without ARIMA-GARCH insights from complete dataset file.

    Why: This dataset evaluates the predictive value of technical indicators
    in isolation (no ARIMA/GARCH insights).

    Args:
        input_file: Complete dataset parquet path (optional; default from constants).
        output_file: Destination parquet path (optional; default from constants).

    Returns:
        The filtered DataFrame that was saved to disk.
    """
    from src.constants import LIGHTGBM_DATASET_COMPLETE_FILE, LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE

    if input_file is None:
        input_file = LIGHTGBM_DATASET_COMPLETE_FILE.with_suffix(".parquet")
    if output_file is None:
        output_file = LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df_complete = pd.read_parquet(input_file, engine="pyarrow")

    # Create dataset without insights
    df_without_insights = create_dataset_without_insights(df_complete)

    # Save both parquet and CSV
    from src.utils import save_parquet_and_csv

    if "ticker" in df_without_insights.columns and "tickers" not in df_without_insights.columns:
        df_without_insights = df_without_insights.rename(columns={"ticker": "tickers"})
    save_parquet_and_csv(df_without_insights, output_file.with_suffix(".parquet"))

    logger.info(
        "Saved dataset without insights: %s (%d rows, %d columns)",
        output_file.with_suffix(".parquet"),
        len(df_without_insights),
        len(df_without_insights.columns),
    )

    return df_without_insights


def create_dataset_without_sigma2(df: pd.DataFrame) -> pd.DataFrame:
    """Create dataset without sigma2_garch columns (ablation study).

    Keeps all other ARIMA-GARCH features (arima_pred_return, arima_residual_return,
    sigma_garch, std_resid_garch) but removes only sigma2_garch and its lags.

    Args:
        df: Complete dataset with all columns.

    Returns:
        Dataset without sigma2_garch columns.
    """
    from src.lightgbm.data_preparation.columns import get_sigma2_columns_to_drop

    logger.info("Creating dataset without sigma2_garch (ablation study)")
    columns_to_drop = get_sigma2_columns_to_drop(df)
    logger.info(f"Dropping {len(columns_to_drop)} sigma2_garch columns: {columns_to_drop}")
    return df.drop(columns=columns_to_drop).copy()


def _collect_insight_columns_with_lags(df: pd.DataFrame) -> list[str]:
    """Collect all ARIMA-GARCH insight columns with their lags."""
    cols: list[str] = []
    for insight_col in LIGHTGBM_ARIMA_GARCH_INSIGHT_COLUMNS:
        if insight_col in df.columns:
            cols.append(insight_col)
        # Include all lags of insight columns
        cols.extend(
            [
                f"{insight_col}_lag_{lag}"
                for lag in LIGHTGBM_LAG_WINDOWS
                if f"{insight_col}_lag_{lag}" in df.columns
            ]
        )
    return cols


def _count_insight_features(keep_cols: list[str]) -> int:
    """Count number of insight features in column list."""
    return len(
        [
            c
            for c in keep_cols
            if any(insight in c for insight in LIGHTGBM_ARIMA_GARCH_INSIGHT_COLUMNS)
        ]
    )


def create_dataset_insights_only_from_sigma_plus_base(
    df_sigma_plus_base: pd.DataFrame,
) -> pd.DataFrame:
    """Create insights-only dataset from sigma-plus-base dataset.

    Keeps only ARIMA-GARCH insight features (with lags) + ticker_id + target (log_volatility).
    Excludes all lags of log_volatility (target only, no lag features).

    Args:
        df_sigma_plus_base: Sigma-plus-base dataset DataFrame.

    Returns:
        Dataset with only ARIMA-GARCH insights + ticker_id + target (no target lags).
    """
    logger.info("Creating insights-only dataset from sigma-plus-base")

    keep_cols: list[str] = []

    # Metadata columns
    keep_cols.extend(select_metadata_columns(df_sigma_plus_base))

    # Target column (log_volatility) without any lags (target only, no lag features)
    target_col = get_target_column_name(df_sigma_plus_base)
    if target_col in df_sigma_plus_base.columns:
        keep_cols.append(target_col)

    # All ARIMA-GARCH insight columns with their lags
    keep_cols.extend(_collect_insight_columns_with_lags(df_sigma_plus_base))

    # Remove duplicates and filter to existing columns
    keep_cols = list(dict.fromkeys(keep_cols))
    keep_cols = [c for c in keep_cols if c in df_sigma_plus_base.columns]

    # Ensure ticker_id is present (required for model)
    if "ticker_id" not in keep_cols and "ticker_id" in df_sigma_plus_base.columns:
        keep_cols.append("ticker_id")

    # Include close column if present
    keep_cols.extend(select_close_column(df_sigma_plus_base))

    insight_count = _count_insight_features(keep_cols)
    logger.info(
        f"Selected {len(keep_cols)} columns for insights-only dataset "
        f"(target: {target_col}, insights: {insight_count} features)"
    )

    return cast(pd.DataFrame, df_sigma_plus_base[keep_cols].copy())


def create_dataset_insights_only_from_file(
    input_file: Path | None = None,
    output_file: Path | None = None,
) -> pd.DataFrame:
    """Create insights-only dataset from sigma-plus-base parquet file.

    Args:
        input_file: Path to sigma-plus-base parquet file. If None, uses default.
        output_file: Path to save insights-only dataset. If None, uses default.

    Returns:
        Insights-only dataset DataFrame.
    """
    if input_file is None:
        input_file = LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE.with_suffix(".parquet")

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    logger.info(f"Loading sigma-plus-base dataset from {input_file}")
    df_sigma = pd.read_parquet(input_file, engine="pyarrow")

    df_insights = create_dataset_insights_only_from_sigma_plus_base(df_sigma)

    if output_file is None:
        output_file = LIGHTGBM_DATASET_INSIGHTS_ONLY_FILE.with_suffix(".parquet")

    # Save parquet and CSV
    from src.utils import save_parquet_and_csv

    if "ticker" in df_insights.columns and "tickers" not in df_insights.columns:
        df_insights = df_insights.rename(columns={"ticker": "tickers"})
    save_parquet_and_csv(df_insights, output_file)

    logger.info(
        f"Saved insights-only dataset: {output_file} "
        f"({len(df_insights)} rows, {len(df_insights.columns)} columns)"
    )

    return df_insights


def ensure_sigma_plus_base_dataset(*, include_lags: bool = True) -> Path:
    """Ensure the sigma-plus-base dataset exists; create it if missing.

    If the dataset doesn't exist, calls prepare_datasets() to create all datasets
    at once, ensuring they all have the same number of rows.

    Args:
        include_lags: If True, include lag features when available.

    Returns:
        Path to the sigma-plus-base dataset.
    """
    dataset_path = LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE.with_suffix(".parquet")

    # Check if complete dataset exists - if not, create all datasets at once
    complete_path = LIGHTGBM_DATASET_COMPLETE_FILE.with_suffix(".parquet")
    if not complete_path.exists():
        logger.info(
            "Complete dataset not found. Creating all datasets at once to ensure consistency."
        )
        from src.lightgbm.data_preparation.dataset_creation import prepare_datasets

        prepare_datasets()
        return dataset_path

    if dataset_path.exists():
        if include_lags:
            df_existing = pd.read_parquet(dataset_path, engine="pyarrow")
            expected_cols = select_base_feature_columns(df_existing, include_lags=True) + [
                "sigma2_garch",
                *[f"sigma2_garch_lag_{lag}" for lag in LIGHTGBM_LAG_WINDOWS],
            ]
            missing_cols = [col for col in expected_cols if col not in df_existing.columns]
            if missing_cols:
                logger.info(
                    "Sigma-plus-base dataset missing columns %s. Recreating all datasets.",
                    missing_cols,
                )
                from src.lightgbm.data_preparation.dataset_creation import prepare_datasets

                prepare_datasets()
        return dataset_path

    # Dataset doesn't exist - create all datasets at once for consistency
    logger.info("Sigma-plus-base dataset not found. Creating all datasets at once.")
    from src.lightgbm.data_preparation.dataset_creation import prepare_datasets

    prepare_datasets()
    return dataset_path


# =============================================================================
# Technical-only and Technical+Insights datasets without target lags
# =============================================================================


def _collect_log_volatility_lag_columns(df: pd.DataFrame) -> list[str]:
    """Collect all lag columns of the target (log_volatility).

    Why: For the dedicated technical models, we explicitly remove target lags
    to ensure features at time t never include y[t-k]. This creates a clean
    setting to assess the value of technical indicators (with/without insights)
    without target autocorrelation signals.
    """
    target_col = get_target_column_name(df)
    return [c for c in df.columns if c.startswith(f"{target_col}_lag_")]


def _drop_columns_and_save(
    df: pd.DataFrame, columns_to_drop: list[str], output_file: Path
) -> pd.DataFrame:
    """Drop specified columns from df and save to both parquet and CSV.

    Args:
        df: Input DataFrame (complete dataset variant).
        columns_to_drop: Columns to remove from the dataset.
        output_file: Output file path (parquet will be written, CSV alongside).

    Returns:
        Filtered DataFrame actually written to disk.
    """
    from src.utils import save_parquet_and_csv

    keep_cols = [c for c in df.columns if c not in columns_to_drop]
    df_out = df[keep_cols].copy()

    # Ensure metadata consistency across saved datasets
    if "ticker" in df_out.columns and "tickers" not in df_out.columns:
        df_out = df_out.rename(columns={"ticker": "tickers"})

    save_parquet_and_csv(df_out, output_file.with_suffix(".parquet"))
    logger.info(
        "Saved dataset: %s (%d rows, %d columns) [dropped %d columns]",
        output_file.with_suffix(".parquet"),
        len(df_out),
        len(df_out.columns),
        len(columns_to_drop),
    )
    return df_out


def create_dataset_technical_only_no_target_lags_from_clean(
    df_clean: pd.DataFrame, *, include_lags: bool = True
) -> pd.DataFrame:
    """Create dataset with technical indicators only and no target lags from clean DataFrame.

    Why: This dataset evaluates the predictive value of technical indicators
    in isolation (no ARIMA/GARCH insights and no target lags).
    All datasets created from the same df_clean will have the same number of rows.

    Args:
        df_clean: Already prepared and cleaned DataFrame with all columns and no missing values.
        include_lags: If True, include lag features (not used here as we exclude target lags).

    Returns:
        Dataset with only technical indicators and no target lags.
    """
    keep_cols = select_metadata_columns(df_clean)
    keep_cols.extend(select_target_columns(df_clean, include_lags=False))  # No target lags

    # Get technical indicators but exclude log_volatility (the target) and its lags
    # to avoid including target lags in this "no target lags" dataset
    tech_cols = select_technical_indicator_columns(df_clean, include_lags)
    target_col = get_target_column_name(df_clean)
    tech_cols = [col for col in tech_cols if not col.startswith(f"{target_col}_lag_")]
    keep_cols.extend(tech_cols)

    # Remove duplicates and select columns
    keep_cols = list(dict.fromkeys(keep_cols))
    keep_cols = [c for c in keep_cols if c in df_clean.columns]

    return cast(pd.DataFrame, df_clean[keep_cols].copy())


def create_dataset_technical_plus_insights_no_target_lags_from_clean(
    df_clean: pd.DataFrame, *, include_lags: bool = True
) -> pd.DataFrame:
    """Create dataset with technical indicators + insights and no target lags from clean DataFrame.

    Why: This dataset measures the added value of ARIMA/GARCH insights when
    combined with technical indicators, while explicitly removing target lags
    to avoid autocorrelation signals.
    All datasets created from the same df_clean will have the same number of rows.

    Args:
        df_clean: Already prepared and cleaned DataFrame with all columns and no missing values.
        include_lags: If True, include lag features for technical indicators and insights.

    Returns:
        Dataset with technical indicators + insights and no target lags.
    """
    keep_cols = select_metadata_columns(df_clean)
    keep_cols.extend(select_target_columns(df_clean, include_lags=False))  # No target lags

    # Get technical indicators but exclude log_volatility (the target) lags
    # to avoid including target lags in this "no target lags" dataset
    tech_cols = select_technical_indicator_columns(df_clean, include_lags)
    target_col = get_target_column_name(df_clean)
    tech_cols = [col for col in tech_cols if not col.startswith(f"{target_col}_lag_")]
    keep_cols.extend(tech_cols)

    keep_cols.extend(select_garch_insight_columns(df_clean, include_lags))

    # Remove duplicates and select columns
    keep_cols = list(dict.fromkeys(keep_cols))
    keep_cols = [c for c in keep_cols if c in df_clean.columns]

    return cast(pd.DataFrame, df_clean[keep_cols].copy())


def ensure_technical_only_no_target_lags_dataset() -> Path:
    """Ensure the technical-only-no-target-lags dataset exists.

    IMPORTANT: To guarantee row consistency, if this dataset is missing,
    we rebuild ALL datasets via prepare_datasets() rather than creating
    it in isolation from the complete dataset.
    """
    from src.constants import LIGHTGBM_DATASET_COMPLETE_FILE
    from src.path import LIGHTGBM_DATASET_TECHNICAL_ONLY_NO_TARGET_LAGS_FILE as OUT

    complete_parquet = LIGHTGBM_DATASET_COMPLETE_FILE.with_suffix(".parquet")
    out_parquet = OUT.with_suffix(".parquet")

    # If either the complete dataset or this specific dataset is missing,
    # recreate ALL datasets to ensure consistency
    if not complete_parquet.exists() or not out_parquet.exists():
        logger.info(
            "Technical-only-no-target-lags dataset missing or complete dataset missing. "
            "Creating all datasets to ensure consistency."
        )
        from src.lightgbm.data_preparation.dataset_creation import prepare_datasets

        prepare_datasets()

    return out_parquet


def ensure_technical_plus_insights_no_target_lags_dataset() -> Path:
    """Ensure the technical-plus-insights-no-target-lags dataset exists.

    IMPORTANT: To guarantee row consistency, if this dataset is missing,
    we rebuild ALL datasets via prepare_datasets() rather than creating
    it in isolation from the complete dataset.
    """
    from src.constants import LIGHTGBM_DATASET_COMPLETE_FILE
    from src.path import LIGHTGBM_DATASET_TECHNICAL_PLUS_INSIGHTS_NO_TARGET_LAGS_FILE as OUT

    complete_parquet = LIGHTGBM_DATASET_COMPLETE_FILE.with_suffix(".parquet")
    out_parquet = OUT.with_suffix(".parquet")

    # If either the complete dataset or this specific dataset is missing,
    # recreate ALL datasets to ensure consistency
    if not complete_parquet.exists() or not out_parquet.exists():
        logger.info(
            "Technical-plus-insights-no-target-lags dataset missing or complete dataset missing. "
            "Creating all datasets to ensure consistency."
        )
        from src.lightgbm.data_preparation.dataset_creation import prepare_datasets

        prepare_datasets()

    return out_parquet


def ensure_dataset_without_insights() -> Path:
    """Ensure the without-insights dataset exists.

    IMPORTANT: To guarantee row consistency, if this dataset is missing,
    we rebuild ALL datasets via prepare_datasets() rather than creating
    it in isolation from the complete dataset.
    """
    from src.constants import LIGHTGBM_DATASET_COMPLETE_FILE
    from src.path import LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE as OUT

    complete_parquet = LIGHTGBM_DATASET_COMPLETE_FILE.with_suffix(".parquet")
    out_parquet = OUT.with_suffix(".parquet")

    # If either the complete dataset or this specific dataset is missing,
    # recreate ALL datasets to ensure consistency
    if not complete_parquet.exists() or not out_parquet.exists():
        logger.info(
            "Without-insights dataset missing or complete dataset missing. "
            "Creating all datasets to ensure consistency."
        )
        from src.lightgbm.data_preparation.dataset_creation import prepare_datasets

        prepare_datasets()

    return out_parquet
