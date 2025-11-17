"""Main dataset creation functions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.constants import (
    DATA_DIR,
    LIGHTGBM_DATASET_INSIGHTS_ONLY_FILE,
    LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE,
    LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE,
    LIGHTGBM_DATASET_TECHNICAL_ONLY_NO_TARGET_LAGS_FILE,
    LIGHTGBM_DATASET_TECHNICAL_PLUS_INSIGHTS_NO_TARGET_LAGS_FILE,
    LIGHTGBM_LAG_WINDOWS,
)
from src.lightgbm.data_preparation.columns import (
    get_non_observable_columns_to_drop,
    remove_missing_values,
)
from src.lightgbm.data_preparation.data_loading import ensure_ticker_id_column
from src.lightgbm.data_preparation.dataset_pipeline import prepare_base_dataframe
from src.lightgbm.data_preparation.dataset_variants import (
    create_dataset_log_volatility_only_from_clean,
    create_dataset_sigma_plus_base_from_clean,
    create_dataset_technical_only_no_target_lags_from_clean,
    create_dataset_technical_plus_insights_no_target_lags_from_clean,
    create_dataset_without_insights,
)
from src.lightgbm.data_preparation.features import (
    add_calendar_features,
    add_custom_ml_indicators_per_ticker,
    add_lag_features,
)
from src.lightgbm.data_preparation.target_creation import (
    create_target_columns,
    get_target_column_name,
)
from src.utils import get_logger, log_split_dates, save_parquet_and_csv, validate_temporal_split

logger = get_logger(__name__)


def _compute_log_volatility_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log_volatility from log_return if missing."""
    if "ticker" in df.columns and "log_return" in df.columns:
        if "log_volatility" not in df.columns:
            logger.info("Computing log volatility per ticker (not saved in files)")
            from src.data_preparation.computations import compute_volatility_for_tickers

            df = compute_volatility_for_tickers(df)
    return df


def _add_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators with strict validation and explicit behavior.

    - If ticker-level data AND required columns ('close', 'volume') exist,
      compute full per-ticker indicators (volume/returns/turnover/OBV + calendar).
    - Otherwise, if 'date' exists, add calendar-only features explicitly.
    - Always log chosen path; never apply silent fallbacks.
    """
    df_in = df.copy()

    if "ticker" in df_in.columns and {"close", "volume"}.issubset(df_in.columns):
        logger.info("Calculating custom ML indicators per ticker")
        return add_custom_ml_indicators_per_ticker(df_in)

    if "date" in df_in.columns:
        missing = {c for c in ("ticker", "close", "volume") if c not in df_in.columns}
        logger.info(
            "Adding calendar-only features (missing columns prevent full indicators): %s",
            sorted(missing),
        )
        return add_calendar_features(df_in)

    logger.info("No 'date' column found; skipping indicator creation entirely")
    return df_in


def _get_lag_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get list of columns to lag based on available columns."""
    lag_feature_columns: list[str] = []

    # Toujours inclure la cible de base (ex: log_volatility)
    return_col = get_target_column_name(df)
    if return_col not in lag_feature_columns:
        lag_feature_columns.append(return_col)

    # Features volume/prix/techniques
    requested_to_lag = [
        "log_return",
        "abs_ret",
        "ret_sq",
        "log_volatility",
        "log_volume",
        "log_volume_rel_ma_5",
        "log_volume_zscore_20",
        "log_turnover",
        "turnover_rel_ma_5",
        "obv",
        "atr",
    ]
    for col in requested_to_lag:
        if col in df.columns and col not in lag_feature_columns:
            lag_feature_columns.append(col)

    # GARCH/ARIMA insights, if present
    # ARIMA features: sarima_pred and sarima_resid (from ARIMA model)
    # GARCH features: sigma2_garch, sigma_garch, std_resid_garch
    insight_cols = [
        "sigma2_garch",
        "sigma_garch",
        "std_resid_garch",
        "sarima_resid",
        "sarima_pred",
        "arima_residual_return",
    ]
    for insight_col in insight_cols:
        if insight_col in df.columns and insight_col not in lag_feature_columns:
            lag_feature_columns.append(insight_col)

    return lag_feature_columns


def _add_features_and_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators, lag features, and then shift target.

    CRITICAL PIPELINE ORDER TO PREVENT DATA LEAKAGE:
    1. Compute log volatility if needed (from returns ≤ t)
    2. Add custom ML indicators per ticker (all based on data ≤ t)
    3. Add lag features on NON-SHIFTED columns (lags look at past: t-1, t-2, etc.)
    4. ONLY THEN shift target with shift(-1) to align features at t with target at t+1
    5. Remove rows with missing values after target shift

    This order ensures that:
    - log_volatility[t] contains volatility computed from returns up to time t
    - log_volatility_lag_k[t] contains volatility from time t-k (true past values)
    - After target shift, features at t predict target at t+1 without leakage
    """
    # Step 1: Compute log volatility from returns ≤ t
    df = _compute_log_volatility_if_needed(df)

    # Step 2: Add custom indicators (all use data ≤ t)
    df = _add_custom_indicators(df)

    # Step 3: Add lag features BEFORE shifting target
    # This ensures lags contain true historical values
    lag_feature_columns = _get_lag_feature_columns(df)
    df = add_lag_features(df, feature_columns=lag_feature_columns, lag_windows=LIGHTGBM_LAG_WINDOWS)

    # Step 4: ONLY NOW shift the target column
    # This aligns features at time t with target at t+1
    df = create_target_columns(df)

    # Step 5: Remove rows with missing values after all transformations
    df = remove_missing_values(df)

    return df


def _create_temporal_split_if_missing(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Create or recreate temporal split aligned with target date (t -> t+1).

    Why: When a pre-existing ``split`` column comes from an earlier stage (based on
    feature dates at t), it may be misaligned with the shifted target (t+1). To
    avoid any leakage at the split boundary, we always recreate the split using
    target dates.

    Returns:
        DataFrame with a ``split`` column ('train'/'test') aligned to target dates.
    """
    from src.lightgbm.data_preparation.target_creation import _create_temporal_split

    logger.info(
        "Recreating temporal split based on target dates to ensure alignment "
        "(no global dropna applied)"
    )
    return _create_temporal_split(df_clean, use_target_date=True)


def _prepare_clean_base_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare clean base DataFrame with temporal split and ticker identifiers.

    Drop non-observable columns to avoid leakage.
    """
    df_clean = df.copy()

    if "date" in df_clean.columns:
        df_clean = _create_temporal_split_if_missing(df_clean)

    df_clean = ensure_ticker_id_column(df_clean)

    non_observable_columns = get_non_observable_columns_to_drop(df_clean)
    columns_to_remove = [col for col in non_observable_columns if col in df_clean.columns]
    if columns_to_remove:
        logger.info("Dropping non-observable columns: %s", columns_to_remove)
        df_clean = df_clean.drop(columns=columns_to_remove)

    return df_clean


def _log_technical_indicators_status(df_clean: pd.DataFrame) -> None:
    """Log status of technical indicator columns."""
    from src.constants import LIGHTGBM_CALENDAR_FEATURE_COLUMNS, LIGHTGBM_TECHNICAL_FEATURE_COLUMNS

    tech_cols_present = [
        col for col in LIGHTGBM_TECHNICAL_FEATURE_COLUMNS if col in df_clean.columns
    ]
    calendar_cols_present = [
        col for col in LIGHTGBM_CALENDAR_FEATURE_COLUMNS if col in df_clean.columns
    ]
    logger.info(
        "Technical indicators present: %d/%d technical, %d/%d calendar",
        len(tech_cols_present),
        len(LIGHTGBM_TECHNICAL_FEATURE_COLUMNS),
        len(calendar_cols_present),
        len(LIGHTGBM_CALENDAR_FEATURE_COLUMNS),
    )
    if len(tech_cols_present) < len(LIGHTGBM_TECHNICAL_FEATURE_COLUMNS):
        missing = set(LIGHTGBM_TECHNICAL_FEATURE_COLUMNS) - set(tech_cols_present)
        logger.warning("Missing technical indicator columns: %s", sorted(missing))


def _create_dataset_variants(
    df_clean: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create all dataset variants from clean DataFrame."""
    df_without_insights = create_dataset_without_insights(df_clean)
    df_without_insights = ensure_ticker_id_column(df_without_insights)

    df_sigma_plus_base = create_dataset_sigma_plus_base_from_clean(df_clean, include_lags=True)
    df_sigma_plus_base = ensure_ticker_id_column(df_sigma_plus_base)

    df_log_volatility_only = create_dataset_log_volatility_only_from_clean(
        df_clean, include_lags=True
    )
    df_log_volatility_only = ensure_ticker_id_column(df_log_volatility_only)

    df_technical_only_no_target_lags = create_dataset_technical_only_no_target_lags_from_clean(
        df_clean, include_lags=True
    )
    df_technical_only_no_target_lags = ensure_ticker_id_column(df_technical_only_no_target_lags)

    df_technical_plus_insights_no_target_lags = (
        create_dataset_technical_plus_insights_no_target_lags_from_clean(
            df_clean, include_lags=True
        )
    )
    df_technical_plus_insights_no_target_lags = ensure_ticker_id_column(
        df_technical_plus_insights_no_target_lags
    )

    return (
        df_without_insights,
        df_sigma_plus_base,
        df_log_volatility_only,
        df_technical_only_no_target_lags,
        df_technical_plus_insights_no_target_lags,
    )


def _verify_datasets_same_length(
    df_complete: pd.DataFrame,
    df_without_insights: pd.DataFrame,
    df_sigma_plus_base: pd.DataFrame,
    df_log_volatility_only: pd.DataFrame,
    df_technical_only_no_target_lags: pd.DataFrame,
    df_technical_plus_insights_no_target_lags: pd.DataFrame,
) -> None:
    """Verify all datasets have the same number of rows."""
    n = len(df_complete)
    assert len(df_without_insights) == n, "without_insights must use same observations as complete"
    assert len(df_sigma_plus_base) == n, "sigma_plus_base must use same observations as complete"
    assert (
        len(df_log_volatility_only) == n
    ), "log_volatility_only must use same observations as complete"
    assert (
        len(df_technical_only_no_target_lags) == n
    ), "technical_only_no_target_lags must use same observations as complete"
    assert (
        len(df_technical_plus_insights_no_target_lags) == n
    ), "technical_plus_insights_no_target_lags must use same observations as complete"


def _clean_and_prepare_datasets(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean data and create all dataset variants.

    Important:
        - Aucun dropna global n'est appliqué ici.
        - Tous les datasets sont des sous-ensembles en colonnes du même df_clean
          et **partagent donc exactement les mêmes lignes**.

    Args:
        df: DataFrame with features and lags.

    Returns:
        Tuple of (complete, without_insights, sigma_plus_base, log_volatility_only,
                 technical_only_no_target_lags, technical_plus_insights_no_target_lags) datasets.
    """
    df_clean = _prepare_clean_base_df(df)

    logger.info("Creating complete dataset with all columns (no additional row filtering)")
    _log_technical_indicators_status(df_clean)

    df_complete = df_clean.copy()
    (
        df_without_insights,
        df_sigma_plus_base,
        df_log_volatility_only,
        df_technical_only_no_target_lags,
        df_technical_plus_insights_no_target_lags,
    ) = _create_dataset_variants(df_clean)

    _verify_datasets_same_length(
        df_complete,
        df_without_insights,
        df_sigma_plus_base,
        df_log_volatility_only,
        df_technical_only_no_target_lags,
        df_technical_plus_insights_no_target_lags,
    )

    return (
        df_complete,
        df_without_insights,
        df_sigma_plus_base,
        df_log_volatility_only,
        df_technical_only_no_target_lags,
        df_technical_plus_insights_no_target_lags,
    )


def _save_main_datasets(
    df_complete: pd.DataFrame, df_without_insights: pd.DataFrame, output_dir: Path
) -> None:
    """Save complete and without-insights datasets."""
    output_complete = output_dir / "lightgbm_dataset_complete.parquet"
    output_without = output_dir / "lightgbm_dataset_without_insights.parquet"

    # Rename metadata column to 'tickers' for saved files
    if "ticker" in df_complete.columns and "tickers" not in df_complete.columns:
        df_complete = df_complete.rename(columns={"ticker": "tickers"})
    if "ticker" in df_without_insights.columns and "tickers" not in df_without_insights.columns:
        df_without_insights = df_without_insights.rename(columns={"ticker": "tickers"})

    save_parquet_and_csv(df_complete, output_complete)
    save_parquet_and_csv(df_without_insights, output_without)

    logger.info(
        "Saved complete dataset: %s (%d rows, %d columns)",
        output_complete,
        len(df_complete),
        len(df_complete.columns),
    )
    logger.info(
        "Saved dataset without insights: %s (%d rows, %d columns)",
        output_without,
        len(df_without_insights),
        len(df_without_insights.columns),
    )


def _normalize_ticker_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ticker column name from 'ticker' to 'tickers' if needed."""
    if "ticker" in df.columns and "tickers" not in df.columns:
        return df.rename(columns={"ticker": "tickers"})
    return df


def _get_output_path(
    base_file: Path,
    variant_name: str,
    output_dir: Path | None,
    variant_dir: Path | None,
) -> Path:
    """Get output path for a dataset variant."""
    if output_dir is not None:
        if variant_dir is not None:
            return variant_dir / f"lightgbm_dataset_{variant_name}.parquet"
        return output_dir / f"lightgbm_dataset_{variant_name}.parquet"
    return base_file.with_suffix(".parquet")


def _save_single_dataset(
    df: pd.DataFrame,
    base_file: Path,
    variant_name: str,
    output_dir: Path | None,
    variant_dir: Path | None,
) -> None:
    """Save a single dataset variant."""
    df_normalized = _normalize_ticker_column(df)
    output_path = _get_output_path(base_file, variant_name, output_dir, variant_dir)
    save_parquet_and_csv(df_normalized, output_path)

    logger.info(
        "Saved %s dataset: %s (%d rows, %d columns)",
        variant_name,
        output_path,
        len(df_normalized),
        len(df_normalized.columns),
    )


def _save_variant_datasets(
    df_sigma_plus_base: pd.DataFrame,
    df_log_volatility_only: pd.DataFrame,
    df_technical_only_no_target_lags: pd.DataFrame,
    df_technical_plus_insights_no_target_lags: pd.DataFrame,
    output_dir: Path | None = None,
) -> None:
    """Save sigma-plus-base, log-volatility-only, and technical datasets."""

    # Define dataset configurations
    datasets = [
        (df_sigma_plus_base, LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE, "sigma_plus_base"),
        (df_log_volatility_only, LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE, "log_volatility_only"),
        (
            df_technical_only_no_target_lags,
            LIGHTGBM_DATASET_TECHNICAL_ONLY_NO_TARGET_LAGS_FILE,
            "technical_only_no_target_lags",
        ),
        (
            df_technical_plus_insights_no_target_lags,
            LIGHTGBM_DATASET_TECHNICAL_PLUS_INSIGHTS_NO_TARGET_LAGS_FILE,
            "technical_plus_insights_no_target_lags",
        ),
    ]

    # Save all datasets
    for df, base_file, variant_name in datasets:
        _save_single_dataset(df, base_file, variant_name, output_dir, None)


def _save_insights_only_dataset(
    df_sigma_plus_base: pd.DataFrame, output_dir: Path | None = None
) -> pd.DataFrame:
    """Save insights-only dataset derived from sigma-plus-base."""
    from src.lightgbm.data_preparation.dataset_variants import (
        create_dataset_insights_only_from_sigma_plus_base,
    )

    df_insights_only = create_dataset_insights_only_from_sigma_plus_base(df_sigma_plus_base)
    if "ticker" in df_insights_only.columns and "tickers" not in df_insights_only.columns:
        df_insights_only = df_insights_only.rename(columns={"ticker": "tickers"})

    if output_dir:
        output_insights = output_dir / "lightgbm_dataset_insights_only.parquet"
    else:
        output_insights = LIGHTGBM_DATASET_INSIGHTS_ONLY_FILE.with_suffix(".parquet")

    save_parquet_and_csv(df_insights_only, output_insights)

    logger.info(
        "Saved insights-only dataset: %s (%d rows, %d columns)",
        output_insights,
        len(df_insights_only),
        len(df_insights_only.columns),
    )

    return df_insights_only


def _validate_dataset_consistency(
    df_complete: pd.DataFrame,
    df_without_insights: pd.DataFrame,
    df_sigma_plus_base: pd.DataFrame,
    df_log_volatility_only: pd.DataFrame,
    df_insights_only: pd.DataFrame,
) -> None:
    """Validate that all datasets have the same number of rows."""
    n = len(df_complete)
    assert len(df_sigma_plus_base) == n
    assert len(df_log_volatility_only) == n
    assert len(df_insights_only) == n
    assert len(df_without_insights) == n


def _validate_temporal_order(df: pd.DataFrame) -> None:
    """Validate temporal order of datasets to prevent look-ahead bias."""
    if "split" not in df.columns or "date" not in df.columns:
        return

    if "ticker" in df.columns:
        validate_temporal_split(
            df, ticker_col="ticker", function_name="prepare_datasets (complete)"
        )
        log_split_dates(df, ticker_col="ticker", function_name="prepare_datasets (complete)")
    else:
        validate_temporal_split(df, function_name="prepare_datasets (complete)")
        log_split_dates(df, function_name="prepare_datasets (complete)")


def _save_all_datasets(
    df_complete: pd.DataFrame,
    df_without_insights: pd.DataFrame,
    output_dir: Path,
    df_sigma_plus_base: pd.DataFrame,
    df_log_volatility_only: pd.DataFrame,
    df_technical_only_no_target_lags: pd.DataFrame,
    df_technical_plus_insights_no_target_lags: pd.DataFrame,
) -> None:
    """Save all dataset variants to files.

    All datasets must have the same number of rows.
    Saves: complete, without_insights, sigma_plus_base, log_volatility_only,
           technical_only_no_target_lags, technical_plus_insights_no_target_lags, insights_only.
    """
    _save_main_datasets(df_complete, df_without_insights, output_dir)
    _save_variant_datasets(
        df_sigma_plus_base,
        df_log_volatility_only,
        df_technical_only_no_target_lags,
        df_technical_plus_insights_no_target_lags,
        output_dir,
    )
    df_insights_only = _save_insights_only_dataset(df_sigma_plus_base, output_dir)
    _validate_dataset_consistency(
        df_complete,
        df_without_insights,
        df_sigma_plus_base,
        df_log_volatility_only,
        df_insights_only,
    )


def prepare_datasets(
    df: pd.DataFrame | None = None, output_dir: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare two main datasets: complete and without ARIMA-GARCH insights.

    Pipeline:

    1. Load / normalize / sort data.
    2. Compute volatility, custom indicators, target, and lags.
    3. Construire un unique df_clean sans dropna global.
    4. En dériver toutes les variantes comme sous-ensembles en colonnes.
    5. Sauvegarder, puis valider l'ordre temporel.

    Garanties:
        - Pas de fuite temporelle (lags et rolling ne regardent que le passé).
        - Tous les datasets ont exactement le même nombre de lignes.
    """
    if output_dir is None:
        output_dir = DATA_DIR

    df_prepared = prepare_base_dataframe(df, prefer_insights=True)
    df_with_features = _add_features_and_lags(df_prepared)
    (
        df_complete,
        df_without_insights,
        df_sigma_plus_base,
        df_log_volatility_only,
        df_technical_only_no_target_lags,
        df_technical_plus_insights_no_target_lags,
    ) = _clean_and_prepare_datasets(df_with_features)

    _validate_temporal_order(df_complete)
    _save_all_datasets(
        df_complete,
        df_without_insights,
        output_dir,
        df_sigma_plus_base,
        df_log_volatility_only,
        df_technical_only_no_target_lags,
        df_technical_plus_insights_no_target_lags,
    )

    return df_complete, df_without_insights
