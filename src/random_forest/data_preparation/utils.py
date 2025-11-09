"""Utility functions for Random Forest data preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, cast

import pandas as pd

from src.constants import (
    DATA_DIR,
    RF_DATASET_TECHNICAL_INDICATORS_FILE,
    RF_DATASET_SIGMA2_ONLY_FILE,
    RF_DATASET_RSI14_ONLY_FILE,
    RF_ARIMA_GARCH_INSIGHT_COLUMNS,
    RF_LAG_FEATURE_COLUMNS,
    RF_LAG_WINDOWS,
    RF_TECHNICAL_FEATURE_COLUMNS,
)
from src.utils import get_logger
from src.random_forest.data_preparation.calculs_indicators import add_technical_indicators


logger = get_logger(__name__)


def _validate_lag_value(lag: int) -> None:
    """Validate that lag value is positive.

    Args:
        lag: Lag value to validate.

    Raises:
        ValueError: If lag is not positive.
    """
    if lag <= 0:
        msg = f"Lag value must be positive, received {lag}"
        raise ValueError(msg)


def _add_single_lag_feature(df: pd.DataFrame, column: str, lag: int) -> pd.DataFrame:
    """Add a single lag feature to dataframe.

    Args:
        df: Input dataframe.
        column: Column name to lag.
        lag: Number of periods to shift.

    Returns:
        DataFrame with added lag feature.
    """
    lag_column_name = f"{column}_lag_{lag}"
    df[lag_column_name] = df[column].shift(lag)
    return df


def _log_missing_columns(feature_columns: Sequence[str], df: pd.DataFrame) -> None:
    """Log missing columns for lagging.

    Args:
        feature_columns: Column names to check.
        df: DataFrame to check columns against.
    """
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        logger.debug("Columns skipped for lagging because they are missing: %s", missing_columns)


def _add_lags_for_column(df: pd.DataFrame, column: str, lag_windows: Sequence[int]) -> pd.DataFrame:
    """Add lag features for a single column.

    Args:
        df: Input dataframe.
        column: Column name to lag.
        lag_windows: Lag windows to apply.

    Returns:
        DataFrame with lag features added for the column.
    """
    for lag in sorted(set(lag_windows)):
        _validate_lag_value(lag)
        df = _add_single_lag_feature(df, column, lag)
    return df


def add_lag_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    lag_windows: Sequence[int],
) -> pd.DataFrame:
    """Return a dataframe with lagged versions of selected columns.

    Args:
        df: Input dataframe sorted by time.
        feature_columns: Column names to lag if present in the dataframe.
        lag_windows: Positive integers indicating how many periods to shift.

    Returns:
        DataFrame including the original data and the requested lag features.
    """
    logger.info(
        "Adding lag features for %d columns with windows %s",
        len(feature_columns),
        lag_windows,
    )
    _log_missing_columns(feature_columns, df)

    df_with_lags = df.copy()
    for column in feature_columns:
        if column not in df_with_lags.columns:
            continue
        df_with_lags = _add_lags_for_column(df_with_lags, column, lag_windows)

    return df_with_lags


def _get_default_garch_file_path() -> Path:
    """Get default GARCH variance file path.

    Returns:
        Path to default GARCH variance file.
    """
    from src.constants import GARCH_ROLLING_VARIANCE_FILE

    return GARCH_ROLLING_VARIANCE_FILE


def _validate_garch_columns(df: pd.DataFrame) -> None:
    """Validate that required columns are present in GARCH data.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = ["date", "weighted_closing", "weighted_open", "log_weighted_return"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def _add_split_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add split column if not present (80/20 train/test temporal split).

    Args:
        df: DataFrame to add split column to.

    Returns:
        DataFrame with split column added.
    """
    if "split" not in df.columns:
        logger.info("Adding split column (80% train, 20% test)")
        split_idx = int(len(df) * 0.8)
        df["split"] = ["train"] * split_idx + ["test"] * (len(df) - split_idx)
    return df


def load_garch_data(file_path: Path | None = None) -> pd.DataFrame:
    """Load GARCH variance data.

    Args:
        file_path: Path to GARCH variance CSV file. If None, uses default from results/rolling/.

    Returns:
        DataFrame with GARCH variance data including log_weighted_return and split column.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required columns are missing.
    """
    if file_path is None:
        file_path = _get_default_garch_file_path()

    if not file_path.exists():
        raise FileNotFoundError(f"GARCH variance file not found: {file_path}")

    logger.info(f"Loading GARCH variance data from {file_path}")
    df = pd.read_csv(file_path, parse_dates=["date"])

    _validate_garch_columns(df)
    df = _add_split_column(df)

    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for consistency.

    Renames log_weighted_return to weighted_log_return and removes weighted_return.

    Args:
        df: DataFrame with potential column naming issues.

    Returns:
        DataFrame with normalized column names.
    """
    df_normalized = df.copy()

    if "log_weighted_return" in df_normalized.columns:
        logger.info("Renaming log_weighted_return to weighted_log_return")
        df_normalized = df_normalized.rename(columns={"log_weighted_return": "weighted_log_return"})

    if "weighted_return" in df_normalized.columns:
        logger.info("Removing weighted_return column (keeping weighted_log_return)")
        df_normalized = df_normalized.drop(columns=["weighted_return"])

    return df_normalized


def _create_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create shifted target columns for next-day prediction.

    Args:
        df: DataFrame containing weighted_log_return and optionally split columns.

    Returns:
        DataFrame with weighted_log_return shifted to represent J+1 and a
        weighted_log_return_t column storing the original series.

    Raises:
        ValueError: If weighted_log_return is missing from the dataframe.
    """

    if "weighted_log_return" not in df.columns:
        msg = "Column weighted_log_return is required to create prediction targets"
        raise ValueError(msg)

    df_shifted = df.copy()
    df_shifted["weighted_log_return_t"] = df_shifted["weighted_log_return"]

    logger.info("Shifting weighted_log_return to align with next-day target")
    df_shifted["weighted_log_return"] = df_shifted["weighted_log_return"].shift(-1)

    subset_columns = ["weighted_log_return"]

    if "split" in df_shifted.columns:
        logger.info("Shifting split column to align with next-day target")
        df_shifted["split"] = df_shifted["split"].shift(-1)
        subset_columns.append("split")

    df_shifted = df_shifted.dropna(subset=subset_columns).reset_index(drop=True)

    return df_shifted


def _remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing values and log statistics.

    Args:
        df: DataFrame that may contain missing values.

    Returns:
        Cleaned DataFrame without missing values.
    """
    initial_rows = len(df)
    logger.info("Removing rows with missing values from technical indicators")
    df_clean = df.dropna().reset_index(drop=True).copy()
    removed_rows = initial_rows - len(df_clean)
    logger.info(
        f"Removed {removed_rows} rows with missing values "
        f"(from {initial_rows} to {len(df_clean)})"
    )
    return df_clean


def _get_base_insight_columns(df: pd.DataFrame) -> list[str]:
    """Get base ARIMA-GARCH insight columns present in dataframe.

    Args:
        df: DataFrame with potential insight columns.

    Returns:
        List of base insight column names present in dataframe.
    """
    return [col for col in RF_ARIMA_GARCH_INSIGHT_COLUMNS if col in df.columns]


def _get_lagged_insight_columns(df: pd.DataFrame) -> list[str]:
    """Get lagged ARIMA-GARCH insight columns present in dataframe.

    Args:
        df: DataFrame with potential lagged insight columns.

    Returns:
        List of lagged insight column names present in dataframe.
    """
    lagged_columns = []
    for col in RF_ARIMA_GARCH_INSIGHT_COLUMNS:
        for lag in RF_LAG_WINDOWS:
            lagged_col = f"{col}_lag_{lag}"
            if lagged_col in df.columns:
                lagged_columns.append(lagged_col)
    return lagged_columns


def _get_insight_columns_to_drop(df: pd.DataFrame) -> list[str]:
    """Get list of ARIMA-GARCH insight columns and their lagged versions to drop.

    Args:
        df: DataFrame with potential insight columns.

    Returns:
        List of column names to drop.
    """
    columns_to_drop = _get_base_insight_columns(df)
    lagged_columns_to_drop = _get_lagged_insight_columns(df)
    columns_to_drop.extend(lagged_columns_to_drop)
    return columns_to_drop


def _get_non_observable_columns_to_drop() -> list[str]:
    """Get weighted price columns and their lags to remove before saving datasets."""

    base_columns = ["weighted_closing", "weighted_open"]
    lagged_columns = [f"{column}_lag_{lag}" for column in base_columns for lag in RF_LAG_WINDOWS]
    return base_columns + lagged_columns


def _get_sigma2_columns_to_drop(df: pd.DataFrame) -> list[str]:
    """Get list of sigma2_garch columns and their lagged versions to drop.

    Args:
        df: DataFrame with potential sigma2_garch columns.

    Returns:
        List of column names to drop (sigma2_garch and its lags only).
    """
    columns_to_drop = []
    # Base column
    if "sigma2_garch" in df.columns:
        columns_to_drop.append("sigma2_garch")
    # Lagged versions
    for lag in RF_LAG_WINDOWS:
        lagged_col = f"sigma2_garch_lag_{lag}"
        if lagged_col in df.columns:
            columns_to_drop.append(lagged_col)
    return columns_to_drop


def _create_dataset_without_sigma2(df: pd.DataFrame) -> pd.DataFrame:
    """Create dataset without sigma2_garch columns (ablation study).

    Keeps all other ARIMA-GARCH features (arima_pred_return, arima_residual_return,
    sigma_garch, std_resid_garch) but removes only sigma2_garch and its lags.

    Args:
        df: Complete dataset with all columns.

    Returns:
        Dataset without sigma2_garch columns.
    """
    logger.info("Creating dataset without sigma2_garch (ablation study)")
    columns_to_drop = _get_sigma2_columns_to_drop(df)
    logger.info(f"Dropping {len(columns_to_drop)} sigma2_garch columns: {columns_to_drop}")
    return df.drop(columns=columns_to_drop).copy()


def _create_dataset_without_insights(df: pd.DataFrame) -> pd.DataFrame:
    """Create dataset without ARIMA-GARCH insight columns.

    Args:
        df: Complete dataset with all columns.

    Returns:
        Dataset without ARIMA-GARCH insight columns.
    """
    logger.info("Creating dataset without ARIMA-GARCH insights")
    columns_to_drop = _get_insight_columns_to_drop(df)
    return df.drop(columns=columns_to_drop).copy()


def _save_datasets(
    df_complete: pd.DataFrame,
    df_without_insights: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save both datasets to CSV files.

    Args:
        df_complete: Complete dataset with all columns.
        df_without_insights: Dataset without ARIMA-GARCH insights.
        output_dir: Directory to save datasets.
    """
    output_complete = output_dir / "rf_dataset_complete.csv"
    output_without = output_dir / "rf_dataset_without_insights.csv"

    df_complete.to_csv(output_complete, index=False)
    df_without_insights.to_csv(output_without, index=False)

    logger.info(
        f"Saved complete dataset: {output_complete} "
        f"({len(df_complete)} rows, {len(df_complete.columns)} columns)"
    )
    logger.info(
        f"Saved dataset without insights: {output_without} "
        f"({len(df_without_insights)} rows, {len(df_without_insights.columns)} columns)"
    )


def create_dataset_without_sigma2(
    df: pd.DataFrame | None = None, output_path: Path | None = None
) -> pd.DataFrame:
    """Create dataset without sigma2_garch columns for ablation study.

    Args:
        df: Complete dataset. If None, loads from RF_DATASET_COMPLETE_FILE.
        output_path: Path to save dataset. If None, uses RF_DATASET_WITHOUT_SIGMA2_FILE.

    Returns:
        Dataset without sigma2_garch columns.
    """
    from src.constants import RF_DATASET_COMPLETE_FILE, RF_DATASET_WITHOUT_SIGMA2_FILE

    if df is None:
        logger.info(f"Loading complete dataset from {RF_DATASET_COMPLETE_FILE}")
        df = pd.read_csv(RF_DATASET_COMPLETE_FILE)

    df_without_sigma2 = _create_dataset_without_sigma2(df)

    if output_path is None:
        output_path = RF_DATASET_WITHOUT_SIGMA2_FILE

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_without_sigma2.to_csv(output_path, index=False)
    logger.info(
        f"Saved dataset without sigma2_garch: {output_path} "
        f"({len(df_without_sigma2)} rows, {len(df_without_sigma2.columns)} columns)"
    )

    return df_without_sigma2


def create_dataset_sigma2_only(
    df: pd.DataFrame | None = None,
    output_path: Path | None = None,
    *,
    include_lags: bool = True,
) -> pd.DataFrame:
    """Create a dataset that keeps only sigma2_garch and its lags as features.

    This supports a focused ablation: evaluate the predictive power of EGARCH
    variance alone on `weighted_log_return` at horizon t+1.

    Args:
        df: Complete dataset. If None, loads from RF_DATASET_COMPLETE_FILE.
        output_path: Path to save dataset. If None, uses RF_DATASET_SIGMA2_ONLY_FILE.
        include_lags: If True, also keep sigma2_garch_lag_{k} for k in RF_LAG_WINDOWS.
            Defaults to True to include all available information.

    Returns:
        Dataset containing only target columns (date, split, weighted_log_return)
        plus sigma2_garch and its lags as the sole features.
    """
    from src.constants import RF_DATASET_COMPLETE_FILE

    if df is None:
        logger.info(f"Loading complete dataset from {RF_DATASET_COMPLETE_FILE}")
        df = pd.read_csv(RF_DATASET_COMPLETE_FILE)

    base_cols = ["date", "split", "weighted_log_return"]
    keep_cols = base_cols + (["sigma2_garch"] if "sigma2_garch" in df.columns else [])

    if include_lags:
        for lag in RF_LAG_WINDOWS:
            col = f"sigma2_garch_lag_{lag}"
            if col in df.columns:
                keep_cols.append(col)

    missing = [c for c in ["sigma2_garch"] if c not in df.columns]
    if missing:
        logger.warning("sigma2_garch not found in dataset; sigma2-only dataset will have no features")

    # Select and drop any NA rows (to keep alignment with training code expectations)
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_sigma2 = cast(pd.DataFrame, df[keep_cols]).dropna().reset_index(drop=True).copy()

    if output_path is None:
        output_path = RF_DATASET_SIGMA2_ONLY_FILE

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_sigma2.to_csv(output_path, index=False)
    logger.info(
        "Saved sigma2-only dataset: %s (%d rows, %d columns)", output_path, len(df_sigma2), len(df_sigma2.columns)
    )

    return df_sigma2


def ensure_sigma2_only_dataset(*, include_lags: bool = True) -> Path:
    """Ensure the sigma2-only dataset exists; create it if missing.

    Args:
        include_lags: If True, include sigma2_garch lag features when available.
            Defaults to True to include all available information.

    Returns:
        Path to the sigma2-only dataset.
    """
    if RF_DATASET_SIGMA2_ONLY_FILE.exists():
        # Verify lags exist in the dataset (they should be included by default)
        if include_lags:
            df_existing = pd.read_csv(RF_DATASET_SIGMA2_ONLY_FILE, nrows=0)  # Read only headers
            expected_lag_cols = [f"sigma2_garch_lag_{lag}" for lag in RF_LAG_WINDOWS]
            missing_lags = [col for col in expected_lag_cols if col not in df_existing.columns]
            if missing_lags:
                logger.info(
                    f"Dataset exists but missing lag columns {missing_lags}. "
                    "Recreating with lags included."
                )
                create_dataset_sigma2_only(include_lags=True)
        return RF_DATASET_SIGMA2_ONLY_FILE

    create_dataset_sigma2_only(include_lags=include_lags)
    return RF_DATASET_SIGMA2_ONLY_FILE


def create_dataset_rsi14_only(
    df: pd.DataFrame | None = None,
    output_path: Path | None = None,
    *,
    include_lags: bool = True,
) -> pd.DataFrame:
    """Create a dataset that keeps only rsi_14 and its lags as feature.

    This supports a focused ablation: evaluate the predictive power of RSI-14
    technical indicator alone on `weighted_log_return` at horizon t+1.

    Args:
        df: GARCH data DataFrame. If None, loads from default GARCH file.
        output_path: Path to save dataset. If None, uses RF_DATASET_RSI14_ONLY_FILE.
        include_lags: If True, also keep rsi_14_lag_{k} for k in RF_LAG_WINDOWS.
            Defaults to True to include all available information.

    Returns:
        Dataset containing only target columns (date, split, weighted_log_return)
        plus rsi_14 and its lags as the sole feature.
    """
    from src.random_forest.data_preparation.calculs_indicators import calculate_rsi

    if df is None:
        df = load_garch_data()

    # Normalize column names and calculate RSI
    df_normalized = _normalize_column_names(df)
    df_normalized["rsi_14"] = calculate_rsi(
        cast(pd.Series, df_normalized["weighted_closing"]), period=14
    )
    logger.info("RSI-14 calculated successfully")

    # Create target columns (shift for next-day prediction)
    df_shifted = _create_target_columns(df_normalized)

    # Add lags for RSI-14
    if include_lags:
        df_with_lags = add_lag_features(
            df_shifted,
            feature_columns=["rsi_14"],
            lag_windows=RF_LAG_WINDOWS,
        )
    else:
        df_with_lags = df_shifted

    # Drop non-observable columns
    non_observable_columns = _get_non_observable_columns_to_drop()
    columns_to_remove = [col for col in non_observable_columns if col in df_with_lags.columns]
    if columns_to_remove:
        logger.info("Dropping non-observable columns: %s", columns_to_remove)
        df_with_lags = df_with_lags.drop(columns=columns_to_remove)

    # Select only RSI-related columns
    base_cols = ["date", "split", "weighted_log_return"]
    keep_cols = base_cols + ["rsi_14"]
    if include_lags:
        for lag in RF_LAG_WINDOWS:
            col = f"rsi_14_lag_{lag}"
            if col in df_with_lags.columns:
                keep_cols.append(col)

    # Remove missing values
    keep_cols = [c for c in keep_cols if c in df_with_lags.columns]
    df_rsi14 = cast(pd.DataFrame, df_with_lags[keep_cols]).dropna().reset_index(drop=True).copy()

    if output_path is None:
        output_path = RF_DATASET_RSI14_ONLY_FILE

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_rsi14.to_csv(output_path, index=False)
    logger.info(
        "Saved rsi14-only dataset: %s (%d rows, %d columns)", output_path, len(df_rsi14), len(df_rsi14.columns)
    )

    return df_rsi14


def ensure_rsi14_only_dataset(*, include_lags: bool = True) -> Path:
    """Ensure the rsi14-only dataset exists; create it if missing.

    Args:
        include_lags: If True, include rsi_14 lag features when available.
            Defaults to True to include all available information.

    Returns:
        Path to the rsi14-only dataset.
    """
    if RF_DATASET_RSI14_ONLY_FILE.exists():
        # Verify lags exist in the dataset (they should be included by default)
        if include_lags:
            df_existing = pd.read_csv(RF_DATASET_RSI14_ONLY_FILE, nrows=0)  # Read only headers
            expected_lag_cols = [f"rsi_14_lag_{lag}" for lag in RF_LAG_WINDOWS]
            missing_lags = [col for col in expected_lag_cols if col not in df_existing.columns]
            if missing_lags:
                logger.info(
                    f"Dataset exists but missing lag columns {missing_lags}. "
                    "Recreating with lags included."
                )
                create_dataset_rsi14_only(include_lags=True)
        return RF_DATASET_RSI14_ONLY_FILE

    create_dataset_rsi14_only(include_lags=include_lags)
    return RF_DATASET_RSI14_ONLY_FILE


def _get_technical_indicator_columns(include_lags: bool) -> list[str]:
    """Return list of technical indicator columns (with optional lags)."""
    base_cols = list(RF_TECHNICAL_FEATURE_COLUMNS)
    if not include_lags:
        return base_cols

    lagged_cols = [
        f"{indicator}_lag_{lag}" for indicator in base_cols for lag in RF_LAG_WINDOWS
    ]
    return base_cols + lagged_cols


def create_dataset_technical_indicators(
    df: pd.DataFrame | None = None,
    output_path: Path | None = None,
    *,
    include_lags: bool = True,
) -> pd.DataFrame:
    """Create dataset with multiple technical indicators and their lags."""
    if df is None:
        df = load_garch_data()

    df_normalized = _normalize_column_names(df)
    df_with_indicators = add_technical_indicators(df_normalized)
    df_shifted = _create_target_columns(df_with_indicators)

    if include_lags:
        df_with_lags = add_lag_features(
            df_shifted,
            feature_columns=RF_TECHNICAL_FEATURE_COLUMNS,
            lag_windows=RF_LAG_WINDOWS,
        )
    else:
        df_with_lags = df_shifted

    non_observable_columns = _get_non_observable_columns_to_drop()
    columns_to_remove = [col for col in non_observable_columns if col in df_with_lags.columns]
    if columns_to_remove:
        logger.info("Dropping non-observable columns: %s", columns_to_remove)
        df_with_lags = df_with_lags.drop(columns=columns_to_remove)

    base_cols = ["date", "split", "weighted_log_return"]
    keep_cols = [col for col in base_cols if col in df_with_lags.columns]
    keep_cols.extend([col for col in _get_technical_indicator_columns(include_lags) if col in df_with_lags.columns])

    df_selected = cast(pd.DataFrame, df_with_lags[keep_cols]).dropna().reset_index(drop=True).copy()

    if output_path is None:
        output_path = RF_DATASET_TECHNICAL_INDICATORS_FILE

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_selected.to_csv(output_path, index=False)
    logger.info(
        "Saved technical-indicators dataset: %s (%d rows, %d columns)",
        output_path,
        len(df_selected),
        len(df_selected.columns),
    )

    return df_selected


def ensure_technical_indicators_dataset(*, include_lags: bool = True) -> Path:
    """Ensure the technical indicators dataset exists with requested lags."""
    dataset_path = RF_DATASET_TECHNICAL_INDICATORS_FILE
    if dataset_path.exists():
        if include_lags:
            df_headers = pd.read_csv(dataset_path, nrows=0)
            missing = [
                col
                for col in _get_technical_indicator_columns(include_lags=True)
                if col not in df_headers.columns
            ]
            if missing:
                logger.info(
                    "Technical dataset missing lag columns %s. Recreating dataset.",
                    missing,
                )
                create_dataset_technical_indicators(include_lags=True)
        return dataset_path

    create_dataset_technical_indicators(include_lags=include_lags)
    return dataset_path


def prepare_datasets(
    df: pd.DataFrame | None = None, output_dir: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare two datasets: complete and without ARIMA-GARCH insights.

    Args:
        df: DataFrame with GARCH data. If None, loads from default file.
        output_dir: Directory to save datasets. If None, uses DATA_DIR.

    Returns:
        Tuple of (dataset_complete, dataset_without_insights).

    Raises:
        ValueError: If required columns are missing.
    """
    if df is None:
        df = load_garch_data()

    if output_dir is None:
        output_dir = DATA_DIR

    df_normalized = _normalize_column_names(df)

    df_shifted = _create_target_columns(df_normalized)

    # Add lags for GARCH features only (RSI is not included in complete dataset)
    lag_feature_columns = [col for col in RF_LAG_FEATURE_COLUMNS if col != "weighted_log_return_t"]
    df_with_lags = add_lag_features(
        df_shifted,
        feature_columns=lag_feature_columns,
        lag_windows=RF_LAG_WINDOWS,
    )

    df_clean = _remove_missing_values(df_with_lags)

    non_observable_columns = _get_non_observable_columns_to_drop()
    columns_to_remove = [col for col in non_observable_columns if col in df_clean.columns]
    if columns_to_remove:
        logger.info("Dropping non-observable columns: %s", columns_to_remove)
        df_clean = df_clean.drop(columns=columns_to_remove)

    logger.info("Creating complete dataset with all columns")
    df_complete = df_clean.copy()

    df_without_insights = _create_dataset_without_insights(df_clean)

    _save_datasets(df_complete, df_without_insights, output_dir)

    return df_complete, df_without_insights
