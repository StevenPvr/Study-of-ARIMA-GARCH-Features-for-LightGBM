"""Clean I/O module for GARCH predictions persistence.

Provides academic-grade functions for saving and loading GARCH variance forecasts
with full metadata, temporal validation, and reproducibility guarantees.

Key Features:
- Structured output format with metadata header
- Temporal causality validation before save
- Split preservation and verification
- Compression support for large datasets
- Atomic writes (temp file + rename) for safety
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.constants import GARCH_ESTIMATION_FILE, GARCH_FORECASTS_FILE, GARCH_ML_DATASET_FILE
from src.garch.training_garch.utils import count_splits
from src.utils import (
    ensure_output_dir,
    get_logger,
    load_dataframe,
    load_json_data,
    save_json_pretty,
    save_parquet_and_csv,
    validate_file_exists,
    validate_required_columns,
    validate_temporal_split,
)

logger = get_logger(__name__)


def _resolve_output_path_with_mkdir(output_path: Path | None, default_path: Path) -> Path:
    """Resolve output path with default and ensure parent directory exists.

    Delegates to src.utils.ensure_output_dir() for consistency.

    Args:
        output_path: Optional custom output path.
        default_path: Default path to use if output_path is None.

    Returns:
        Resolved output path.
    """
    out_path = output_path if output_path else default_path
    ensure_output_dir(out_path)
    return out_path


@dataclass
class ForecastMetadata:
    """Metadata for variance forecast outputs.

    Attributes:
        n_observations: Total number of forecasts.
        n_train: Number of train forecasts.
        n_test: Number of test forecasts.
        date_range: (start_date, end_date) as ISO strings.
        model_type: EGARCH specification (e.g., "EGARCH(1,1)").
        distribution: Innovation distribution used.
        refit_frequency: Refit frequency used.
        window_type: Window type (expanding/rolling).
        generated_at: ISO timestamp of generation.
        anti_leakage_validated: Whether temporal causality was validated.
        window_size: Rolling window size (optional).
        initial_window_size: Initial window size for expanding window (optional).
        min_window_size: Minimum window size to start forecasting (optional).
        n_refits_train: Number of refits in train period (optional).
        n_refits_total: Total number of refits (optional).
        convergence_rate: Proportion of successful refits (optional).
    """

    n_observations: int
    n_train: int
    n_test: int
    date_range: tuple[str, str]
    model_type: str
    distribution: str
    refit_frequency: int
    window_type: str
    generated_at: str
    anti_leakage_validated: bool
    window_size: int | None = None
    initial_window_size: int | None = None
    min_window_size: int | None = None
    n_refits_train: int | None = None
    n_refits_total: int | None = None
    convergence_rate: float | None = None


def _validate_forecast_dataframe(df: pd.DataFrame) -> None:
    """Validate forecast DataFrame structure.

    Args:
        df: Forecast DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    validate_required_columns(df, {"date", "split"}, "Forecast DataFrame")

    if df.empty:
        msg = "Cannot save empty forecast DataFrame"
        raise ValueError(msg)


def _extract_metadata(
    df: pd.DataFrame,
    model_type: str,
    distribution: str,
    refit_frequency: int,
    window_type: str,
    window_size: int | None = None,
    initial_window_size: int | None = None,
    min_window_size: int | None = None,
    n_refits_train: int | None = None,
    n_refits_total: int | None = None,
    convergence_rate: float | None = None,
) -> ForecastMetadata:
    """Extract metadata from forecast DataFrame.

    Args:
        df: Forecast DataFrame with date and split columns.
        model_type: EGARCH specification string.
        distribution: Innovation distribution name.
        refit_frequency: Refit frequency parameter.
        window_type: Window type (expanding/rolling).
        window_size: Rolling window size (optional).
        initial_window_size: Initial window size for expanding window (optional).
        min_window_size: Minimum window size to start forecasting (optional).
        n_refits_train: Number of refits in train period (optional).
        n_refits_total: Total number of refits (optional).
        convergence_rate: Proportion of successful refits (optional).

    Returns:
        ForecastMetadata object.
    """
    n_train, n_test = count_splits(df)

    # Extract date range
    date_series = pd.to_datetime(df["date"])
    date_min = date_series.min()
    date_max = date_series.max()

    if pd.isna(date_min) or pd.isna(date_max):
        msg = "Cannot extract date range: missing dates in DataFrame"
        raise ValueError(msg)

    # Convert to Timestamp and validate (type narrowing)
    timestamp_min = pd.Timestamp(date_min)
    timestamp_max = pd.Timestamp(date_max)
    if not isinstance(timestamp_min, pd.Timestamp) or not isinstance(timestamp_max, pd.Timestamp):
        msg = "Cannot extract date range: invalid timestamp conversion"
        raise ValueError(msg)
    if pd.isna(timestamp_min) or pd.isna(timestamp_max):
        msg = "Cannot extract date range: timestamp is NaT"
        raise ValueError(msg)

    date_range = (timestamp_min.isoformat(), timestamp_max.isoformat())

    return ForecastMetadata(
        n_observations=len(df),
        n_train=n_train,
        n_test=n_test,
        date_range=date_range,
        model_type=model_type,
        distribution=distribution,
        refit_frequency=refit_frequency,
        window_type=window_type,
        generated_at=datetime.now(timezone.utc).isoformat(),
        anti_leakage_validated=True,
        window_size=window_size,
        initial_window_size=initial_window_size,
        min_window_size=min_window_size,
        n_refits_train=n_refits_train,
        n_refits_total=n_refits_total,
        convergence_rate=convergence_rate,
    )


def _save_metadata_to_file(metadata: ForecastMetadata, output_path: Path) -> None:
    """Save metadata to JSON file.

    Delegates to src.utils.save_json_pretty() for consistency.

    Args:
        metadata: Metadata object to save.
        output_path: Path to save metadata file.
    """
    metadata_path = output_path.with_suffix(".meta.json")
    save_json_pretty(asdict(metadata), metadata_path)
    logger.info("Saved forecast metadata: %s", metadata_path)


def save_garch_forecasts(
    forecasts: pd.DataFrame,
    *,
    model_type: str = "EGARCH(1,1)",
    distribution: str = "auto",
    refit_frequency: int = 20,
    window_type: str = "expanding",
    output_path: Path | None = None,
    save_metadata: bool = True,
    window_size: int | None = None,
    initial_window_size: int | None = None,
    min_window_size: int | None = None,
    n_refits_train: int | None = None,
    n_refits_total: int | None = None,
    convergence_rate: float | None = None,
) -> None:
    """Save GARCH variance forecasts with metadata.

    Performs temporal validation before saving to ensure anti-leakage guarantees.

    Args:
        forecasts: DataFrame with columns: date, split, garch_forecast_h1, arima_resid.
        model_type: EGARCH specification (default: EGARCH(1,1)).
        distribution: Innovation distribution used (default: auto).
        refit_frequency: Refit frequency parameter (default: 20).
        window_type: Window type - expanding or rolling (default: expanding).
        output_path: Optional custom output path (default: GARCH_FORECASTS_FILE).
        save_metadata: Whether to save metadata JSON file (default: True).
        window_size: Rolling window size (optional).
        initial_window_size: Initial window size for expanding window (optional).
        min_window_size: Minimum window size to start forecasting (optional).
        n_refits_train: Number of refits in train period (optional).
        n_refits_total: Total number of refits (optional).
        convergence_rate: Proportion of successful refits (optional).

    Raises:
        ValueError: If DataFrame is invalid or temporal validation fails.
    """
    _validate_forecast_dataframe(forecasts)
    validate_temporal_split(forecasts, function_name="save_garch_forecasts")

    metadata = _extract_metadata(
        forecasts,
        model_type,
        distribution,
        refit_frequency,
        window_type,
        window_size,
        initial_window_size,
        min_window_size,
        n_refits_train,
        n_refits_total,
        convergence_rate,
    )

    if not metadata.anti_leakage_validated:
        msg = "Forecast metadata must set anti_leakage_validated=True after temporal validation"
        raise ValueError(msg)

    out_path = _resolve_output_path_with_mkdir(output_path, GARCH_FORECASTS_FILE)
    save_parquet_and_csv(forecasts, out_path)

    if save_metadata:
        _save_metadata_to_file(metadata, out_path)

    logger.info(
        "Saved GARCH forecasts: %d obs (%d train, %d test) → %s",
        metadata.n_observations,
        metadata.n_train,
        metadata.n_test,
        out_path,
    )


def save_ml_dataset(
    df: pd.DataFrame,
    *,
    output_path: Path | None = None,
    validate_splits: bool = True,
) -> None:
    """Save ML-ready dataset with GARCH features.

    Validates temporal ordering and split consistency before saving.

    Args:
        df: ML dataset DataFrame with features and target.
        output_path: Optional custom output path (default: GARCH_ML_DATASET_FILE).
        validate_splits: Whether to validate temporal split (default: True).

    Raises:
        ValueError: If required columns are missing or validation fails.
    """
    validate_required_columns(df, {"date"}, "ML dataset")

    if df.empty:
        msg = "Cannot save empty ML dataset"
        raise ValueError(msg)

    # Temporal validation
    if validate_splits and "split" in df.columns:
        validate_temporal_split(df, function_name="save_ml_dataset")

    # Resolve output path
    out_path = _resolve_output_path_with_mkdir(output_path, GARCH_ML_DATASET_FILE)

    # Save (CSV + Parquet)
    save_parquet_and_csv(df, out_path)

    n_train, n_test = count_splits(df)

    logger.info(
        "Saved ML dataset: %d rows (%d train, %d test) → %s",
        len(df),
        n_train,
        n_test,
        out_path,
    )


def load_garch_forecasts(
    path: Path | None = None,
    load_metadata: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Load GARCH forecasts with optional metadata.

    Args:
        path: Optional custom path (default: GARCH_FORECASTS_FILE).
        load_metadata: Whether to load metadata JSON (default: True).

    Returns:
        Tuple of (forecasts_df, metadata_dict or None).

    Raises:
        FileNotFoundError: If forecast file not found.
    """
    in_path = path if path else GARCH_FORECASTS_FILE

    # Load DataFrame (automatically tries Parquet first, then CSV)
    df = load_dataframe(in_path, date_columns=["date"], validate_not_empty=False)

    # Load metadata
    metadata = None
    if load_metadata:
        metadata_path = in_path.with_suffix(".meta.json")
        if metadata_path.exists():
            metadata = load_json_data(metadata_path)

    logger.info("Loaded GARCH forecasts: %d observations from %s", len(df), in_path)
    return df, metadata


def _validate_estimation_fits(fits: dict[str, dict[str, Any]]) -> None:
    """Validate estimation fits dictionary.

    Args:
        fits: Dict mapping distribution names to estimation results.

    Raises:
        ValueError: If fits dict is empty or invalid.
    """
    if not fits:
        msg = "Cannot save empty estimation results"
        raise ValueError(msg)

    for dist_name, fit_result in fits.items():
        if "converged" not in fit_result or "params" not in fit_result:
            msg = f"Fit result for '{dist_name}' missing required keys"
            raise ValueError(msg)


def _build_estimation_document(
    fits: dict[str, dict[str, Any]], n_observations: int
) -> dict[str, Any]:
    """Build estimation document from fits.

    Args:
        fits: Dict mapping distribution names to estimation results.
        n_observations: Number of observations.

    Returns:
        Estimation document dictionary.
    """
    estimation_doc = {
        "n_observations": n_observations,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "distributions_tested": list(fits.keys()),
    }

    for dist_name, fit_result in fits.items():
        key = f"egarch_{dist_name}"
        estimation_doc[key] = _build_distribution_result(dist_name, fit_result)

    return estimation_doc


def save_estimation_results(
    fits: dict[str, dict[str, Any]],
    *,
    n_observations: int,
    output_path: Path | None = None,
) -> None:
    """Save EGARCH parameter estimation results.

    Args:
        fits: Dict mapping distribution names to estimation results.
        n_observations: Number of observations used for estimation.
        output_path: Optional custom output path (default: GARCH_ESTIMATION_FILE).

    Raises:
        ValueError: If fits dict is empty or invalid.
    """
    _validate_estimation_fits(fits)
    out_path = _resolve_output_path_with_mkdir(output_path, GARCH_ESTIMATION_FILE)
    estimation_doc = _build_estimation_document(fits, n_observations)

    # Use atomic write via temp file
    temp_path = out_path.with_suffix(".tmp")
    save_json_pretty(estimation_doc, temp_path)
    temp_path.replace(out_path)

    logger.info(
        "Saved estimation results: %d distributions, %d obs → %s",
        len(fits),
        n_observations,
        out_path,
    )


def load_estimation_results(
    path: Path | None = None,
) -> dict[str, Any]:
    """Load EGARCH parameter estimation results.

    Args:
        path: Optional custom path (default: GARCH_ESTIMATION_FILE).

    Returns:
        Estimation results dictionary.

    Raises:
        FileNotFoundError: If estimation file not found.
    """
    in_path = path if path else GARCH_ESTIMATION_FILE

    validate_file_exists(in_path, "Estimation file")

    estimation_doc = load_json_data(in_path)

    logger.info(
        "Loaded estimation results: %d distributions from %s",
        len(estimation_doc.get("distributions_tested", [])),
        in_path,
    )

    return estimation_doc


def _build_distribution_result(dist_name: str, fit_result: dict[str, Any]) -> dict[str, Any]:
    """Build estimation result for a single distribution.

    Args:
        dist_name: Distribution name.
        fit_result: Fit result dictionary.

    Returns:
        Formatted result dictionary.
    """
    result = {
        "converged": bool(fit_result.get("converged", False)),
        "params": fit_result["params"],
        "log_likelihood": float(fit_result.get("log_likelihood", float("nan"))),
        "iterations": int(fit_result.get("iterations", 0)),
    }

    # Add optional keys if present
    if "convergence_message" in fit_result:
        result["convergence_message"] = str(fit_result["convergence_message"])
    if "aic" in fit_result:
        result["aic"] = float(fit_result["aic"])
    if "bic" in fit_result:
        result["bic"] = float(fit_result["bic"])

    return result


__all__ = [
    "ForecastMetadata",
    "save_garch_forecasts",
    "save_ml_dataset",
    "load_garch_forecasts",
    "save_estimation_results",
    "load_estimation_results",
]
