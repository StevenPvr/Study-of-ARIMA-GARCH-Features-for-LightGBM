"""GARCH forecasting orchestration for full sample (TRAIN + TEST).

This module provides the main entry point for generating leak-free GARCH
variance forecasts on the complete sample:

1. TRAIN split: Expanding window forecasts with periodic refit
2. TEST split: Forecasts generated separately using model trained on TRAIN
3. Combined output: garch_forecasts.csv with h=1 forecasts for LightGBM

CRITICAL TEMPORAL SEPARATION:
- Forecaster is trained ONLY on TRAIN data
- TEST forecasts are generated using the trained model
- This ensures NO data leakage and complete temporal separation

All forecasts are σ²_t+1|t (one-step-ahead), never σ²_t|t (filtered).
This ensures NO data leakage when used as features in downstream models.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.constants import (
    GARCH_DATASET_FILE,
    GARCH_EVALUATION_DIR,
    GARCH_INITIAL_WINDOW_SIZE_DEFAULT,
    GARCH_MIN_WINDOW_SIZE,
    GARCH_MODEL_FILE,
)
from src.garch.training_garch.forecaster import EGARCHForecaster, ForecastResult
from src.garch.training_garch.utils import count_splits, load_optimized_hyperparameters
from src.utils import get_logger

logger = get_logger(__name__)


def load_garch_dataset() -> pd.DataFrame:
    """Load GARCH dataset (ARIMA residuals).

    Returns:
        DataFrame with columns: date, arima_resid (or sarima_resid), split.

    Raises:
        FileNotFoundError: If dataset not found.
        ValueError: If required columns are missing.
    """
    if not GARCH_DATASET_FILE.exists():
        msg = f"GARCH dataset not found: {GARCH_DATASET_FILE}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(GARCH_DATASET_FILE, parse_dates=["date"])

    required_cols = ["date", "split"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    # Check for residual columns or columns needed to compute residuals
    # Accept either: arima_resid/sarima_resid OR weighted_log_return + arima_fitted_in_sample
    has_residual_col = "arima_resid" in df.columns or "sarima_resid" in df.columns
    has_compute_cols = "weighted_log_return" in df.columns and "arima_fitted_in_sample" in df.columns
    if not has_residual_col and not has_compute_cols:
        msg = (
            "Missing required columns: either residual column ('arima_resid' or 'sarima_resid') "
            "or columns to compute residuals ('weighted_log_return' and 'arima_fitted_in_sample')"
        )
        raise ValueError(msg)

    n_train, n_test = count_splits(df)
    logger.info(
        "Loaded GARCH dataset: %d observations, %d train, %d test",
        len(df),
        n_train,
        n_test,
    )

    return df


def _calculate_weighted_convergence_rate(
    result_train: ForecastResult,
    result_test: ForecastResult,
) -> float:
    """Calculate weighted average convergence rate across train and test splits.

    Args:
        result_train: ForecastResult from training split.
        result_test: ForecastResult from test split.

    Returns:
        Weighted average convergence rate, or 1.0 if no refits occurred.
    """
    n_refits_total = result_train.n_refits + result_test.n_refits

    if n_refits_total > 0:
        convergence_rate = (
            result_train.convergence_rate * result_train.n_refits
            + result_test.convergence_rate * result_test.n_refits
        ) / n_refits_total
    else:
        convergence_rate = 1.0

    return convergence_rate


def _resolve_optimized_hyperparameters(
    o: int | None,
    p: int | None,
    dist: str | None,
    refit_frequency: int | None,
    window_type: str | None,
    window_size: int | None,
) -> tuple[int, int, str, int, str, int | None]:
    """Resolve hyperparameters from optimization results with optional overrides."""
    opt_params = load_optimized_hyperparameters()

    resolved_o = o if o is not None else opt_params["o"]
    resolved_p = p if p is not None else opt_params["p"]
    resolved_dist = dist if dist is not None else opt_params["distribution"]
    resolved_refit_freq = (
        refit_frequency if refit_frequency is not None else opt_params["refit_freq"]
    )
    resolved_window_type = window_type if window_type is not None else opt_params["window_type"]
    resolved_window_size = window_size if window_size is not None else opt_params.get("window_size")

    logger.info(
        "Using optimized hyperparameters: EGARCH(%d,%d), distribution=%s, "
        "refit_freq=%d, window_type=%s, window_size=%s",
        resolved_o,
        resolved_p,
        resolved_dist,
        resolved_refit_freq,
        resolved_window_type,
        resolved_window_size,
    )

    return (
        resolved_o,
        resolved_p,
        resolved_dist,
        resolved_refit_freq,
        resolved_window_type,
        resolved_window_size,
    )


def _validate_explicit_hyperparameters(
    o: int | None,
    p: int | None,
    dist: str | None,
    refit_frequency: int | None,
    window_type: str | None,
) -> None:
    """Validate that all required hyperparameters are provided explicitly."""
    if o is None or p is None or dist is None or refit_frequency is None or window_type is None:
        msg = (
            "When use_optimized_params=False, all hyperparameters must be "
            "provided explicitly. "
            f"Missing: o={o}, p={p}, dist={dist}, "
            f"refit_frequency={refit_frequency}, window_type={window_type}"
        )
        raise ValueError(msg)


def _resolve_hyperparameters(
    use_optimized: bool,
    o: int | None,
    p: int | None,
    dist: str | None,
    refit_frequency: int | None,
    window_type: str | None,
    window_size: int | None,
) -> tuple[int, int, str, int, str, int | None]:
    """Resolve hyperparameters from optimization or explicit arguments.

    Args:
        use_optimized: If True, load from optimization results.
        o: ARCH order (overrides optimized if provided).
        p: GARCH order (overrides optimized if provided).
        dist: Distribution (overrides optimized if provided).
        refit_frequency: Refit frequency (overrides optimized if provided).
        window_type: Window type (overrides optimized if provided).
        window_size: Window size (overrides optimized if provided).

    Returns:
        Tuple of (o, p, dist, refit_frequency, window_type, window_size).

    Raises:
        FileNotFoundError: If use_optimized=True and optimization results not found.
        ValueError: If use_optimized=False and required parameters not provided.
    """
    if use_optimized:
        return _resolve_optimized_hyperparameters(
            o, p, dist, refit_frequency, window_type, window_size
        )

    # use_optimized=False: all parameters must be provided explicitly
    _validate_explicit_hyperparameters(o, p, dist, refit_frequency, window_type)

    # After validation, cast to non-None types since validation guarantees they exist
    from typing import cast

    return (
        cast(int, o),
        cast(int, p),
        cast(str, dist),
        cast(int, refit_frequency),
        cast(str, window_type),
        window_size,
    )


def _generate_train_forecasts(
    forecaster: EGARCHForecaster,
    resid_train: np.ndarray,
    dates_train: pd.DatetimeIndex,
) -> ForecastResult:
    """Generate forecasts on TRAIN data with expanding window.

    Args:
        forecaster: EGARCH forecaster instance.
        resid_train: Training residuals.
        dates_train: Training dates.

    Returns:
        ForecastResult with train forecasts.
    """
    logger.info("Generating TRAIN forecasts (expanding window)...")
    return forecaster.forecast_expanding(resid_train, dates=dates_train)


def _generate_test_forecasts(
    forecaster: EGARCHForecaster,
    resid_train: np.ndarray,
    resid_test: np.ndarray,
    dates_test: pd.DatetimeIndex,
) -> ForecastResult:
    """Generate forecasts on TEST data using model trained on TRAIN.

    CRITICAL: This function ensures temporal separation by:
    1. Starting with model trained only on TRAIN data
    2. Generating TEST forecasts sequentially from that state
    3. Each TEST forecast at time t uses only data up to t-1

    Args:
        forecaster: EGARCH forecaster instance (already trained on TRAIN).
        resid_train: Training residuals (used for refit windows).
        resid_test: Test residuals.
        dates_test: Test dates.

    Returns:
        ForecastResult with test forecasts.
    """
    logger.info("Generating TEST forecasts (rolling/expanding from trained model)...")

    # Continue forecasts from current state without regenerating train forecasts
    # This uses the forecaster's current parameters and refit manager state
    return forecaster.forecast_continuing(resid_train, resid_test, dates_test)


def _assign_forecasts_to_dataframes(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    result_train: ForecastResult,
    result_test: ForecastResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign forecast results to train and test DataFrames.

    Returns:
        Tuple of (df_train_with_forecasts, df_test_with_forecasts).
    """
    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train["garch_forecast_h1"] = result_train.forecasts
    df_train["garch_vol_h1"] = result_train.volatility
    df_train["forecast_type"] = "expanding"
    df_train["refit_occurred"] = result_train.refit_mask

    df_test["garch_forecast_h1"] = result_test.forecasts
    df_test["garch_vol_h1"] = result_test.volatility
    df_test["forecast_type"] = "rolling"
    df_test["refit_occurred"] = result_test.refit_mask

    return df_train, df_test


def _save_combined_forecasts(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    result_train: ForecastResult,
    result_test: ForecastResult,
    o: int,
    p: int,
    dist: str,
    refit_frequency: int,
    window_type: str,
    window_size: int | None,
    initial_window_size: int,
    min_window_size: int,
    output_dir: Path,
) -> Path:
    """Combine train and test forecasts and save with metadata.

    Returns:
        Path to saved forecast file.
    """
    from src.garch.training_garch.predictions_io import save_garch_forecasts

    df_combined: pd.DataFrame = pd.concat([df_train, df_test], ignore_index=True)  # type: ignore[assignment]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "garch_forecasts.parquet"

    n_refits_train = result_train.n_refits
    n_refits_total = result_train.n_refits + result_test.n_refits
    convergence_rate = _calculate_weighted_convergence_rate(result_train, result_test)

    save_garch_forecasts(
        df_combined,
        model_type=f"EGARCH({o},{p})",
        distribution=dist,
        refit_frequency=refit_frequency,
        window_type=window_type,
        output_path=output_file,
        save_metadata=True,
        window_size=window_size,
        initial_window_size=initial_window_size,
        min_window_size=min_window_size,
        n_refits_train=n_refits_train,
        n_refits_total=n_refits_total,
        convergence_rate=convergence_rate,
    )

    return output_file


def _prepare_forecast_data(
    df_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Prepare data for forecasting by splitting and extracting arrays.

    Args:
        df_data: Full dataset with 'split' column.

    Returns:
        Tuple of (df_train, df_test, resid_train, resid_test, dates_train, dates_test).
    """
    # Defensive: ensure strict temporal ordering before any split.
    # Why: Upstream sources should already be sorted, but we avoid relying on it.
    df_sorted = df_data.sort_values("date").reset_index(drop=True)

    df_train = pd.DataFrame(df_sorted[df_sorted["split"] == "train"]).copy()
    df_test = pd.DataFrame(df_sorted[df_sorted["split"] == "test"]).copy()

    from src.garch.training_garch.utils import _extract_direct_residuals

    resid_train = _extract_direct_residuals(df_train)
    # For test data, try to extract directly (may need residual column)
    if "arima_resid" in df_test.columns:
        resid_test = np.asarray(df_test["arima_resid"], dtype=float)
    elif "sarima_resid" in df_test.columns:
        resid_test = np.asarray(df_test["sarima_resid"], dtype=float)
    else:
        # Test residuals should come from residual column, not computed
        msg = (
            "Test residuals require 'arima_resid' or 'sarima_resid' column. "
            "Cannot compute test residuals from fitted values."
        )
        raise ValueError(msg)
    dates_train = pd.DatetimeIndex(df_train["date"])
    dates_test = pd.DatetimeIndex(df_test["date"])

    return df_train, df_test, resid_train, resid_test, dates_train, dates_test


def _create_forecaster_from_params(
    o: int,
    p: int,
    dist: str,
    refit_frequency: int,
    window_type: str,
    window_size: int | None,
    initial_window_size: int,
    min_window_size: int,
    use_fixed_params: bool = False,
    anchor_at_min_window: bool = False,
) -> EGARCHForecaster:
    """Create EGARCHForecaster from parameters.

    Args:
        o: ARCH order.
        p: GARCH order.
        dist: Distribution.
        refit_frequency: Refit frequency.
        window_type: Window type.
        window_size: Window size.
        initial_window_size: Initial window size.
        min_window_size: Minimum window size.

    Returns:
        Configured EGARCHForecaster instance.
    """
    return EGARCHForecaster(
        o=o,
        p=p,
        dist=dist,
        refit_frequency=refit_frequency,
        window_type=window_type,
        window_size=window_size,
        initial_window_size=initial_window_size,
        min_window_size=min_window_size,
        use_fixed_params=use_fixed_params,
        anchor_at_min_window=anchor_at_min_window,
    )


def _log_summary(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    result_train: ForecastResult,
    result_test: ForecastResult,
    o: int,
    p: int,
    dist: str,
    output_file: Path | None,
) -> None:
    """Log summary of forecasting results."""
    n_refits_total = result_train.n_refits + result_test.n_refits
    n_total = len(df_train) + len(df_test)
    convergence_rate = _calculate_weighted_convergence_rate(result_train, result_test)

    logger.info("=" * 60)
    logger.info("GARCH FORECASTING COMPLETE")
    logger.info("=" * 60)
    logger.info("  Model: EGARCH(%d,%d), dist=%s", o, p, dist)
    logger.info("  Train: %d obs, %d refits", len(df_train), result_train.n_refits)
    logger.info("  Test: %d obs, %d refits", len(df_test), result_test.n_refits)
    logger.info(
        "  Total: %d obs, %d refits, %.1f%% converged",
        n_total,
        n_refits_total,
        convergence_rate * 100,
    )
    if output_file is not None:
        logger.info("  Output: %s (+ CSV)", output_file)
        logger.info("  Metadata: %s", output_file.with_suffix(".meta.json"))
    else:
        logger.info("  Output: None (+ CSV)")
    logger.info("=" * 60)


def generate_full_sample_forecasts(
    *,
    use_optimized_params: bool = True,
    o: int | None = None,
    p: int | None = None,
    dist: str | None = None,
    refit_frequency: int | None = None,
    window_type: str | None = None,
    window_size: int | None = None,
    initial_window_size: int = GARCH_INITIAL_WINDOW_SIZE_DEFAULT,
    min_window_size: int = GARCH_MIN_WINDOW_SIZE,
    output_dir: Path = GARCH_EVALUATION_DIR,
    anchor_at_min_window: bool = False,
) -> pd.DataFrame:
    """Generate GARCH forecasts for full sample (TRAIN + TEST).

    CRITICAL TEMPORAL SEPARATION:
    This function ensures complete temporal separation by:
    1. Training forecaster ONLY on TRAIN data
    2. Generating TEST forecasts using the trained model
    3. Each forecast uses only information available at that time

    Args:
        use_optimized_params: Use optimized hyperparameters if True.
        o: ARCH order (overrides optimized).
        p: GARCH order (overrides optimized).
        dist: Distribution (overrides optimized).
        refit_frequency: Refit frequency (overrides optimized).
        window_type: Window type (overrides optimized).
        window_size: Rolling window size.
        initial_window_size: Initial training window size.
        min_window_size: Minimum window size to start forecasting.
        output_dir: Output directory for results.
        anchor_at_min_window: If True, anchor initial fit/scheduling at
            ``min_window_size`` to start forecasts earlier (evaluation mode).

    Returns:
        Combined DataFrame with train and test forecasts.
    """
    df_data = load_garch_dataset()
    df_train, df_test, resid_train, resid_test, dates_train, dates_test = _prepare_forecast_data(
        df_data
    )

    o, p, dist, refit_frequency, window_type, window_size = _resolve_hyperparameters(
        use_optimized_params, o, p, dist, refit_frequency, window_type, window_size
    )

    logger.info(
        "Generating forecasts with EGARCH(%d,%d), dist=%s, refit_freq=%d, window=%s",
        o,
        p,
        dist,
        refit_frequency,
        window_type,
    )

    forecaster = _create_forecaster_from_params(
        o,
        p,
        dist,
        refit_frequency,
        window_type,
        window_size,
        initial_window_size,
        min_window_size,
        use_fixed_params=False,
        anchor_at_min_window=anchor_at_min_window,
    )

    result_train = _generate_train_forecasts(forecaster, resid_train, dates_train)
    result_test = _generate_test_forecasts(forecaster, resid_train, resid_test, dates_test)

    df_train, df_test = _assign_forecasts_to_dataframes(
        df_train, df_test, result_train, result_test
    )

    output_file = _save_combined_forecasts(
        df_train,
        df_test,
        result_train,
        result_test,
        o,
        p,
        dist,
        refit_frequency,
        window_type,
        window_size,
        initial_window_size,
        min_window_size,
        output_dir,
    )

    _log_summary(df_train, df_test, result_train, result_test, o, p, dist, output_file)

    df_combined: pd.DataFrame = pd.concat([df_train, df_test], ignore_index=True)  # type: ignore[assignment]
    return df_combined


def generate_full_sample_forecasts_from_trained_model(
    *,
    min_window_size: int = GARCH_MIN_WINDOW_SIZE,
    anchor_at_min_window: bool = False,
) -> pd.DataFrame:
    """Generate forecasts using the trained model from model.joblib.

    This function loads the trained EGARCH forecaster from model.joblib
    and uses it directly to generate forecasts, avoiding re-estimation.

    Returns:
        Combined DataFrame with train and test forecasts.

    Raises:
        FileNotFoundError: If model.joblib is not found.
        RuntimeError: If forecast generation fails.
    """
    if not GARCH_MODEL_FILE.exists():
        msg = f"Trained model file not found: {GARCH_MODEL_FILE}"
        raise FileNotFoundError(msg)

    logger.info("Loading trained EGARCH model from %s", GARCH_MODEL_FILE)
    model_data = joblib.load(GARCH_MODEL_FILE)

    # Extract required keys from model and metadata without silent fallbacks
    o = int(model_data["o"])  # always required
    p = int(model_data["p"])  # always required
    dist = str(model_data["dist"])  # always required

    # Load complementary configuration from metadata.json (written at training time)
    from src.constants import GARCH_MODEL_METADATA_FILE

    if not GARCH_MODEL_METADATA_FILE.exists():
        msg = (
            "Trained model metadata not found: "
            f"{GARCH_MODEL_METADATA_FILE}. "
            "Cannot reconstruct trained forecaster without metadata."
        )
        raise FileNotFoundError(msg)

    import json

    with open(GARCH_MODEL_METADATA_FILE, "r") as f:
        metadata = json.load(f)

    try:
        window_type = str(metadata["window_type"])  # 'expanding' or 'rolling'
        refit_frequency = int(metadata["refit_frequency"])  # strict
        # window_size may be None for expanding
        window_size_val = metadata.get("window_size", None)
        window_size = int(window_size_val) if window_size_val is not None else None
        initial_window_size = int(metadata["initial_window_size"])  # anchor for start position
    except KeyError as exc:
        # Explicit failure to avoid hidden defaults
        raise KeyError(
            f"Missing key in model metadata: {exc}. Re-run training to regenerate metadata."
        ) from exc

    logger.info(
        "Using trained model: EGARCH(%d,%d), dist=%s, refit_freq=%d, "
        "window=%s, initial_window=%d, window_size=%s",
        o,
        p,
        dist,
        refit_frequency,
        window_type,
        initial_window_size,
        str(window_size),
    )

    # Load and prepare data
    df_data = load_garch_dataset()
    df_train, df_test, resid_train, resid_test, dates_train, dates_test = _prepare_forecast_data(
        df_data
    )

    # Reconstruct the forecaster with the saved configuration
    # Note: We use the saved hyperparameters but the model parameters are already trained
    forecaster = _create_forecaster_from_params(
        o,
        p,
        dist,
        refit_frequency,
        window_type,
        window_size,
        initial_window_size=initial_window_size,
        min_window_size=min_window_size,
        use_fixed_params=True,  # Use fixed pre-trained parameters
        anchor_at_min_window=anchor_at_min_window,
    )

    # Load the trained parameters into the refit manager
    params_dict = model_data["params"]

    # Update the forecaster with parameters
    forecaster.refit_manager.current_params = params_dict

    logger.info(
        "Using pre-trained parameters: omega=%.4f, alpha=%.4f, gamma=%.4f",
        params_dict["omega"],
        params_dict.get("alpha", 0),
        params_dict.get("gamma", 0),
    )

    # Generate forecasts using the trained model
    result_train = _generate_train_forecasts(forecaster, resid_train, dates_train)
    result_test = _generate_test_forecasts(forecaster, resid_train, resid_test, dates_test)

    df_train, df_test = _assign_forecasts_to_dataframes(
        df_train, df_test, result_train, result_test
    )

    _log_summary(df_train, df_test, result_train, result_test, o, p, dist, None)

    df_combined: pd.DataFrame = pd.concat([df_train, df_test], ignore_index=True)  # type: ignore[assignment]
    return df_combined


def generate_full_sample_forecasts_hybrid(
    *,
    min_window_size: int = GARCH_MIN_WINDOW_SIZE,
    anchor_at_min_window: bool = False,
) -> pd.DataFrame:
    """Generate forecasts using trained params to seed, and optimized hyperparams for refits.

    Behavior:
    - Loads best hyperparameters (o, p, distribution, refit_freq, window_type, window_size)
      from optimization results.
    - Loads trained parameters from model.joblib and seeds the forecaster's initial state.
    - Enables refits (use_fixed_params=False) with the optimized refit schedule and window.

    Returns:
        Combined DataFrame with train and test forecasts.

    Raises:
        FileNotFoundError: If model.joblib or optimization results are missing.
        ValueError: If trained model orders/dist mismatch optimized hyperparameters.
    """
    # Load data
    df_data = load_garch_dataset()
    df_train, df_test, resid_train, resid_test, dates_train, dates_test = _prepare_forecast_data(
        df_data
    )

    # Load optimized hyperparameters
    opt = load_optimized_hyperparameters()
    o_opt = int(opt["o"])  # required
    p_opt = int(opt["p"])  # required
    dist_opt = str(opt["distribution"])  # required
    refit_freq = int(opt["refit_freq"])  # required
    window_type = str(opt["window_type"])  # required
    window_size = int(opt["window_size"]) if opt.get("window_size") is not None else None

    # Load trained model params to seed initial state
    if not GARCH_MODEL_FILE.exists():
        raise FileNotFoundError(f"Trained model file not found: {GARCH_MODEL_FILE}")
    import joblib

    logger.info("Loading trained EGARCH model from %s", str(GARCH_MODEL_FILE.resolve()))
    model_data = joblib.load(GARCH_MODEL_FILE)
    o_tr = int(model_data["o"])  # strict
    p_tr = int(model_data["p"])  # strict
    dist_tr = str(model_data["dist"])  # strict
    params_seed: dict[str, float] = model_data["params"]

    # Strict consistency checks to avoid mismatched parameterization
    if (o_tr != o_opt) or (p_tr != p_opt):
        raise ValueError(
            f"Order mismatch between trained model and optimized hyperparams: "
            f"trained EGARCH({o_tr},{p_tr}) vs optimized EGARCH({o_opt},{p_opt})."
        )
    if dist_tr.lower() != dist_opt.lower():
        raise ValueError(
            "Distribution mismatch: trained model uses '"
            f"{dist_tr}', optimized hyperparams use '{dist_opt}'."
        )

    logger.info(
        "Hybrid mode: seed with trained params and refit with optimized schedule: "
        "EGARCH(%d,%d), dist=%s, refit_freq=%d, window=%s, window_size=%s",
        o_opt,
        p_opt,
        dist_opt,
        refit_freq,
        window_type,
        str(window_size),
    )

    # Build forecaster with refits enabled
    forecaster = _create_forecaster_from_params(
        o_opt,
        p_opt,
        dist_opt,
        refit_freq,
        window_type,
        window_size,
        initial_window_size=GARCH_INITIAL_WINDOW_SIZE_DEFAULT,
        min_window_size=min_window_size,
        use_fixed_params=False,  # refits enabled
        anchor_at_min_window=anchor_at_min_window,
    )
    # Seed initial state with trained parameters and enable hybrid seeding
    forecaster.refit_manager.current_params = params_seed
    forecaster.seed_with_current_params = True

    # Generate forecasts
    result_train = _generate_train_forecasts(forecaster, resid_train, dates_train)
    # Disable seeding for continuation; state is in refit_manager now
    forecaster.seed_with_current_params = False
    result_test = _generate_test_forecasts(forecaster, resid_train, resid_test, dates_test)

    df_train, df_test = _assign_forecasts_to_dataframes(
        df_train, df_test, result_train, result_test
    )

    output_file = _save_combined_forecasts(
        df_train,
        df_test,
        result_train,
        result_test,
        o_opt,
        p_opt,
        dist_opt,
        refit_freq,
        window_type,
        window_size,
        GARCH_INITIAL_WINDOW_SIZE_DEFAULT,
        GARCH_MIN_WINDOW_SIZE,
        GARCH_EVALUATION_DIR,
    )

    _log_summary(df_train, df_test, result_train, result_test, o_opt, p_opt, dist_opt, output_file)

    df_combined: pd.DataFrame = pd.concat([df_train, df_test], ignore_index=True)  # type: ignore[assignment]
    return df_combined
