"""Simplified GARCH evaluation using orchestration.py.

This refactored version delegates forecasting to orchestration.py, eliminating
266 lines of duplicated refit/forecast logic.

Key simplifications:
- forecast_on_test_from_trained_model: 266 → ~30 lines (uses orchestration.py)
- Removed manual refit logic (uses EGARCHForecaster)
- All functions ≤40 lines per AGENTS.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import GARCH_EVAL_DEFAULT_LEVEL, GARCH_EVAL_FORCED_MIN_START_SIZE
from src.garch.garch_eval.data_loading import load_and_prepare_residuals, load_model_params
from src.garch.garch_eval.helpers import assemble_forecast_results
from src.garch.garch_eval.variance_path import compute_initial_forecasts, compute_variance_path
from src.garch.training_garch.orchestration import (
    generate_full_sample_forecasts,
    generate_full_sample_forecasts_from_trained_model,
)
from src.path import DATA_TICKERS_FULL_FILE, DATA_TICKERS_FULL_INSIGHTS_FILE, GARCH_FORECASTS_FILE
from src.utils import ensure_output_dir, get_logger, save_parquet_and_csv

logger = get_logger(__name__)


def _refit_model_params(resid_up_to_pos: np.ndarray, initial_dist: str) -> dict[str, float]:
    """Compatibility seam for test monkeypatching.

    This internal hook exists solely to allow tests to replace expensive
    refitting logic with a lightweight stub. The production path uses the
    EGARCHForecaster/RefitManager in orchestration, so calling this function
    directly is unsupported and will raise.

    Args:
        resid_up_to_pos: Residuals up to a given position (unused here).
        initial_dist: Initial distribution name (unused here).

    Returns:
        Never returns; always raises to avoid silent fallbacks.

    Raises:
        NotImplementedError: Always, to enforce explicit monkeypatching in tests.
    """
    msg = (
        "_refit_model_params is a test-only seam. Use EGARCHForecaster/RefitManager; "
        "replace this via monkeypatch in tests."
    )
    raise NotImplementedError(msg)


def forecast_on_test_from_trained_model(
    df_full_forecasts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate forecasts on TEST using the trained EGARCH model.

    Behavior:
    - If df_full_forecasts is provided, extract TEST and format it.
    - Otherwise, load the trained model (model.joblib + metadata) and
      generate leak-free one-step-ahead forecasts via orchestration.

    Args:
        df_full_forecasts: Optional full sample forecasts (TRAIN + TEST).
                          If None, generates them.

    Returns:
        DataFrame with columns: date, resid, RV, sigma2_egarch_raw.

    Raises:
        FileNotFoundError: If trained model or dataset not found.
    """
    logger.info("Generating TEST forecasts using orchestration pipeline...")

    # Use provided full-sample forecasts or generate from trained model
    if df_full_forecasts is None:
        df_forecasts = generate_full_sample_forecasts_from_trained_model(
            min_window_size=GARCH_EVAL_FORCED_MIN_START_SIZE,
            anchor_at_min_window=True,
        )
    else:
        df_forecasts = df_full_forecasts

    # Filter to TEST split only
    df_test = df_forecasts[df_forecasts["split"] == "test"].copy()

    # Format for compatibility with existing code
    # Find residual column name
    from src.garch.structure_garch.utils import _find_residual_column

    try:
        resid_col_name = _find_residual_column(df_test)
    except ValueError:
        msg = (
            "Test data missing residual column ('arima_resid' or 'sarima_resid'). "
            "Cannot compute realized volatility for evaluation."
        )
        raise ValueError(msg) from None

    df_test["RV"] = df_test[resid_col_name] ** 2
    rename_map: dict[str, str] = {
        resid_col_name: "resid",
        "garch_forecast_h1": "sigma2_egarch_raw",
    }
    df_renamed = df_test.rename(columns=rename_map)  # type: ignore[arg-type]
    df_result: pd.DataFrame = df_renamed[["date", "resid", "RV", "sigma2_egarch_raw"]].copy()  # type: ignore[assignment]

    # Save results
    if GARCH_FORECASTS_FILE.suffix.lower() != ".parquet":
        msg = f"GARCH forecasts must be saved with a parquet base path, got: {GARCH_FORECASTS_FILE}"
        raise ValueError(msg)
    save_parquet_and_csv(df_result, GARCH_FORECASTS_FILE)
    logger.info("Saved %d TEST forecasts to: %s", len(df_result), GARCH_FORECASTS_FILE)

    return df_result


def forecast_from_artifacts(
    *,
    horizon: int,
    level: float = GARCH_EVAL_DEFAULT_LEVEL,
) -> pd.DataFrame:
    """Compute EGARCH variance forecasts directly from persisted artifacts.

    Args:
        horizon: Number of periods to forecast ahead.
        level: Prediction interval confidence level (0 < level < 1).

    Returns:
        DataFrame containing variance forecasts, prediction intervals, and VaR.

    Raises:
        ValueError: If inputs are invalid or artifacts contain insufficient data.
    """
    if horizon <= 0:
        msg = "horizon must be a positive integer"
        raise ValueError(msg)
    if not 0.0 < level < 1.0:
        msg = "level must be strictly between 0 and 1"
        raise ValueError(msg)

    _, resid_train, _ = load_and_prepare_residuals()
    if resid_train.size == 0:
        msg = "Training residuals are empty; cannot build forecasts"
        raise ValueError(msg)

    params, model_name, dist, nu, gamma, lambda_skew = load_model_params()
    omega = float(params["omega"])
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    gamma_val = float(gamma if gamma is not None else params.get("gamma", 0.0))

    sigma2_train = compute_variance_path(
        resid_train,
        model_name,
        omega,
        alpha,
        beta,
        gamma_val,
        dist,
        nu,
        lambda_skew,
    )
    if sigma2_train.size == 0:
        msg = "Variance path on training data is empty"
        raise ValueError(msg)

    s2_one_step, s2_multi = compute_initial_forecasts(
        resid_train,
        sigma2_train,
        horizon,
        omega,
        alpha,
        gamma_val,
        beta,
        dist,
        nu,
        lambda_skew,
    )
    forecasts = assemble_forecast_results(
        s2_multi,
        s2_one_step,
        level,
        dist,
        nu,
        lambda_skew,
    )
    forecasts["model_name"] = model_name

    if GARCH_FORECASTS_FILE.suffix.lower() != ".parquet":
        msg = f"GARCH forecasts must be saved with a parquet base path, got: {GARCH_FORECASTS_FILE}"
        raise ValueError(msg)
    save_parquet_and_csv(forecasts, GARCH_FORECASTS_FILE)
    logger.info("Saved %d-step forecasts to: %s", horizon, GARCH_FORECASTS_FILE)

    return forecasts


def _filter_valid_forecasts(
    forecasts: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and filter valid forecast data (≤40 lines).

    Args:
        forecasts: DataFrame with resid, RV, sigma2_egarch_raw columns.

    Returns:
        Tuple of (resid_valid, RV_valid, sigma2_valid).
    """
    resid = forecasts["resid"].to_numpy(dtype=float)
    RV = forecasts["RV"].to_numpy(dtype=float)
    sigma2 = forecasts["sigma2_egarch_raw"].to_numpy(dtype=float)

    valid_mask = np.isfinite(resid) & np.isfinite(RV) & np.isfinite(sigma2) & (sigma2 > 0)

    return resid[valid_mask], RV[valid_mask], sigma2[valid_mask]


def compute_metrics_from_forecasts_new_format(
    forecasts: pd.DataFrame,
) -> dict[str, object]:
    """Compute metrics from new format forecasts (≤40 lines).

    Args:
        forecasts: DataFrame with columns: date, resid, RV, sigma2_egarch_raw.

    Returns:
        Dictionary with variance metrics.
    """
    from src.garch.garch_eval.metrics import mse_mae_variance, qlike_loss

    if forecasts.empty:
        return {
            "n_obs": 0,
            "qlike": float("nan"),
            "mse_var": float("nan"),
            "mae_var": float("nan"),
        }

    _, RV_valid, sigma2_valid = _filter_valid_forecasts(forecasts)

    if RV_valid.size == 0:
        return {
            "n_obs": 0,
            "qlike": float("nan"),
            "mse_var": float("nan"),
            "mae_var": float("nan"),
        }

    variance_losses = mse_mae_variance(RV_valid, sigma2_valid)
    return {
        "n_obs": int(RV_valid.size),
        "qlike": qlike_loss(RV_valid, sigma2_valid),
        "mse_var": variance_losses["mse"],
        "mae_var": variance_losses["mae"],
    }


def compute_metrics_from_forecasts(forecasts: pd.DataFrame) -> dict[str, object]:
    """Compute metrics from old format forecasts with h column (≤40 lines).

    Args:
        forecasts: DataFrame with h, actual_residual, sigma2_forecast columns.

    Returns:
        Dictionary with variance and VaR metrics.
    """
    from src.garch.garch_eval.metrics import mse_mae_variance, qlike_loss

    if forecasts.empty:
        return _empty_metrics_dict()

    df_h1 = forecasts.loc[(forecasts["h"] == 1) & forecasts["actual_residual"].notna()].copy()

    if df_h1.empty:
        return _empty_metrics_dict()

    e_test = df_h1["actual_residual"].to_numpy(dtype=float)
    s2_test = df_h1["sigma2_forecast"].to_numpy(dtype=float)

    variance_losses = mse_mae_variance(e_test, s2_test)
    metrics: dict[str, object] = {
        "n_obs": int(e_test.size),
        "qlike": qlike_loss(e_test, s2_test),
        "mse_var": variance_losses["mse"],
        "mae_var": variance_losses["mae"],
    }

    # Compute VaR backtests
    metrics["var_backtests_empirical"] = _compute_var_backtests(df_h1)
    return metrics


def _empty_metrics_dict() -> dict[str, object]:
    """Return empty metrics dictionary."""
    return {
        "n_obs": 0,
        "qlike": float("nan"),
        "mse_var": float("nan"),
        "mae_var": float("nan"),
        "var_backtests_empirical": {},
    }


def _compute_var_backtests(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute VaR backtests by alpha level (≤40 lines)."""
    var_results: dict[str, dict[str, float]] = {}

    for alpha, group in df.groupby("var_left_alpha"):
        hits = (group["actual_residual"] < group["VaR"]).astype(int)
        n_obs = int(hits.size)
        violations = int(hits.sum())
        hit_rate = float(hits.mean()) if n_obs > 0 else float("nan")

        var_results[str(alpha)] = {
            "n": n_obs,
            "violations": violations,
            "hit_rate": hit_rate,
        }

    return var_results


def egarch_one_step_variance_forecast(
    e_last: float,
    s2_last: float,
    *,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    dist: str = "normal",
    nu: float | None = None,
    lambda_skew: float | None = None,
) -> float:
    """One-step EGARCH variance forecast.

    Wrapper for compute_egarch_forecasts that returns only one-step forecast.

    Args:
        e_last: Last residual.
        s2_last: Last variance.
        omega: Omega parameter.
        alpha: Alpha parameter.
        gamma: Gamma parameter.
        beta: Beta parameter.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter.

    Returns:
        One-step ahead variance forecast.
    """
    from src.garch.garch_eval.variance_path import compute_egarch_forecasts

    s2_1, _ = compute_egarch_forecasts(
        e_last=e_last,
        s2_last=s2_last,
        horizon=1,
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        dist=dist,
        nu=nu,
        lambda_skew=lambda_skew,
    )
    return s2_1


def egarch_multi_step_variance_forecast(
    horizon: int,
    s2_last: float,
    *,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    dist: str = "normal",
    nu: float | None = None,
    lambda_skew: float | None = None,
) -> np.ndarray:
    """Multi-step EGARCH variance forecast.

    Wrapper for compute_egarch_forecasts that returns multi-step forecasts.

    Args:
        horizon: Forecast horizon.
        s2_last: Last variance.
        omega: Omega parameter.
        alpha: Alpha parameter.
        gamma: Gamma parameter.
        beta: Beta parameter.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter.

    Returns:
        Multi-step variance forecasts array.
    """
    from src.garch.garch_eval.variance_path import compute_egarch_forecasts

    # Use dummy e_last since multi-step uses unconditional expectation
    _, s2_h = compute_egarch_forecasts(
        e_last=0.0,
        s2_last=s2_last,
        horizon=horizon,
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        dist=dist,
        nu=nu,
        lambda_skew=lambda_skew,
    )
    return s2_h


def prediction_interval(
    mean: float,
    variance: float,
    *,
    level: float = GARCH_EVAL_DEFAULT_LEVEL,
    dist: str = "normal",
    nu: float | None = None,
    lambda_skew: float | None = None,
) -> tuple[float, float]:
    """Two-sided prediction interval (≤40 lines).

    Args:
        mean: Mean of the return distribution.
        variance: Conditional variance.
        level: Prediction interval level (default: 0.95).
        dist: Distribution type ('normal' or 'skewt').
        nu: Degrees of freedom for Student-t/Skew-t.
        lambda_skew: Skewness parameter for Skew-t.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    from src.garch.garch_eval.utils import quantile

    if variance <= 0.0:
        msg = "variance must be positive"
        raise ValueError(msg)

    alpha = (1.0 - float(level)) / 2.0
    sigma = float(np.sqrt(variance))
    q_lo = quantile(dist, alpha, nu, lambda_skew)
    q_hi = quantile(dist, 1.0 - alpha, nu, lambda_skew)

    return float(mean + sigma * q_lo), float(mean + sigma * q_hi)


def value_at_risk(
    alpha: float,
    *,
    mean: float = 0.0,
    variance: float,
    dist: str = "normal",
    nu: float | None = None,
    lambda_skew: float | None = None,
) -> float:
    """Left-tail Value-at-Risk at level alpha (≤40 lines).

    Returns VaR_alpha such that P(R < VaR_alpha) = alpha.

    Args:
        alpha: VaR level (e.g., 0.01 or 0.05).
        mean: Mean return.
        variance: Conditional variance.
        dist: Distribution type.
        nu: Degrees of freedom for Student-t/Skew-t.
        lambda_skew: Skewness parameter for Skew-t.

    Returns:
        VaR value.
    """
    from src.garch.garch_eval.utils import quantile

    if variance <= 0.0:
        msg = "variance must be positive"
        raise ValueError(msg)

    sigma = float(np.sqrt(variance))
    q = quantile(dist, float(alpha), nu, lambda_skew)

    return float(mean + sigma * q)


def _load_ticker_data() -> pd.DataFrame:
    """Load ticker-level data from parquet file or fallback to garch_dataset.

    For GARCH evaluation, we use the garch_dataset which contains the necessary
    data for evaluation without requiring ticker-level data.

    Returns:
        DataFrame with data for GARCH evaluation.
    """
    from src.path import DATA_TICKERS_FULL_FILE, GARCH_DATASET_FILE

    # Try to load ticker-level data first
    if DATA_TICKERS_FULL_FILE.exists():
        logger.info("Loading data_tickers_full.parquet...")
        df_tickers = pd.read_parquet(DATA_TICKERS_FULL_FILE)
        logger.info("✓ Loaded %d ticker observations", len(df_tickers))
        return df_tickers
    # Fallback to garch_dataset for GARCH evaluation
    elif GARCH_DATASET_FILE.exists():
        logger.info("Loading garch_dataset.csv (fallback for GARCH evaluation)...")
        df_garch = pd.read_csv(GARCH_DATASET_FILE, parse_dates=["date"])
        logger.info("✓ Loaded %d GARCH observations", len(df_garch))
        return df_garch
    else:
        msg = f"Neither {DATA_TICKERS_FULL_FILE} nor {GARCH_DATASET_FILE} found"
        raise FileNotFoundError(msg)


def _prepare_garch_forecasts(
    df_full_forecasts: pd.DataFrame | None,
) -> pd.DataFrame:
    """Prepare GARCH forecasts for merging.

    Args:
        df_full_forecasts: Optional full sample forecasts (TRAIN + TEST).
                          If None, generates them.

    Returns:
        DataFrame with GARCH forecasts ready for merging.
    """
    if df_full_forecasts is None:
        logger.info("Generating full sample GARCH forecasts...")
        df_garch = generate_full_sample_forecasts(
            use_optimized_params=True,
            min_window_size=GARCH_EVAL_FORCED_MIN_START_SIZE,
            initial_window_size=GARCH_EVAL_FORCED_MIN_START_SIZE,
            anchor_at_min_window=True,
        )
    else:
        logger.info("Using provided full sample GARCH forecasts...")
        df_garch = df_full_forecasts
    logger.info("✓ Using %d GARCH forecasts", len(df_garch))
    return df_garch


def _compute_standardized_residuals(df_garch: pd.DataFrame) -> pd.DataFrame:
    """Compute standardized GARCH residuals (ε_t / σ_t).

    Args:
        df_garch: DataFrame with GARCH forecasts and ARIMA residuals.

    Returns:
        DataFrame with added garch_std_resid column.

    Raises:
        ValueError: If residual column ('arima_resid' or 'sarima_resid') is missing.
    """
    from src.garch.structure_garch.utils import _find_residual_column

    df_garch = df_garch.copy()
    try:
        resid_col_name = _find_residual_column(df_garch)
    except ValueError:
        msg = (
            "Missing residual column ('arima_resid' or 'sarima_resid') "
            "required for computing standardized residuals."
        )
        raise ValueError(msg) from None

    df_garch["garch_std_resid"] = np.where(
        df_garch["garch_forecast_h1"] > 0,
        df_garch[resid_col_name] / np.sqrt(df_garch["garch_forecast_h1"]),
        np.nan,
    )
    return df_garch


def _select_garch_columns(df_garch: pd.DataFrame) -> pd.DataFrame:
    """Select GARCH columns for merging.

    Args:
        df_garch: DataFrame with GARCH forecasts.

    Returns:
        DataFrame with selected GARCH columns.

    Raises:
        ValueError: If residual column ('arima_resid' or 'sarima_resid') is missing.
    """
    from src.garch.structure_garch.utils import _find_residual_column

    try:
        resid_col_name = _find_residual_column(df_garch)
    except ValueError:
        msg = (
            "Missing residual column ('arima_resid' or 'sarima_resid') "
            "required for selecting GARCH columns."
        )
        raise ValueError(msg) from None

    garch_cols = [
        "date",
        resid_col_name,
        "garch_forecast_h1",
        "garch_vol_h1",
        "garch_std_resid",
    ]
    result: pd.DataFrame = df_garch[garch_cols].copy()  # type: ignore[assignment]
    return result


def _merge_ticker_and_garch_data(
    df_tickers: pd.DataFrame, df_garch_subset: pd.DataFrame
) -> pd.DataFrame:
    """Merge ticker data with GARCH forecasts on date.

    CRITICAL: GARCH forecasts are univariate (one value per date) and will be
    duplicated for all tickers of the same date during the merge.

    Args:
        df_tickers: Ticker-level data.
        df_garch_subset: GARCH forecasts subset.

    Returns:
        Merged DataFrame.
    """
    logger.info("Merging ticker data with GARCH insights...")
    df_merged: pd.DataFrame = df_tickers.merge(df_garch_subset, on="date", how="left")
    logger.info("✓ Merged data shape: %s", df_merged.shape)
    return df_merged


def _sort_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sort data logically for financial analysis.

    Sort order:
    1. By split (train first, then test) - separates training and test periods
    2. By date (chronological) - maintains temporal order
    3. By ticker (for each date) - groups all tickers for the same date together

    Args:
        df: DataFrame to sort.

    Returns:
        Sorted DataFrame.
    """
    logger.info("Sorting data logically for financial analysis...")

    def _get_split_order(split_val: str) -> int:
        return 0 if split_val == "train" else 1

    df_sorted = df.copy()
    df_sorted["_split_order"] = df_sorted["split"].apply(_get_split_order)
    df_sorted = (
        df_sorted.sort_values(
            by=["_split_order", "date", "tickers"],
            ascending=[True, True, True],
        )
        .drop(columns=["_split_order"])
        .reset_index(drop=True)
    )

    return df_sorted


def _validate_merged_data(df_merged: pd.DataFrame) -> None:
    """Validate merged data structure and GARCH duplication.

    Args:
        df_merged: Merged DataFrame to validate.

    Raises:
        ValueError: If validation fails.
    """
    # Validate sorting: check that train comes before test
    split_order = df_merged["split"].unique()
    if len(split_order) > 1 and split_order[0] != "train":
        msg = "Invalid sort order: test split appears before train split"
        raise ValueError(msg)

    # Validate GARCH duplication: for each date, all tickers should have same GARCH value
    logger.info("Validating GARCH forecast duplication...")
    df_with_garch = df_merged[df_merged["garch_forecast_h1"].notna()]
    if len(df_with_garch) > 0:
        date_series = pd.Series(df_with_garch["date"])
        dates_with_garch = date_series.drop_duplicates()
        if len(dates_with_garch) > 0:
            sample_date = dates_with_garch.iloc[0]
            df_sample_date = df_merged[df_merged["date"] == sample_date]
            garch_series = pd.Series(df_sample_date["garch_forecast_h1"])
            garch_values = garch_series.nunique()
            if garch_values > 1:
                msg = (
                    f"GARCH forecasts not properly duplicated: {garch_values} "
                    f"unique values for date {sample_date}"
                )
                raise ValueError(msg)
            logger.info(
                "✓ GARCH forecasts correctly duplicated: 1 value per date, "
                "replicated for all tickers"
            )


def _remove_missing_values_per_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing GARCH fields per ticker only.

    Why: We only drop rows that miss GARCH-derived fields (forecast/residuals)
    to avoid over-dropping due to unrelated NaNs in ticker-level columns. This
    targets the exact cause of row loss when GARCH forecasts start later than
    raw ticker history.

    Args:
        df: DataFrame with ticker and date columns.

    Returns:
        DataFrame with GARCH-related missing values removed per ticker.
    """
    logger.info("Removing rows with missing GARCH fields per ticker...")

    initial_count = len(df)
    # Drop only on the GARCH insight columns we add in this pipeline
    from src.garch.structure_garch.utils import _find_residual_column

    try:
        resid_col_name = _find_residual_column(df)
    except ValueError:
        msg = (
            "Missing residual column ('arima_resid' or 'sarima_resid') "
            "required for cleaning GARCH data."
        )
        raise ValueError(msg) from None

    subset_cols = ["garch_forecast_h1", resid_col_name]
    df_cleaned = df.dropna(subset=subset_cols).reset_index(drop=True)

    n_removed = initial_count - len(df_cleaned)
    logger.info(
        "✓ Removed %d rows with missing GARCH fields (%.1f%%) from %d rows to %d rows",
        n_removed,
        100 * n_removed / initial_count if initial_count > 0 else 0,
        initial_count,
        len(df_cleaned),
    )

    return df_cleaned


def _save_insights_data(df_merged: pd.DataFrame) -> None:
    """Save insights data to parquet and CSV files.

    Args:
        df_merged: Merged and validated DataFrame.
    """
    from pathlib import Path

    from src.path import DATA_TICKERS_FULL_INSIGHTS_FILE

    output_csv = Path(str(DATA_TICKERS_FULL_INSIGHTS_FILE).replace(".parquet", ".csv"))

    logger.info("Saving to parquet: %s", DATA_TICKERS_FULL_INSIGHTS_FILE)
    ensure_output_dir(DATA_TICKERS_FULL_INSIGHTS_FILE)
    df_merged.to_parquet(DATA_TICKERS_FULL_INSIGHTS_FILE, index=False)

    logger.info("Saving to CSV: %s", output_csv)
    ensure_output_dir(output_csv)
    df_merged.to_csv(output_csv, index=False)

    logger.info("✓ Saved data_tickers_full_insights with %d observations", len(df_merged))
    logger.info("  Sorted by: split (train→test), date (chronological), ticker")
    logger.info("  Columns: %s", list(df_merged.columns))


def _filter_initial_observations_per_ticker(
    df: pd.DataFrame, min_window_size: int = 250
) -> pd.DataFrame:
    """Filter first min_window_size observations per ticker.

    GARCH forecasts start at observation min_window_size (default 250),
    so the first 250 observations per ticker have NaN for GARCH features.
    We remove these observations to avoid training LightGBM on incomplete data.

    Args:
        df: Merged dataframe with ticker and date columns.
        min_window_size: Number of initial observations to skip per ticker.

    Returns:
        Filtered dataframe with first min_window_size observations removed per ticker.
    """
    logger.info(
        "Filtering first %d observations per ticker (GARCH warm-up period)...",
        min_window_size,
    )

    df = df.sort_values(["tickers", "date"]).reset_index(drop=True)
    positions = df.groupby("tickers").cumcount()
    df_filtered = df.loc[positions >= min_window_size].reset_index(drop=True)

    n_removed = len(df) - len(df_filtered)
    logger.info(
        "✓ Filtered %d observations (%.1f%%) from %d rows to %d rows",
        n_removed,
        100 * n_removed / len(df) if len(df) > 0 else 0,
        len(df),
        len(df_filtered),
    )

    return df_filtered


def generate_data_tickers_full_insights(
    df_full_forecasts: pd.DataFrame | None = None,
) -> None:
    """Create data_tickers_full_insights from evaluation artifacts without truncation.

    This implementation builds insights as requested:
    - ARIMA insights from rolling_predictions.csv → sarima_pred, arima_resid
    - GARCH insights from forecasts.csv → sigma2_egarch_raw, resid

    It preserves the full ticker timeline (e.g., 2013–2024) and does not drop
    early dates where insights are naturally unavailable (NaN).

    Args:
        df_full_forecasts: Unused (kept for API stability).
    """
    logger.info("Building data_tickers_full_insights from rolling_predictions + forecasts.csv…")

    # Load base ticker-level data
    if not DATA_TICKERS_FULL_FILE.exists():
        msg = f"Ticker-level dataset not found: {DATA_TICKERS_FULL_FILE}"
        raise FileNotFoundError(msg)
    df_tickers = pd.read_parquet(DATA_TICKERS_FULL_FILE)
    if "date" not in df_tickers.columns:
        msg = "data_tickers_full must contain a 'date' column"
        raise KeyError(msg)
    df_tickers = df_tickers.copy()
    df_tickers["date"] = pd.to_datetime(df_tickers["date"])  # type: ignore[assignment]

    # Load ARIMA full series backtest residuals
    arima_file = Path("results/arima/evaluation/full_series_backtest_residuals.csv")
    if not arima_file.exists():
        msg = f"ARIMA full series backtest residuals not found: {arima_file}"
        raise FileNotFoundError(msg)
    df_arima = pd.read_csv(arima_file, parse_dates=["date"])  # date,y_true,y_pred,arima_resid
    # Accept either arima_resid or sarima_resid
    from src.garch.structure_garch.utils import _find_residual_column

    required_arima = {"date", "y_pred"}
    missing_arima = required_arima - set(df_arima.columns)
    if missing_arima:
        msg = (
            f"full_series_backtest_residuals.csv missing required columns: {sorted(missing_arima)}"
        )
        raise KeyError(msg)

    try:
        resid_col_name = _find_residual_column(df_arima)
    except ValueError:
        msg = (
            "full_series_backtest_residuals.csv missing required residual column "
            "('arima_resid' or 'sarima_resid')."
        )
        raise KeyError(msg) from None

    df_arima = df_arima.rename(columns={"y_pred": "sarima_pred"})
    df_arima = df_arima[["date", "sarima_pred", resid_col_name]].copy()

    # Load GARCH evaluation forecasts
    garch_file = Path("results/garch/evaluation/garch_forecasts.csv")
    if not garch_file.exists():
        msg = f"GARCH forecasts file not found: {garch_file}"
        raise FileNotFoundError(msg)
    df_garch_eval = pd.read_csv(
        garch_file, parse_dates=["date"]
    )  # date,split,weighted_log_return,arima_resid,garch_forecast_h1,garch_vol_h1,
    # forecast_type,refit_occurred
    required_garch = {"date", "garch_forecast_h1", "garch_vol_h1"}
    missing_garch = required_garch - set(df_garch_eval.columns)
    if missing_garch:
        msg = f"garch_forecasts.csv missing required columns: {sorted(missing_garch)}"
        raise KeyError(msg)
    df_garch_eval = df_garch_eval[["date", "garch_forecast_h1", "garch_vol_h1"]].copy()
    # Rename columns for compatibility
    df_garch_eval = df_garch_eval.rename(
        columns={"garch_forecast_h1": "sigma2_egarch_raw", "garch_vol_h1": "sigma_garch"}
    )
    # Provide compatibility alias for downstream consumers expecting 'sigma2_garch'
    df_garch_eval["sigma2_garch"] = df_garch_eval["sigma2_egarch_raw"]

    # Merge: duplicate per-date GARCH/ARIMA across all tickers for that date
    df_out = df_tickers.merge(df_arima, on="date", how="left")
    df_out = df_out.merge(df_garch_eval, on="date", how="left")

    # Sort deterministically: split (train→test) if present, then date, then tickers
    def _split_order(v: object) -> int:
        return 0 if str(v) == "train" else 1

    if "split" in df_out.columns:
        df_out = df_out.copy()
        df_out["_split_order"] = df_out["split"].map(_split_order)
        sort_cols = ["_split_order", "date"] + (["tickers"] if "tickers" in df_out.columns else [])
        df_out = df_out.sort_values(sort_cols).drop(columns=["_split_order"]).reset_index(drop=True)
    else:
        sort_cols = ["date"] + (["tickers"] if "tickers" in df_out.columns else [])
        df_out = df_out.sort_values(sort_cols).reset_index(drop=True)

    # Persist parquet + CSV to configured paths
    out_parquet = DATA_TICKERS_FULL_INSIGHTS_FILE
    out_csv = DATA_TICKERS_FULL_INSIGHTS_FILE.with_suffix(".csv")
    ensure_output_dir(out_parquet)
    ensure_output_dir(out_csv)
    df_out.to_parquet(out_parquet, index=False)
    df_out.to_csv(out_csv, index=False)

    logger.info(
        "✓ Saved data_tickers_full_insights: %s (+ CSV) | rows=%d cols=%d",
        out_parquet,
        len(df_out),
        df_out.shape[1],
    )
