"""GARCH evaluation - simple post-training evaluation on test split."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable, Iterable, Literal, Sequence

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd

from src.constants import (
    GARCH_EVAL_DEFAULT_ALPHAS,
    GARCH_EVAL_FORCED_MIN_START_SIZE,
    GARCH_EVAL_FORECAST_MODE_CHOICES,
    GARCH_EVAL_FORECAST_MODE_DEFAULT,
    GARCH_EVAL_FORECAST_MODE_HYBRID,
    GARCH_EVAL_FORECAST_MODE_NO_REFIT,
    GARCH_EVALUATION_DIR,
)
from src.garch.garch_eval.eval import (
    compute_metrics_from_forecasts_new_format,
    forecast_on_test_from_trained_model,
    generate_data_tickers_full_insights,
)
from src.garch.garch_eval.metrics import (
    apply_mz_calibration,
    compute_classic_metrics_from_artifacts,
    mincer_zarnowitz,
    save_metrics_json,
    save_var_summary_json,
    summarize_var_backtests,
)
from src.garch.garch_eval.plotting import generate_eval_plots_from_artifacts
from src.garch.garch_eval.utils import load_best_model
from src.garch.training_garch.orchestration import (
    generate_full_sample_forecasts_from_trained_model,
    generate_full_sample_forecasts_hybrid,
)
from src.utils import get_logger

logger = get_logger(__name__)

ForecastMode = Literal[
    GARCH_EVAL_FORECAST_MODE_NO_REFIT,
    GARCH_EVAL_FORECAST_MODE_HYBRID,
]

_FORECAST_GENERATORS: dict[str, Callable[..., pd.DataFrame]] = {
    GARCH_EVAL_FORECAST_MODE_NO_REFIT: generate_full_sample_forecasts_from_trained_model,
    GARCH_EVAL_FORECAST_MODE_HYBRID: generate_full_sample_forecasts_hybrid,
}


def _compute_and_save_basic_metrics(
    forecasts: pd.DataFrame,
    alphas: list[float],
    apply_mz_cal: bool = False,
) -> None:
    """Compute and save basic GARCH metrics and plots.

    Args:
        forecasts: TEST forecasts DataFrame.
        alphas: VaR alpha levels for backtests.
        apply_mz_cal: If True, applies MZ calibration (DIAGNOSTIC ONLY - causes look-ahead bias).
                     Defaults to False to avoid data leakage in out-of-sample evaluation.
    """
    required_cols = {"resid", "sigma2_egarch_raw"}
    missing_cols = required_cols.difference(forecasts.columns)
    if missing_cols:
        missing_str = ", ".join(sorted(missing_cols))
        raise KeyError(f"Forecasts missing required columns for VaR computation: {missing_str}")

    resid = forecasts["resid"].to_numpy(dtype=float)
    sigma2 = forecasts["sigma2_egarch_raw"].to_numpy(dtype=float)
    valid_mask = np.isfinite(resid) & np.isfinite(sigma2) & (sigma2 > 0.0)
    resid_valid = resid[valid_mask]
    sigma2_valid = sigma2[valid_mask]

    rolling_metrics = compute_metrics_from_forecasts_new_format(forecasts)

    payload: dict[str, object] = {
        "rolling_metrics": rolling_metrics,
    }

    try:
        params, name, dist, nu, lambda_skew = load_best_model()
        payload["classic_metrics"] = compute_classic_metrics_from_artifacts(
            params=params,
            model_name=name,
            dist=dist,
            nu=nu,
            lambda_skew=lambda_skew,
            alphas=alphas,
            apply_mz_calibration=apply_mz_cal,
        )
        generate_eval_plots_from_artifacts(
            params=params,
            model_name=name,
            dist=dist,
            nu=nu,
            lambda_skew=lambda_skew,
            alphas=alphas,
        )
        var_summary = summarize_var_backtests(
            resid_valid,
            sigma2_valid,
            dist=dist,
            nu=nu,
            lambda_skew=lambda_skew,
            alphas=alphas,
        )
        payload["var_summary"] = var_summary
        save_var_summary_json(var_summary)
        logger.info("✓ Basic metrics and plots generated")
    except Exception as ex:
        logger.warning("Could not generate all plots: %s", ex)

    save_metrics_json(payload)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the GARCH evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained GARCH models and export LightGBM-ready features.",
    )
    parser.add_argument(
        "--forecast-mode",
        choices=GARCH_EVAL_FORECAST_MODE_CHOICES,
        default=GARCH_EVAL_FORECAST_MODE_DEFAULT,
        help=(
            "Select the full-sample forecast generator. 'no_refit' reuses the trained "
            "model without refitting on TRAIN/TEST (official evaluation default)."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _resolve_forecast_generator(
    forecast_mode: ForecastMode,
) -> Callable[..., pd.DataFrame]:
    """Return the forecast generator associated with the provided mode."""
    try:
        return _FORECAST_GENERATORS[forecast_mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported forecast mode: {forecast_mode}") from exc


def _generate_full_sample_forecasts(
    forecast_mode: ForecastMode,
    *,
    min_window_size: int,
) -> pd.DataFrame:
    """Generate full-sample forecasts using the requested mode."""
    generator = _resolve_forecast_generator(forecast_mode)
    mode_details = (
        "no refits on TRAIN/TEST splits; LightGBM features stay consistent"
        if forecast_mode == GARCH_EVAL_FORECAST_MODE_NO_REFIT
        else "hybrid mode with scheduled refits"
    )
    logger.info(
        "Selected GARCH forecast generation mode: %s (%s).",
        forecast_mode,
        mode_details,
    )
    return generator(
        min_window_size=min_window_size,
        anchor_at_min_window=True,
    )


def _log_header() -> None:
    """Display the CLI banner."""
    logger.info("\n")
    logger.info("=" * 70)
    logger.info(" GARCH MODEL EVALUATION (Test Split)")
    logger.info("=" * 70)


def _log_footer() -> None:
    """Display the summary once the evaluation is complete."""
    logger.info("\n" + "=" * 70)
    logger.info(" EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nResults saved to: {GARCH_EVALUATION_DIR}")
    logger.info("\nMetrics computed:")
    logger.info("  • QLIKE, MSE, MAE for variance forecasts")
    logger.info("  • VaR backtests")
    logger.info("  • Mincer-Zarnowitz calibration diagnostics")
    logger.info("  • Evaluation plots")
    logger.info("  • data_tickers_full_insights.csv and .parquet")
    logger.info("\n" + "=" * 70)


def _run_pipeline(forecast_mode: ForecastMode) -> None:
    """Execute the evaluation pipeline for the requested mode."""
    logger.info("\nGenerating full-sample forecasts...")
    df_full_forecasts = _generate_full_sample_forecasts(
        forecast_mode,
        min_window_size=GARCH_EVAL_FORCED_MIN_START_SIZE,
    )
    logger.info(
        "✓ Generated %d full sample forecasts using '%s' mode",
        len(df_full_forecasts),
        forecast_mode,
    )

    logger.info("\nExtracting TEST forecasts for evaluation...")
    forecasts = forecast_on_test_from_trained_model(df_full_forecasts=df_full_forecasts)
    logger.info("✓ Extracted %d TEST forecasts", len(forecasts))

    logger.info("\nComputing evaluation metrics...")
    _compute_and_save_basic_metrics(
        forecasts,
        alphas=list(GARCH_EVAL_DEFAULT_ALPHAS),
        apply_mz_cal=False,
    )

    logger.info("\nGenerating data_tickers_full_insights...")
    generate_data_tickers_full_insights(df_full_forecasts=df_full_forecasts)
    logger.info(
        "✓ data_tickers_full_insights generated using '%s' forecasts",
        forecast_mode,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Main entry point - simple post-training evaluation on test split."""
    args = parse_args(argv)
    forecast_mode: ForecastMode = args.forecast_mode

    _log_header()

    try:
        _run_pipeline(forecast_mode)
        _log_footer()
    except FileNotFoundError:
        logger.error("\nGARCH forecasts not found. Please run training first:")
        logger.error("  python -m src.garch.training_garch.main")
        sys.exit(1)
    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()


def run_complete_evaluation(
    forecasts_df: pd.DataFrame | None = None,
    *,
    alphas: Iterable[float] | None = None,
    apply_mz_cal: bool = False,
    forecast_mode: ForecastMode = GARCH_EVAL_FORECAST_MODE_DEFAULT,
) -> dict[str, object]:
    """Run full evaluation pipeline and return summary results."""
    alpha_list = [float(a) for a in (alphas if alphas is not None else GARCH_EVAL_DEFAULT_ALPHAS)]
    if not alpha_list:
        raise ValueError("alphas must contain at least one value.")

    if forecasts_df is None:
        df_full_forecasts = _generate_full_sample_forecasts(
            forecast_mode,
            min_window_size=GARCH_EVAL_FORCED_MIN_START_SIZE,
        )
        forecasts = forecast_on_test_from_trained_model(df_full_forecasts=df_full_forecasts)
        _compute_and_save_basic_metrics(forecasts, alpha_list, apply_mz_cal=apply_mz_cal)
        generate_data_tickers_full_insights(df_full_forecasts=df_full_forecasts)
    else:
        if forecasts_df.empty:
            raise ValueError("forecasts_df must contain at least one row.")
        required_cols = {"date", "resid", "RV", "sigma2_egarch_raw"}
        missing = required_cols.difference(forecasts_df.columns)
        if missing:
            raise ValueError(f"forecasts_df missing required columns: {sorted(missing)}")
        forecasts = forecasts_df.copy()

    basic_stats = compute_metrics_from_forecasts_new_format(forecasts)
    resid = forecasts["resid"].to_numpy(dtype=float)
    sigma2 = forecasts["sigma2_egarch_raw"].to_numpy(dtype=float)
    mz_params = mincer_zarnowitz(resid, sigma2)
    slope_val = float(mz_params.get("slope", np.nan))
    if not np.isfinite(slope_val):
        raise ValueError("Mincer-Zarnowitz regression returned a non-finite slope.")
    sigma2_calibrated = apply_mz_calibration(
        sigma2,
        intercept=0.0,
        slope=slope_val,
        use_intercept=False,
    )

    return {
        "basic_stats": basic_stats,
        "mz_calibration": {
            "params": {
                "intercept": float(mz_params.get("intercept", np.nan)),
                "slope": float(mz_params.get("slope", np.nan)),
                "r2": float(mz_params.get("r2", np.nan)),
                "p_intercept": float(mz_params.get("p_intercept", np.nan)),
                "p_slope": float(mz_params.get("p_slope", np.nan)),
            },
            "sigma2_calibrated": sigma2_calibrated.tolist(),
        },
        "alphas": alpha_list,
        "apply_mz_calibration": apply_mz_cal,
    }
