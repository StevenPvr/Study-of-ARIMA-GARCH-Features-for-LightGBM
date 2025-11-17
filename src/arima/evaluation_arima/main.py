"""CLI entry point for ARIMA evaluation.

Keeps `src.*` imports intact as requested.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd

# Add project root to Python path for direct execution.
# This must be done before importing src modules.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


from src.arima.evaluation_arima.evaluation_arima import (  # type: ignore
    backtest_full_series,
    evaluate_model,
    ljung_box_on_residuals,
    plot_residuals_acf_with_ljungbox,
    run_all_normality_tests,
    save_evaluation_results,
    save_ljung_box_results,
)
from src.arima.evaluation_arima.save_data_for_garch import (  # type: ignore
    regenerate_garch_dataset_from_rolling_predictions,
    save_garch_dataset,
)
from src.arima.evaluation_arima.model_performance import (  # type: ignore
    plot_predictions_vs_actual,
)
from src.arima.evaluation_arima.utils import detect_value_column  # type: ignore
# No default constants imported - all parameters must be provided explicitly
from src.path import (  # type: ignore
    ARIMA_RESULTS_DIR,
    PREDICTIONS_VS_ACTUAL_ARIMA_PLOT,
    ROLLING_PREDICTIONS_ARIMA_FILE,
    WEIGHTED_LOG_RETURNS_SPLIT_FILE,
)
from src.utils import ensure_output_dir, get_logger, load_csv_file, save_json_pretty  # type: ignore

logger = get_logger(__name__)


def _load_split_df() -> pd.DataFrame:
    path = Path(WEIGHTED_LOG_RETURNS_SPLIT_FILE)
    df = load_csv_file(path)
    if "date" not in df.columns:
        raise ValueError("Split file must contain a 'date' column.")
    return df


def _get_train_test_split_date(df: pd.DataFrame) -> str:
    """Extract the train/test split date from the dataframe."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    if "split" in df.columns:
        if set(df["split"].unique()) >= {"train", "test"}:
            # Find the transition from train to test
            split_changes = df["split"] != df["split"].shift(1)
            test_starts = df[(df["split"] == "test") & split_changes]
            if not test_starts.empty:
                split_date = test_starts["date"].iloc[0]
                return split_date.strftime("%Y-%m-%d")

    if "is_test" in df.columns:
        # Find where is_test becomes True
        test_starts = df[(df["is_test"] == True) & (df["is_test"] != df["is_test"].shift(1))]
        if not test_starts.empty:
            split_date = test_starts["date"].iloc[0]
            return split_date.strftime("%Y-%m-%d")

    raise ValueError("Could not determine train/test split date from dataframe.")


def _split_train_test(df: pd.DataFrame, value_col: str) -> tuple[pd.Series, pd.Series]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    if "split" in df.columns:
        if set(df["split"].unique()) >= {"train", "test"}:
            train_df = df[df["split"] == "train"].set_index("date")
            test_df = df[df["split"] == "test"].set_index("date")
            train: pd.Series = train_df[value_col]  # type: ignore[assignment]
            test: pd.Series = test_df[value_col]  # type: ignore[assignment]
            return train, test

    if "is_test" in df.columns:
        mask = df["is_test"].astype(bool)
        train_df = df[~mask].set_index("date")
        test_df = df[mask].set_index("date")
        train_series: pd.Series = train_df[value_col]  # type: ignore[assignment]
        test_series: pd.Series = test_df[value_col]  # type: ignore[assignment]
        return train_series, test_series

    # No implicit fallback allowed: explicit split markers required
    raise ValueError(
        "No explicit split markers found in dataframe. "
        "Expected 'split' column with values {'train','test'} or a boolean 'is_test' column."
    )


def _load_trained_model_and_order() -> tuple[Any, tuple[int, int, int]]:
    """Load a trained ARIMA model and extract order.

    Raises explicit errors instead of silently falling back to defaults.
    """
    try:
        from src.arima.training_arima.training_arima import load_trained_model  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("Failed to import ARIMA training loader 'load_trained_model'.") from exc

    fitted_model, model_info = load_trained_model()

    order: tuple[int, int, int] | None = None

    # Prefer explicit metadata
    if isinstance(model_info, dict):
        maybe_order = model_info.get("order")
        if maybe_order is not None:
            order = tuple(maybe_order)  # type: ignore[arg-type]

    # Fallback extraction from fitted model attributes is not allowed; enforce explicit presence
    if order is None:
        raise ValueError(
            "Trained model metadata must include 'order'. "
            "Extraction from fitted model attributes is not allowed."
        )

    return fitted_model, order


def _run_residual_diagnostics(residuals: Sequence[float], lags: int) -> None:
    """Generate residual diagnostics (ACF plot + Ljung–Box results)."""
    try:
        plot_residuals_acf_with_ljungbox(residuals, lags=lags)
        lb_result = ljung_box_on_residuals(residuals, lags=lags)
        save_ljung_box_results(lb_result)
    except Exception as exc:  # pragma: no cover
        logger.warning("Diagnostics could not be generated: %s", exc)


def _save_normality_tests(residuals: Sequence[float]) -> None:
    """Run and persist residual normality tests."""
    logger.info("=" * 60)
    logger.info("RUNNING NORMALITY TESTS ON RESIDUALS")
    logger.info("=" * 60)
    try:
        results = run_all_normality_tests(residuals)
        logger.info(
            "Jarque-Bera: statistic=%.4f, p-value=%.4f",
            results["jarque_bera"]["statistic"],
            results["jarque_bera"]["p_value"],
        )
        logger.info(
            "Shapiro-Wilk: statistic=%.4f, p-value=%.4f",
            results["shapiro_wilk"]["statistic"],
            results["shapiro_wilk"]["p_value"],
        )
        logger.info(
            "Anderson-Darling: statistic=%.4f",
            results["anderson_darling"]["statistic"],
        )
        output = Path(ARIMA_RESULTS_DIR) / "evaluation" / "normality_tests.json"
        ensure_output_dir(output.parent)
        save_json_pretty(results, output)
        logger.info("Saved normality test results → %s", output)
    except Exception as exc:  # pragma: no cover
        logger.warning("Normality tests could not be generated: %s", exc)


def _compute_backtest_metrics(history: pd.DataFrame) -> dict[str, Any]:
    """Compute evaluation metrics from backtest history.

    Args:
        history: DataFrame with y_true and y_pred columns.

    Returns:
        Dict with MSE, RMSE, MAE metrics.
    """
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mse = mean_squared_error(history["y_true"], history["y_pred"])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(history["y_true"], history["y_pred"])

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "n_observations": len(history),
    }


def _save_backtest_results(
    history: pd.DataFrame,
    metrics: dict[str, Any],
    refit_every: int,
) -> None:
    """Save backtest results and summary to disk.

    Args:
        history: DataFrame with backtest results.
        metrics: Dict with computed metrics.
        refit_every: Refit interval used.
    """
    output_dir = Path(ARIMA_RESULTS_DIR) / "evaluation"
    ensure_output_dir(output_dir)

    if not history.empty:
        backtest_file = output_dir / "full_series_backtest_residuals.csv"
        ensure_output_dir(backtest_file)
        history.to_csv(backtest_file, index=False)
        logger.info("Saved full series backtest outputs → %s", backtest_file)

    summary = {**metrics, "refit_every": refit_every}
    save_json_pretty(summary, output_dir / "full_series_backtest_summary.json")


def _run_full_series_backtest(
    train: pd.Series,
    test: pd.Series,
    order: tuple[int, int, int],
    refit_every: int,
) -> pd.DataFrame:
    """Execute backtest on full series (train+test) and persist outputs.

    This backtest generates residuals for all dates (train+test) needed by GARCH,
    using a rolling forecast with periodic refits every 20 days.
    """
    logger.info("=" * 60)
    logger.info("RUNNING FULL SERIES BACKTEST (train+test)")
    logger.info("Any forecasting error will abort the CLI execution.")
    history = backtest_full_series(
        train_series=train,
        test_series=test,
        order=order,
        refit_every=refit_every,
        verbose=True,
    )

    metrics = _compute_backtest_metrics(history)

    logger.info(
        "Backtest metrics: MSE=%.6f | RMSE=%.6f | MAE=%.6f",
        metrics["mse"],
        metrics["rmse"],
        metrics["mae"],
    )

    _save_backtest_results(history, metrics, refit_every)
    return history


def _build_garch_outputs(
    results: dict[str, Any],
    fitted_model: Any | None,
    backtest_residuals: pd.DataFrame | None,
) -> None:
    """Persist datasets required by the GARCH pipeline."""
    logger.info("=" * 60)
    logger.info("BUILDING GARCH DATASET")
    logger.info("=" * 60)
    try:
        save_garch_dataset(
            results,
            fitted_model=fitted_model,
            backtest_residuals=backtest_residuals,
        )
        return
    except Exception as exc:  # pragma: no cover
        logger.warning("GARCH dataset generation failed: %s", exc)
    try:
        logger.info("Attempting to regenerate GARCH dataset from rolling_predictions.csv…")
        regenerate_garch_dataset_from_rolling_predictions()
    except Exception as exc:  # pragma: no cover
        logger.warning("GARCH dataset regeneration also failed: %s", exc)


def main() -> None:
    logger.info("=" * 60)
    logger.info("ARIMA evaluation starting…")

    df = _load_split_df()
    value_col = detect_value_column(df)
    train, test = _split_train_test(df, value_col=value_col)

    fitted_model, order = _load_trained_model_and_order()

    # Load model_info to get trend and refit_every parameters
    from src.arima.training_arima.training_arima import load_trained_model

    _, model_info = load_trained_model()
    if not isinstance(model_info, dict) or "params" not in model_info:
        raise ValueError("Model info must contain params with trend and refit_every")

    params = model_info["params"]
    trend = params.get("trend")
    refit_every = params.get("refit_every")

    if trend is None:
        raise ValueError("Model parameters must include 'trend'")
    if refit_every is None:
        raise ValueError("Model parameters must include 'refit_every'")

    results = evaluate_model(
        train_series=train,
        test_series=test,
        order=order,
        refit_every=refit_every,
        verbose=True,
        model_info=model_info,
        trend=trend,
    )

    save_evaluation_results(results)

    # Generate predictions vs actual plot
    try:
        split_df = _load_split_df()
        split_date = _get_train_test_split_date(split_df)
        plot_predictions_vs_actual(
            predictions_file=str(ROLLING_PREDICTIONS_ARIMA_FILE),
            output_file=str(PREDICTIONS_VS_ACTUAL_ARIMA_PLOT),
            train_test_split_date=split_date,
        )
        logger.info("Generated ARIMA predictions vs actual plot")
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not generate predictions vs actual plot: %s", exc)

    residuals = results.get("residuals")
    if residuals is None:
        raise KeyError("Evaluation results must include 'residuals'.")

    # Use standard Ljung-Box lags for residual diagnostics
    ljungbox_lags = 20
    _run_residual_diagnostics(residuals, ljungbox_lags)
    _save_normality_tests(residuals)

    # Generate full series residuals using optimal refit_every from optimization
    # This uses rolling forecast with periodic refits (more consistent with optimization)
    optimal_refit_every = results.get("refit_every")
    if optimal_refit_every is None:
        raise ValueError("Evaluation results must include 'refit_every' parameter")
    logger.info(
        f"Generating full series residuals for GARCH using "
        f"optimal refit_every={optimal_refit_every}"
    )

    from src.arima.evaluation_arima.evaluation_arima import backtest_full_series

    try:
        logger.info("Running full series backtest for GARCH artifacts (errors propagate to CLI).")
        backtest_history = backtest_full_series(
            train_series=train,
            test_series=test,
            order=order,
            refit_every=optimal_refit_every,
            verbose=True,
            trend=trend,
        )
    except RuntimeError as exc:
        logger.error("Full series backtest failed: %s", exc)
        raise

    # Extract fitted model from evaluate_model results for GARCH
    full_fitted_model = results.get("_fitted_model")

    _build_garch_outputs(results, full_fitted_model, backtest_history)

    logger.info("=" * 60)
    logger.info("ARIMA evaluation complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
