"""Results processing utilities for ARIMA optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, cast

import pandas as pd

from src.path import ARIMA_BEST_MODELS_FILE, ARIMA_OPTIMIZATION_RESULTS_FILE
from src.utils import ensure_output_dir, get_logger, save_json_pretty

logger = get_logger(__name__)


def to_dataframe(results: Iterable[dict]) -> pd.DataFrame:
    """Convert evaluation results to a structured DataFrame.

    Flattens nested 'params' dictionary into columns prefixed with 'param_'
    and orders columns with metrics first, then parameters, then other fields.

    Args:
        results: Iterable of result dictionaries from model evaluation.

    Returns:
        DataFrame with ordered columns: metrics, parameters, other fields.
    """
    df = pd.DataFrame(results).copy()

    if df.empty:
        return df

    ensured_metric_cols = {
        "rmse": pd.NA,
        "mae": pd.NA,
        "lb_pvalue": pd.NA,
        "lb_reject_5pct": pd.NA,
    }
    for col, default_value in ensured_metric_cols.items():
        if col not in df.columns:
            df[col] = default_value

    params_column = "params"
    param_cols: list[str] = []

    if params_column in df.columns:
        normalized = pd.json_normalize(
            df[params_column].apply(lambda x: x if isinstance(x, dict) else {}).tolist(),
        ).add_prefix("param_")
        param_cols = list(normalized.columns)
        df = df.drop(columns=[params_column]).join(normalized)

    front = [
        "aic",
        "bic",
        "rmse",
        "mae",
        "lb_pvalue",
        "lb_reject_5pct",
        "error",
    ]
    front_cols = [c for c in front if c in df.columns]
    other_cols = [c for c in df.columns if c not in front_cols + param_cols]

    ordered_cols = front_cols + param_cols + other_cols
    return df.loc[:, ordered_cols]


def _filter_successful_models(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out models with errors.

    Args:
        df: DataFrame with evaluation results.

    Returns:
        DataFrame with only successful models.

    Raises:
        RuntimeError: If no successful models found.
    """
    df_ok = df.copy()
    if "error" in df_ok.columns:
        mask = df_ok["error"].isna()
        df_ok = df_ok.loc[mask].copy()
    if df_ok.empty:
        raise RuntimeError("No successful models.")
    return df_ok


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    """Convert DataFrame row to model dictionary.

    Args:
        row: DataFrame row with model results.

    Returns:
        Dictionary with params and metrics.
    """
    params = {
        str(k).replace("param_", ""): (str(row[k]) if k == "param_trend" else int(row[k]))
        for k in row.index
        if isinstance(k, str) and k.startswith("param_")
    }
    result: dict[str, Any] = {
        "params": params,
        "aic": float(row["aic"]),
        "bic": float(row["bic"]),
    }
    return result


def pick_best(df: object) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select best models based on AIC and BIC.

    The best model is selected using AIC (theoretically optimal for forecasting).

    Args:
        df: DataFrame with evaluation results.

    Returns:
        Tuple of (best_aic_dict, best_bic_dict).

    Raises:
        RuntimeError: If no successful models found.
        TypeError: If df is not a DataFrame.
        ValueError: If required columns are missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    df_typed: pd.DataFrame = cast(pd.DataFrame, df)

    # Check for successful models first (this raises RuntimeError if empty)
    df_ok = _filter_successful_models(df_typed)

    # Prioritize models that pass Ljung-Box when diagnostics are available
    lb_col = "lb_reject_5pct"
    if lb_col in df_ok.columns:
        lb_series = df_ok[lb_col].astype("boolean")
        lb_mask = lb_series.fillna(True).astype(bool)
        candidates = df_ok.loc[~lb_mask].copy()
        if not candidates.empty:
            df_ok = candidates
        else:
            logger.warning("All models reject Ljung-Box at 5%%; retaining full candidate set.")

    # Check for required columns
    if "aic" not in df_ok.columns:
        raise ValueError("DataFrame must contain 'aic' column for model selection.")
    if "bic" not in df_ok.columns:
        logger.warning("BIC column not found")

    # Select best model based on AIC (theoretically optimal for forecasting)
    best_aic_idx = df_ok["aic"].astype(float).idxmin()
    best_bic_idx = df_ok["bic"].astype(float).idxmin()

    best_aic_row = cast(pd.Series, df_ok.loc[best_aic_idx])
    best_bic_row = cast(pd.Series, df_ok.loc[best_bic_idx])

    logger.info(
        "Best model selected using AIC minimization (theoretically optimal for forecasting)"
    )

    return _row_to_dict(best_aic_row), _row_to_dict(best_bic_row)


def determine_sort_columns(df: pd.DataFrame, criterion: str) -> list[str]:
    """Determine sort columns for DataFrame based on available metrics.

    Returns a list of column names to use for sorting, prioritizing:
    1. lb_reject_5pct (if available) - prefer models that don't reject Ljung-Box test
    2. lb_pvalue (if available) - prioritize larger Ljung-Box p-values
    3. The specified criterion (aic or bic)

    Args:
        df: DataFrame with model evaluation results.
        criterion: Criterion to use for sorting ('aic' or 'bic').

    Returns:
        List of column names for sorting.
    """
    sort_cols: list[str] = []

    # Prioritize models that don't reject Ljung-Box test and have larger p-values
    if "lb_reject_5pct" in df.columns:
        sort_cols.append("lb_reject_5pct")
    if "lb_pvalue" in df.columns:
        sort_cols.append("lb_pvalue")

    # Always include the criterion
    sort_cols.append(criterion)

    return sort_cols


def build_best_model_dict(row: pd.Series) -> dict[str, Any]:
    """Build a model dictionary from a pandas Series row.

    Extracts ARIMA parameters and metrics from a DataFrame row and constructs
    a dictionary with all parameters plus a formatted params string.

    Args:
        row: Pandas Series row containing ARIMA parameters and metrics.

    Returns:
        Dictionary with all parameters (p, d, q, P, D, Q, s, aic, bic) and params string.

    Raises:
        ValueError: If required ARIMA parameters are missing.
    """
    required_params = ["p", "d", "q", "P", "D", "Q", "s"]
    missing_params = [p for p in required_params if p not in row.index]
    if missing_params:
        msg = f"Missing required ARIMA parameters: {missing_params}"
        raise ValueError(msg)

    model_dict: dict[str, Any] = {
        "p": int(row["p"]),
        "d": int(row["d"]),
        "q": int(row["q"]),
        "P": int(row["P"]),
        "D": int(row["D"]),
        "Q": int(row["Q"]),
        "s": int(row["s"]),
    }

    # Add metrics if present
    if "aic" in row.index:
        model_dict["aic"] = float(row["aic"])
    if "bic" in row.index:
        model_dict["bic"] = float(row["bic"])

    # Add params string
    model_dict["params"] = f"ARIMA({model_dict['p']},{model_dict['d']},{model_dict['q']})"

    # Add any other columns from the row
    for key in row.index:
        if key not in model_dict:
            model_dict[str(key)] = row[key]

    return model_dict


def _resolve_path(file_path: Path, out_dir: Path) -> Path:
    """Resolve file path relative to output directory.

    Args:
        file_path: File path (absolute or relative).
        out_dir: Output directory.

    Returns:
        Resolved absolute path.
    """
    if file_path.is_absolute():
        return file_path
    return out_dir / file_path


def save_results(
    df: pd.DataFrame,
    best_aic: dict,
    best_bic: Optional[dict],
    out_dir: Path,
    *,
    best_models_file: Path = ARIMA_BEST_MODELS_FILE,
    results_file: Path = ARIMA_OPTIMIZATION_RESULTS_FILE,
) -> None:
    """Save optimization results to CSV and JSON files.

    Args:
        df: DataFrame with all evaluation results.
        best_aic: Dictionary with best AIC model parameters and metrics.
        best_bic: Dictionary with best BIC model parameters and metrics, or None.
        out_dir: Output directory path (created if it doesn't exist).
        best_models_file: Target path for best models JSON file.
        results_file: Target path for optimization results CSV file.

    Raises:
        ValueError: If df is empty.
    """
    if df.empty:
        msg = "Optimization results DataFrame is empty"
        raise ValueError(msg)

    ensure_output_dir(out_dir)

    results_path = _resolve_path(results_file, out_dir)
    # Ensure parent directory exists (create all parent directories)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Saved optimization table to CSV: {results_path}")

    best_models_path = _resolve_path(best_models_file, out_dir)
    # Ensure parent directory exists (create all parent directories)
    best_models_path.parent.mkdir(parents=True, exist_ok=True)
    best_models_data = {"best_aic": best_aic, "best_bic": best_bic}
    save_json_pretty(best_models_data, best_models_path)
    logger.info(f"Saved best-models JSON: {best_models_path}")
