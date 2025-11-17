"""Block permutation importance for time series LightGBM models.

Computes feature importance by permuting contiguous blocks of each feature,
to respect local temporal structure. Reports the degradation in R² and
increase in RMSE relative to the baseline.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Union

import matplotlib

# Non-interactive backend for headless execution
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import lightgbm as lgb
from src.constants import DEFAULT_RANDOM_STATE
from src.path import LIGHTGBM_PERMUTATION_PLOTS_DIR, LIGHTGBM_PERMUTATION_RESULTS_FILE
from src.utils import ensure_output_dir, get_logger, save_json_pretty

logger = get_logger(__name__)


def _permute_by_blocks(n: int, block_size: int, rng: np.random.RandomState) -> np.ndarray:
    """Return index array that permutes the order of contiguous blocks.

    Within a block, the order is preserved. The blocks themselves are shuffled.
    Fully vectorized with NumPy for performance.
    """
    n_blocks = (n + block_size - 1) // block_size  # Ceiling division

    # Create block ranges vectorized
    block_starts = np.arange(0, n, block_size, dtype=np.int32)
    block_ends = np.minimum(block_starts + block_size, n)
    block_sizes = block_ends - block_starts

    # Shuffle block order
    block_order = rng.permutation(n_blocks)

    # Build indices fully vectorized
    # Pre-allocate output array
    indices = np.empty(n, dtype=np.int32)

    # Use advanced indexing to fill indices
    pos = 0
    for block_idx in block_order:
        start = block_starts[block_idx]
        size = block_sizes[block_idx]
        indices[pos : pos + size] = np.arange(start, start + size, dtype=np.int32)
        pos += size

    return indices


def _baseline_metrics(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float]:
    """Return baseline (r2, rmse)."""
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return float(r2), float(rmse)


def _validate_permutation_params(
    X: pd.DataFrame,
    y: pd.Series,
    block_size: int,
    n_repeats: int,
    sample_fraction: float,
) -> None:
    """Validate parameters for permutation importance computation.

    Args:
        X: Test features.
        y: Test target.
        block_size: Contiguous block size for permutation.
        n_repeats: Number of random block permutations.
        sample_fraction: Fraction of test data to use.

    Raises:
        ValueError: If any parameter is invalid.
    """
    if X.empty or y.empty:
        raise ValueError("X and y must be non-empty for permutation importance")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive")
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be between 0 and 1")


def _sample_data_for_permutation(
    X: pd.DataFrame,
    y: pd.Series,
    sample_fraction: float,
    rng: np.random.RandomState,
) -> tuple[pd.DataFrame, pd.Series]:
    """Sample a fraction of data for permutation importance computation.

    Samples contiguous blocks to preserve temporal structure.

    Args:
        X: Test features.
        y: Test target.
        sample_fraction: Fraction of data to sample.
        rng: Random number generator.

    Returns:
        Tuple of (sampled_X, sampled_y).
    """
    n_total = len(X)
    n_sample = int(n_total * sample_fraction)
    if n_sample < n_total:
        max_start = n_total - n_sample
        start_idx = rng.randint(0, max_start + 1) if max_start > 0 else 0
        end_idx = start_idx + n_sample
        X_sampled = X.iloc[start_idx:end_idx].copy()
        y_sampled = y.iloc[start_idx:end_idx].copy()
        logger.info(
            f"Sampling {n_sample}/{n_total} rows ({sample_fraction*100:.1f}%) "
            f"for permutation importance (indices {start_idx}:{end_idx})"
        )
        return X_sampled, y_sampled
    return X.copy(), y.copy()


def _compute_baseline_metrics_for_permutation(
    model: Union[lgb.LGBMRegressor, RandomForestRegressor],
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[float, float]:
    """Compute baseline R² and RMSE metrics.

    Args:
        model: Trained regression model.
        X: Test features.
        y: Test target.

    Returns:
        Tuple of (r2_base, rmse_base).
    """
    y_pred_base = np.asarray(model.predict(X), dtype=np.float32)
    r2_base, rmse_base = _baseline_metrics(y, y_pred_base)
    logger.info("Permutation baseline - R2: %.6f, RMSE: %.6f", r2_base, rmse_base)
    return r2_base, rmse_base


def _compute_single_feature_permutation_importance(
    model: Union[lgb.LGBMRegressor, RandomForestRegressor],
    X: pd.DataFrame,
    y: pd.Series,
    col: str,
    r2_base: float,
    rmse_base: float,
    block_size: int,
    n_repeats: int,
    rng: np.random.RandomState,
) -> dict[str, float]:
    """Compute permutation importance for a single feature.

    Args:
        model: Trained regression model.
        X: Test features.
        y: Test target.
        col: Feature column name.
        r2_base: Baseline R² score.
        rmse_base: Baseline RMSE.
        block_size: Contiguous block size for permutation.
        n_repeats: Number of random block permutations.
        rng: Random number generator.

    Returns:
        Dictionary with delta_r2_mean, delta_r2_std, delta_rmse_mean, delta_rmse_std.
    """
    n = len(X)
    deltas_r2 = np.empty(n_repeats, dtype=np.float32)
    deltas_rmse = np.empty(n_repeats, dtype=np.float32)

    col_values = X[col].to_numpy(dtype=np.float32)
    X_array = X.to_numpy(dtype=np.float32)
    col_idx = X.columns.get_loc(col)

    for i in range(n_repeats):
        perm_idx = _permute_by_blocks(n, block_size, rng)
        perm_values = col_values[perm_idx]

        X_perm_array = X_array.copy()
        X_perm_array[:, col_idx] = perm_values
        X_perm = pd.DataFrame(X_perm_array, columns=X.columns, index=X.index)

        y_pred_perm = np.asarray(model.predict(X_perm), dtype=np.float32)
        r2_perm, rmse_perm = _baseline_metrics(y, y_pred_perm)

        deltas_r2[i] = r2_base - r2_perm
        deltas_rmse[i] = rmse_perm - rmse_base

    return {
        "delta_r2_mean": float(np.mean(deltas_r2)),
        "delta_r2_std": float(np.std(deltas_r2, ddof=1)) if n_repeats > 1 else 0.0,
        "delta_rmse_mean": float(np.mean(deltas_rmse)),
        "delta_rmse_std": float(np.std(deltas_rmse, ddof=1)) if n_repeats > 1 else 0.0,
    }


def compute_block_permutation_importance(
    model: Union[lgb.LGBMRegressor, RandomForestRegressor],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    block_size: int = 20,
    n_repeats: int = 300,
    sample_fraction: float = 0.5,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, dict[str, float]]:
    """Compute block permutation importance for each feature.

    Args:
        model: Trained regression model (LightGBM or RandomForest).
        X: Test features (ordered in time).
        y: Test target aligned with X.
        block_size: Contiguous block size for permutation.
        n_repeats: Number of random block permutations (default: 300, sufficient for 300k obs).
        sample_fraction: Fraction of test data to use (default: 0.2 for 20%).
        random_state: Seed for reproducibility.

    Returns:
        Mapping feature -> statistics with keys:
          - delta_r2_mean, delta_r2_std
          - delta_rmse_mean, delta_rmse_std
    """
    _validate_permutation_params(X, y, block_size, n_repeats, sample_fraction)

    rng = np.random.RandomState(random_state)
    X_sampled, y_sampled = _sample_data_for_permutation(X, y, sample_fraction, rng)
    r2_base, rmse_base = _compute_baseline_metrics_for_permutation(model, X_sampled, y_sampled)

    results: dict[str, dict[str, float]] = {}
    for col in X_sampled.columns:
        results[col] = _compute_single_feature_permutation_importance(
            model,
            X_sampled,
            y_sampled,
            col,
            r2_base,
            rmse_base,
            block_size,
            n_repeats,
            rng,
        )

    return results


def save_permutation_results(
    per_model_results: dict[str, dict[str, dict[str, float]]],
    *,
    output_json: Path = LIGHTGBM_PERMUTATION_RESULTS_FILE,
) -> None:
    """Save permutation importance results to JSON."""
    save_json_pretty(per_model_results, output_json)
    logger.info("Permutation importance results saved to %s", output_json)


def plot_permutation_bars(
    per_model_results: dict[str, dict[str, dict[str, float]]],
    *,
    top_k: int = 20,
) -> list[Path]:
    """Create bar plots of delta R² per model.

    Args:
        per_model_results: Mapping model_name -> feature -> stats.
        top_k: Number of top features to display by mean delta R².

    Returns:
        List of plot paths written.
    """
    paths: list[Path] = []
    for model_name, stats in per_model_results.items():
        df = pd.DataFrame(stats).T.sort_values("delta_r2_mean", ascending=False).head(top_k)
        plt.figure(figsize=(10, 6))
        plt.barh(df.index[::-1], df["delta_r2_mean"][::-1])
        plt.title(f"Block Permutation Importance (ΔR²) - {model_name}")
        plt.xlabel("ΔR² (higher = more important)")
        plt.tight_layout()
        out_path = LIGHTGBM_PERMUTATION_PLOTS_DIR / f"permutation_{model_name}.png"
        ensure_output_dir(out_path)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved permutation importance plot for %s: %s", model_name, out_path)
        paths.append(out_path)
    return paths
