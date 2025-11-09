"""Block permutation importance for time series Random Forest models.

Computes feature importance by permuting contiguous blocks of each feature,
to respect local temporal structure. Reports the degradation in R² and
increase in RMSE relative to the baseline.
"""

from __future__ import annotations

from typing import Any

import json
import math
from pathlib import Path

import matplotlib

# Non-interactive backend for headless execution
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.constants import (
    DEFAULT_RANDOM_STATE,
    RF_PERMUTATION_PLOTS_DIR,
    RF_PERMUTATION_RESULTS_FILE,
)
from src.utils import get_logger

logger = get_logger(__name__)


def _chunk_blocks(n: int, block_size: int) -> list[slice]:
    """Return list of slices delineating contiguous blocks."""
    blocks: list[slice] = []
    start = 0
    while start < n:
        end = min(start + block_size, n)
        blocks.append(slice(start, end))
        start = end
    return blocks


def _permute_by_blocks(n: int, block_size: int, rng: np.random.RandomState) -> np.ndarray:
    """Return index array that permutes the order of contiguous blocks.

    Within a block, the order is preserved. The blocks themselves are shuffled.
    """
    blocks = _chunk_blocks(n, block_size)
    order = np.arange(len(blocks))
    rng.shuffle(order)
    indices: list[int] = []
    for b in order:
        s = blocks[int(b)]
        indices.extend(range(s.start, s.stop))
    return np.asarray(indices, dtype=int)


def _baseline_metrics(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float]:
    """Return baseline (r2, rmse)."""
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return float(r2), float(rmse)


def compute_block_permutation_importance(
    model: RandomForestRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    block_size: int = 20,
    n_repeats: int = 50,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, dict[str, float]]:
    """Compute block permutation importance for each feature.

    Args:
        model: Trained RandomForestRegressor.
        X: Test features (ordered in time).
        y: Test target aligned with X.
        block_size: Contiguous block size for permutation.
        n_repeats: Number of random block permutations.
        random_state: Seed for reproducibility.

    Returns:
        Mapping feature -> statistics with keys:
          - delta_r2_mean, delta_r2_std
          - delta_rmse_mean, delta_rmse_std
    """
    if X.empty or y.empty:
        raise ValueError("X and y must be non-empty for permutation importance")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive")

    n = len(X)
    rng = np.random.RandomState(random_state)

    # Baseline metrics
    y_pred_base = model.predict(X)
    r2_base, rmse_base = _baseline_metrics(y, y_pred_base)
    logger.info("Permutation baseline - R2: %.6f, RMSE: %.6f", r2_base, rmse_base)

    results: dict[str, dict[str, float]] = {}

    for col in X.columns:
        deltas_r2: list[float] = []
        deltas_rmse: list[float] = []

        col_values = X[col].to_numpy()

        for _ in range(n_repeats):
            perm_idx = _permute_by_blocks(n, block_size, rng)
            perm_values = col_values[perm_idx]
            X_perm = X.copy()
            X_perm[col] = perm_values

            y_pred_perm = model.predict(X_perm)
            r2_perm, rmse_perm = _baseline_metrics(y, y_pred_perm)

            # Larger delta indicates more important feature
            deltas_r2.append(r2_base - r2_perm)
            deltas_rmse.append(rmse_perm - rmse_base)

        results[col] = {
            "delta_r2_mean": float(np.mean(deltas_r2)),
            "delta_r2_std": float(np.std(deltas_r2, ddof=1)) if len(deltas_r2) > 1 else 0.0,
            "delta_rmse_mean": float(np.mean(deltas_rmse)),
            "delta_rmse_std": float(np.std(deltas_rmse, ddof=1)) if len(deltas_rmse) > 1 else 0.0,
        }

    return results


def save_permutation_results(
    per_model_results: dict[str, dict[str, dict[str, float]]],
    *,
    output_json: Path = RF_PERMUTATION_RESULTS_FILE,
) -> None:
    """Save permutation importance results to JSON."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(per_model_results, f, indent=2)
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
    RF_PERMUTATION_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for model_name, stats in per_model_results.items():
        df = (
            pd.DataFrame(stats).T.sort_values("delta_r2_mean", ascending=False).head(top_k)
        )
        plt.figure(figsize=(10, 6))
        plt.barh(df.index[::-1], df["delta_r2_mean"][::-1])
        plt.title(f"Block Permutation Importance (ΔR²) - {model_name}")
        plt.xlabel("ΔR² (higher = more important)")
        plt.tight_layout()
        out_path = RF_PERMUTATION_PLOTS_DIR / f"permutation_{model_name}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved permutation importance plot for %s: %s", model_name, out_path)
        paths.append(out_path)
    return paths

