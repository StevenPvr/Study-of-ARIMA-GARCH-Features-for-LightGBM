"""SHAP analysis utilities for LightGBM evaluation."""

from __future__ import annotations

import gc
from pathlib import Path
from threading import Lock
from typing import Union

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless/parallel execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb
# Constants imported explicitly - no defaults used
from src.path import LIGHTGBM_SHAP_PLOTS_DIR
from src.utils import ensure_output_dir, get_logger

# Lock for matplotlib operations (not thread-safe)
_matplotlib_lock = Lock()

logger = get_logger(__name__)


def _extract_shap_values(explainer: shap.TreeExplainer, X_sample: pd.DataFrame) -> np.ndarray:
    """Extract and validate SHAP values from explainer.

    Args:
        explainer: SHAP TreeExplainer instance.
        X_sample: Features DataFrame.

    Returns:
        Validated SHAP values array.

    Raises:
        ValueError: If SHAP values shape doesn't match input shape.
    """
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.array(shap_values)

    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)

    if shap_values.shape[0] != len(X_sample) or shap_values.shape[1] != X_sample.shape[1]:
        raise ValueError(
            f"SHAP values shape {shap_values.shape} doesn't match "
            f"X_sample shape {X_sample.shape}"
        )

    logger.info(
        f"SHAP values computed: shape={shap_values.shape}, "
        f"min={shap_values.min():.6f}, max={shap_values.max():.6f}, "
        f"mean={shap_values.mean():.6f}, std={shap_values.std():.6f}"
    )

    return shap_values


def _create_shap_explanation(
    shap_values: np.ndarray,
    explainer: shap.TreeExplainer,
    X_sample: pd.DataFrame,
) -> shap.Explanation:
    """Create SHAP Explanation object.

    Args:
        shap_values: SHAP values array.
        explainer: SHAP TreeExplainer instance.
        X_sample: Features DataFrame.

    Returns:
        SHAP Explanation object.
    """
    return shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_sample.values,
        feature_names=X_sample.columns.tolist(),
    )


def _calculate_display_max(
    model_name: str,
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    max_display: int,
) -> int:
    """Calculate max_display ensuring sigma2_garch is shown for lightgbm_complete.

    Args:
        model_name: Name of the model.
        X_sample: Features DataFrame.
        shap_values: SHAP values array.
        max_display: Initial maximum display value.

    Returns:
        Adjusted max_display value.
    """
    if model_name != "lightgbm_complete" or "sigma2_garch" not in X_sample.columns:
        return max_display

    feature_importances = np.abs(shap_values).mean(axis=0)
    feature_names_list = X_sample.columns.tolist()
    sorted_indices = np.argsort(feature_importances)[::-1]
    sigma2_garch_idx = feature_names_list.index("sigma2_garch")
    sigma2_garch_rank = np.where(sorted_indices == sigma2_garch_idx)[0][0]

    if sigma2_garch_rank >= max_display:
        display_max = sigma2_garch_rank + 1
        logger.info(
            f"Increased max_display to {display_max} to ensure sigma2_garch "
            f"(rank {sigma2_garch_rank + 1}) is displayed for {model_name}"
        )
        return display_max

    return max_display


def _save_shap_plot(
    explanation: shap.Explanation,
    model_name: str,
    output_dir: Path,
    display_max: int,
) -> Path:
    """Save SHAP beeswarm plot to file.

    Args:
        explanation: SHAP Explanation object.
        model_name: Name of the model.
        output_dir: Directory to save plot.
        display_max: Maximum number of features to display.

    Returns:
        Path to saved plot file.
    """
    plot_path = output_dir / f"{model_name}_shap_summary.png"
    ensure_output_dir(plot_path)

    with _matplotlib_lock:
        shap.plots.beeswarm(explanation, max_display=display_max, show=False)
        fig = plt.gcf()
        fig.set_size_inches(12, 8)
        fig.suptitle(f"SHAP Feature Importance - {model_name}", fontsize=14, y=0.995)
        plt.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"SHAP plot saved to {plot_path}")
    return plot_path


def compute_shap_values(
    model: Union[lgb.LGBMRegressor, RandomForestRegressor],
    X: pd.DataFrame,
    model_name: str,
    max_display: int,
    output_dir: Path = LIGHTGBM_SHAP_PLOTS_DIR,
) -> tuple[shap.Explanation, Path]:
    """Compute SHAP values and create visualization.

    Args:
        model: Trained regression model (LightGBM or RandomForest).
        X: Features DataFrame.
        model_name: Name for the model (for plot title).
        output_dir: Directory to save SHAP plots.
        max_display: Maximum number of features to display.

    Returns:
        Tuple of (SHAP explanation object, path to saved plot).
    """
    logger.info(f"Computing SHAP values for {model_name}")

    explainer = shap.TreeExplainer(model)
    try:
        shap_values = _extract_shap_values(explainer, X)
        explanation = _create_shap_explanation(shap_values, explainer, X)
        display_max = _calculate_display_max(model_name, X, shap_values, max_display)
        plot_path = _save_shap_plot(explanation, model_name, output_dir, display_max)
        return explanation, plot_path
    finally:
        # Free SHAP explainer resources
        del explainer
        gc.collect()
