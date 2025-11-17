"""Visualization utilities for LightGBM data preparation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.constants import PLOTS_DIR
from src.utils import get_logger
from src.visualization import save_plot_wrapper

logger = get_logger(__name__)
_PLOT_ALPHA_DEFAULT = 0.8
_FONTSIZE_TITLE = 14
_FONTSIZE_LABEL = 12
_PLOT_DPI = 300


def plot_log_volatility_distribution(
    df: pd.DataFrame,
    *,
    output_dir: Path | None = None,
) -> None:
    """Plot the distribution of log_volatility column.

    Args:
        df: DataFrame containing the log_volatility column.
        output_dir: Directory to save the plot. If None, uses default plots directory.
    """
    if output_dir is None:
        output_dir = PLOTS_DIR / "lightgbm" / "data_preparation"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    if "log_volatility" not in df.columns:
        logger.warning("log_volatility column not found in DataFrame, skipping plot")
        return

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Histogram with KDE
    sns.histplot(
        data=df,
        x="log_volatility",
        kde=True,
        ax=ax1,
        alpha=_PLOT_ALPHA_DEFAULT,
        color="skyblue",
    )
    ax1.set_title(
        "Distribution de log_volatility (Histogramme + KDE)", fontsize=_FONTSIZE_TITLE
    )
    ax1.set_xlabel("log_volatility", fontsize=_FONTSIZE_LABEL)
    ax1.set_ylabel("Fréquence", fontsize=_FONTSIZE_LABEL)
    ax1.grid(True, alpha=0.3)

    # Box plot
    sns.boxplot(
        data=df,
        y="log_volatility",
        ax=ax2,
        color="lightcoral",
        width=0.3,
    )
    ax2.set_title("Distribution de log_volatility (Box Plot)", fontsize=_FONTSIZE_TITLE)
    ax2.set_ylabel("log_volatility", fontsize=_FONTSIZE_LABEL)
    ax2.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save plot
    output_file = output_dir / "log_volatility_distribution.png"
    save_plot_wrapper(str(output_file), dpi=_PLOT_DPI)

    logger.info("Saved log_volatility distribution plot to: %s", output_file)
