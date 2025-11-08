"""Spearman correlation calculation and visualization for Random Forest datasets."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Set non-interactive backend before importing pyplot
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.constants import (
    RF_CORRELATION_PLOTS_DIR,
    RF_DATASET_COMPLETE_FILE,
    RF_DATASET_WITHOUT_INSIGHTS_FILE,
)
from src.utils import get_logger

logger = get_logger(__name__)


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load a Random Forest dataset.

    Args:
        file_path: Path to the dataset CSV file.

    Returns:
        DataFrame with the dataset.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    logger.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path, parse_dates=["date"])
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def calculate_spearman_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Spearman rank correlation matrix.

    Excludes non-numeric columns (date, split) and calculates correlation
    only on numeric features.

    Args:
        df: DataFrame with numeric and non-numeric columns.

    Returns:
        Correlation matrix as DataFrame.

    Raises:
        ValueError: If DataFrame is empty or has no numeric columns.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    # Select only numeric columns (exclude date, split, etc.)
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    if not numeric_cols:
        raise ValueError("No numeric columns found in DataFrame")

    logger.info(f"Calculating Spearman correlation for {len(numeric_cols)} numeric columns")
    numeric_df = df[numeric_cols]
    # DataFrame.corr() can be called without 'other' parameter to compute correlation matrix
    corr_matrix = numeric_df.corr(method="spearman")  # type: ignore[call-arg]

    logger.info("Spearman correlation matrix calculated successfully")
    return corr_matrix


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    output_path: Path,
    dataset_name: str,
    figsize: tuple[int, int] = (12, 10),
) -> None:
    """Create and save a heatmap visualization of the correlation matrix.

    Args:
        corr_matrix: Correlation matrix DataFrame.
        output_path: Path to save the plot.
        dataset_name: Name of the dataset (for logging).
        figsize: Figure size (width, height) in inches.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1,
    )
    plt.title(f"Spearman Correlation Matrix - {dataset_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved correlation plot for {dataset_name}: {output_path}")


def compute_correlations(
    complete_dataset_path: Path | None = None,
    without_insights_dataset_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Spearman correlations and create visualizations for both Random Forest datasets.

    Args:
        complete_dataset_path: Path to complete dataset. If None, uses default.
        without_insights_dataset_path: Path to dataset without insights. If None, uses default.
        output_dir: Directory to save correlation plots. If None, uses RF_CORRELATION_PLOTS_DIR.

    Returns:
        Tuple of (complete_correlation_matrix, without_insights_correlation_matrix).

    Raises:
        FileNotFoundError: If dataset files don't exist.
        ValueError: If datasets are empty or have no numeric columns.
    """
    if complete_dataset_path is None:
        complete_dataset_path = RF_DATASET_COMPLETE_FILE
    if without_insights_dataset_path is None:
        without_insights_dataset_path = RF_DATASET_WITHOUT_INSIGHTS_FILE
    if output_dir is None:
        output_dir = RF_CORRELATION_PLOTS_DIR

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    df_complete = load_dataset(complete_dataset_path)
    df_without = load_dataset(without_insights_dataset_path)

    # Calculate correlations
    corr_complete = calculate_spearman_correlation(df_complete)
    corr_without = calculate_spearman_correlation(df_without)

    # Create and save correlation plots
    output_complete = output_dir / "rf_correlation_complete.png"
    output_without = output_dir / "rf_correlation_without_insights.png"

    plot_correlation_matrix(corr_complete, output_complete, "Complete Dataset")
    plot_correlation_matrix(corr_without, output_without, "Dataset Without Insights")

    return corr_complete, corr_without
