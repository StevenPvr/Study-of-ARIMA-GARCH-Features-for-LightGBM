"""Main entry point for block permutation importance analysis.

This module computes permutation importance for all trained LightGBM models
and saves results to JSON and plots. It is separated from evaluation as it
is computationally expensive (can take several hours).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config_logging import setup_logging
from src.constants import (
    DEFAULT_RANDOM_STATE,
    LIGHTGBM_DATASET_COMPLETE_FILE,
    LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE,
    LIGHTGBM_MODELS_DIR,
)
from src.lightgbm.eval.data_loading import load_dataset, load_model
from src.lightgbm.model_utils import filter_existing_models, get_optional_model_configs
from src.lightgbm.permutation.permutation import (
    compute_block_permutation_importance,
    plot_permutation_bars,
    save_permutation_results,
)
from src.utils import get_logger

# Setup logging first
setup_logging()
logger = get_logger(__name__)


def _get_required_model_configs() -> list[tuple[Path, Path, str]]:
    """Get list of required model configurations (always included)."""
    return [
        (
            LIGHTGBM_DATASET_COMPLETE_FILE,
            LIGHTGBM_MODELS_DIR / "lightgbm_complete.joblib",
            "lightgbm_complete",
        ),
        (
            LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE,
            LIGHTGBM_MODELS_DIR / "lightgbm_without_insights.joblib",
            "lightgbm_without_insights",
        ),
    ]


def _prepare_permutation_tasks() -> list[tuple[Path, Path, str]]:
    """Prepare permutation tasks for all trained models.

    Returns:
        List of tuples (dataset_path, model_path, model_name).
    """
    tasks = _get_required_model_configs()
    optional_configs = get_optional_model_configs()
    existing_optional = filter_existing_models(optional_configs)
    tasks.extend(existing_optional)

    return tasks


def _run_permutation_for_model(
    dataset_path: Path,
    model_path: Path,
    model_name: str,
    block_size: int,
    n_repeats: int,
    sample_fraction: float,
    random_state: int,
) -> dict[str, dict[str, float]]:
    """Run permutation importance for a single model.

    Args:
        dataset_path: Path to the dataset CSV file.
        model_path: Path to the trained model joblib file.
        model_name: Name for the model.
        block_size: Contiguous block size for permutation.
        n_repeats: Number of random block permutations.
        sample_fraction: Fraction of test data to use.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary mapping feature -> importance statistics.
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"PERMUTATION IMPORTANCE: {model_name}")
    logger.info(f"{'=' * 70}")

    # Load test data
    X_test, y_test = load_dataset(dataset_path, split="test")

    # Load model
    model = load_model(model_path)

    # Compute permutation importance
    logger.info(
        f"Computing permutation importance with block_size={block_size}, "
        f"n_repeats={n_repeats}, sample_fraction={sample_fraction}"
    )
    results = compute_block_permutation_importance(
        model,
        X_test,
        y_test,
        block_size=block_size,
        n_repeats=n_repeats,
        sample_fraction=sample_fraction,
        random_state=random_state,
    )

    logger.info(f"Permutation importance computed for {len(results)} features")
    return results


def main(
    block_size: int = 20,
    n_repeats: int = 300,
    sample_fraction: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> None:
    """Compute block permutation importance for all trained models.

    Args:
        block_size: Contiguous block size for permutation (default: 20).
        n_repeats: Number of random block permutations (default: 300).
        sample_fraction: Fraction of test data to use (default: 0.2 = 20%).
        random_state: Seed for reproducibility.
    """
    logger.info("=" * 80)
    logger.info("BLOCK PERMUTATION IMPORTANCE ANALYSIS")
    logger.info("=" * 80)

    # Prepare tasks
    tasks = _prepare_permutation_tasks()

    if not tasks:
        logger.error("No trained models found. Please run training first.")
        return

    logger.info(f"Found {len(tasks)} model(s) to analyze")

    # Run permutation for each model
    per_model_results: dict[str, dict[str, dict[str, float]]] = {}
    for i, (dataset_path, model_path, model_name) in enumerate(tasks, 1):
        logger.info(f"\n[{i}/{len(tasks)}] Processing {model_name}...")
        results = _run_permutation_for_model(
            dataset_path,
            model_path,
            model_name,
            block_size,
            n_repeats,
            sample_fraction,
            random_state,
        )
        per_model_results[model_name] = results

    # Save results
    logger.info("\n" + "=" * 70)
    logger.info("SAVING RESULTS")
    logger.info("=" * 70)
    save_permutation_results(per_model_results)

    # Create plots
    logger.info("\n" + "=" * 70)
    logger.info("CREATING PLOTS")
    logger.info("=" * 70)
    plot_paths = plot_permutation_bars(per_model_results, top_k=20)
    logger.info(f"Created {len(plot_paths)} plot(s)")

    logger.info("\n" + "=" * 80)
    logger.info("PERMUTATION IMPORTANCE ANALYSIS COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute block permutation importance for LightGBM models"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=20,
        help="Contiguous block size for permutation (default: 20)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=300,
        help="Number of random block permutations (default: 300)",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.2,
        help="Fraction of test data to use (default: 0.2 = 20%%)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_STATE})",
    )

    args = parser.parse_args()

    main(
        block_size=args.block_size,
        n_repeats=args.n_repeats,
        sample_fraction=args.sample_fraction,
        random_state=args.random_state,
    )
