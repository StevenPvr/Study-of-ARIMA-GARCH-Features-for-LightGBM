"""CLI for estimating EGARCH(1,1) parameters via conditional MLE.

Estimates parameters for three distributions simultaneously:
- Normal
- Student-t
- Skew-t

Uses conditional maximum likelihood estimation with variance recursion.
Results saved to results/garch/eval/
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

from src.constants import (
    GARCH_DATASET_FILE,
    GARCH_ESTIMATION_FILE,
    GARCH_EVAL_DIR,
)
from src.garch.garch_params.estimation import estimate_egarch_mle
from src.garch.structure_garch.detection import load_garch_dataset, prepare_residuals
from src.utils import get_logger

logger = get_logger(__name__)


def _load_and_prepare_data() -> tuple[np.ndarray, np.ndarray]:
    """Load dataset and prepare training/test residuals."""
    try:
        df = load_garch_dataset(str(GARCH_DATASET_FILE))
    except Exception as ex:
        logger.error("Failed to load GARCH dataset: %s", ex)
        raise

    df_train = df.loc[df["split"] == "train"].copy()
    if df_train.empty:
        msg = "No training data found in dataset"
        logger.error(msg)
        raise ValueError(msg)

    try:
        resid_train = prepare_residuals(df_train, use_test_only=False)
        resid_test = prepare_residuals(df, use_test_only=True)
    except Exception as ex:
        logger.error("Failed to prepare residuals: %s", ex)
        raise

    resid_train = resid_train[np.isfinite(resid_train)]
    resid_test = resid_test[np.isfinite(resid_test)]

    if resid_train.size < 10:
        msg = f"Insufficient training residuals: {resid_train.size} < 10"
        logger.error(msg)
        raise ValueError(msg)

    return resid_train, resid_test


def _estimate_single_model(
    resid_train: np.ndarray, dist: str
) -> tuple[str, dict[str, float] | None]:
    """Estimate a single EGARCH model (helper for parallel execution).

    Args:
        resid_train: Training residuals from SARIMA model.
        dist: Distribution name ('normal', 'student', or 'skewt').

    Returns:
        Tuple of (distribution_name, parameter_dict or None).
    """
    try:
        logger.info("Optimizing EGARCH(1,1) with %s innovations...", dist.capitalize())
        result = estimate_egarch_mle(resid_train, dist=dist)
        return dist, result
    except Exception as ex:
        logger.warning("EGARCH MLE (%s) failed: %s", dist, ex)
        return dist, None


def _estimate_egarch_models(
    resid_train: np.ndarray,
) -> tuple[
    dict[str, float] | None,
    dict[str, float] | None,
    dict[str, float] | None,
]:
    """Estimate EGARCH models for normal, student, and skewt distributions.

    Optimizes all three models in parallel using conditional MLE.

    Args:
        resid_train: Training residuals from SARIMA model.

    Returns:
        Tuple of (egarch_normal, egarch_student, egarch_skewt) parameter dicts.
    """
    distributions = ["normal", "student", "skewt"]
    results: dict[str, dict[str, float] | None] = {}

    # Optimize all three models in parallel
    logger.info("Starting parallel optimization of 3 EGARCH models...")
    with ProcessPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_to_dist = {
            executor.submit(_estimate_single_model, resid_train, dist): dist
            for dist in distributions
        }

        # Collect results as they complete
        for future in as_completed(future_to_dist):
            dist, result = future.result()
            results[dist] = result

    return results.get("normal"), results.get("student"), results.get("skewt")


def main() -> None:
    """Estimate EGARCH(1,1) parameters via conditional MLE.

    Methodology:
    - Assumes parametric distribution for innovations zt (Normal, Student-t, Skew-t)
    - Maximizes conditional log-likelihood by recursing conditional variance σt²
    - Optimizes all three distributions simultaneously

    Results saved to results/garch/eval/
    """
    logger.info("=" * 60)
    logger.info("GARCH ESTIMATION (Conditional MLE)")
    logger.info("Optimizing: Normal, Student-t, Skew-t")
    logger.info("=" * 60)

    resid_train, resid_test = _load_and_prepare_data()
    egarch_normal, egarch_student, egarch_skewt = _estimate_egarch_models(resid_train)

    payload = {
        "source": str(GARCH_DATASET_FILE),
        "methodology": "Conditional maximum likelihood estimation",
        "n_obs_train": int(resid_train.size),
        "n_obs_test": int(resid_test.size),
        "egarch_normal": egarch_normal,
        "egarch_student": egarch_student,
        "egarch_skewt": egarch_skewt,
    }

    GARCH_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with GARCH_ESTIMATION_FILE.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved GARCH MLE results: %s", GARCH_ESTIMATION_FILE)


if __name__ == "__main__":
    main()
