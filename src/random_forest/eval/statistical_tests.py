"""Statistical tests for comparing Random Forest model performance."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

from src.utils import get_logger

logger = get_logger(__name__)


def diebold_mariano_test(
    errors_model1: np.ndarray | pd.Series,
    errors_model2: np.ndarray | pd.Series,
    h: int = 1,
    power: int = 2,
) -> dict[str, Any]:
    """Perform Diebold-Mariano test to compare forecast accuracy of two models.

    The Diebold-Mariano test tests the null hypothesis that two forecasts have
    equal predictive accuracy. A significant result indicates that one model
    is significantly better than the other.

    Args:
        errors_model1: Forecast errors from model 1 (y_true - y_pred).
        errors_model2: Forecast errors from model 2 (y_true - y_pred).
        h: Forecast horizon (default: 1 for one-step ahead).
        power: Power for loss differential (2 for MSE, 1 for MAE).

    Returns:
        Dictionary with test statistic, p-value, and interpretation.

    Raises:
        ValueError: If error arrays have different lengths or are empty.

    References:
        Diebold, F.X. and Mariano, R.S. (1995) "Comparing Predictive Accuracy",
        Journal of Business and Economic Statistics, 13, 253-263.
    """
    e1 = np.asarray(errors_model1).flatten()
    e2 = np.asarray(errors_model2).flatten()

    if len(e1) != len(e2):
        msg = f"Error arrays must have same length: {len(e1)} vs {len(e2)}"
        raise ValueError(msg)

    if len(e1) == 0:
        raise ValueError("Error arrays cannot be empty")

    # Calculate loss differential
    d = np.abs(e1) ** power - np.abs(e2) ** power

    # Mean of loss differential
    d_mean = np.mean(d)

    # Variance of loss differential with autocorrelation correction
    n = len(d)
    gamma_0 = np.var(d, ddof=1)

    # Harvey, Leybourne & Newbold (1997) correction for small samples
    if h > 1:
        gamma = [gamma_0]
        for k in range(1, h):
            gamma.append(np.cov(d[:-k], d[k:])[0, 1])
        variance = gamma_0 + 2 * np.sum(gamma[1:])
    else:
        variance = gamma_0

    # Handle case where models are identical (variance = 0)
    if variance < 1e-10 or np.isclose(variance, 0.0):
        dm_stat = 0.0
        p_value = 1.0
    else:
        # Diebold-Mariano statistic
        dm_stat = d_mean / np.sqrt(variance / n)

        # Two-tailed p-value (using t-distribution for small samples)
        p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n - 1))

    # Interpretation
    if p_value < 0.01:
        significance = "highly significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "significant (p < 0.05)"
    elif p_value < 0.10:
        significance = "marginally significant (p < 0.10)"
    else:
        significance = "not significant (p >= 0.10)"

    if dm_stat > 0:
        better_model = "model_2"
        interpretation = f"Model 2 performs better ({significance})"
    elif dm_stat < 0:
        better_model = "model_1"
        interpretation = f"Model 1 performs better ({significance})"
    else:
        better_model = "equal"
        interpretation = f"Models perform equally ({significance})"

    logger.info(f"Diebold-Mariano test: DM={dm_stat:.4f}, p={p_value:.4f}")
    logger.info(f"Interpretation: {interpretation}")

    return {
        "dm_statistic": float(dm_stat),
        "p_value": float(p_value),
        "better_model": better_model,
        "significance": significance,
        "interpretation": interpretation,
        "mean_loss_diff": float(d_mean),
        "n_observations": int(n),
    }


def compare_models_statistical(
    y_true: np.ndarray | pd.Series,
    y_pred_model1: np.ndarray | pd.Series,
    y_pred_model2: np.ndarray | pd.Series,
    model1_name: str = "model_1",
    model2_name: str = "model_2",
) -> dict[str, Any]:
    """Compare two models using Diebold-Mariano test.

    Args:
        y_true: True target values.
        y_pred_model1: Predictions from model 1.
        y_pred_model2: Predictions from model 2.
        model1_name: Name of model 1 (for logging).
        model2_name: Name of model 2 (for logging).

    Returns:
        Dictionary with test results for MSE and MAE comparisons.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred1 = np.asarray(y_pred_model1).flatten()
    y_pred2 = np.asarray(y_pred_model2).flatten()

    # Calculate errors
    errors1 = y_true - y_pred1
    errors2 = y_true - y_pred2

    logger.info("=" * 70)
    logger.info(f"STATISTICAL COMPARISON: {model1_name} vs {model2_name}")
    logger.info("=" * 70)

    # Test based on MSE (power=2)
    logger.info("\nDiebold-Mariano Test (MSE-based):")
    dm_mse = diebold_mariano_test(errors1, errors2, h=1, power=2)

    # Test based on MAE (power=1)
    logger.info("\nDiebold-Mariano Test (MAE-based):")
    dm_mae = diebold_mariano_test(errors1, errors2, h=1, power=1)

    logger.info("=" * 70)

    return {
        "mse_based": dm_mse,
        "mae_based": dm_mae,
        "model1_name": model1_name,
        "model2_name": model2_name,
    }


def bootstrap_r2_comparison(
    y_true: np.ndarray | pd.Series,
    y_pred_model1: np.ndarray | pd.Series,
    y_pred_model2: np.ndarray | pd.Series,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compare R² scores using bootstrap resampling.

    Args:
        y_true: True target values.
        y_pred_model1: Predictions from model 1.
        y_pred_model2: Predictions from model 2.
        n_bootstrap: Number of bootstrap samples.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with bootstrap test results for R² comparison.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred1 = np.asarray(y_pred_model1).flatten()
    y_pred2 = np.asarray(y_pred_model2).flatten()

    n = len(y_true)
    rng = np.random.RandomState(random_state)

    # Calculate observed R² for both models
    r2_model1 = r2_score(y_true, y_pred1)
    r2_model2 = r2_score(y_true, y_pred2)
    r2_diff_observed = r2_model1 - r2_model2

    # Bootstrap resampling
    r2_diffs = []
    for _ in range(n_bootstrap):
        # Resample indices with replacement
        indices = rng.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_pred1_boot = y_pred1[indices]
        y_pred2_boot = y_pred2[indices]

        # Calculate R² for bootstrap sample
        r2_1_boot = r2_score(y_true_boot, y_pred1_boot)
        r2_2_boot = r2_score(y_true_boot, y_pred2_boot)
        r2_diffs.append(r2_1_boot - r2_2_boot)

    r2_diffs = np.array(r2_diffs)

    # Calculate statistics
    mean_diff = float(np.mean(r2_diffs))
    std_diff = float(np.std(r2_diffs))

    # Calculate p-value (two-tailed test)
    # H0: R² difference = 0
    p_value = float(np.mean(np.abs(r2_diffs) >= np.abs(r2_diff_observed)))

    # Confidence interval (95%)
    ci_lower = float(np.percentile(r2_diffs, 2.5))
    ci_upper = float(np.percentile(r2_diffs, 97.5))

    # Interpretation
    if p_value < 0.01:
        significance = "highly significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "significant (p < 0.05)"
    elif p_value < 0.10:
        significance = "marginally significant (p < 0.10)"
    else:
        significance = "not significant (p >= 0.10)"

    if r2_diff_observed > 0:
        better_model = "model_1"
        interpretation = f"Model 1 has higher R² ({significance})"
    elif r2_diff_observed < 0:
        better_model = "model_2"
        interpretation = f"Model 2 has higher R² ({significance})"
    else:
        better_model = "equal"
        interpretation = f"Models have equal R² ({significance})"

    logger.info(f"Bootstrap R² comparison: diff={r2_diff_observed:.6f}, p={p_value:.4f}")
    logger.info(f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
    logger.info(f"Interpretation: {interpretation}")

    return {
        "r2_model1": float(r2_model1),
        "r2_model2": float(r2_model2),
        "r2_diff_observed": float(r2_diff_observed),
        "mean_diff_bootstrap": mean_diff,
        "std_diff_bootstrap": std_diff,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "better_model": better_model,
        "significance": significance,
        "interpretation": interpretation,
        "n_bootstrap": n_bootstrap,
    }

