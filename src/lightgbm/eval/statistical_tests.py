"""Statistical tests for comparing LightGBM model performance."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

from src.utils import get_logger

logger = get_logger(__name__)


def _calculate_variance_with_autocorrelation(d: np.ndarray, h: int) -> float:
    """Calculate variance with autocorrelation correction (HAC estimator).

    Uses Newey-West style HAC variance estimator:
    V = gamma_0 + 2 * sum_{k=1}^{h-1} gamma_k

    where gamma_k is the autocovariance at lag k.

    Args:
        d: Loss differential array d_t = L_t(1) - L_t(2).
        h: Forecast horizon.

    Returns:
        Variance of the mean with autocorrelation correction.
    """
    n = len(d)
    d_mean = float(np.mean(d))
    d_centered = d - d_mean

    # Variance (gamma_0)
    gamma_0 = float(np.var(d_centered, ddof=1))

    if h <= 1:
        # Variance of the mean
        return gamma_0 / n

    # Calculate autocovariances gamma_k for k = 1 to h-1
    gamma_sum = 0.0
    for k in range(1, min(h, n)):
        if n - k > 0:
            gamma_k = float(np.mean(d_centered[k:] * d_centered[:-k]))
            gamma_sum += gamma_k

    variance_hac = gamma_0 + 2.0 * gamma_sum

    # Variance of the mean
    return variance_hac / n


def _calculate_dm_statistic(d_mean: float, variance: float, n: int) -> tuple[float, float]:
    """Calculate Diebold-Mariano statistic and p-value.

    For large samples (n > 1000), uses normal distribution instead of t-distribution.

    Args:
        d_mean: Mean of loss differential.
        variance: Variance of the mean (already divided by n).
        n: Number of observations.

    Returns:
        Tuple of (dm_statistic, p_value).
    """
    # Only treat truly degenerate cases as "no information"
    if variance <= 0.0 or not np.isfinite(variance):
        return 0.0, 1.0

    dm_stat = float(d_mean / np.sqrt(variance))

    if n > 1000:
        p_value = float(2 * (1 - stats.norm.cdf(np.abs(dm_stat))))
    else:
        p_value = float(2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n - 1)))

    return dm_stat, p_value


def _interpret_significance(p_value: float) -> str:
    """Interpret p-value significance level."""
    if p_value < 0.01:
        return "highly significant (p < 0.01)"
    if p_value < 0.05:
        return "significant (p < 0.05)"
    if p_value < 0.10:
        return "marginally significant (p < 0.10)"
    return "not significant (p >= 0.10)"


def _interpret_dm_result(dm_stat: float, significance: str) -> tuple[str, str]:
    """Interpret Diebold-Mariano test result.

    Args:
        dm_stat: DM statistic value.
        significance: Significance level string.

    Returns:
        Tuple of (better_model, interpretation).
    """
    if dm_stat > 0:
        return "model_2", f"Model 2 performs better ({significance})"
    if dm_stat < 0:
        return "model_1", f"Model 1 performs better ({significance})"
    return "equal", f"Models perform equally ({significance})"


def diebold_mariano_test(
    errors_model1: np.ndarray | pd.Series,
    errors_model2: np.ndarray | pd.Series,
    h: int = 1,
    power: int = 2,
) -> dict[str, Any]:
    """Perform Diebold-Mariano test to compare forecast accuracy of two models.

    H0: equal predictive accuracy.

    Args:
        errors_model1: Forecast errors from model 1 (y_true - y_pred).
        errors_model2: Forecast errors from model 2 (y_true - y_pred).
        h: Forecast horizon (default: 1).
        power: Power for loss differential (2 for MSE, 1 for MAE).

    Returns:
        Dictionary with test statistic, p-value, effect size, and interpretation.
    """
    e1 = np.asarray(errors_model1).flatten()
    e2 = np.asarray(errors_model2).flatten()

    if len(e1) != len(e2):
        msg = f"Error arrays must have same length: {len(e1)} vs {len(e2)}"
        raise ValueError(msg)
    if len(e1) == 0:
        raise ValueError("Error arrays cannot be empty")

    # Loss differential
    d = np.abs(e1) ** power - np.abs(e2) ** power
    d_mean = cast(float, float(np.mean(d)))
    n = len(d)

    variance = _calculate_variance_with_autocorrelation(d, h)
    dm_stat, p_value = _calculate_dm_statistic(d_mean, variance, n)

    significance = _interpret_significance(p_value)
    better_model, interpretation = _interpret_dm_result(dm_stat, significance)

    # Effect size = DM statistic (standardised mean difference using var of mean)
    if variance > 0.0 and np.isfinite(variance):
        std_d_mean = float(np.sqrt(variance))
        effect_size = float(d_mean / std_d_mean)
    else:
        effect_size = 0.0

    logger.info(f"Diebold-Mariano test: DM={dm_stat:.4f}, p={p_value:.4f}")
    logger.info(f"Mean loss difference: {d_mean:.6f} (effect size: {effect_size:.4f})")
    logger.info(f"Interpretation: {interpretation}")

    return {
        "dm_statistic": float(dm_stat),
        "p_value": float(p_value),
        "better_model": better_model,
        "significance": significance,
        "interpretation": interpretation,
        "mean_loss_diff": float(d_mean),
        "effect_size": float(effect_size),
        "n_observations": int(n),
    }


def compare_models_statistical(
    y_true: np.ndarray | pd.Series,
    y_pred_model1: np.ndarray | pd.Series,
    y_pred_model2: np.ndarray | pd.Series,
    model1_name: str = "model_1",
    model2_name: str = "model_2",
) -> dict[str, Any]:
    """Compare two models using Diebold-Mariano test (MSE and MAE)."""
    y_true = np.asarray(y_true).flatten()
    y_pred1 = np.asarray(y_pred_model1).flatten()
    y_pred2 = np.asarray(y_pred_model2).flatten()

    errors1 = y_true - y_pred1
    errors2 = y_true - y_pred2

    logger.info("=" * 70)
    logger.info(f"STATISTICAL COMPARISON: {model1_name} vs {model2_name}")
    logger.info("=" * 70)

    logger.info("\nDiebold-Mariano Test (MSE-based):")
    dm_mse = diebold_mariano_test(errors1, errors2, h=1, power=2)

    logger.info("\nDiebold-Mariano Test (MAE-based):")
    dm_mae = diebold_mariano_test(errors1, errors2, h=1, power=1)

    logger.info("=" * 70)

    return {
        "mse_based": dm_mse,
        "mae_based": dm_mae,
        "model1_name": model1_name,
        "model2_name": model2_name,
    }


def _determine_bootstrap_size(n: int, n_bootstrap: int | None) -> int:
    """Determine number of bootstrap samples based on data size.

    Args:
        n: Number of observations.
        n_bootstrap: User-specified bootstrap size. If None, auto-determined.

    Returns:
        Number of bootstrap samples to use.
    """
    if n_bootstrap is not None:
        return n_bootstrap
    if n < 1000:
        return 1000
    if n < 10000:
        return 2000
    return 5000


def _compute_bootstrap_r2_differences(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    n_bootstrap: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Compute R² differences via bootstrap resampling.

    Args:
        y_true: True target values.
        y_pred1: Predictions from model 1.
        y_pred2: Predictions from model 2.
        n_bootstrap: Number of bootstrap samples.
        rng: Random number generator.

    Returns:
        Array of R² differences from bootstrap samples.
    """
    n = len(y_true)
    r2_diffs = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_pred1_boot = y_pred1[indices]
        y_pred2_boot = y_pred2[indices]

        r2_1_boot = r2_score(y_true_boot, y_pred1_boot)
        r2_2_boot = r2_score(y_true_boot, y_pred2_boot)
        r2_diffs.append(r2_1_boot - r2_2_boot)

    return np.asarray(r2_diffs)


def _compute_bootstrap_statistics(r2_diffs: np.ndarray) -> dict[str, float]:
    """Compute bootstrap statistics (mean, std, p-value, confidence intervals).

    Args:
        r2_diffs: Array of R² differences from bootstrap samples.

    Returns:
        Dictionary with mean_diff, std_diff, p_value, ci_lower, ci_upper.
    """
    mean_diff = float(np.mean(r2_diffs))
    std_diff = float(np.std(r2_diffs))

    prop_positive = float(np.mean(r2_diffs >= 0))
    prop_negative = float(np.mean(r2_diffs <= 0))
    p_value = float(2 * min(prop_positive, prop_negative))

    ci_lower = float(np.percentile(r2_diffs, 2.5))
    ci_upper = float(np.percentile(r2_diffs, 97.5))

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def _interpret_bootstrap_r2_result(r2_diff_observed: float, p_value: float) -> tuple[str, str, str]:
    """Interpret bootstrap R² comparison result.

    Args:
        r2_diff_observed: Observed R² difference (model1 - model2).
        p_value: Bootstrap p-value.

    Returns:
        Tuple of (better_model, significance, interpretation).
    """
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

    return better_model, significance, interpretation


def bootstrap_r2_comparison(
    y_true: np.ndarray | pd.Series,
    y_pred_model1: np.ndarray | pd.Series,
    y_pred_model2: np.ndarray | pd.Series,
    n_bootstrap: int | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compare R² scores using bootstrap resampling.

    H0: R² difference = 0.

    Args:
        y_true: True target values.
        y_pred_model1: Predictions from model 1.
        y_pred_model2: Predictions from model 2.
        n_bootstrap: Number of bootstrap samples. If None, auto based on n.
        random_state: Random seed.

    Returns:
        Dictionary with bootstrap test results for R² comparison.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred1 = np.asarray(y_pred_model1).flatten()
    y_pred2 = np.asarray(y_pred_model2).flatten()

    n = len(y_true)
    n_bootstrap = _determine_bootstrap_size(n, n_bootstrap)
    rng = np.random.RandomState(random_state)

    r2_model1 = r2_score(y_true, y_pred1)
    r2_model2 = r2_score(y_true, y_pred2)
    r2_diff_observed = r2_model1 - r2_model2

    r2_diffs = _compute_bootstrap_r2_differences(y_true, y_pred1, y_pred2, n_bootstrap, rng)
    stats_dict = _compute_bootstrap_statistics(r2_diffs)
    better_model, significance, interpretation = _interpret_bootstrap_r2_result(
        r2_diff_observed, stats_dict["p_value"]
    )

    logger.info(
        f"Bootstrap R² comparison: diff={r2_diff_observed:.6f}, p={stats_dict['p_value']:.4f}"
    )
    logger.info(f"95% CI: [{stats_dict['ci_lower']:.6f}, {stats_dict['ci_upper']:.6f}]")
    logger.info(f"Interpretation: {interpretation}")

    return {
        "r2_model1": float(r2_model1),
        "r2_model2": float(r2_model2),
        "r2_diff_observed": float(r2_diff_observed),
        "mean_diff_bootstrap": stats_dict["mean_diff"],
        "std_diff_bootstrap": stats_dict["std_diff"],
        "p_value": stats_dict["p_value"],
        "ci_lower": stats_dict["ci_lower"],
        "ci_upper": stats_dict["ci_upper"],
        "better_model": better_model,
        "significance": significance,
        "interpretation": interpretation,
        "n_bootstrap": int(n_bootstrap),
    }
