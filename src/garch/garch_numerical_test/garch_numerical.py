"""Numerical tests for GARCH structure detection (pre-EGARCH, post-SARIMA).

Provides implementations of:
- Ljung-Box test on residuals
- Ljung-Box test on squared residuals
- Engle ARCH-LM test
- McLeod-Li test
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.constants import (
    GARCH_LJUNG_BOX_LAGS_DEFAULT,
    GARCH_LM_LAGS_DEFAULT,
    LJUNGBOX_SIGNIFICANCE_LEVEL,
)
from src.utils import get_logger

logger = get_logger(__name__)


def _prepare_lags_list(lags: int | list[int]) -> list[int]:
    """Convert lags to a list of integers.

    Args:
        lags: Single lag or list of lags.

    Returns:
        List of lag integers.
    """
    if isinstance(lags, int):
        return list(range(1, lags + 1))
    return sorted(set(int(lag) for lag in lags if lag > 0))


def _chi2_sf(x: float, df: int) -> float:
    """Chi-square survival function P[X >= x].

    Args:
        x: Test statistic value.
        df: Degrees of freedom.

    Returns:
        P-value.
    """
    try:
        from scipy.stats import chi2  # type: ignore

        return float(chi2.sf(x, df))
    except Exception:
        return float("nan")


def _compute_ljung_box_manual(
    res_centered: np.ndarray, lags_list: list[int], n: float
) -> tuple[list[float], list[float]]:
    """Compute Ljung-Box statistics manually.

    Args:
        res_centered: Centered residual series.
        lags_list: List of lag values to test.
        n: Sample size.

    Returns:
        Tuple of (lb_stats, lb_pvalues) lists.
    """
    lb_stats = []
    lb_pvalues = []
    s = 0.0
    for h in lags_list:
        if h >= n:
            break
        acf_h = np.sum(res_centered[h:] * res_centered[:-h]) / np.sum(res_centered**2)
        s += (acf_h * acf_h) / max(1.0, (n - h))
        q = n * (n + 2.0) * s
        lb_stats.append(float(q))
        p_val = _chi2_sf(q, h)
        lb_pvalues.append(p_val)
    return lb_stats, lb_pvalues


def _compute_ljung_box_stats(
    res: np.ndarray, lags_list: list[int]
) -> tuple[list[float], list[float]]:
    """Compute Ljung-Box statistics using statsmodels or fallback.

    Args:
        res: Residual series (already cleaned).
        lags_list: List of lag values to test.

    Returns:
        Tuple of (lb_stats, lb_pvalues) lists.
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox  # type: ignore

        df = acorr_ljungbox(res, lags=lags_list, return_df=True)
        return [float(x) for x in df["lb_stat"]], [float(x) for x in df["lb_pvalue"]]
    except Exception as e:
        logger.debug(f"Fallback in Ljung-Box computation: {e}")
        n = float(res.size)
        mean_res = np.mean(res)
        res_centered = res - mean_res
        return _compute_ljung_box_manual(res_centered, lags_list, n)


def ljung_box_test(
    residuals: np.ndarray,
    lags: int = GARCH_LJUNG_BOX_LAGS_DEFAULT,
    alpha: float = LJUNGBOX_SIGNIFICANCE_LEVEL,
) -> dict[str, Any]:
    """Ljung-Box test on residuals to assess whiteness.

    Tests the null hypothesis that the residuals are independently
    distributed (no autocorrelation).

    Args:
        residuals: Residual series.
        lags: Number of lags to test.
        alpha: Significance level.

    Returns:
        Dict with lags, lb_stat, lb_pvalue, reject (bool), and n.
    """
    res = np.asarray(residuals, dtype=float)
    res = res[np.isfinite(res)]
    lags_list = _prepare_lags_list(lags)

    if res.size == 0 or not lags_list:
        return {
            "lags": [],
            "lb_stat": [],
            "lb_pvalue": [],
            "reject": False,
            "n": 0,
        }

    lb_stats, lb_pvalues = _compute_ljung_box_stats(res, lags_list)
    reject = bool(lb_pvalues[-1] < alpha) if lb_pvalues else False
    return {
        "lags": lags_list[: len(lb_stats)],
        "lb_stat": lb_stats,
        "lb_pvalue": lb_pvalues,
        "reject": reject,
        "n": int(res.size),
    }


def ljung_box_squared_test(
    residuals: np.ndarray,
    lags: int = GARCH_LJUNG_BOX_LAGS_DEFAULT,
    alpha: float = LJUNGBOX_SIGNIFICANCE_LEVEL,
) -> dict[str, Any]:
    """Ljung-Box test on squared residuals to detect ARCH effects.

    Tests the null hypothesis that squared residuals are independently
    distributed (no autocorrelation in squared residuals).

    Args:
        residuals: Residual series.
        lags: Number of lags to test.
        alpha: Significance level.

    Returns:
        Dict with lags, lb_stat, lb_pvalue, reject (bool), and n.
    """
    res = np.asarray(residuals, dtype=float)
    res = res[np.isfinite(res)]
    res_squared = res**2
    return ljung_box_test(res_squared, lags=lags, alpha=alpha)


def _compute_arch_lm_statistic(e2: np.ndarray, lags: int) -> float:
    """Compute ARCH-LM test statistic.

    Args:
        e2: Squared residuals.
        lags: Number of lags in regression.

    Returns:
        LM test statistic value.
    """
    n = int(e2.size)
    Y = e2[lags:]
    X = np.ones((n - lags, lags + 1), dtype=float)
    for j in range(1, lags + 1):
        X[:, j] = e2[lags - j : n - j]

    try:
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        Y_hat = X @ beta
        ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
        ss_res = float(np.sum((Y - Y_hat) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return (n - lags) * max(0.0, r2)
    except Exception:
        return float("nan")


def engle_arch_lm_test(
    residuals: np.ndarray,
    lags: int = GARCH_LM_LAGS_DEFAULT,
    alpha: float = LJUNGBOX_SIGNIFICANCE_LEVEL,
) -> dict[str, Any]:
    """Engle's ARCH-LM test for heteroskedasticity.

    Tests the null hypothesis of no ARCH effects using Lagrange multiplier
    test on squared residuals regressed on lagged squared residuals.

    Args:
        residuals: Residual series.
        lags: Number of lags in regression.
        alpha: Significance level.

    Returns:
        Dict with lm_stat, p_value, df, reject (bool), and n.
    """
    e2 = np.asarray(residuals, dtype=float) ** 2
    e2 = e2[~np.isnan(e2)]
    n = int(e2.size)

    if n <= lags:
        return {
            "lm_stat": float("nan"),
            "p_value": float("nan"),
            "df": float(lags),
            "reject": False,
            "n": n,
        }

    lm = _compute_arch_lm_statistic(e2, lags)
    p_val = _chi2_sf(lm, lags)
    reject = bool(np.isfinite(p_val) and p_val < alpha)

    return {
        "lm_stat": float(lm),
        "p_value": float(p_val),
        "df": float(lags),
        "reject": reject,
        "n": n,
    }


def mcleod_li_test(
    residuals: np.ndarray,
    lags: int = GARCH_LJUNG_BOX_LAGS_DEFAULT,
    alpha: float = LJUNGBOX_SIGNIFICANCE_LEVEL,
) -> dict[str, Any]:
    """McLeod-Li test for ARCH effects.

    Tests for ARCH effects by applying Ljung-Box test to squared residuals.
    This is similar to Ljung-Box on squared residuals but with a different
    interpretation focused on detecting ARCH/GARCH structure.

    Args:
        residuals: Residual series.
        lags: Number of lags to test.
        alpha: Significance level.

    Returns:
        Dict with lags, lb_stat, lb_pvalue, reject (bool), and n.
    """
    res = np.asarray(residuals, dtype=float)
    res = res[np.isfinite(res)]
    res_squared = res**2
    lags_list = _prepare_lags_list(lags)

    if res_squared.size == 0 or not lags_list:
        return {
            "lags": [],
            "lb_stat": [],
            "lb_pvalue": [],
            "reject": False,
            "n": 0,
        }

    lb_stats, lb_pvalues = _compute_ljung_box_stats(res_squared, lags_list)
    reject = bool(lb_pvalues[-1] < alpha) if lb_pvalues else False
    return {
        "lags": lags_list[: len(lb_stats)],
        "lb_stat": lb_stats,
        "lb_pvalue": lb_pvalues,
        "reject": reject,
        "n": int(res_squared.size),
    }


def run_all_tests(
    residuals: np.ndarray,
    *,
    ljung_box_lags: int = GARCH_LJUNG_BOX_LAGS_DEFAULT,
    arch_lm_lags: int = GARCH_LM_LAGS_DEFAULT,
    alpha: float = LJUNGBOX_SIGNIFICANCE_LEVEL,
) -> dict[str, Any]:
    """Run all numerical tests for GARCH structure detection.

    Args:
        residuals: Residual series from SARIMA model.
        ljung_box_lags: Lags for Ljung-Box tests.
        arch_lm_lags: Lags for ARCH-LM test.
        alpha: Significance level.

    Returns:
        Dict containing results from all tests.
    """
    results = {
        "ljung_box_residuals": ljung_box_test(residuals, lags=ljung_box_lags, alpha=alpha),
        "ljung_box_squared": ljung_box_squared_test(residuals, lags=ljung_box_lags, alpha=alpha),
        "engle_arch_lm": engle_arch_lm_test(residuals, lags=arch_lm_lags, alpha=alpha),
        "mcleod_li": mcleod_li_test(residuals, lags=ljung_box_lags, alpha=alpha),
    }
    return results


__all__ = [
    "ljung_box_test",
    "ljung_box_squared_test",
    "engle_arch_lm_test",
    "mcleod_li_test",
    "run_all_tests",
]
