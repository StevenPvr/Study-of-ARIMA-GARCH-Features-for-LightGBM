"""Stationarity checks for time series (ADF + KPSS).

This module provides small, focused helpers to:
- run ADF and KPSS on a pandas Series
- combine results into a single verdict
- load the project's weighted returns and persist a JSON report

All functions are short, typed, and log meaningful progress.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from src.constants import STATIONARITY_REPORT_FILE, STATIONARITY_RESAMPLE_FREQ
from src.utils import get_logger

from .utils import _load_csv_series, _validate_series

logger = get_logger(__name__)


class TestResult(TypedDict):
    statistic: float
    p_value: float
    lags: int | None
    nobs: int | None
    critical_values: dict[str, float] | None


@dataclass(frozen=True)
class StationarityReport:
    stationary: bool
    alpha: float
    adf: TestResult
    kpss: TestResult


def adf_test(series: pd.Series, *, autolag: str = "AIC") -> TestResult:
    """Run Augmented Dickey–Fuller test.

    The number of lags is automatically selected based on the specified criterion
    (default: AIC). The lag value in the result reflects the optimal number chosen
    by the algorithm for the given series, not a fixed value.

    Args:
        series: Input time series.
        autolag: Criterion for lag selection ("AIC", "BIC", "t-stat", or None).
                 Default "AIC" minimizes Akaike Information Criterion.

    Returns:
        TestResult with statistic, p-value, lags (auto-selected), nobs, and critical values.
    """
    s = _validate_series(series)
    result = adfuller(s, autolag=autolag)
    if len(result) < 5:
        raise RuntimeError(
            f"Unexpected ADF result: expected at least 5 components, got {len(result)}"
        )
    stat = result[0]
    pval = result[1]
    lags = result[2]
    nobs = result[3]
    crit = cast(dict[str, float], result[4])
    return {
        "statistic": float(stat),
        "p_value": float(pval),
        "lags": int(lags) if lags is not None else None,
        "nobs": int(nobs) if nobs is not None else None,
        "critical_values": {k: float(v) for k, v in crit.items()},
    }


def kpss_test(series: pd.Series, *, regression: Literal["c", "ct"] = "c") -> TestResult:
    """Run KPSS test for (trend-)stationarity.

    The number of lags is automatically calculated based on the series length
    using Newey-West bandwidth selection. The lag value in the result reflects
    the optimal number chosen for the given series, not a fixed value.

    Args:
        series: Input time series.
        regression: 'c' (level) or 'ct' (trend).

    Returns:
        TestResult with statistic, p-value, lags (auto-calculated), and critical values.
    """
    s = _validate_series(series)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InterpolationWarning)
        stat, pval, lags, crit = kpss(s, regression=regression, nlags="auto")
    return {
        "statistic": float(stat),
        "p_value": float(pval),
        "lags": int(lags) if lags is not None else None,
        "nobs": int(s.size),
        "critical_values": {k: float(v) for k, v in crit.items()},
    }


def evaluate_stationarity(series: pd.Series, *, alpha: float = 0.05) -> StationarityReport:
    """Combine ADF and KPSS into a single verdict.

    Rule of thumb:
    - ADF p < alpha (reject unit root) and KPSS p > alpha (do not reject stationarity)
      ⇒ stationary = True; otherwise False.

    Args:
        series: Input time series.
        alpha: Significance level (must be between 0 and 1).

    Returns:
        StationarityReport with combined verdict and test results.

    Raises:
        ValueError: If alpha is not in (0, 1).
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    s = _validate_series(series)
    adf_res = adf_test(s)
    # Strict behavior: no fallback. Any failure in KPSS raises and stops the pipeline.
    kpss_res = kpss_test(s)
    adf_rejects_unit_root = adf_res["p_value"] < alpha
    kpss_accepts_stationarity = np.isnan(kpss_res["p_value"]) or kpss_res["p_value"] > alpha
    stationary = bool(adf_rejects_unit_root and kpss_accepts_stationarity)
    return StationarityReport(stationary=stationary, alpha=float(alpha), adf=adf_res, kpss=kpss_res)


def run_stationarity_pipeline(
    *,
    data_file: str,
    column: str = "weighted_log_return",
    alpha: float = 0.05,
) -> StationarityReport:
    """Load series, run tests, return structured report."""
    logger.info("Running stationarity checks (ADF + KPSS) on %s::%s", data_file, column)
    series = _load_csv_series(data_file=data_file, column=column)
    # Weekly stationarity: aggregate log-returns by calendar week using sum
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    weekly_series = series.resample(STATIONARITY_RESAMPLE_FREQ).sum(min_count=1).dropna()
    report = evaluate_stationarity(weekly_series, alpha=alpha)
    logger.info("Stationary=%s (alpha=%.3f)", report.stationary, report.alpha)
    return report


def save_stationarity_report(report: StationarityReport, out_path: Path | None = None) -> Path:
    """Persist report as JSON to the configured path."""
    target = Path(out_path) if out_path is not None else STATIONARITY_REPORT_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as f:
        json.dump(asdict(report), f, indent=2)
    logger.info("Saved stationarity report: %s", target)
    return target
