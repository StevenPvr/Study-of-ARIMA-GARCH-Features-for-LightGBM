"""CLI for post-estimation diagnostics of GARCH models.

Implements methodology for verifying GARCH model adequacy:
1. Verify standardized residuals εt/σt behave as centered white noise
2. Verify squared standardized residuals show no significant autocorrelation
   (ACF/PACF plots + Ljung-Box tests)
3. Verify distribution adequacy for zt (Normal or Student-t)
   (graphical diagnostics + normality tests)

Results saved to results/garch/diagnostic/
Plots saved to plots/garch/diagnostics/
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

from src.constants import (
    GARCH_ACF_LAGS_DEFAULT,
    GARCH_DATASET_FILE,
    GARCH_DIAGNOSTICS_DIR,
    GARCH_DIAGNOSTICS_PLOTS_DIR,
    GARCH_DISTRIBUTION_DIAGNOSTICS_FILE,
    GARCH_ESTIMATION_FILE,
    GARCH_LJUNG_BOX_LAGS_DEFAULT,
    GARCH_LJUNGBOX_FILE,
    GARCH_STD_ACF_PACF_PLOT,
    GARCH_STD_QQ_PLOT,
    GARCH_STD_SQUARED_ACF_PACF_PLOT,
)
from src.garch.garch_diagnostic.diagnostics import (
    compute_distribution_diagnostics,
    compute_ljung_box_on_std,
    compute_ljung_box_on_std_squared,
    save_acf_pacf_std_plots,
    save_acf_pacf_std_squared_plots,
    save_qq_plot_std_residuals,
)
from src.garch.structure_garch.detection import load_garch_dataset, prepare_residuals
from src.utils import get_logger

logger = get_logger(__name__)


def _check_converged_params(params: dict | None) -> bool:
    """Check if parameters dictionary indicates convergence."""
    return isinstance(params, dict) and params.get("converged", False)


def _try_new_format_params(est_payload: dict) -> tuple[str | None, dict | None]:
    """Try to extract parameters from new format keys.

    Checks in preference order: egarch_skewt → egarch_student → egarch_normal.
    """
    egarch_skewt = est_payload.get("egarch_skewt")
    if _check_converged_params(egarch_skewt):
        return "skewt", egarch_skewt
    egarch_student = est_payload.get("egarch_student")
    if _check_converged_params(egarch_student):
        return "student", egarch_student
    egarch_normal = est_payload.get("egarch_normal")
    if _check_converged_params(egarch_normal):
        return "normal", egarch_normal
    return None, None


def _try_legacy_format_params(est_payload: dict) -> tuple[str | None, dict | None]:
    """Try to extract parameters from legacy format (student, normal)."""
    student = est_payload.get("student")
    if _check_converged_params(student):
        return "student", student
    normal = est_payload.get("normal")
    if _check_converged_params(normal):
        return "normal", normal
    return None, None


def _choose_best_params(est_payload: dict) -> tuple[str | None, dict | None]:
    """Choose best converged EGARCH parameters from estimation payload."""
    dist, params = _try_new_format_params(est_payload)
    if params is not None:
        return dist, params
    return _try_legacy_format_params(est_payload)


def _load_estimation_file() -> dict:
    """Load estimation JSON file.

    Raises:
        FileNotFoundError: If file is missing.
        ValueError: If JSON is invalid.
    """
    try:
        with open(GARCH_ESTIMATION_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Estimation file not found: %s", GARCH_ESTIMATION_FILE)
        raise
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in estimation file: %s", e)
        raise ValueError(f"Invalid JSON in {GARCH_ESTIMATION_FILE}") from e


def _extract_nu_from_params(best: dict) -> float | None:
    """Extract nu parameter from best params dictionary."""
    nu_value = best.get("nu")
    return float(nu_value) if nu_value is not None else None  # type: ignore[arg-type]


def _load_and_prepare_residuals() -> np.ndarray:
    """Load dataset and prepare test residuals.

    Raises:
        ValueError: If no valid residuals found.
    """
    data_frame = load_garch_dataset(str(GARCH_DATASET_FILE))
    resid_test = prepare_residuals(data_frame, use_test_only=True)
    resid_test = resid_test[np.isfinite(resid_test)]
    if resid_test.size == 0:
        logger.error("No valid residuals found in test set")
        raise ValueError("No valid residuals found in test set")
    return resid_test


def _load_data_and_params() -> tuple[np.ndarray, str | None, dict, float | None] | None:
    """Load dataset, residuals, and best EGARCH parameters.

    Returns:
        Tuple of (residuals, distribution, params, nu) or None if no converged model found.

    Raises:
        FileNotFoundError: If required files are missing.
        ValueError: If data loading fails.
    """
    est = _load_estimation_file()
    dist, best = _choose_best_params(est)
    if best is None:
        logger.warning("No converged EGARCH model found in %s", GARCH_ESTIMATION_FILE)
        return None

    nu = _extract_nu_from_params(best)
    try:
        resid_test = _load_and_prepare_residuals()
        return resid_test, dist, best, nu
    except Exception as e:
        logger.error("Failed to load dataset or prepare residuals: %s", e)
        raise


def _generate_acf_pacf_plots(
    resid_test: np.ndarray,
    best_params: dict,
    dist: str | None,
    nu: float | None,
) -> None:
    """Generate and save ACF/PACF plots for standardized residuals.

    Verifies:
    1. Standardized residuals (z) behave as centered white noise
    2. Squared standardized residuals (z²) show no significant autocorrelation

    Args:
        resid_test: Test residuals array εt.
        best_params: Best GARCH parameters dictionary.
        dist: Distribution name (normal or student).
        nu: Degrees of freedom for Student-t distribution (if applicable).
    """
    try:
        save_acf_pacf_std_squared_plots(
            resid_test,
            best_params,
            lags=GARCH_ACF_LAGS_DEFAULT,
            outdir=GARCH_DIAGNOSTICS_PLOTS_DIR,
            filename=GARCH_STD_SQUARED_ACF_PACF_PLOT.name,
            dist=(dist or "normal"),
            nu=nu,
        )
        save_acf_pacf_std_plots(
            resid_test,
            best_params,
            lags=GARCH_ACF_LAGS_DEFAULT,
            outdir=GARCH_DIAGNOSTICS_PLOTS_DIR,
            filename=GARCH_STD_ACF_PACF_PLOT.name,
            dist=(dist or "normal"),
            nu=nu,
        )
        logger.info("Saved ACF/PACF plots to %s", GARCH_DIAGNOSTICS_PLOTS_DIR)
    except Exception as ex:
        logger.warning("ACF/PACF plotting failed: %s", ex)


def _run_ljung_box_tests(
    resid_test: np.ndarray,
    best_params: dict,
    dist: str | None,
    nu: float | None,
) -> None:
    """Run Ljung-Box tests on standardized and squared standardized residuals.

    Tests for white noise behavior:
    - z should show no autocorrelation (white noise)
    - z² should show no autocorrelation (volatility correctly captured)

    Results saved to results/garch/diagnostic/

    Args:
        resid_test: Test residuals array εt.
        best_params: Best GARCH parameters dictionary.
        dist: Distribution name (normal or student).
        nu: Degrees of freedom for Student-t distribution (if applicable).
    """
    try:
        lb2 = compute_ljung_box_on_std_squared(
            resid_test,
            best_params,
            lags=GARCH_LJUNG_BOX_LAGS_DEFAULT,
            dist=(dist or "normal"),
            nu=nu,
        )
        GARCH_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
        with Path(GARCH_LJUNGBOX_FILE).open("w") as f:
            json.dump(lb2, f, indent=2)
        logger.info("Saved Ljung-Box(z^2) to: %s", GARCH_LJUNGBOX_FILE)
    except Exception as ex:
        logger.warning("Ljung-Box(z^2) failed: %s", ex)
    try:
        compute_ljung_box_on_std(
            resid_test,
            best_params,
            lags=GARCH_LJUNG_BOX_LAGS_DEFAULT,
            dist=(dist or "normal"),
            nu=nu,
        )
    except Exception as ex:
        logger.warning("Ljung-Box(z) failed: %s", ex)


def _run_distribution_diagnostics(
    resid_test: np.ndarray,
    best_params: dict,
    dist: str | None,
) -> None:
    """Run distribution diagnostics and generate QQ plot.

    Verifies adequacy of chosen distribution (Normal or Student-t) for zt:
    - Graphical diagnostics: QQ plot
    - Numerical tests: Jarque-Bera, Kolmogorov-Smirnov

    Results saved to results/garch/diagnostic/
    Plots saved to plots/garch/diagnostics/

    Args:
        resid_test: Test residuals array εt.
        best_params: Best GARCH parameters dictionary.
        dist: Distribution name (normal or student).
    """
    try:
        nu = float(best_params.get("nu")) if best_params.get("nu") is not None else None  # type: ignore[arg-type]
        diag = compute_distribution_diagnostics(
            resid_test, best_params, dist=(dist or "normal"), nu=nu
        )
        save_qq_plot_std_residuals(
            resid_test,
            best_params,
            dist=(dist or "normal"),
            nu=nu,
            outdir=GARCH_DIAGNOSTICS_PLOTS_DIR,
            filename=GARCH_STD_QQ_PLOT.name,
        )
        GARCH_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
        with Path(GARCH_DISTRIBUTION_DIAGNOSTICS_FILE).open("w") as f:
            json.dump(diag, f, indent=2)
        logger.info("Saved distribution diagnostics to: %s", GARCH_DISTRIBUTION_DIAGNOSTICS_FILE)
        logger.info("Distribution diagnostics: %s", diag)
    except Exception as ex:
        logger.warning("Distribution diagnostics failed: %s", ex)


def main() -> None:
    """Run post-estimation GARCH diagnostics.

    Methodology:
    1. Verify standardized residuals εt/σt behave as centered white noise
    2. Verify squared standardized residuals show no significant autocorrelation
       (ACF/PACF plots + Ljung-Box tests)
    3. Verify distribution adequacy for zt (Normal or Student-t)
       (graphical diagnostics + normality tests)

    Results saved to results/garch/diagnostic/
    Plots saved to plots/garch/diagnostics/
    """
    logger.info("=" * 60)
    logger.info("GARCH DIAGNOSTICS (post-estimation)")
    logger.info("=" * 60)

    result = _load_data_and_params()
    if result is None:
        return
    resid_test, dist, best, nu = result

    _generate_acf_pacf_plots(resid_test, best, dist, nu)
    _run_ljung_box_tests(resid_test, best, dist, nu)
    _run_distribution_diagnostics(resid_test, best, dist)


if __name__ == "__main__":
    main()
