"""I/O and loading utilities for GARCH diagnostics.

Merges JSON I/O helpers and estimation/data loading to simplify the module
structure while keeping explicit validation and error handling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.constants import GARCH_DATASET_FILE, GARCH_ESTIMATION_FILE
from src.garch.structure_garch.utils import load_garch_dataset, prepare_residuals
from src.utils import get_logger, save_json_pretty

logger = get_logger(__name__)


# =============================
# JSON save helpers
# =============================


def save_diagnostics_json(
    data: dict[str, Any],
    output_file: Path,
    description: str = "diagnostics",
) -> None:
    """Save diagnostics dictionary to JSON file using project utility."""
    save_json_pretty(data, output_file)
    logger.info("Saved %s to: %s", description, output_file)


def validate_dict_field(data: dict[str, Any], field_name: str) -> None:
    """Validate that a field exists and is a dict."""
    if field_name not in data:
        raise KeyError(f"Missing required field: {field_name}")
    if not isinstance(data[field_name], dict):
        raise TypeError(f"{field_name} must be a dict, got {type(data[field_name]).__name__}")


# =============================
# Estimation parameters helpers
# =============================


def check_converged_params(params: dict | None) -> bool:
    """Return True if params indicate convergence."""
    return isinstance(params, dict) and params.get("converged", False)


def try_new_format_params(est_payload: dict) -> tuple[str | None, dict | None]:
    """Extract converged params from new format keys in priority order."""
    for key in ("egarch_skewt", "egarch_student", "egarch_normal"):
        candidate = est_payload.get(key)
        if check_converged_params(candidate):
            dist = (
                "skewt"
                if key == "egarch_skewt"
                else ("student" if key == "egarch_student" else "normal")
            )
            return dist, candidate
    return None, None


def try_legacy_format_params(est_payload: dict) -> tuple[str | None, dict | None]:
    """Extract converged params from legacy format keys."""
    for key, dist in (("student", "student"), ("normal", "normal")):
        candidate = est_payload.get(key)
        if check_converged_params(candidate):
            return dist, candidate
    return None, None


def choose_best_params(est_payload: dict) -> tuple[str, dict]:
    """Choose best converged EGARCH parameters from estimation payload.

    Raises ValueError if none found.
    """
    dist, params = try_new_format_params(est_payload)
    if params is not None:
        return dist, params  # type: ignore[return-value]
    dist, params = try_legacy_format_params(est_payload)
    if params is not None:
        return dist, params  # type: ignore[return-value]
    raise ValueError("No converged EGARCH model found in estimation payload")


def load_estimation_file() -> dict:
    """Load estimation JSON or raise explicit error."""
    try:
        with open(GARCH_ESTIMATION_FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Estimation file not found: %s", GARCH_ESTIMATION_FILE)
        raise
    except json.JSONDecodeError as e:  # pragma: no cover - explicit error path
        logger.error("Invalid JSON in estimation file: %s", e)
        raise ValueError(f"Invalid JSON in {GARCH_ESTIMATION_FILE}") from e


def extract_params_dict(best: dict) -> dict:
    """Extract parameters dict from estimation results (new or legacy)."""
    params = best.get("params")
    return params if isinstance(params, dict) else best


def extract_nu_from_params(best: dict) -> float | None:
    """Extract nu parameter from estimation results (new or legacy)."""
    params = best.get("params", {})
    if isinstance(params, dict) and params.get("nu") is not None:
        return float(params["nu"])  # type: ignore[index]
    nu_value = best.get("nu")
    return float(nu_value) if nu_value is not None else None


# =============================
# Data loading helpers
# =============================


def load_and_prepare_residuals() -> np.ndarray:
    """Load dataset and return finite test residuals or raise error."""
    data_frame = load_garch_dataset(str(GARCH_DATASET_FILE))
    resid_test = prepare_residuals(data_frame, use_test_only=True)
    resid_test = resid_test[np.isfinite(resid_test)]
    if resid_test.size == 0:
        logger.error("No valid residuals found in test set")
        raise ValueError("No valid residuals found in test set")
    return resid_test


def load_data_and_params() -> tuple[np.ndarray, str, dict, float | None]:
    """Load residuals and best EGARCH params; return (residuals, dist, params, nu)."""
    est = load_estimation_file()
    dist, best = choose_best_params(est)
    params_dict = extract_params_dict(best)
    nu = extract_nu_from_params(best)
    resid_test = load_and_prepare_residuals()
    return resid_test, dist, params_dict, nu


__all__ = [
    # JSON helpers
    "save_diagnostics_json",
    "validate_dict_field",
    # Estimation helpers
    "check_converged_params",
    "try_new_format_params",
    "try_legacy_format_params",
    "choose_best_params",
    "load_estimation_file",
    "extract_params_dict",
    "extract_nu_from_params",
    # Data loading
    "load_and_prepare_residuals",
    "load_data_and_params",
]
