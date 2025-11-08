"""SARIMA model training module."""

from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd

from src.arima.models.sarima_model import FittedSARIMAModel, fit_sarima_model
from src.constants import (
    SARIMA_BEST_MODELS_FILE,
    SARIMA_TRAINED_MODEL_FILE,
    SARIMA_TRAINED_MODEL_METADATA_FILE,
)
from src.utils import get_logger

from .utils import (
    _extract_model_parameters,
    _validate_sarima_parameters,
)

logger = get_logger(__name__)


def load_best_models() -> dict[str, Any]:
    """
    Load best SARIMA models from saved file.

    Returns:
        Dictionary with best models info

    Raises:
        FileNotFoundError: If best models file doesn't exist
        RuntimeError: If file is empty or invalid
    """
    if not SARIMA_BEST_MODELS_FILE.exists():
        msg = f"Best models file not found: {SARIMA_BEST_MODELS_FILE}. " "Run optimization first."
        raise FileNotFoundError(msg)

    try:
        # Use built-in open so tests can patch it reliably
        with open(SARIMA_BEST_MODELS_FILE, "r") as f:
            best_models = json.load(f)
    except Exception as e:
        raise RuntimeError("Failed to load best models: " + str(e)) from e

    if not best_models:
        msg = "Best models file is empty. Run optimization first."
        raise RuntimeError(msg)

    return best_models


def train_sarima_model(
    train_series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 12),
) -> FittedSARIMAModel:
    """
    Train a SARIMA model with specified order and seasonal order.

    Args:
        train_series: Training time series data
        order: SARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, s)

    Returns:
        Fitted SARIMA model (SARIMAXResults)

    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If model training fails
    """
    _validate_sarima_parameters(train_series, order, seasonal_order)

    logger.info(f"Training SARIMA{order}{seasonal_order} model on {len(train_series)} observations")
    try:
        fitted_model = fit_sarima_model(
            train_series, order=order, seasonal_order=seasonal_order, verbose=False
        )
        logger.info(f"Model trained successfully - AIC: {fitted_model.aic:.2f}")
        return fitted_model
    except RuntimeError:
        # Re-raise RuntimeError as-is
        raise
    except Exception as e:
        msg = f"Failed to train SARIMA{order}{seasonal_order} model: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e


def train_best_model(
    train_series: pd.Series,
    prefer: str = "aic",
) -> tuple[FittedSARIMAModel, dict[str, Any]]:
    """
    Train the best SARIMA model based on optimization results.

    Loads best model parameters from saved file and trains the model.

    Args:
        train_series: Training time series data
        prefer: Which criterion to prefer ('aic' or 'bic')

    Returns:
        Tuple of (fitted_model, model_info)

    Raises:
        ValueError: If prefer parameter is invalid or model_info is missing required keys
        FileNotFoundError: If best models file doesn't exist
        RuntimeError: If model training fails
    """
    if prefer.lower() not in ("aic", "bic"):
        msg = f"Invalid prefer parameter: {prefer}. Must be 'aic' or 'bic'"
        raise ValueError(msg)

    if train_series.empty:
        msg = "Training series cannot be empty"
        raise ValueError(msg)

    best_models = load_best_models()

    key = "best_aic" if prefer.lower() == "aic" else "best_bic"
    if key not in best_models:
        msg = f"Best model '{key}' not found in saved results. Run optimization first."
        raise RuntimeError(msg)

    model_info = best_models[key]
    order, seasonal_order = _extract_model_parameters(model_info)

    logger.info(f"Training best {prefer.upper()} model: {model_info.get('params', 'N/A')}")
    fitted_model = train_sarima_model(train_series, order, seasonal_order)

    return fitted_model, model_info


def save_trained_model(fitted_model: FittedSARIMAModel, model_info: dict[str, Any]) -> None:
    """
    Save trained SARIMA model to disk.

    Args:
        fitted_model: Fitted SARIMA model (SARIMAXResults)
        model_info: Dictionary with model information

    Raises:
        ValueError: If fitted_model or model_info is None
        RuntimeError: If saving fails
    """
    if fitted_model is None:
        msg = "fitted_model cannot be None"
        raise ValueError(msg)

    if model_info is None:
        msg = "model_info cannot be None"
        raise ValueError(msg)

    try:
        SARIMA_TRAINED_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted_model, SARIMA_TRAINED_MODEL_FILE)
        logger.info(f"Saved trained model: {SARIMA_TRAINED_MODEL_FILE}")

        # Save model metadata
        # Use built-in open so tests can patch it reliably
        with open(SARIMA_TRAINED_MODEL_METADATA_FILE, "w") as f:
            json.dump(model_info, f, indent=2)
        logger.info(f"Saved model metadata: {SARIMA_TRAINED_MODEL_METADATA_FILE}")
    except Exception as e:
        msg = f"Failed to save trained model: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e


def load_trained_model() -> tuple[FittedSARIMAModel, dict[str, Any]]:
    """
    Load trained SARIMA model from disk.

    Returns:
        Tuple of (fitted_model, model_info)

    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If loading fails
    """
    if not SARIMA_TRAINED_MODEL_FILE.exists():
        msg = (
            f"Trained model file not found: {SARIMA_TRAINED_MODEL_FILE}. " "Train the model first."
        )
        raise FileNotFoundError(msg)

    try:
        fitted_model = joblib.load(SARIMA_TRAINED_MODEL_FILE)
    except Exception as e:
        msg = f"Failed to load trained model from {SARIMA_TRAINED_MODEL_FILE}: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e

    # Load metadata
    model_info = {}
    if SARIMA_TRAINED_MODEL_METADATA_FILE.exists():
        try:
            # Use built-in open so tests can patch it reliably
            with open(SARIMA_TRAINED_MODEL_METADATA_FILE, "r") as f:
                model_info = json.load(f)
        except Exception as e:
            logger.warning(
                f"Failed to load model metadata from {SARIMA_TRAINED_MODEL_METADATA_FILE}: {e}. "
                "Continuing with empty metadata."
            )

    logger.info(f"Loaded trained model: {SARIMA_TRAINED_MODEL_FILE}")
    return fitted_model, model_info
