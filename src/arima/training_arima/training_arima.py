"""ARIMA model training module."""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from src.arima.models.arima_model import FittedARIMAModel, fit_arima_model
from src.path import (
    ARIMA_BEST_MODELS_FILE,
    ARIMA_TRAINED_MODEL_FILE,
    ARIMA_TRAINED_MODEL_METADATA_FILE,
)
from src.utils import (
    ensure_output_dir,
    get_logger,
    load_json_data,
    save_json_pretty,
    validate_file_exists,
)

from .utils import extract_model_parameters, validate_arima_parameters

logger = get_logger(__name__)


def _validate_prefer_parameter(prefer: str) -> None:
    """
    Validate the prefer parameter.

    Args:
        prefer: Which criterion to prefer ('aic' or 'bic')

    Raises:
        ValueError: If prefer parameter is invalid
    """
    valid_options = ("aic", "bic")
    if prefer.lower() not in valid_options:
        msg = f"Invalid prefer parameter: {prefer}. Must be one of {valid_options}"
        raise ValueError(msg)


def _get_best_model_key(prefer: str) -> str:
    """
    Get the key for the best model based on prefer parameter.

    Args:
        prefer: Which criterion to prefer ('aic' or 'bic')

    Returns:
        Key string ('best_aic' or 'best_bic')
    """
    return "best_aic" if prefer.lower() == "aic" else "best_bic"


def _get_model_info_from_best_models(best_models: dict[str, Any], prefer: str) -> dict[str, Any]:
    """
    Extract model info from best_models dictionary.

    Args:
        best_models: Dictionary with best models info
        prefer: Which criterion to prefer ('aic' or 'bic')

    Returns:
        Model info dictionary

    Raises:
        RuntimeError: If model key not found in best_models
    """
    key = _get_best_model_key(prefer)
    if key not in best_models:
        msg = f"Best model '{key}' not found in saved results. Run optimization first."
        raise RuntimeError(msg)
    return best_models[key]


def load_best_models() -> dict[str, Any]:
    """
    Load best ARIMA models from saved file.

    Delegates to src.utils.load_json_data() for consistency.

    Returns:
        Dictionary with best models info

    Raises:
        FileNotFoundError: If best models file doesn't exist
        RuntimeError: If file is empty or invalid
    """
    validate_file_exists(ARIMA_BEST_MODELS_FILE, "Best models file")

    best_models = load_json_data(ARIMA_BEST_MODELS_FILE)

    if not best_models:
        msg = "Best models file is empty. Run optimization first."
        raise RuntimeError(msg)

    return best_models


def train_arima_model(
    train_series: pd.Series,
    order: tuple[int, int, int],
) -> FittedARIMAModel:
    """
    Train an ARIMA model with specified order.

    Args:
        train_series: Training time series data
        order: ARIMA order (p, d, q)

    Returns:
        Fitted ARIMA model (SARIMAXResults - seasonal_order=(0,0,0,0))

    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If model training fails
    """
    validate_arima_parameters(train_series, order)

    logger.info(f"Training ARIMA{order} model on {len(train_series)} observations")
    try:
        fitted_model = fit_arima_model(train_series, order=order, verbose=False)
        logger.info(f"Model trained successfully - AIC: {fitted_model.aic:.2f}")
        return fitted_model
    except RuntimeError:
        # Re-raise RuntimeError as-is
        raise
    except Exception as e:
        msg = f"Failed to train ARIMA{order} model: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e


def train_best_model(
    train_series: pd.Series,
    prefer: str = "aic",
) -> tuple[FittedARIMAModel, dict[str, Any]]:
    """
    Train the best ARIMA model based on optimization results.

    Loads best model parameters from saved file and trains the model.
    Only uses training data to prevent look-ahead bias.

    Args:
        train_series: Training time series data (must be training set only)
        prefer: Which criterion to prefer ('aic' or 'bic')

    Returns:
        Tuple of (fitted_model, model_info)

    Raises:
        ValueError: If prefer parameter is invalid or model_info is missing required keys
        FileNotFoundError: If best models file doesn't exist
        RuntimeError: If model training fails
    """
    _validate_prefer_parameter(prefer)

    if train_series.empty:
        msg = "Training series cannot be empty"
        raise ValueError(msg)

    best_models = load_best_models()
    model_info = _get_model_info_from_best_models(best_models, prefer)
    order = extract_model_parameters(model_info)

    logger.info(f"Training best {prefer.upper()} model: {model_info.get('params', 'N/A')}")
    fitted_model = train_arima_model(train_series, order)

    # Add order to model_info for easy retrieval in evaluation
    model_info["order"] = order

    return fitted_model, model_info


def _save_model_file(fitted_model: FittedARIMAModel) -> None:
    """
    Save fitted model to disk.

    Args:
        fitted_model: Fitted ARIMA model (SARIMAXResults)

    Raises:
        RuntimeError: If saving fails
    """
    ensure_output_dir(ARIMA_TRAINED_MODEL_FILE)
    joblib.dump(fitted_model, ARIMA_TRAINED_MODEL_FILE)
    logger.info(f"Saved trained model: {ARIMA_TRAINED_MODEL_FILE}")


def _save_model_metadata(model_info: dict[str, Any]) -> None:
    """
    Save model metadata to disk.

    Delegates to src.utils.save_json_pretty() for consistency.

    Args:
        model_info: Dictionary with model information

    Raises:
        RuntimeError: If saving fails
    """
    save_json_pretty(model_info, ARIMA_TRAINED_MODEL_METADATA_FILE)
    logger.info(f"Saved model metadata: {ARIMA_TRAINED_MODEL_METADATA_FILE}")


def save_trained_model(fitted_model: FittedARIMAModel, model_info: dict[str, Any] | None) -> None:
    """
    Save trained ARIMA model to disk.

    Args:
        fitted_model: Fitted ARIMA model (SARIMAXResults)
        model_info: Dictionary with model information. Must not be None.

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
        _save_model_file(fitted_model)
        _save_model_metadata(model_info)
    except Exception as e:
        msg = f"Failed to save trained model: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e


def load_trained_model() -> tuple[FittedARIMAModel, dict[str, Any]]:
    """
    Load trained ARIMA model from disk.

    Returns:
        Tuple of (fitted_model, model_info)

    Raises:
        FileNotFoundError: If model file or metadata file doesn't exist
        RuntimeError: If loading fails
    """
    validate_file_exists(ARIMA_TRAINED_MODEL_FILE, "Trained model file")
    validate_file_exists(ARIMA_TRAINED_MODEL_METADATA_FILE, "Model metadata file")

    try:
        fitted_model = joblib.load(ARIMA_TRAINED_MODEL_FILE)
        logger.info(f"Loaded trained model: {ARIMA_TRAINED_MODEL_FILE}")

        model_info = load_json_data(ARIMA_TRAINED_MODEL_METADATA_FILE)
        logger.info(f"Loaded model metadata: {ARIMA_TRAINED_MODEL_METADATA_FILE}")

        return fitted_model, model_info
    except Exception as e:
        msg = f"Failed to load trained model: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e
