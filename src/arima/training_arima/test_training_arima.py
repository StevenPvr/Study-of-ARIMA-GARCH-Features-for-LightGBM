"""Unit tests for training_arima module."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.arima.training_arima.training_arima import (
    load_best_models,
    load_trained_model,
    save_trained_model,
    train_sarima_model,
    train_best_model,
)


class TestLoadBestModels:
    """Tests for load_best_models function."""

    @patch("src.arima.training_arima.training_arima.SARIMA_BEST_MODELS_FILE")
    def test_load_best_models_success(self, mock_file: MagicMock) -> None:
        """Test successful loading of best models."""
        mock_file.exists.return_value = True
        best_models = {
            "best_aic": {"params": "SARIMA(1,0,1)(0,0,0)[12]", "p": 1, "d": 0, "q": 1},
            "best_bic": {"params": "SARIMA(1,1,1)(0,0,0)[12]", "p": 1, "d": 1, "q": 1},
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(best_models))):
            result = load_best_models()

        assert isinstance(result, dict)
        assert "best_aic" in result
        assert "best_bic" in result

    @patch("src.arima.training_arima.training_arima.SARIMA_BEST_MODELS_FILE")
    def test_load_best_models_file_not_found(self, mock_file: MagicMock) -> None:
        """Test when best models file doesn't exist."""
        mock_file.exists.return_value = False

        with pytest.raises(FileNotFoundError):
            load_best_models()

    @patch("src.arima.training_arima.training_arima.SARIMA_BEST_MODELS_FILE")
    def test_load_best_models_empty_file(self, mock_file: MagicMock) -> None:
        """Test when best models file is empty."""
        mock_file.exists.return_value = True

        with patch("builtins.open", mock_open(read_data="{}")):
            with pytest.raises(RuntimeError, match="empty"):
                load_best_models()


class TestTrainSarimaModel:
    """Tests for train_sarima_model function."""

    @patch("src.arima.training_arima.training_arima.fit_sarima_model")
    @patch("src.arima.training_arima.training_arima.logger")
    def test_train_sarima_model_success(
        self,
        mock_logger: MagicMock,
        mock_fit_sarima: MagicMock,
    ) -> None:
        """Test successful SARIMA model training."""
        train_series = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 100)
        order = (1, 0, 1)
        seasonal_order = (0, 0, 0, 12)

        mock_fitted = MagicMock()
        mock_fitted.aic = -100.0
        mock_fit_sarima.return_value = mock_fitted

        fitted_model = train_sarima_model(train_series, order, seasonal_order)

        assert fitted_model == mock_fitted
        mock_fit_sarima.assert_called_once_with(
            train_series, order=order, seasonal_order=seasonal_order, verbose=False
        )

    @patch("src.arima.training_arima.training_arima.fit_sarima_model")
    def test_train_sarima_model_failure(self, mock_fit_sarima: MagicMock) -> None:
        """Test SARIMA model training failure."""
        train_series = pd.Series([0.01, -0.02, 0.015] * 10)
        order = (1, 0, 1)
        seasonal_order = (0, 0, 0, 12)

        mock_fit_sarima.side_effect = Exception("Training failed")

        with pytest.raises(RuntimeError, match="Failed to train"):
            train_sarima_model(train_series, order, seasonal_order)


class TestTrainBestModel:
    """Tests for train_best_model function."""

    @patch("src.arima.training_arima.training_arima.train_sarima_model")
    @patch("src.arima.training_arima.training_arima.load_best_models")
    @patch("src.arima.training_arima.training_arima.logger")
    def test_train_best_model_success(
        self,
        mock_logger: MagicMock,
        mock_load: MagicMock,
        mock_train: MagicMock,
    ) -> None:
        """Test successful training of best model."""
        train_series = pd.Series([0.01, -0.02, 0.015] * 100)
        best_models = {
            "best_aic": {
                "params": "SARIMA(1,0,1)(0,0,0)[12]",
                "p": 1,
                "d": 0,
                "q": 1,
                "P": 0,
                "D": 0,
                "Q": 0,
                "s": 12,
                "aic": -100.0,
                "bic": -95.0,
            },
            "best_bic": {
                "params": "SARIMA(1,1,1)(0,0,0)[12]",
                "p": 1,
                "d": 1,
                "q": 1,
                "P": 0,
                "D": 0,
                "Q": 0,
                "s": 12,
                "aic": -98.0,
                "bic": -96.0,
            },
        }

        mock_load.return_value = best_models
        mock_fitted = MagicMock()
        mock_fitted.aic = -100.0
        mock_train.return_value = mock_fitted

        fitted_model, model_info = train_best_model(train_series, prefer="aic")

        assert fitted_model == mock_fitted
        assert model_info == best_models["best_aic"]
        assert model_info["params"] == "SARIMA(1,0,1)(0,0,0)[12]"
        mock_train.assert_called_once_with(train_series, (1, 0, 1), (0, 0, 0, 12))


class TestSaveTrainedModel:
    """Tests for save_trained_model function."""

    @patch("src.arima.training_arima.training_arima.SARIMA_TRAINED_MODEL_FILE")
    @patch("src.arima.training_arima.training_arima.SARIMA_TRAINED_MODEL_METADATA_FILE")
    @patch("src.arima.training_arima.training_arima.joblib.dump")
    @patch("src.arima.training_arima.training_arima.logger")
    def test_save_trained_model_success(
        self,
        mock_logger: MagicMock,
        mock_dump: MagicMock,
        mock_metadata_file: MagicMock,
        mock_model_file: MagicMock,
    ) -> None:
        """Test successful saving of trained model."""
        mock_fitted = MagicMock()
        model_info = {"params": "SARIMA(1,0,1)(0,0,0)[12]", "p": 1, "d": 0, "q": 1}

        mock_model_file.parent.mkdir = MagicMock()

        with patch("builtins.open", mock_open()):
            save_trained_model(mock_fitted, model_info)

        mock_dump.assert_called_once()
        mock_logger.info.assert_called()


class TestLoadTrainedModel:
    """Tests for load_trained_model function."""

    @patch("src.arima.training_arima.training_arima.SARIMA_TRAINED_MODEL_FILE")
    @patch("src.arima.training_arima.training_arima.SARIMA_TRAINED_MODEL_METADATA_FILE")
    @patch("src.arima.training_arima.training_arima.joblib.load")
    @patch("src.arima.training_arima.training_arima.logger")
    def test_load_trained_model_success(
        self,
        mock_logger: MagicMock,
        mock_load: MagicMock,
        mock_metadata_file: MagicMock,
        mock_model_file: MagicMock,
    ) -> None:
        """Test successful loading of trained model."""
        mock_model_file.exists.return_value = True
        mock_fitted = MagicMock()
        mock_load.return_value = mock_fitted

        mock_metadata_file.exists.return_value = True
        model_info = {"params": "SARIMA(1,0,1)(0,0,0)[12]", "p": 1, "d": 0, "q": 1}

        with patch("builtins.open", mock_open(read_data=json.dumps(model_info))):
            fitted_model, info = load_trained_model()

        assert fitted_model == mock_fitted
        assert info == model_info

    @patch("src.arima.training_arima.training_arima.SARIMA_TRAINED_MODEL_FILE")
    def test_load_trained_model_file_not_found(self, mock_file: MagicMock) -> None:
        """Test when trained model file doesn't exist."""
        mock_file.exists.return_value = False

        with pytest.raises(FileNotFoundError):
            load_trained_model()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
