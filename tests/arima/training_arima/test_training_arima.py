"""Unit tests for training_arima module."""

from __future__ import annotations

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
    train_arima_model,
    train_best_model,
)
from src.arima.training_arima.utils import extract_model_parameters, validate_arima_parameters


class TestLoadBestModels:
    """Tests for load_best_models function."""

    @patch("src.arima.training_arima.training_arima.validate_file_exists")
    @patch("src.arima.training_arima.training_arima.load_json_data")
    def test_load_best_models_success(
        self, mock_load_json: MagicMock, mock_validate: MagicMock
    ) -> None:
        """Test successful loading of best models."""
        best_models = {
            "best_aic": {"params": "ARIMA(1,0,1)", "p": 1, "d": 0, "q": 1},
            "best_bic": {"params": "ARIMA(1,1,1)", "p": 1, "d": 1, "q": 1},
        }
        mock_load_json.return_value = best_models

        result = load_best_models()

        assert isinstance(result, dict)
        assert "best_aic" in result
        assert "best_bic" in result
        mock_load_json.assert_called_once()

    @patch("src.arima.training_arima.training_arima.load_json_data")
    def test_load_best_models_file_not_found(self, mock_load_json: MagicMock) -> None:
        """Test when best models file doesn't exist."""
        mock_load_json.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            load_best_models()

    @patch("src.arima.training_arima.training_arima.validate_file_exists")
    @patch("src.arima.training_arima.training_arima.load_json_data")
    def test_load_best_models_empty_file(
        self, mock_load_json: MagicMock, mock_validate: MagicMock
    ) -> None:
        """Test when best models file is empty."""
        mock_load_json.return_value = {}

        with pytest.raises(RuntimeError, match="empty"):
            load_best_models()


class TestTrainArimaModel:
    """Tests for train_arima_model function."""

    @patch("src.arima.training_arima.training_arima.fit_arima_model")
    @patch("src.arima.training_arima.training_arima.logger")
    def test_train_arima_model_success(
        self,
        mock_logger: MagicMock,
        mock_fit_arima: MagicMock,
    ) -> None:
        """Test successful ARIMA model training."""
        train_series = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 100)
        order = (1, 0, 1)

        mock_fitted = MagicMock()
        mock_fitted.aic = -100.0
        mock_fit_arima.return_value = mock_fitted

        fitted_model = train_arima_model(train_series, order)

        assert fitted_model == mock_fitted
        mock_fit_arima.assert_called_once_with(train_series, order=order, verbose=False)

    @patch("src.arima.training_arima.training_arima.fit_arima_model")
    def test_train_arima_model_failure(self, mock_fit_arima: MagicMock) -> None:
        """Test ARIMA model training failure."""
        train_series = pd.Series([0.01, -0.02, 0.015] * 10)
        order = (1, 0, 1)

        mock_fit_arima.side_effect = Exception("Training failed")

        with pytest.raises(RuntimeError, match="Failed to train"):
            train_arima_model(train_series, order)

    @patch("src.arima.training_arima.training_arima.fit_arima_model")
    def test_train_arima_model_runtime_error(self, mock_fit_arima: MagicMock) -> None:
        """Test ARIMA model training with RuntimeError."""
        train_series = pd.Series([0.01, -0.02, 0.015] * 10)
        order = (1, 0, 1)

        mock_fit_arima.side_effect = RuntimeError("Model convergence failed")

        with pytest.raises(RuntimeError, match="Model convergence failed"):
            train_arima_model(train_series, order)


class TestTrainBestModel:
    """Tests for train_best_model function."""

    @patch("src.arima.training_arima.training_arima.train_arima_model")
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
                "params": "ARIMA(1,0,1)",
                "p": 1,
                "d": 0,
                "q": 1,
                "aic": -100.0,
                "bic": -95.0,
            },
            "best_bic": {
                "params": "ARIMA(1,1,1)",
                "p": 1,
                "d": 1,
                "q": 1,
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
        assert model_info["params"] == "ARIMA(1,0,1)"
        mock_train.assert_called_once_with(train_series, (1, 0, 1))

    @patch("src.arima.training_arima.training_arima.train_arima_model")
    @patch("src.arima.training_arima.training_arima.load_best_models")
    @patch("src.arima.training_arima.training_arima.logger")
    def test_train_best_model_success_bic(
        self,
        mock_logger: MagicMock,
        mock_load: MagicMock,
        mock_train: MagicMock,
    ) -> None:
        """Test successful training of best BIC model."""
        train_series = pd.Series([0.01, -0.02, 0.015] * 100)
        best_models = {
            "best_aic": {
                "params": "ARIMA(1,0,1)(0,0,0)[12]",
                "p": 1,
                "d": 0,
                "q": 1,
                "P": 0,
                "D": 0,
                "Q": 0,
                "s": 12,
            },
            "best_bic": {
                "params": "ARIMA(1,1,1)",
                "p": 1,
                "d": 1,
                "q": 1,
                "P": 0,
                "D": 0,
                "Q": 0,
                "s": 12,
            },
        }

        mock_load.return_value = best_models
        mock_fitted = MagicMock()
        mock_fitted.aic = -98.0
        mock_train.return_value = mock_fitted

        fitted_model, model_info = train_best_model(train_series, prefer="bic")

        assert fitted_model == mock_fitted
        assert model_info == best_models["best_bic"]
        assert model_info["params"] == "ARIMA(1,1,1)"
        mock_train.assert_called_once_with(train_series, (1, 1, 1))

    @patch("src.arima.training_arima.training_arima.load_best_models")
    def test_train_best_model_invalid_prefer(self, mock_load: MagicMock) -> None:
        """Test train_best_model with invalid prefer parameter."""
        train_series = pd.Series([0.01, -0.02, 0.015] * 100)

        with pytest.raises(ValueError, match="Invalid prefer parameter"):
            train_best_model(train_series, prefer="invalid")

    @patch("src.arima.training_arima.training_arima.load_best_models")
    def test_train_best_model_empty_series(self, mock_load: MagicMock) -> None:
        """Test train_best_model with empty training series."""
        train_series = pd.Series([], dtype=float)

        with pytest.raises(ValueError, match="Training series cannot be empty"):
            train_best_model(train_series, prefer="aic")

    @patch("src.arima.training_arima.training_arima.load_best_models")
    def test_train_best_model_missing_key(self, mock_load: MagicMock) -> None:
        """Test train_best_model when best model key is missing."""
        train_series = pd.Series([0.01, -0.02, 0.015] * 100)
        best_models = {
            "best_aic": {
                "params": "ARIMA(1,0,1)(0,0,0)[12]",
                "p": 1,
                "d": 0,
                "q": 1,
                "P": 0,
                "D": 0,
                "Q": 0,
                "s": 12,
            },
        }

        mock_load.return_value = best_models

        with pytest.raises(RuntimeError, match="Best model 'best_bic' not found"):
            train_best_model(train_series, prefer="bic")


class TestSaveTrainedModel:
    """Tests for save_trained_model function."""

    @patch("src.arima.training_arima.training_arima.ARIMA_TRAINED_MODEL_FILE")
    @patch("src.arima.training_arima.training_arima.ARIMA_TRAINED_MODEL_METADATA_FILE")
    @patch("src.arima.training_arima.training_arima.joblib.dump")
    @patch("src.arima.training_arima.training_arima.logger")
    def test_save_trained_model_success(
        self,
        mock_logger: MagicMock,
        mock_dump: MagicMock,
        _mock_metadata_file: MagicMock,
        mock_model_file: MagicMock,
    ) -> None:
        """Test successful saving of trained model."""
        mock_fitted = MagicMock()
        model_info = {"params": "ARIMA(1,0,1)(0,0,0)[12]", "p": 1, "d": 0, "q": 1}

        mock_model_file.parent.mkdir = MagicMock()

        with patch("builtins.open", mock_open()):
            save_trained_model(mock_fitted, model_info)

        mock_dump.assert_called_once()
        mock_logger.info.assert_called()

    def test_save_trained_model_fitted_model_none(self) -> None:
        """Test save_trained_model when fitted_model is None."""
        model_info = {"params": "ARIMA(1,0,1)(0,0,0)[12]", "p": 1, "d": 0, "q": 1}

        with pytest.raises(ValueError, match="fitted_model cannot be None"):
            save_trained_model(None, model_info)

    def test_save_trained_model_model_info_none(self) -> None:
        """Test save_trained_model when model_info is None."""
        mock_fitted = MagicMock()

        with pytest.raises(ValueError, match="model_info cannot be None"):
            save_trained_model(mock_fitted, None)

    @patch("src.arima.training_arima.training_arima.ARIMA_TRAINED_MODEL_FILE")
    @patch("src.arima.training_arima.training_arima.joblib.dump")
    def test_save_trained_model_save_error(
        self, mock_dump: MagicMock, mock_model_file: MagicMock
    ) -> None:
        """Test save_trained_model when saving fails."""
        mock_fitted = MagicMock()
        model_info = {"params": "ARIMA(1,0,1)(0,0,0)[12]", "p": 1, "d": 0, "q": 1}

        mock_model_file.parent.mkdir = MagicMock()
        mock_dump.side_effect = Exception("Save failed")

        with pytest.raises(RuntimeError, match="Failed to save trained model"):
            save_trained_model(mock_fitted, model_info)


class TestValidateArimaParameters:
    """Tests for validate_arima_parameters function."""

    def test_validate_arima_parameters_success(self) -> None:
        """Test successful validation of ARIMA parameters."""
        train_series = pd.Series([0.01, -0.02, 0.015] * 100)
        order = (1, 0, 1)

        # Should not raise
        validate_arima_parameters(train_series, order)

    def test_validate_arima_parameters_empty_series(self) -> None:
        """Test validation with empty series."""
        train_series = pd.Series([], dtype=float)
        order = (1, 0, 1)
        with pytest.raises(ValueError, match="Training series cannot be empty"):
            validate_arima_parameters(train_series, order)

    def test_validate_arima_parameters_invalid_order_length(self) -> None:
        """Test validation with invalid order length."""
        train_series = pd.Series([0.01, -0.02, 0.015] * 100)
        order = (1, 0)  # Invalid: should be 3 values

        with pytest.raises(ValueError, match="Invalid order"):
            validate_arima_parameters(train_series, order)  # type: ignore[arg-type]

    def test_validate_arima_parameters_invalid_order_negative(self) -> None:
        """Test validation with negative order values."""
        train_series = pd.Series([0.01, -0.02, 0.015] * 100)
        order = (1, -1, 1)  # Invalid: negative value

        with pytest.raises(ValueError, match="Invalid order"):
            validate_arima_parameters(train_series, order)


class TestLoadTrainedModel:
    """Tests for load_trained_model function."""

    @patch("src.arima.training_arima.training_arima.load_json_data")
    @patch("src.arima.training_arima.training_arima.joblib.load")
    @patch("src.arima.training_arima.training_arima.validate_file_exists")
    def test_load_trained_model_success(
        self,
        mock_validate: MagicMock,
        mock_joblib_load: MagicMock,
        mock_load_json: MagicMock,
    ) -> None:
        """Test successful loading of trained model."""
        mock_fitted = MagicMock()
        mock_fitted.aic = -100.0
        model_info = {"params": "ARIMA(1,0,1)(0,0,0)[12]", "p": 1, "d": 0, "q": 1}

        mock_joblib_load.return_value = mock_fitted
        mock_load_json.return_value = model_info

        fitted_model, loaded_info = load_trained_model()

        assert fitted_model == mock_fitted
        assert loaded_info == model_info
        assert mock_validate.call_count == 2  # Called for both model file and metadata file

    @patch("src.arima.training_arima.training_arima.validate_file_exists")
    def test_load_trained_model_file_not_found(self, mock_validate: MagicMock) -> None:
        """Test load_trained_model when file doesn't exist."""
        mock_validate.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            load_trained_model()

    @patch("src.arima.training_arima.training_arima.load_json_data")
    @patch("src.arima.training_arima.training_arima.joblib.load")
    @patch("src.arima.training_arima.training_arima.validate_file_exists")
    def test_load_trained_model_loading_error(
        self,
        mock_validate: MagicMock,
        mock_joblib_load: MagicMock,
        mock_load_json: MagicMock,
    ) -> None:
        """Test load_trained_model when loading fails."""
        mock_joblib_load.side_effect = Exception("Loading failed")

        with pytest.raises(RuntimeError, match="Failed to load trained model"):
            load_trained_model()


class TestExtractModelParameters:
    """Tests for extract_model_parameters function."""

    def test_extract_model_parameters_success(self) -> None:
        """Test successful extraction of model parameters."""
        model_info = {
            "params": "ARIMA(1,0,1)",
            "p": 1,
            "d": 0,
            "q": 1,
        }

        order = extract_model_parameters(model_info)

        assert order == (1, 0, 1)

    def test_extract_model_parameters_nested_params(self) -> None:
        """Test extraction with nested params dictionary."""
        model_info = {
            "params": {
                "p": 2,
                "d": 1,
                "q": 2,
            },
        }

        order = extract_model_parameters(model_info)

        assert order == (2, 1, 2)

    def test_extract_model_parameters_missing_keys(self) -> None:
        """Test extraction with missing required keys."""
        model_info = {
            "params": "ARIMA(1,0,1)",
            "p": 1,
            "d": 0,
            # Missing q
        }

        with pytest.raises(ValueError, match="Model info missing required keys"):
            extract_model_parameters(model_info)

    def test_extract_model_parameters_empty_dict(self) -> None:
        """Test extraction with empty dictionary."""
        model_info = {}  # type: dict[str, object]

        with pytest.raises(ValueError, match="Model info missing required keys"):
            extract_model_parameters(model_info)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
