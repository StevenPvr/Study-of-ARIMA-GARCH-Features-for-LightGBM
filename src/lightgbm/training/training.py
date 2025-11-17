"""LightGBM training module using optimized hyperparameters.

This module provides backward compatibility by re-exporting functions
from the refactored submodules.
"""

from __future__ import annotations

# Re-export all public functions for backward compatibility
from src.lightgbm.training.data_loading import load_dataset, load_optimization_results
from src.lightgbm.training.model_training import _run_single_training, save_model, train_lightgbm
from src.lightgbm.training.training_orchestration import run_training

# Re-export constants for backward compatibility (used in tests)

# Backward compatibility alias
train_lightgbm = train_lightgbm

__all__ = [
    "load_dataset",
    "load_optimization_results",
    "run_training",
    "save_model",
    "train_lightgbm",
    "train_lightgbm",  # Backward compatibility
    "_run_single_training",  # For tests
]
