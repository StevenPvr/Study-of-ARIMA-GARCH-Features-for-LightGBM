"""LightGBM training module."""

from __future__ import annotations

from src.lightgbm.training.training import (
    load_dataset,
    load_optimization_results,
    run_training,
    save_model,
    train_lightgbm,
)

# Backward compatibility alias
train_lightgbm = train_lightgbm

__all__ = [
    "load_dataset",
    "load_optimization_results",
    "run_training",
    "save_model",
    "train_lightgbm",
    "train_lightgbm",  # Backward compatibility
]
