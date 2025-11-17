"""Data leakage detection tests for LightGBM models."""

from __future__ import annotations

from src.lightgbm.data_leakage_checkup.leakage_test import run_target_shuffle_test

__all__ = [
    "run_target_shuffle_test",
]
