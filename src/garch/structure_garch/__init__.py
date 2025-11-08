"""GARCH structure detection module.

Provides utilities for ARCH/GARCH identification and diagnostics:
- Loading GARCH datasets
- Computing ACF of returns and squared returns
- Running Engle's ARCH-LM test
- Detecting heteroskedasticity
- Generating diagnostic plots
"""

from __future__ import annotations

from src.garch.structure_garch.detection import (
    compute_acf,
    compute_arch_lm_test,
    compute_squared_acf,
    detect_heteroskedasticity,
    load_garch_dataset,
    plot_arch_diagnostics,
    prepare_residuals,
)

__all__ = [
    "compute_acf",
    "compute_arch_lm_test",
    "compute_squared_acf",
    "detect_heteroskedasticity",
    "load_garch_dataset",
    "plot_arch_diagnostics",
    "prepare_residuals",
]
