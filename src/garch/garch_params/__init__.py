"""GARCH parameter estimation module.

Provides EGARCH(1,1) parameter estimation via conditional MLE.
"""

from __future__ import annotations

from src.garch.garch_params.estimation import (
    egarch11_variance,
    estimate_egarch_mle,
)

__all__ = [
    "egarch11_variance",
    "estimate_egarch_mle",
]
