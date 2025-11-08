"""Rolling EGARCH backtesting package.

Exposes a minimal API used by wrappers and benchmarks:
- `run_from_artifacts`: load dataset and run rolling EGARCH
- `run_rolling_egarch`: run from an in-memory DataFrame
- `save_rolling_outputs`: persist CSV/JSON outputs

The implementation focuses on one-step-ahead volatility forecasts with
periodic refits, avoiding information leakage.
"""

from __future__ import annotations

from .rolling import (
    EgarchParams,
    GarchParams,
    run_from_artifacts,
    run_rolling_egarch,
    save_rolling_outputs,
)

__all__ = [
    "EgarchParams",
    "GarchParams",
    "run_from_artifacts",
    "run_rolling_egarch",
    "save_rolling_outputs",
]
