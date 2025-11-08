"""Model wrappers used by benchmarks and CLIs.

This package provides thin, typed facades over internal modeling engines
so that benchmarks can import models from a stable location: `src.models`.

We keep wrappers minimal to satisfy KISS/DRY and avoid duplication.
"""

from __future__ import annotations

__all__ = [
    # Expose EGARCH wrapper
    "egarch",
]

