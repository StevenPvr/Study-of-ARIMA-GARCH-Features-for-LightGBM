"""Stable model import facade.

This package re-exports model wrappers from their internal locations so that
downstream code can import from `src.models` without depending on internal
module layouts.
"""

from __future__ import annotations

__all__ = ["egarch"]

