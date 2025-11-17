"""Tests for MLE estimation cache.

Tests the thread-safe LRU cache for EGARCH MLE estimations.
"""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pytest

from src.garch.garch_params.estimation.common import (
    MLECache,
    _compute_residual_hash,
    _create_cache_key,
    clear_global_cache,
    get_global_cache,
)
from src.garch.garch_params.estimation.common import ConvergenceResult


class TestMLECache:
    """Test suite for MLECache class."""

    def test_cache_initialization(self) -> None:
        """Test cache initialization with valid max_size."""
        cache = MLECache(max_size=100)
        assert cache.max_size == 100
        assert cache.hits == 0
        assert cache.misses == 0
        assert len(cache.cache) == 0

    def test_cache_initialization_invalid_size(self) -> None:
        """Test cache initialization with invalid max_size raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be positive"):
            MLECache(max_size=0)

        with pytest.raises(ValueError, match="max_size must be positive"):
            MLECache(max_size=-10)

    def test_cache_miss(self) -> None:
        """Test cache miss on first request."""
        cache = MLECache(max_size=10)
        residuals = np.random.randn(100)

        result = cache.get(residuals, o=1, p=1, dist="normal")

        assert result is None
        assert cache.hits == 0
        assert cache.misses == 1

    def test_cache_hit(self) -> None:
        """Test cache hit after storing result."""
        cache = MLECache(max_size=10)
        residuals = np.random.randn(100)
        params = {"omega": 0.1, "alpha": 0.05, "beta": 0.9, "gamma": 0.0}
        convergence = ConvergenceResult(
            converged=True, n_iterations=50, final_loglik=-100.0, message="success"
        )

        # Store in cache
        cache.put(residuals, o=1, p=1, dist="normal", params=params, convergence=convergence)

        # Retrieve from cache
        result = cache.get(residuals, o=1, p=1, dist="normal")

        assert result is not None
        retrieved_params, retrieved_conv = result
        assert retrieved_params == params
        assert retrieved_conv == convergence
        assert cache.hits == 1
        assert cache.misses == 0

    def test_cache_different_parameters(self) -> None:
        """Test that different parameters create different cache entries."""
        cache = MLECache(max_size=10)
        residuals = np.random.randn(100)
        params1 = {"omega": 0.1, "alpha": 0.05, "beta": 0.9, "gamma": 0.0}
        params2 = {"omega": 0.2, "alpha": 0.1, "beta": 0.8, "gamma": 0.1}
        conv = ConvergenceResult(
            converged=True, n_iterations=50, final_loglik=-100.0, message="success"
        )

        # Store with different (o, p, dist)
        cache.put(residuals, o=1, p=1, dist="normal", params=params1, convergence=conv)
        cache.put(residuals, o=2, p=1, dist="normal", params=params2, convergence=conv)

        # Retrieve both
        result1 = cache.get(residuals, o=1, p=1, dist="normal")
        result2 = cache.get(residuals, o=2, p=1, dist="normal")

        assert result1 is not None
        assert result2 is not None
        assert result1[0] == params1
        assert result2[0] == params2
        assert len(cache.cache) == 2

    def test_cache_lru_eviction(self) -> None:
        """Test LRU eviction when cache is full."""
        cache = MLECache(max_size=2)
        residuals1 = np.random.randn(100)
        residuals2 = np.random.randn(100)
        residuals3 = np.random.randn(100)
        params = {"omega": 0.1, "alpha": 0.05, "beta": 0.9, "gamma": 0.0}
        conv = ConvergenceResult(
            converged=True, n_iterations=50, final_loglik=-100.0, message="success"
        )

        # Fill cache
        cache.put(residuals1, o=1, p=1, dist="normal", params=params, convergence=conv)
        cache.put(residuals2, o=1, p=1, dist="normal", params=params, convergence=conv)
        assert len(cache.cache) == 2

        # Add third entry, should evict first
        cache.put(residuals3, o=1, p=1, dist="normal", params=params, convergence=conv)
        assert len(cache.cache) == 2

        # First entry should be evicted
        result1 = cache.get(residuals1, o=1, p=1, dist="normal")
        assert result1 is None

        # Second and third should still be present
        result2 = cache.get(residuals2, o=1, p=1, dist="normal")
        result3 = cache.get(residuals3, o=1, p=1, dist="normal")
        assert result2 is not None
        assert result3 is not None

    def test_cache_clear(self) -> None:
        """Test cache clearing."""
        cache = MLECache(max_size=10)
        residuals = np.random.randn(100)
        params = {"omega": 0.1, "alpha": 0.05, "beta": 0.9, "gamma": 0.0}
        conv = ConvergenceResult(
            converged=True, n_iterations=50, final_loglik=-100.0, message="success"
        )

        cache.put(residuals, o=1, p=1, dist="normal", params=params, convergence=conv)
        assert len(cache.cache) == 1
        assert cache.hits == 0

        cache.get(residuals, o=1, p=1, dist="normal")
        assert cache.hits == 1

        cache.clear()
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_cache_stats(self) -> None:
        """Test cache statistics computation."""
        cache = MLECache(max_size=10)
        residuals = np.random.randn(100)
        params = {"omega": 0.1, "alpha": 0.05, "beta": 0.9, "gamma": 0.0}
        conv = ConvergenceResult(
            converged=True, n_iterations=50, final_loglik=-100.0, message="success"
        )

        # Initial stats
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0
        assert stats["hit_rate"] == 0.0

        # After miss
        cache.get(residuals, o=1, p=1, dist="normal")
        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.0

        # After put and hit
        cache.put(residuals, o=1, p=1, dist="normal", params=params, convergence=conv)
        cache.get(residuals, o=1, p=1, dist="normal")
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["hit_rate"] == 0.5


class TestCacheHelpers:
    """Test suite for cache helper functions."""

    def test_compute_residual_hash(self) -> None:
        """Test residual hash computation is consistent."""
        residuals1 = np.array([1.0, 2.0, 3.0])
        residuals2 = np.array([1.0, 2.0, 3.0])
        residuals3 = np.array([1.0, 2.0, 4.0])

        hash1 = _compute_residual_hash(residuals1)
        hash2 = _compute_residual_hash(residuals2)
        hash3 = _compute_residual_hash(residuals3)

        # Same residuals should give same hash
        assert hash1 == hash2
        # Different residuals should give different hash
        assert hash1 != hash3

    def test_create_cache_key(self) -> None:
        """Test cache key creation is unique for different inputs."""
        residuals = np.random.randn(100)

        key1 = _create_cache_key(residuals, o=1, p=1, dist="normal")
        key2 = _create_cache_key(residuals, o=1, p=1, dist="normal")
        key3 = _create_cache_key(residuals, o=2, p=1, dist="normal")
        key4 = _create_cache_key(residuals, o=1, p=2, dist="normal")
        key5 = _create_cache_key(residuals, o=1, p=1, dist="student")

        # Same inputs should give same key
        assert key1 == key2
        # Different parameters should give different keys
        assert key1 != key3
        assert key1 != key4
        assert key1 != key5


class TestGlobalCache:
    """Test suite for global cache instance."""

    def test_get_global_cache(self) -> None:
        """Test getting global cache instance."""
        cache1 = get_global_cache()
        cache2 = get_global_cache()

        # Should return same instance
        assert cache1 is cache2
        assert isinstance(cache1, MLECache)

    def test_clear_global_cache(self) -> None:
        """Test clearing global cache."""
        cache = get_global_cache()
        residuals = np.random.randn(100)
        params = {"omega": 0.1, "alpha": 0.05, "beta": 0.9, "gamma": 0.0}
        conv = ConvergenceResult(
            converged=True, n_iterations=50, final_loglik=-100.0, message="success"
        )

        cache.put(residuals, o=1, p=1, dist="normal", params=params, convergence=conv)
        assert len(cache.cache) > 0

        clear_global_cache()
        assert len(cache.cache) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
