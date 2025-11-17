"""Unit tests for GARCH model parameter classes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.garch.garch_params.models import EGARCHParams

# ==================== EGARCHParams Tests ====================


def test_egarch_params_normal_11() -> None:
    """Test EGARCHParams creation for normal distribution, o=1, p=1."""
    params = EGARCHParams(
        omega=-5.0,
        alpha=0.1,
        gamma=0.0,
        beta=0.95,
        nu=None,
        lambda_skew=None,
        o=1,
        p=1,
        dist="normal",
    )
    assert params.omega == -5.0
    assert params.alpha == 0.1
    assert params.gamma == 0.0
    assert params.beta == 0.95
    assert params.o == 1
    assert params.p == 1
    assert params.dist == "normal"


def test_egarch_params_student_11() -> None:
    """Test EGARCHParams creation for student distribution, o=1, p=1."""
    params = EGARCHParams(
        omega=-5.0,
        alpha=0.1,
        gamma=0.0,
        beta=0.95,
        nu=5.0,
        lambda_skew=None,
        o=1,
        p=1,
        dist="student",
    )
    assert params.nu == 5.0
    assert params.lambda_skew is None


def test_egarch_params_skewt_11() -> None:
    """Test EGARCHParams creation for skewt distribution, o=1, p=1."""
    params = EGARCHParams(
        omega=-5.0,
        alpha=0.1,
        gamma=0.0,
        beta=0.95,
        nu=5.0,
        lambda_skew=0.1,
        o=1,
        p=1,
        dist="skewt",
    )
    assert params.nu == 5.0
    assert params.lambda_skew == 0.1


def test_egarch_params_12() -> None:
    """Test EGARCHParams creation for o=1, p=2."""
    params = EGARCHParams(
        omega=-5.0,
        alpha=0.1,
        gamma=0.0,
        beta=(0.5, 0.4),
        nu=None,
        lambda_skew=None,
        o=1,
        p=2,
        dist="normal",
    )
    assert isinstance(params.beta, tuple)
    assert len(params.beta) == 2


def test_egarch_params_21() -> None:
    """Test EGARCHParams creation for o=2, p=1."""
    params = EGARCHParams(
        omega=-5.0,
        alpha=(0.05, 0.05),
        gamma=(0.0, 0.0),
        beta=0.9,
        nu=None,
        lambda_skew=None,
        o=2,
        p=1,
        dist="normal",
    )
    assert isinstance(params.alpha, tuple)
    assert isinstance(params.gamma, tuple)
    assert len(params.alpha) == 2
    assert len(params.gamma) == 2


def test_egarch_params_22() -> None:
    """Test EGARCHParams creation for o=2, p=2."""
    params = EGARCHParams(
        omega=-5.0,
        alpha=(0.05, 0.05),
        gamma=(0.0, 0.0),
        beta=(0.5, 0.4),
        nu=None,
        lambda_skew=None,
        o=2,
        p=2,
        dist="normal",
    )
    assert isinstance(params.alpha, tuple)
    assert isinstance(params.gamma, tuple)
    assert isinstance(params.beta, tuple)


def test_egarch_params_invalid_o() -> None:
    """Test EGARCHParams raises ValueError for invalid o."""
    with pytest.raises(ValueError, match="ARCH order"):
        EGARCHParams(
            omega=-5.0,
            alpha=0.1,
            gamma=0.0,
            beta=0.95,
            nu=None,
            lambda_skew=None,
            o=3,
            p=1,
            dist="normal",
        )


def test_egarch_params_invalid_p() -> None:
    """Test EGARCHParams raises ValueError for invalid p."""
    with pytest.raises(ValueError, match="GARCH order"):
        EGARCHParams(
            omega=-5.0,
            alpha=0.1,
            gamma=0.0,
            beta=0.95,
            nu=None,
            lambda_skew=None,
            o=1,
            p=4,
            dist="normal",
        )


def test_egarch_params_invalid_alpha_type_o1() -> None:
    """Test EGARCHParams raises TypeError for wrong alpha type with o=1."""
    with pytest.raises(TypeError, match="alpha must be float"):
        EGARCHParams(
            omega=-5.0,
            alpha=(0.1, 0.1),  # Should be float for o=1
            gamma=0.0,
            beta=0.95,
            nu=None,
            lambda_skew=None,
            o=1,
            p=1,
            dist="normal",
        )


def test_egarch_params_invalid_alpha_type_o2() -> None:
    """Test EGARCHParams raises TypeError for wrong alpha type with o=2."""
    with pytest.raises(TypeError, match="alpha must be tuple"):
        EGARCHParams(
            omega=-5.0,
            alpha=0.1,  # Should be tuple for o=2
            gamma=(0.0, 0.0),
            beta=0.95,
            nu=None,
            lambda_skew=None,
            o=2,
            p=1,
            dist="normal",
        )


def test_egarch_params_invalid_distribution() -> None:
    """Test EGARCHParams raises ValueError for invalid distribution."""
    with pytest.raises(ValueError, match="not supported"):
        EGARCHParams(
            omega=-5.0,
            alpha=0.1,
            gamma=0.0,
            beta=0.95,
            nu=None,
            lambda_skew=None,
            o=1,
            p=1,
            dist="invalid",
        )


def test_egarch_params_to_dict_normal_11() -> None:
    """Test EGARCHParams.to_dict for normal, o=1, p=1."""
    params = EGARCHParams(
        omega=-5.0,
        alpha=0.1,
        gamma=0.0,
        beta=0.95,
        nu=None,
        lambda_skew=None,
        o=1,
        p=1,
        dist="normal",
        loglik=-100.0,
        converged=True,
    )
    d = params.to_dict()
    assert d["omega"] == -5.0
    assert d["alpha"] == 0.1
    assert d["gamma"] == 0.0
    assert d["beta"] == 0.95
    assert d["loglik"] == -100.0
    assert d["converged"] == 1.0


def test_egarch_params_to_dict_12() -> None:
    """Test EGARCHParams.to_dict for o=1, p=2."""
    params = EGARCHParams(
        omega=-5.0,
        alpha=0.1,
        gamma=0.0,
        beta=(0.5, 0.4),
        nu=None,
        lambda_skew=None,
        o=1,
        p=2,
        dist="normal",
    )
    d = params.to_dict()
    assert d["beta1"] == 0.5
    assert d["beta2"] == 0.4


def test_egarch_params_to_array() -> None:
    """Test EGARCHParams.to_array."""
    params = EGARCHParams(
        omega=-5.0,
        alpha=0.1,
        gamma=0.0,
        beta=0.95,
        nu=None,
        lambda_skew=None,
        o=1,
        p=1,
        dist="normal",
    )
    arr = params.to_array()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == float
    assert len(arr) == 4  # omega, alpha, gamma, beta


def test_egarch_params_from_array() -> None:
    """Test EGARCHParams.from_array."""
    arr = np.array([-5.0, 0.1, 0.0, 0.95])
    params = EGARCHParams.from_array(arr, o=1, p=1, dist="normal")
    assert params.omega == -5.0
    assert params.alpha == 0.1
    assert params.gamma == 0.0
    assert params.beta == 0.95


def test_egarch_params_from_array_student() -> None:
    """Test EGARCHParams.from_array for student distribution."""
    arr = np.array([-5.0, 0.1, 0.0, 0.95, 5.0])
    params = EGARCHParams.from_array(arr, o=1, p=1, dist="student")
    assert params.nu == 5.0
    assert params.lambda_skew is None


def test_egarch_params_from_array_skewt() -> None:
    """Test EGARCHParams.from_array for skewt distribution."""
    arr = np.array([-5.0, 0.1, 0.0, 0.95, 5.0, 0.1])
    params = EGARCHParams.from_array(arr, o=1, p=1, dist="skewt")
    assert params.nu == 5.0
    assert params.lambda_skew == 0.1


# ==================== GARCHParams Tests ====================


def test_garch_params_normal() -> None:
    """Test GARCHParams creation for normal distribution."""
    params = EGARCHParams(
        omega=0.05,
        alpha=0.05,
        gamma=0.0,
        beta=0.9,
        nu=None,
        lambda_skew=None,
        o=1,
        p=1,
        dist="normal",
    )
    assert params.omega == 0.05
    assert params.alpha == 0.05
    assert params.beta == 0.9
    assert params.dist == "normal"


def test_garch_params_student() -> None:
    """Test GARCHParams creation for student distribution."""
    params = EGARCHParams(
        omega=0.05,
        alpha=0.05,
        gamma=0.0,
        beta=0.9,
        nu=5.0,
        lambda_skew=None,
        o=1,
        p=1,
        dist="student",
    )
    assert params.nu == 5.0


def test_garch_params_to_dict() -> None:
    """Test GARCHParams.to_dict."""
    params = EGARCHParams(
        omega=0.05,
        alpha=0.05,
        gamma=0.0,
        beta=0.9,
        nu=None,
        lambda_skew=None,
        o=1,
        p=1,
        dist="normal",
        loglik=-100.0,
        converged=True,
    )
    d = params.to_dict()
    assert d["omega"] == 0.05
    assert d["alpha"] == 0.05
    assert d["beta"] == 0.9
    assert d["loglik"] == -100.0
    assert d["converged"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
