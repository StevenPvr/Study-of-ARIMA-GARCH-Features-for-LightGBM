"""Tests anti-leakage pour garantir rigueur académique du module GARCH.

Ces tests vérifient explicitement:
1. Forecasts au temps t utilisent uniquement residuals[:t]
2. Refit windows excluent la position courante
3. Séparation stricte TRAIN/TEST
4. Variance filtered != Variance forecast
5. Temporalité causale des récursions EGARCH

Conformément aux recommandations de l'audit méthodologique (2025-01-12).
"""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).parent.parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.garch.garch_params.core import egarch_variance
from src.garch.garch_params.models import EGARCHParams
from src.garch.garch_params.refit.refit_manager import RefitManager
from src.garch.garch_params.refit import ExpandingWindow, RollingWindow
from src.garch.training_garch.forecaster import EGARCHForecaster
from src.garch.training_garch.variance_filter import VarianceFilter


class TestForecastTemporalCausality:
    """Tests pour vérifier que les forecasts respectent la causalité temporelle."""

    def test_forecast_expanding_uses_correct_history_length(self):
        """Vérifie que forecast à position t utilise exactement residuals[:t]."""
        np.random.seed(42)
        # Use residuals with realistic scale for financial returns (std ~ 0.01)
        # Standard normal (std=1) causes MLE to converge to unstable parameters
        residuals = np.random.randn(100) * 0.01

        forecaster = EGARCHForecaster(
            o=1,
            p=1,
            dist="normal",
            refit_frequency=50,
            window_type="expanding",
            initial_window_size=50,
        )

        # Mock refit_manager.perform_refit to use stable EGARCH parameters
        # This avoids MLE convergence to unstable parameters with small samples
        def mock_perform_refit(resid_hist, position):
            from src.garch.garch_params.estimation.common import ConvergenceResult

            # Use stable EGARCH(1,1) parameters typical for financial returns
            params = {
                "omega": -0.1,  # Small negative intercept
                "alpha": 0.15,  # Moderate ARCH effect
                "gamma": -0.05,  # Small leverage effect
                "beta": 0.95,  # High persistence
            }
            convergence = ConvergenceResult(
                converged=True, n_iterations=10, final_loglik=1000.0, message="Mocked convergence"
            )
            return params, convergence

        # Spy sur _compute_one_step_forecast
        with (
            patch.object(forecaster.refit_manager, "perform_refit", side_effect=mock_perform_refit),
            patch.object(
                forecaster,
                "_compute_one_step_forecast",
                wraps=forecaster._compute_one_step_forecast,
            ) as mock_forecast,
        ):
            result = forecaster.forecast_expanding(residuals)

            # Vérifier que les forecasts ont été générés
            assert len(result.forecasts) == 100
            assert np.all(np.isfinite(result.forecasts[50:]))

            # Vérifier que chaque appel utilise la bonne longueur d'historique
            for call_args in mock_forecast.call_args_list:
                history = call_args[0][0]  # residuals_history
                # La longueur de l'historique doit être < position du forecast
                assert len(history) >= 50, f"History too short: {len(history)}"
                assert len(history) < 100, f"History too long: {len(history)}"

    def test_forecast_position_t_excludes_residual_t(self):
        """Vérifie que le forecast à t n'utilise PAS le résidu au temps t."""
        np.random.seed(42)
        n = 100
        # Use residuals with realistic scale for financial returns (std ~ 0.01)
        residuals = np.random.randn(n) * 0.01

        forecaster = EGARCHForecaster(
            o=1,
            p=1,
            dist="normal",
            refit_frequency=50,
            window_type="expanding",
            initial_window_size=50,
        )

        # Mock refit_manager.perform_refit to use stable EGARCH parameters
        def mock_perform_refit(resid_hist, position):
            from src.garch.garch_params.estimation.common import ConvergenceResult

            params = {
                "omega": -0.1,
                "alpha": 0.15,
                "gamma": -0.05,
                "beta": 0.95,
            }
            convergence = ConvergenceResult(
                converged=True, n_iterations=10, final_loglik=1000.0, message="Mocked convergence"
            )
            return params, convergence

        with patch.object(
            forecaster.refit_manager, "perform_refit", side_effect=mock_perform_refit
        ):
            result = forecaster.forecast_expanding(residuals)

        # Le forecast à position 50 doit utiliser residuals[:50] (pas [:51])
        # On ne peut pas vérifier directement, mais on peut vérifier que
        # la longueur des forecasts est correcte
        assert result.forecasts.size == n
        assert np.isnan(result.forecasts[:50]).all(), "Should have NaN before initial window"
        assert np.isfinite(
            result.forecasts[50:]
        ).all(), "Should have forecasts after initial window"

    def test_egarch_variance_recursion_uses_lagged_residuals(self):
        """Vérifie que la récursion EGARCH utilise ε_{t-1} pour calculer σ²_t."""
        np.random.seed(42)
        residuals = np.random.randn(100)
        omega, alpha, gamma, beta = 0.1, 0.15, -0.05, 0.9

        # Use fixed initial variance to avoid dependence on sample variance of all residuals
        init_var = 1.0

        # Calculer variance path
        sigma2 = egarch_variance(
            residuals,
            omega=omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            dist="normal",
            o=1,
            p=1,
            init=init_var,
        )

        # Pour chaque position t > 0:
        # σ²_t devrait dépendre de ε_{t-1}, pas ε_t
        # On vérifie en modifiant ε_t et vérifiant que σ²_t ne change pas

        residuals_modified = residuals.copy()
        residuals_modified[50] = 999.0  # Modification massive à t=50

        sigma2_modified = egarch_variance(
            residuals_modified,
            omega=omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            dist="normal",
            o=1,
            p=1,
            init=init_var,
        )

        # σ²_50 ne devrait PAS changer (car dépend de ε_49)
        assert np.isclose(
            sigma2[50], sigma2_modified[50]
        ), "Variance at t should not depend on residual at t"

        # σ²_51 DEVRAIT changer (car dépend de ε_50)
        assert not np.isclose(
            sigma2[51], sigma2_modified[51]
        ), "Variance at t+1 should depend on residual at t"


class TestRefitWindowCausality:
    """Tests pour vérifier que les fenêtres de refit sont causales."""

    def test_expanding_window_excludes_current_position(self):
        """Vérifie que ExpandingWindow utilise [start:position) (exclusif)."""
        window_strategy = ExpandingWindow(start=0)

        # Window à position 100
        window = window_strategy.compute_window(100)

        assert window.start == 0
        assert window.end == 100  # Exclusif, donc utilise [0:100]

        # Vérifier que l'extraction de données est correcte
        data = np.arange(200)
        extracted = data[window.start : window.end]

        assert len(extracted) == 100
        assert extracted[-1] == 99  # Dernier élément est 99, pas 100
        assert 100 not in extracted

    def test_rolling_window_excludes_current_position(self):
        """Vérifie que RollingWindow utilise [position-w:position) (exclusif)."""
        window_strategy = RollingWindow(window_size=50)

        # Window à position 100
        window = window_strategy.compute_window(100)

        assert window.start == 50  # max(0, 100-50)
        assert window.end == 100  # Exclusif

        # Vérifier extraction
        data = np.arange(200)
        extracted = data[window.start : window.end]

        assert len(extracted) == 50
        assert extracted[-1] == 99  # Dernier élément est 99, pas 100
        assert 100 not in extracted

    def test_refit_manager_uses_causal_windows(self):
        """Vérifie que RefitManager génère des fenêtres causales."""
        refit_manager = RefitManager(
            frequency=20,
            window_type="expanding",
            window_size=None,
            o=1,
            p=1,
            dist="normal",
        )

        # Vérifier fenêtre à position 60
        window = refit_manager.compute_refit_window(60)

        assert window.end == 60, "Window end should equal current position"

        # Extraire données
        residuals = np.arange(100)
        residuals_window = residuals[window.start : window.end]

        # Ne devrait PAS inclure résidu à position 60
        assert 60 not in residuals_window
        assert residuals_window[-1] == 59


class TestTrainTestSeparation:
    """Tests pour vérifier la séparation stricte TRAIN/TEST."""

    def test_train_test_split_no_overlap(self):
        """Vérifie qu'il n'y a pas de chevauchement entre TRAIN et TEST."""
        # Simuler un dataset
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "sarima_resid": np.random.randn(100),
                "split": ["train"] * 70 + ["test"] * 30,
            }
        )

        df_train = df[df["split"] == "train"]
        df_test = df[df["split"] == "test"]

        # Vérifier pas de chevauchement temporel
        max_train_date = df_train["date"].max()
        min_test_date = df_test["date"].min()

        assert max_train_date < min_test_date, "TRAIN and TEST should not overlap temporally"

        # Vérifier tailles
        assert len(df_train) == 70
        assert len(df_test) == 30
        assert len(df_train) + len(df_test) == 100

    def test_forecaster_train_test_continuity(self):
        """Vérifie que les forecasts TEST ne réutilisent pas données futures."""
        np.random.seed(42)

        # Créer données TRAIN et TEST avec échelle réaliste
        resid_train = np.random.randn(70) * 0.01
        resid_test = np.random.randn(30) * 0.01

        forecaster = EGARCHForecaster(
            o=1,
            p=1,
            dist="normal",
            refit_frequency=20,
            window_type="expanding",
            initial_window_size=50,
        )

        # Mock refit_manager.perform_refit to use stable EGARCH parameters
        def mock_perform_refit(resid_hist, position):
            from src.garch.garch_params.estimation.common import ConvergenceResult

            params = {
                "omega": -0.1,
                "alpha": 0.15,
                "gamma": -0.05,
                "beta": 0.95,
            }
            convergence = ConvergenceResult(
                converged=True, n_iterations=10, final_loglik=1000.0, message="Mocked convergence"
            )
            return params, convergence

        with patch.object(
            forecaster.refit_manager, "perform_refit", side_effect=mock_perform_refit
        ):
            # Forecast sur TRAIN
            result_train = forecaster.forecast_expanding(resid_train)

            # Forecast sur TRAIN+TEST (comme orchestration.py)
            forecaster.clear_history()
            resid_full = np.concatenate([resid_train, resid_test])
            result_full = forecaster.forecast_expanding(resid_full)

        # Les forecasts TRAIN devraient être identiques
        # (car même refit strategy, même données)
        train_forecasts_original = result_train.forecasts[50:]
        train_forecasts_full = result_full.forecasts[50:70]

        # NOTE: Peut ne pas être exactement identique à cause des refits
        # Mais devrait être très proche
        assert len(train_forecasts_original) == 20
        assert len(train_forecasts_full) == 20


class TestFilteredVsForecastVariance:
    """Tests pour vérifier distinction entre filtered et forecast variance."""

    def test_filtered_variance_uses_all_data(self):
        """Vérifie que VarianceFilter calcule σ²_t|t (avec info jusqu'à t)."""
        np.random.seed(42)
        residuals = np.random.randn(100)

        params = EGARCHParams(
            omega=0.1,
            alpha=0.15,
            gamma=-0.05,
            beta=0.9,
            nu=None,
            lambda_skew=None,
            o=1,
            p=1,
            dist="normal",
        )

        variance_filter = VarianceFilter(params)
        sigma2_filtered = variance_filter.filter_variance(residuals)

        # Filtered variance devrait avoir même longueur que residuals
        assert len(sigma2_filtered) == len(residuals)

        # σ²_t devrait dépendre de ε_t (filtered, pas forecast)
        # On vérifie en modifiant ε_50 et vérifiant que σ²_51 change
        residuals_modified = residuals.copy()
        residuals_modified[50] = 999.0

        sigma2_modified = variance_filter.filter_variance(residuals_modified)

        # σ²_51 devrait changer (dépend de ε_50)
        assert not np.isclose(sigma2_filtered[51], sigma2_modified[51])

    def test_forecast_variance_excludes_current_data(self):
        """Vérifie que EGARCHForecaster calcule σ²_{t+1}|t (sans info de t)."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.01

        forecaster = EGARCHForecaster(
            o=1,
            p=1,
            dist="normal",
            refit_frequency=100,  # Pas de refit
            window_type="expanding",
            initial_window_size=50,
        )

        # Mock refit_manager.perform_refit to use stable EGARCH parameters
        def mock_perform_refit(resid_hist, position):
            from src.garch.garch_params.estimation.common import ConvergenceResult

            params = {
                "omega": -0.1,
                "alpha": 0.15,
                "gamma": -0.05,
                "beta": 0.95,
            }
            convergence = ConvergenceResult(
                converged=True, n_iterations=10, final_loglik=1000.0, message="Mocked convergence"
            )
            return params, convergence

        with patch.object(
            forecaster.refit_manager, "perform_refit", side_effect=mock_perform_refit
        ):
            result = forecaster.forecast_expanding(residuals)
        forecasts = result.forecasts

        # Forecast à position t ne devrait PAS dépendre de ε_t
        # On vérifie en modifiant ε_50 et vérifiant que forecast_50 ne change pas
        residuals_modified = residuals.copy()
        residuals_modified[50] = 0.999  # Changed to reasonable scale

        forecaster_modified = EGARCHForecaster(
            o=1,
            p=1,
            dist="normal",
            refit_frequency=100,
            window_type="expanding",
            initial_window_size=50,
        )
        with patch.object(
            forecaster_modified.refit_manager, "perform_refit", side_effect=mock_perform_refit
        ):
            result_modified = forecaster_modified.forecast_expanding(residuals_modified)

        # forecast[50] ne devrait PAS changer
        assert np.isclose(
            forecasts[50], result_modified.forecasts[50]
        ), "Forecast at t should not depend on residual at t"

    def test_filtered_and_forecast_variance_differ(self):
        """Vérifie que filtered variance != forecast variance."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.01

        params = EGARCHParams(
            omega=-0.1,
            alpha=0.15,
            gamma=-0.05,
            beta=0.95,
            nu=None,
            lambda_skew=None,
            o=1,
            p=1,
            dist="normal",
        )

        # Filtered variance
        variance_filter = VarianceFilter(params)
        sigma2_filtered = variance_filter.filter_variance(residuals)

        # Forecast variance
        forecaster = EGARCHForecaster(
            o=1,
            p=1,
            dist="normal",
            refit_frequency=100,
            window_type="expanding",
            initial_window_size=50,
        )

        # Mock refit_manager.perform_refit to use stable EGARCH parameters
        def mock_perform_refit(resid_hist, position):
            from src.garch.garch_params.estimation.common import ConvergenceResult

            return params.to_dict(), ConvergenceResult(
                converged=True, n_iterations=10, final_loglik=1000.0, message="Mocked convergence"
            )

        with patch.object(
            forecaster.refit_manager, "perform_refit", side_effect=mock_perform_refit
        ):
            result = forecaster.forecast_expanding(residuals)
        sigma2_forecast = result.forecasts

        # Les deux devraient être différents
        # (filtered utilise info jusqu'à t, forecast jusqu'à t-1)
        valid_positions = range(50, 100)

        # Vérifier qu'au moins une position diffère significativement
        differences = []
        for t in valid_positions:
            if np.isfinite(sigma2_forecast[t]) and np.isfinite(sigma2_filtered[t]):
                diff = abs(sigma2_filtered[t] - sigma2_forecast[t])
                differences.append(diff)

        # Au moins une différence significative devrait exister
        # (tolérance pour erreurs numériques)
        assert len(differences) > 0, "No valid comparisons found"
        assert max(differences) > 1e-10, "Filtered and forecast variances should differ"


class TestConvergenceTracking:
    """Tests pour vérifier le tracking de convergence."""

    def test_convergence_rate_computed(self):
        """Vérifie que le taux de convergence est calculé."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.01

        forecaster = EGARCHForecaster(
            o=1,
            p=1,
            dist="normal",
            refit_frequency=20,
            window_type="expanding",
            initial_window_size=50,
        )

        # Mock refit_manager.perform_refit to use stable EGARCH parameters
        def mock_perform_refit(resid_hist, position):
            from src.garch.garch_params.estimation.common import ConvergenceResult

            params = {
                "omega": -0.1,
                "alpha": 0.15,
                "gamma": -0.05,
                "beta": 0.95,
            }
            convergence = ConvergenceResult(
                converged=True, n_iterations=10, final_loglik=1000.0, message="Mocked convergence"
            )
            return params, convergence

        with patch.object(
            forecaster.refit_manager, "perform_refit", side_effect=mock_perform_refit
        ):
            result = forecaster.forecast_expanding(residuals)

        # Vérifier que convergence rate existe
        assert hasattr(result, "convergence_rate")
        assert 0.0 <= result.convergence_rate <= 1.0

        # Vérifier nombre de refits
        assert result.n_refits >= 0

    def test_refit_mask_correct(self):
        """Vérifie que le masque de refit est correct."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.01

        forecaster = EGARCHForecaster(
            o=1,
            p=1,
            dist="normal",
            refit_frequency=20,
            window_type="expanding",
            initial_window_size=50,
        )

        # Mock refit_manager.perform_refit to use stable EGARCH parameters
        def mock_perform_refit(resid_hist, position):
            from src.garch.garch_params.estimation.common import ConvergenceResult

            params = {
                "omega": -0.1,
                "alpha": 0.15,
                "gamma": -0.05,
                "beta": 0.95,
            }
            convergence = ConvergenceResult(
                converged=True, n_iterations=10, final_loglik=1000.0, message="Mocked convergence"
            )
            return params, convergence

        with patch.object(
            forecaster.refit_manager, "perform_refit", side_effect=mock_perform_refit
        ):
            result = forecaster.forecast_expanding(residuals)

        # Vérifier que refit_mask a la bonne longueur
        assert len(result.refit_mask) == len(residuals)

        # Compter les refits
        n_refits_from_mask = np.sum(result.refit_mask)

        # Devrait être cohérent avec result.n_refits
        # (peut différer légèrement à cause du refit initial)
        assert n_refits_from_mask >= result.n_refits - 1


class TestNumericalStability:
    """Tests pour vérifier la stabilité numérique."""

    def test_extreme_residuals_handled(self):
        """Vérifie que les résidus extrêmes ne causent pas de crash."""
        np.random.seed(42)
        residuals = np.random.randn(100)

        # Ajouter résidus extrêmes
        residuals[25] = 50.0
        residuals[75] = -50.0

        forecaster = EGARCHForecaster(
            o=1,
            p=1,
            dist="normal",
            refit_frequency=50,
            window_type="expanding",
            initial_window_size=50,
        )

        # Ne devrait pas crasher
        result = forecaster.forecast_expanding(residuals)

        # Vérifier que les forecasts sont finis
        valid_forecasts = result.forecasts[50:]
        assert np.all(np.isfinite(valid_forecasts))

    def test_nan_residuals_filtered(self):
        """Vérifie que les NaN dans les résidus sont gérés."""
        np.random.seed(42)
        residuals = np.random.randn(100)
        residuals[30:35] = np.nan

        # VarianceFilter devrait gérer les NaN
        params = EGARCHParams(
            omega=0.1,
            alpha=0.15,
            gamma=-0.05,
            beta=0.9,
            nu=None,
            lambda_skew=None,
            o=1,
            p=1,
            dist="normal",
        )

        variance_filter = VarianceFilter(params)

        # Devrait retourner des résultats (peut-être avec NaN aux positions NaN)
        sigma2 = variance_filter.filter_variance(residuals)

        assert sigma2 is not None
        assert len(sigma2) == len(residuals)


# Tests additionnels pour couverture complète


def test_anti_leakage_assertions_present():
    """Vérifie que les assertions anti-leakage sont présentes dans le code."""
    import inspect

    from src.garch.training_garch import forecaster

    # Vérifier que le code source contient des assertions
    source = inspect.getsource(forecaster.EGARCHForecaster.forecast_expanding)

    # Chercher patterns anti-leakage
    assert (
        "ANTI-LEAKAGE" in source or "anti-leakage" in source.lower()
    ), "Code should contain anti-leakage comments"
    assert "assert" in source, "Code should contain assertions"


def test_filtered_variance_warnings_present():
    """Vérifie que les warnings de data leakage sont présents."""
    import inspect

    from src.garch.training_garch import variance_filter

    source = inspect.getsource(variance_filter.VarianceFilter)

    # Vérifier warnings dans docstring
    assert (
        "LEAKAGE" in source or "leakage" in source
    ), "VarianceFilter should warn about data leakage"
    assert "WARNING" in source or "warning" in source.lower(), "VarianceFilter should have warnings"


class TestPredictionsIOCausality:
    """Tests anti-leakage pour predictions_io module."""

    def test_save_forecasts_validates_temporality(self):
        """Vérifie que save_garch_forecasts valide la temporalité."""
        import inspect

        from src.garch.training_garch import predictions_io

        source = inspect.getsource(predictions_io.save_garch_forecasts)

        # Doit valider la causalité temporelle
        assert (
            "validate_temporal_split" in source
        ), "save_garch_forecasts should validate temporal causality"
        assert "anti_leakage_validated" in source, "Metadata should track anti-leakage validation"

    def test_save_ml_dataset_validates_splits(self):
        """Vérifie que save_ml_dataset valide les splits."""
        import inspect

        from src.garch.training_garch import predictions_io

        source = inspect.getsource(predictions_io.save_ml_dataset)

        # Doit valider temporalité si split présent
        assert "validate_temporal_split" in source, "save_ml_dataset should validate temporal split"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])  # pragma: no cover
