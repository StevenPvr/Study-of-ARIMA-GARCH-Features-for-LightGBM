"""Unit tests for the EGARCH model wrapper (src.models.egarch)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fake_dataset(n_train: int = 20, n_test: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=n_train + n_test, freq="D")
    e = np.concatenate([
        np.full(n_train, 0.0),  # simple, stable train
        np.full(n_test, 0.01),  # constant test residuals
    ])
    return pd.DataFrame(
        {
            "date": dates,
            "split": ["train"] * n_train + ["test"] * n_test,
            "weighted_log_return": e,
        }
    )


def test_run_from_artifacts_delegates(monkeypatch) -> None:
    import src.models.egarch as eg

    def fake_run_from_artifacts(**kwargs):
        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=3),
            "e": [0.0, 0.0, 0.0],
            "sigma2_forecast": [0.001, 0.001, 0.001],
        })
        return df, {"refit_count": 0}

    monkeypatch.setattr(eg, "_rolling_run_from_artifacts", lambda **kw: fake_run_from_artifacts(**kw))

    fore, metrics = eg.run_from_artifacts(
        refit_every=21,
        window="expanding",
        window_size=1000,
        dist_preference="auto",
        keep_nu_between_refits=True,
        var_alphas=[0.05],
    )

    assert set(["date", "e", "sigma2_forecast"]).issubset(fore.columns)
    assert metrics.get("refit_count") == 0


def test_run_from_df_delegates(monkeypatch) -> None:
    import src.models.egarch as eg

    def fake_run_rolling_egarch(df, **kwargs):
        n = int((df["split"] == "test").sum())
        dates = df.loc[df["split"] == "test", "date"].to_numpy()
        e = np.zeros(n)
        s2 = np.full(n, 0.001)
        out = pd.DataFrame({"date": dates, "e": e, "sigma2_forecast": s2})
        return out, {"refit_count": 1}

    monkeypatch.setattr(eg, "_run_rolling_egarch", fake_run_rolling_egarch)

    df = _fake_dataset()
    fore, metrics = eg.run_from_df(
        df,
        refit_every=21,
        window="expanding",
        window_size=1000,
        dist_preference="auto",
        keep_nu_between_refits=True,
        var_alphas=[0.05],
        calibrate_mz=True,
        calibrate_var=True,
    )

    assert fore.shape[0] == (df["split"] == "test").sum()
    assert metrics.get("refit_count") == 1

