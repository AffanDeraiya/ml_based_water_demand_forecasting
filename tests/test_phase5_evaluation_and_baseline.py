from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.forecasting_evaluation import (
    compute_regression_metrics,
    compute_residual_diagnostics,
    load_phase5_artifacts,
)
from src.models.baselines import SeasonalNaiveBaseline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase5_loads_phase4_artifacts():
    artifacts = load_phase5_artifacts(PROJECT_ROOT)

    assert artifacts["train_X"].shape[0] == 108
    assert artifacts["validation_X"].shape[0] == 24
    assert artifacts["test_X"].shape[0] == 36
    assert artifacts["train_y"].shape[0] == 108
    assert artifacts["validation_y"].shape[0] == 24
    assert artifacts["test_y"].shape[0] == 36
    assert artifacts["target_name"] == "residential_water_demand_m3"
    assert artifacts["feature_count"] == 28


def test_metrics_function_returns_expected_values():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 32.0, 38.0])

    metrics = compute_regression_metrics(y_true, y_pred)

    assert metrics["mae"] == np.round(np.mean(np.abs(y_true - y_pred)), 10)
    assert metrics["rmse"] == np.round(np.sqrt(np.mean((y_true - y_pred) ** 2)), 10)
    assert "smape" in metrics
    assert "r2" in metrics


def test_residual_diagnostics_report_signed_and_absolute_errors():
    diagnostics = compute_residual_diagnostics(
        np.array([10.0, 20.0, 30.0]),
        np.array([8.0, 21.0, 27.0]),
    )

    assert np.isclose(diagnostics["mean_residual"], 4.0 / 3.0)
    assert np.isclose(diagnostics["residual_mae"], 2.0)
    assert diagnostics["max_absolute_residual"] == 3.0


def test_seasonal_naive_uses_last_year_same_month():
    dates = pd.date_range("2019-01-01", periods=24, freq="MS")
    values = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210] * 2)
    train = pd.Series(values, index=dates)

    model = SeasonalNaiveBaseline(seasonal_period=12)
    model.fit(train)
    preds = model.predict(pd.date_range("2020-01-01", periods=12, freq="MS"))

    assert preds.shape[0] == 12
    assert np.allclose(preds.to_numpy(), np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210]))


def test_seasonal_naive_handles_dates_after_training_horizon():
    train_dates = pd.date_range("2019-01-01", periods=12, freq="MS")
    train_values = np.arange(1, 13, dtype=float)
    train = pd.Series(train_values, index=train_dates)

    model = SeasonalNaiveBaseline(seasonal_period=12)
    model.fit(train)
    pred_dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    preds = model.predict(pred_dates)

    expected = np.concatenate([np.arange(1, 13, dtype=float), np.arange(1, 13, dtype=float)])
    assert preds.shape[0] == 24
    assert np.allclose(preds.to_numpy(), expected)
