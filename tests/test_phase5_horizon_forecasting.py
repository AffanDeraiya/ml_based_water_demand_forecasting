from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation.horizon_forecasting import (
    build_all_model_horizon_forecasts,
    build_horizon_forecast_table,
    save_horizon_forecasts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_one_and_three_month_recursive_forecasts_are_generated():
    models = {
        "Random Forest": joblib.load(PROJECT_ROOT / "outputs/models/random_forest_forecaster.joblib"),
        "ANN": joblib.load(PROJECT_ROOT / "outputs/models/ann_forecaster.joblib"),
    }
    forecasts = build_horizon_forecast_table(models, PROJECT_ROOT, horizon=3)

    assert len(forecasts) == 6
    assert set(forecasts["horizon_step"]) == {1, 2, 3}
    assert forecasts.groupby("model").size().to_dict() == {"ANN": 3, "Random Forest": 3}
    assert forecasts["target_month"].min() == "2022-01-01"
    assert np.isfinite(forecasts["forecast_m3"]).all()


def test_all_four_models_generate_horizon_forecasts():
    forecasts = build_all_model_horizon_forecasts(PROJECT_ROOT, horizon=3)

    assert set(forecasts["model"]) == {"Seasonal naive", "Random Forest", "ANN", "LSTM"}
    assert forecasts.groupby("model").size().to_dict() == {
        "ANN": 3,
        "LSTM": 3,
        "Random Forest": 3,
        "Seasonal naive": 3,
    }
    assert np.isfinite(forecasts["forecast_m3"]).all()


def test_horizon_persists_quarterly_totals():
    paths = save_horizon_forecasts(PROJECT_ROOT)
    quarterly = pd.read_csv(paths["quarterly_forecasts"])

    assert len(quarterly) == 4
    assert set(quarterly["model"]) == {"Seasonal naive", "Random Forest", "ANN", "LSTM"}
    assert np.isfinite(quarterly["quarterly_forecast_m3"]).all()
    assert np.isfinite(quarterly["quarterly_actual_m3"]).all()