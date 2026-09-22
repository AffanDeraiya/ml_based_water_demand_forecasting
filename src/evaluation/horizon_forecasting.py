"""One-step and recursive multi-step forecasts from the Phase 5 test origin."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .forecasting_evaluation import load_phase5_artifacts
from src.models.baselines import SeasonalNaiveBaseline
from src.models.lstm import LSTMForecaster, prepare_lstm_sequences


TARGET_LAG_COLUMNS = {
    "demand_lag_1": 1,
    "demand_lag_2": 2,
    "demand_lag_3": 3,
    "demand_lag_6": 6,
    "demand_lag_12": 12,
}
ROLLING_COLUMNS = {
    "demand_rolling_mean_3": ("mean", 3),
    "demand_rolling_std_3": ("std", 3),
    "demand_rolling_mean_6": ("mean", 6),
    "demand_rolling_std_6": ("std", 6),
}


def _recursive_feature_row(
    base_row: pd.Series,
    history: list[float],
) -> pd.Series:
    """Replace target-derived features with values available at the forecast origin."""
    row = base_row.copy()
    for column, lag in TARGET_LAG_COLUMNS.items():
        row[column] = history[-lag]
    for column, (operation, window) in ROLLING_COLUMNS.items():
        values = np.asarray(history[-window:], dtype=float)
        row[column] = values.mean() if operation == "mean" else values.std(ddof=1)
    return row


def recursive_tabular_forecast(
    model: Any,
    feature_history: pd.DataFrame,
    target_history: pd.Series,
    future_features: pd.DataFrame,
    *,
    horizon: int = 3,
) -> pd.Series:
    """Generate recursive forecasts while updating target-derived features."""
    if horizon <= 0 or horizon > len(future_features):
        raise ValueError("horizon must be positive and fit within future_features")
    history = list(target_history.astype(float).to_numpy())
    predictions = []
    for timestamp in future_features.index[:horizon]:
        row = _recursive_feature_row(future_features.loc[timestamp], history)
        prediction = float(np.asarray(model.predict(row.to_frame().T)).reshape(-1)[0])
        predictions.append(prediction)
        history.append(prediction)
    return pd.Series(predictions, index=future_features.index[:horizon], name="forecast")


def build_horizon_forecast_table(
    models: Mapping[str, Any],
    project_root: Path | str = ".",
    *,
    horizon: int = 3,
) -> pd.DataFrame:
    """Build one-month and three-month recursive forecasts for fitted tabular models."""
    artifacts = load_phase5_artifacts(project_root)
    origin = artifacts["test_X"].index[0]
    future_features = artifacts["test_X"].iloc[:horizon]
    target_history = pd.concat([artifacts["train_y"], artifacts["validation_y"]])
    rows = []
    for model_name, model in models.items():
        forecast = recursive_tabular_forecast(
            model,
            pd.concat([artifacts["train_X"], artifacts["validation_X"]]),
            target_history,
            future_features,
            horizon=horizon,
        )
        for step, (timestamp, value) in enumerate(forecast.items(), start=1):
            rows.append({
                "model": model_name,
                "forecast_origin": origin.strftime("%Y-%m-%d"),
                "target_month": timestamp.strftime("%Y-%m-%d"),
                "horizon_step": step,
                "forecast_m3": value,
                "actual_m3": artifacts["test_y"].loc[timestamp],
            })
    return pd.DataFrame(rows)


def load_persisted_lstm(project_root: Path | str = ".") -> tuple[LSTMForecaster, dict[str, Any]]:
    """Reconstruct the persisted LSTM with training-only scalers and sequence context."""
    root = Path(project_root)
    artifacts = load_phase5_artifacts(root)
    prepared = prepare_lstm_sequences(
        artifacts["train_X"], artifacts["validation_X"], artifacts["test_X"],
        artifacts["train_y"], artifacts["validation_y"], artifacts["test_y"], lookback=12,
    )
    import torch

    checkpoint = torch.load(root / "outputs" / "models" / "lstm_forecaster.pt", map_location="cpu", weights_only=False)
    model = LSTMForecaster(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        learning_rate=checkpoint["learning_rate"],
        random_state=checkpoint["random_state"],
    )
    model._model = __import__("src.models.lstm", fromlist=["_LSTMRegressor"])._LSTMRegressor(
        checkpoint["input_size"], checkpoint["hidden_size"], checkpoint["num_layers"]
    )
    model._model.load_state_dict(checkpoint["state_dict"])
    model.x_scaler = prepared["x_scaler"]
    model.y_scaler = prepared["y_scaler"]
    return model, prepared


def build_all_model_horizon_forecasts(project_root: Path | str = ".", *, horizon: int = 3) -> pd.DataFrame:
    """Generate one-step and three-step recursive forecasts for all four models."""
    root = Path(project_root)
    artifacts = load_phase5_artifacts(root)
    import joblib

    tabular_models = {
        "Random Forest": joblib.load(root / "outputs/models/random_forest_forecaster.joblib"),
        "ANN": joblib.load(root / "outputs/models/ann_forecaster.joblib"),
    }
    table = build_horizon_forecast_table(tabular_models, root, horizon=horizon)
    origin = artifacts["test_X"].index[0]
    baseline_dates = artifacts["test_X"].index[:horizon]
    baseline = SeasonalNaiveBaseline(seasonal_period=12).fit(
        pd.concat([artifacts["train_y"], artifacts["validation_y"]])
    )
    baseline_forecast = baseline.predict(baseline_dates)
    baseline_rows = pd.DataFrame({
        "model": "Seasonal naive",
        "forecast_origin": origin.strftime("%Y-%m-%d"),
        "target_month": baseline_forecast.index.strftime("%Y-%m-%d"),
        "horizon_step": range(1, horizon + 1),
        "forecast_m3": baseline_forecast.to_numpy(),
        "actual_m3": artifacts["test_y"].iloc[:horizon].to_numpy(),
    })

    lstm, prepared = load_persisted_lstm(root)
    combined_features = pd.concat([artifacts["train_X"], artifacts["validation_X"]])
    history = list(pd.concat([artifacts["train_y"], artifacts["validation_y"]]).astype(float).to_numpy())
    context = prepared["x_scaler"].transform(combined_features.iloc[-12:]).astype(np.float32)
    lstm_rows = []
    for step, timestamp in enumerate(artifacts["test_X"].index[:horizon], start=1):
        row = _recursive_feature_row(artifacts["test_X"].loc[timestamp], history)
        scaled_row = prepared["x_scaler"].transform(row.to_frame().T).astype(np.float32)
        sequence = np.concatenate([context, scaled_row], axis=0)[-12:][None, :, :]
        prediction = float(lstm.predict(sequence)[0])
        history.append(prediction)
        context = np.concatenate([context, scaled_row], axis=0)[-12:]
        lstm_rows.append({
            "model": "LSTM",
            "forecast_origin": origin.strftime("%Y-%m-%d"),
            "target_month": timestamp.strftime("%Y-%m-%d"),
            "horizon_step": step,
            "forecast_m3": prediction,
            "actual_m3": artifacts["test_y"].loc[timestamp],
        })
    return pd.concat([table, baseline_rows, pd.DataFrame(lstm_rows)], ignore_index=True)


def save_horizon_forecasts(project_root: Path | str = ".", *, horizon: int = 3) -> dict[str, Path]:
    """Persist four-model one-step and recursive three-step forecast outputs."""
    root = Path(project_root)
    table = build_all_model_horizon_forecasts(root, horizon=horizon)
    report_dir = root / "outputs" / "reports"
    metrics_dir = root / "outputs" / "metrics"
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = report_dir / "phase5_horizon_forecasts.csv"
    metrics_path = metrics_dir / "phase5_horizon_metrics.json"
    table.to_csv(forecast_path, index=False)
    metrics = {}
    quarterly_rows = []
    for model, group in table.groupby("model"):
        group = group.sort_values("horizon_step")
        errors = group["actual_m3"] - group["forecast_m3"]
        quarterly_rows.append({
            "model": model,
            "forecast_origin": group["forecast_origin"].iloc[0],
            "quarterly_forecast_m3": float(group["forecast_m3"].sum()),
            "quarterly_actual_m3": float(group["actual_m3"].sum()),
            "quarterly_error_m3": float(errors.sum()),
        })
        metrics[model] = {
            "one_month_absolute_error": float(abs(errors.iloc[0])),
            "three_month_mae": float(errors.abs().mean()),
            "three_month_rmse": float(np.sqrt(np.mean(errors.to_numpy() ** 2))),
            "quarterly_forecast_m3": float(group["forecast_m3"].sum()),
            "quarterly_actual_m3": float(group["actual_m3"].sum()),
            "quarterly_error_m3": float(errors.sum()),
        }
    quarterly_path = report_dir / "phase5_quarterly_forecasts.csv"
    pd.DataFrame(quarterly_rows).to_csv(quarterly_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return {"forecasts": forecast_path, "metrics": metrics_path, "quarterly_forecasts": quarterly_path}