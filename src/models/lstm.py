"""Chronology-safe LSTM forecasting for the shared monthly feature matrix."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.evaluation.forecasting_evaluation import (
    compute_regression_metrics,
    compute_residual_diagnostics,
)


DEFAULT_LSTM_CONFIGS = (
    {"hidden_size": 16, "num_layers": 1, "learning_rate": 0.005},
    {"hidden_size": 32, "num_layers": 1, "learning_rate": 0.003},
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.lstm(values)
        return self.output(sequence_output[:, -1, :]).squeeze(-1)


def prepare_lstm_sequences(
    train_X: pd.DataFrame,
    validation_X: pd.DataFrame,
    test_X: pd.DataFrame,
    train_y: pd.Series,
    validation_y: pd.Series,
    test_y: pd.Series,
    *,
    lookback: int = 12,
) -> dict[str, Any]:
    """Build monthly sequences with context from earlier rows only.

    Feature and target scalers are fitted on the training window only. Validation
    and test sequences may use preceding feature rows as context, but never their
    target values or future feature rows.
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if not train_X.index.is_monotonic_increasing:
        raise ValueError("Training features must be chronological")

    x_scaler = StandardScaler().fit(train_X)
    y_scaler = StandardScaler().fit(np.asarray(train_y).reshape(-1, 1))
    combined_X = pd.concat([train_X, validation_X, test_X])
    combined_scaled_X = x_scaler.transform(combined_X).astype(np.float32)
    split_frames = {
        "train": (0, len(train_X), train_y.index),
        "validation": (len(train_X), len(train_X) + len(validation_X), validation_y.index),
        "test": (len(train_X) + len(validation_X), len(combined_X), test_y.index),
    }

    sequences: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    indexes: dict[str, pd.DatetimeIndex] = {}
    for split, (start, stop, target_index) in split_frames.items():
        sequence_values = []
        for row_position in range(start, stop):
            context_start = row_position - lookback
            if context_start < 0:
                if split == "train":
                    continue
                raise ValueError(f"Insufficient historical context for {split} sequence")
            sequence_values.append(combined_scaled_X[context_start:row_position])
        sequences[split] = np.asarray(sequence_values, dtype=np.float32)
        indexes[split] = target_index[-len(sequence_values):]
        if split == "train":
            target_values = train_y.iloc[lookback:].to_numpy()
        elif split == "validation":
            target_values = validation_y.to_numpy()
        else:
            target_values = test_y.to_numpy()
        targets[split] = y_scaler.transform(np.asarray(target_values).reshape(-1, 1)).ravel().astype(np.float32)

    return {
        "sequences": sequences,
        "targets": targets,
        "indexes": indexes,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "lookback": lookback,
        "feature_columns": list(train_X.columns),
    }


class LSTMForecaster:
    """Small monitored LSTM regressor for monthly feature sequences."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int = 16,
        num_layers: int = 1,
        learning_rate: float = 0.005,
        epochs: int = 150,
        patience: int = 20,
        random_state: int = 2026,
    ):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.random_state = int(random_state)
        self._model: _LSTMRegressor | None = None
        self.training_loss_curve_: list[float] = []
        self.validation_loss_curve_: list[float] = []
        self.best_epoch_: int | None = None
        self.x_scaler: StandardScaler | None = None
        self.y_scaler: StandardScaler | None = None

    def fit(
        self,
        train_sequences: np.ndarray,
        train_targets: np.ndarray,
        validation_sequences: np.ndarray,
        validation_targets: np.ndarray,
        *,
        x_scaler: StandardScaler,
        y_scaler: StandardScaler,
    ) -> "LSTMForecaster":
        _set_seed(self.random_state)
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self._model = _LSTMRegressor(self.input_size, self.hidden_size, self.num_layers)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loss_function = nn.MSELoss()
        train_values = torch.from_numpy(train_sequences)
        train_labels = torch.from_numpy(train_targets)
        validation_values = torch.from_numpy(validation_sequences)
        validation_labels = torch.from_numpy(validation_targets)
        best_loss = float("inf")
        best_state = None
        stale_epochs = 0

        for epoch in range(self.epochs):
            self._model.train()
            optimizer.zero_grad()
            train_prediction = self._model(train_values)
            train_loss = loss_function(train_prediction, train_labels)
            train_loss.backward()
            optimizer.step()
            self._model.eval()
            with torch.no_grad():
                validation_prediction = self._model(validation_values)
                validation_loss = loss_function(validation_prediction, validation_labels)
            self.training_loss_curve_.append(float(train_loss.item()))
            self.validation_loss_curve_.append(float(validation_loss.item()))
            if validation_loss.item() < best_loss - 1e-5:
                best_loss = float(validation_loss.item())
                best_state = copy.deepcopy(self._model.state_dict())
                self.best_epoch_ = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.patience:
                break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        if self._model is None or self.y_scaler is None:
            raise ValueError("Model must be fit before predicting")
        self._model.eval()
        with torch.no_grad():
            scaled_prediction = self._model(torch.from_numpy(sequences)).numpy()
        return self.y_scaler.inverse_transform(scaled_prediction.reshape(-1, 1)).ravel()

    def checkpoint(self) -> dict[str, Any]:
        if self._model is None:
            raise ValueError("Model must be fit before saving")
        return {
            "state_dict": self._model.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
        }


def tune_lstm(
    prepared: Mapping[str, Any],
    configurations: Iterable[Mapping[str, Any]] = DEFAULT_LSTM_CONFIGS,
    *,
    random_state: int = 2026,
) -> tuple[LSTMForecaster, pd.DataFrame]:
    """Select the LSTM configuration by validation RMSE."""
    configuration_list = [dict(configuration) for configuration in configurations]
    records = []
    for configuration_index, configuration in enumerate(configuration_list):
        model = LSTMForecaster(
            input_size=prepared["sequences"]["train"].shape[2],
            random_state=random_state,
            **configuration,
        )
        model.fit(
            prepared["sequences"]["train"],
            prepared["targets"]["train"],
            prepared["sequences"]["validation"],
            prepared["targets"]["validation"],
            x_scaler=prepared["x_scaler"],
            y_scaler=prepared["y_scaler"],
        )
        prediction = model.predict(prepared["sequences"]["validation"])
        actual = prepared["y_scaler"].inverse_transform(
            prepared["targets"]["validation"].reshape(-1, 1)
        ).ravel()
        metrics = compute_regression_metrics(actual, prediction)
        records.append({"configuration_index": configuration_index, **configuration, **metrics})

    if not records:
        raise ValueError("At least one LSTM configuration is required")
    tuning_results = pd.DataFrame(records).sort_values(["rmse", "mae"]).reset_index(drop=True)
    selected_index = int(tuning_results.iloc[0]["configuration_index"])
    selected = configuration_list[selected_index]
    best_model = LSTMForecaster(
        input_size=prepared["sequences"]["train"].shape[2],
        random_state=random_state,
        **selected,
    )
    best_model.fit(
        prepared["sequences"]["train"],
        prepared["targets"]["train"],
        prepared["sequences"]["validation"],
        prepared["targets"]["validation"],
        x_scaler=prepared["x_scaler"],
        y_scaler=prepared["y_scaler"],
    )
    return best_model, tuning_results


def save_lstm_run(
    project_root: Path | str,
    model: LSTMForecaster,
    prepared: Mapping[str, Any],
    predictions: Mapping[str, pd.Series],
    targets: Mapping[str, pd.Series],
    metrics: Mapping[str, Mapping[str, float]],
    configuration: Mapping[str, Any],
) -> dict[str, Path]:
    """Persist LSTM weights, forecasts, metrics, diagnostics, and metadata."""
    root = Path(project_root)
    model_dir = root / "outputs" / "models"
    metrics_dir = root / "outputs" / "metrics"
    report_dir = root / "outputs" / "reports"
    for directory in (model_dir, metrics_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "lstm_forecaster.pt"
    torch.save(model.checkpoint(), model_path)
    predictions_path = report_dir / "lstm_forecaster_predictions.csv"
    pd.concat(predictions.values(), axis=1).set_axis(predictions.keys(), axis=1).to_csv(
        predictions_path, index_label="date"
    )
    metrics_path = metrics_dir / "lstm_forecaster_metrics.json"
    configuration_path = metrics_dir / "lstm_forecaster_configuration.json"
    diagnostics_path = metrics_dir / "lstm_forecaster_residual_diagnostics.json"
    residuals_path = report_dir / "lstm_forecaster_residuals.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    configuration_path.write_text(json.dumps(dict(configuration), indent=2))
    diagnostics = {
        split: compute_residual_diagnostics(targets[split].to_numpy(), predictions[split].to_numpy())
        for split in predictions
    }
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2))
    pd.concat({split: targets[split] - predictions[split] for split in predictions}, axis=1).to_csv(
        residuals_path, index_label="date"
    )
    loss_path = metrics_dir / "lstm_training_history.json"
    loss_path.write_text(json.dumps({
        "training_loss": model.training_loss_curve_,
        "validation_loss": model.validation_loss_curve_,
        "best_epoch": model.best_epoch_,
        "lookback": prepared["lookback"],
    }, indent=2))
    return {
        "model": model_path,
        "predictions": predictions_path,
        "metrics": metrics_path,
        "configuration": configuration_path,
        "residuals": residuals_path,
        "residual_diagnostics": diagnostics_path,
        "training_history": loss_path,
    }