"""Scaled tabular ANN forecaster and validation-driven configuration search."""

from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.evaluation.forecasting_evaluation import compute_regression_metrics


DEFAULT_ANN_CONFIGS = (
    {"hidden_layer_sizes": (32,), "learning_rate_init": 0.001, "alpha": 0.0001},
    {"hidden_layer_sizes": (64,), "learning_rate_init": 0.001, "alpha": 0.0001},
    {"hidden_layer_sizes": (64, 32), "learning_rate_init": 0.0005, "alpha": 0.0001},
)


class ANNForecaster:
    """A small feed-forward network for the shared tabular feature matrix."""

    def __init__(self, *, random_state: int = 2026, **parameters: Any):
        self.random_state = int(random_state)
        self.parameters = dict(parameters)
        self._x_scaler = StandardScaler()
        self._y_scaler = StandardScaler()
        self._model: MLPRegressor | None = None
        self._validation_loss_curve: list[float] = []
        self._best_iteration: int | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_X: pd.DataFrame | None = None,
        validation_y: pd.Series | None = None,
    ) -> "ANNForecaster":
        if (validation_X is None) != (validation_y is None):
            raise ValueError("validation_X and validation_y must be provided together")
        scaled_X = self._x_scaler.fit_transform(X)
        scaled_y = self._y_scaler.fit_transform(np.asarray(y).reshape(-1, 1)).ravel()
        model_parameters = dict(self.parameters)
        max_epochs = int(model_parameters.pop("max_epochs", 300))
        patience = int(model_parameters.pop("patience", 30))
        min_delta = float(model_parameters.pop("min_delta", 1e-5))
        self._model = MLPRegressor(
            activation="relu",
            solver="adam",
            max_iter=1 if validation_X is not None else max_epochs,
            random_state=self.random_state,
            warm_start=validation_X is not None,
            **model_parameters,
        )
        self._validation_loss_curve = []
        self._best_iteration = None

        if validation_X is None:
            self._model.fit(scaled_X, scaled_y)
            return self

        scaled_validation_X = self._x_scaler.transform(validation_X)
        scaled_validation_y = self._y_scaler.transform(
            np.asarray(validation_y).reshape(-1, 1)
        ).ravel()
        best_loss = float("inf")
        best_model = None
        stale_epochs = 0
        for iteration in range(max_epochs):
            self._model.partial_fit(scaled_X, scaled_y)
            validation_prediction = self._model.predict(scaled_validation_X)
            validation_loss = float(
                np.mean((scaled_validation_y - validation_prediction) ** 2)
            )
            self._validation_loss_curve.append(validation_loss)
            if validation_loss < best_loss - min_delta:
                best_loss = validation_loss
                best_model = copy.deepcopy(self._model)
                self._best_iteration = iteration + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= patience:
                break

        if best_model is not None:
            self._model = best_model
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise ValueError("Model must be fit before predicting")
        scaled_prediction = self._model.predict(self._x_scaler.transform(X))
        prediction = self._y_scaler.inverse_transform(
            np.asarray(scaled_prediction).reshape(-1, 1)
        ).ravel()
        return pd.Series(prediction, index=X.index, name="prediction")

    @property
    def loss_curve_(self) -> list[float]:
        if self._model is None:
            raise ValueError("Model must be fit before reading training loss")
        return list(self._model.loss_curve_)

    @property
    def validation_loss_curve_(self) -> list[float]:
        if self._model is None:
            raise ValueError("Model must be fit before reading validation loss")
        return list(self._validation_loss_curve)

    @property
    def best_iteration_(self) -> int:
        if self._best_iteration is None:
            raise ValueError("Model was not trained with validation monitoring")
        return self._best_iteration


def tune_ann(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    validation_X: pd.DataFrame,
    validation_y: pd.Series,
    configurations: Iterable[Mapping[str, Any]] = DEFAULT_ANN_CONFIGS,
    *,
    random_state: int = 2026,
) -> tuple[ANNForecaster, pd.DataFrame]:
    """Select an ANN configuration by validation RMSE, then refit on train only."""
    configuration_list = [dict(configuration) for configuration in configurations]
    records = []
    for configuration_index, configuration in enumerate(configuration_list):
        model = ANNForecaster(random_state=random_state, **configuration)
        model.fit(train_X, train_y, validation_X, validation_y)
        prediction = model.predict(validation_X)
        metrics = compute_regression_metrics(validation_y.to_numpy(), prediction.to_numpy())
        records.append({"configuration_index": configuration_index, **configuration, **metrics})

    if not records:
        raise ValueError("At least one ANN configuration is required")

    tuning_results = pd.DataFrame(records).sort_values(
        by=["rmse", "mae"], ascending=True
    ).reset_index(drop=True)
    selected_index = int(tuning_results.iloc[0]["configuration_index"])
    best_model = ANNForecaster(
        random_state=random_state,
        **configuration_list[selected_index],
    )
    best_model.fit(train_X, train_y, validation_X, validation_y)
    return best_model, tuning_results