"""Random Forest forecasting model and validation-driven configuration search."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.evaluation.forecasting_evaluation import compute_regression_metrics


DEFAULT_RANDOM_FOREST_CONFIGS = (
    {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 1},
    {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 1},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},
)


class RandomForestForecaster:
    """Tabular Random Forest regressor using the shared Phase 4 feature matrix."""

    def __init__(self, *, random_state: int = 2026, **parameters: Any):
        self.random_state = int(random_state)
        self.parameters = dict(parameters)
        self._model: RandomForestRegressor | None = None
        self._feature_names: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestForecaster":
        self._feature_names = list(X.columns)
        self._model = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=-1,
            **self.parameters,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise ValueError("Model must be fit before predicting")
        return pd.Series(self._model.predict(X), index=X.index, name="prediction")

    @property
    def feature_importances_(self) -> pd.Series:
        if self._model is None:
            raise ValueError("Model must be fit before reading feature importances")
        return pd.Series(
            self._model.feature_importances_, index=self._feature_names
        ).sort_values(ascending=False)


def tune_random_forest(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    validation_X: pd.DataFrame,
    validation_y: pd.Series,
    configurations: Iterable[Mapping[str, Any]] = DEFAULT_RANDOM_FOREST_CONFIGS,
    *,
    random_state: int = 2026,
) -> tuple[RandomForestForecaster, pd.DataFrame]:
    """Select a Random Forest configuration by validation RMSE, then refit on train only."""
    configuration_list = [dict(configuration) for configuration in configurations]
    records = []
    for configuration in configuration_list:
        model = RandomForestForecaster(random_state=random_state, **dict(configuration))
        model.fit(train_X, train_y)
        prediction = model.predict(validation_X)
        metrics = compute_regression_metrics(validation_y.to_numpy(), prediction.to_numpy())
        records.append({**dict(configuration), **metrics})

    if not records:
        raise ValueError("At least one Random Forest configuration is required")

    tuning_results = pd.DataFrame(records).sort_values(
        by=["rmse", "mae"], ascending=True
    ).reset_index(drop=True)
    best_configuration = {
        key: tuning_results.iloc[0][key] for key in configuration_list[0].keys()
    }
    for key, value in list(best_configuration.items()):
        if pd.isna(value):
            best_configuration[key] = None
        elif key in {"n_estimators", "max_depth", "min_samples_leaf"}:
            best_configuration[key] = int(value)

    best_model = RandomForestForecaster(random_state=random_state, **best_configuration)
    best_model.fit(train_X, train_y)
    return best_model, tuning_results


def assess_feature_importance_stability(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    configurations: Iterable[Mapping[str, Any]] = DEFAULT_RANDOM_FOREST_CONFIGS,
    *,
    random_states: Iterable[int] = (2026, 2027, 2028),
    top_k: int = 10,
) -> dict[str, Any]:
    """Measure feature-importance rank stability across configurations and seeds."""
    configuration_list = [dict(configuration) for configuration in configurations]
    state_list = [int(random_state) for random_state in random_states]
    if not configuration_list or not state_list:
        raise ValueError("Configurations and random_states must not be empty")

    importance_rows = []
    labels = []
    for configuration_index, configuration in enumerate(configuration_list):
        for random_state in state_list:
            model = RandomForestForecaster(
                random_state=random_state,
                **configuration,
            )
            model.fit(train_X, train_y)
            importance_rows.append(model.feature_importances_.reindex(train_X.columns))
            labels.append(f"configuration_{configuration_index + 1}_seed_{random_state}")

    importance = pd.DataFrame(importance_rows, index=labels)
    ranks = importance.rank(axis=1, ascending=False, method="average")
    rank_correlation = ranks.T.corr(method="spearman")
    upper_triangle = rank_correlation.to_numpy()[np.triu_indices(len(rank_correlation), k=1)]
    top_features = [set(row.nlargest(top_k).index) for _, row in importance.iterrows()]
    top_overlaps = [
        len(left & right) / min(top_k, len(train_X.columns))
        for index, left in enumerate(top_features)
        for right in top_features[index + 1 :]
    ]

    return {
        "importance": importance,
        "rank_correlation": rank_correlation,
        "mean_spearman_rank_correlation": float(upper_triangle.mean()) if len(upper_triangle) else 1.0,
        "minimum_spearman_rank_correlation": float(upper_triangle.min()) if len(upper_triangle) else 1.0,
        "mean_top_k_overlap": float(sum(top_overlaps) / len(top_overlaps)) if top_overlaps else 1.0,
        "top_k": int(top_k),
    }