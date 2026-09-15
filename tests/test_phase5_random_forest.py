from pathlib import Path

import numpy as np

from src.evaluation.forecasting_evaluation import (
    evaluate_supervised_model,
    load_phase5_artifacts,
)
from src.models.random_forest import (
    assess_feature_importance_stability,
    tune_random_forest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_random_forest_tuning_uses_validation_and_returns_feature_importances():
    artifacts = load_phase5_artifacts(PROJECT_ROOT)

    model, tuning_results = tune_random_forest(
        artifacts["train_X"],
        artifacts["train_y"],
        artifacts["validation_X"],
        artifacts["validation_y"],
    )

    assert len(tuning_results) == 3
    assert tuning_results.iloc[0]["rmse"] <= tuning_results.iloc[-1]["rmse"]
    assert set(model.feature_importances_.index) == set(artifacts["feature_columns"])
    assert np.isclose(model.feature_importances_.sum(), 1.0)


def test_random_forest_evaluates_validation_and_test_windows():
    artifacts = load_phase5_artifacts(PROJECT_ROOT)
    model, _ = tune_random_forest(
        artifacts["train_X"],
        artifacts["train_y"],
        artifacts["validation_X"],
        artifacts["validation_y"],
    )

    result = evaluate_supervised_model(
        model,
        artifacts["train_X"],
        artifacts["train_y"],
        artifacts["validation_X"],
        artifacts["validation_y"],
        artifacts["test_X"],
        artifacts["test_y"],
    )

    assert set(result["metrics"]) == {"validation", "test"}
    assert len(result["predictions"]["validation"]) == 24
    assert len(result["predictions"]["test"]) == 36
    assert np.isfinite(result["predictions"]["test"]).all()


def test_random_forest_feature_importance_rank_stability_is_reported():
    artifacts = load_phase5_artifacts(PROJECT_ROOT)

    stability = assess_feature_importance_stability(
        artifacts["train_X"],
        artifacts["train_y"],
        random_states=(2026, 2027),
    )

    assert stability["importance"].shape == (6, 28)
    assert 0.0 <= stability["mean_top_k_overlap"] <= 1.0
    assert -1.0 <= stability["minimum_spearman_rank_correlation"] <= 1.0