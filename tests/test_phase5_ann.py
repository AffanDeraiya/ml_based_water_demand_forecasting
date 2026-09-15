from pathlib import Path

import numpy as np

from src.evaluation.forecasting_evaluation import (
    evaluate_supervised_model,
    load_phase5_artifacts,
)
from src.models.ann import tune_ann


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ann_scales_inputs_and_selects_validation_configuration():
    artifacts = load_phase5_artifacts(PROJECT_ROOT)

    model, tuning_results = tune_ann(
        artifacts["train_X"],
        artifacts["train_y"],
        artifacts["validation_X"],
        artifacts["validation_y"],
    )

    assert len(tuning_results) == 3
    assert tuning_results.iloc[0]["rmse"] <= tuning_results.iloc[-1]["rmse"]
    assert len(model.loss_curve_) > 0
    assert len(model.validation_loss_curve_) == len(model.loss_curve_)
    assert model.best_iteration_ <= len(model.validation_loss_curve_)


def test_ann_evaluates_validation_and_test_in_original_units():
    artifacts = load_phase5_artifacts(PROJECT_ROOT)
    model, _ = tune_ann(
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
        fit_model=False,
    )

    assert set(result["metrics"]) == {"validation", "test"}
    assert len(result["predictions"]["validation"]) == 24
    assert len(result["predictions"]["test"]) == 36
    assert np.isfinite(result["predictions"]["test"]).all()
    assert result["predictions"]["test"].mean() > 1_000_000