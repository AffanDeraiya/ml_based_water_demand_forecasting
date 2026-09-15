from pathlib import Path

import numpy as np

from src.evaluation.forecasting_evaluation import load_phase5_artifacts
from src.models.lstm import prepare_lstm_sequences, tune_lstm


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _prepared():
    artifacts = load_phase5_artifacts(PROJECT_ROOT)
    return prepare_lstm_sequences(
        artifacts["train_X"],
        artifacts["validation_X"],
        artifacts["test_X"],
        artifacts["train_y"],
        artifacts["validation_y"],
        artifacts["test_y"],
        lookback=12,
    )


def test_lstm_sequences_preserve_chronology_and_training_only_context():
    prepared = _prepared()

    assert prepared["sequences"]["train"].shape == (96, 12, 28)
    assert prepared["sequences"]["validation"].shape == (24, 12, 28)
    assert prepared["sequences"]["test"].shape == (36, 12, 28)
    assert prepared["indexes"]["validation"][0].strftime("%Y-%m-%d") == "2020-01-01"
    assert prepared["indexes"]["test"][-1].strftime("%Y-%m-%d") == "2024-12-01"
    assert np.isfinite(prepared["sequences"]["test"]).all()


def test_lstm_tuning_monitors_validation_and_returns_forecasts():
    prepared = _prepared()
    model, tuning_results = tune_lstm(prepared, configurations=({"hidden_size": 8, "num_layers": 1, "learning_rate": 0.005, "epochs": 20, "patience": 5},))
    validation_prediction = model.predict(prepared["sequences"]["validation"])

    assert len(tuning_results) == 1
    assert model.best_epoch_ is not None
    assert len(model.training_loss_curve_) == len(model.validation_loss_curve_)
    assert np.isfinite(validation_prediction).all()