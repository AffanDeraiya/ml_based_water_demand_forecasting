from pathlib import Path

import pandas as pd

from src.evaluation.model_comparison import (
    build_comparison_tables,
    load_saved_predictions,
    save_comparison_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_saved_predictions_align_with_shared_forecast_windows():
    predictions = load_saved_predictions(PROJECT_ROOT)

    assert set(predictions) == {"Seasonal naive", "Random Forest", "ANN", "LSTM"}
    assert all(len(model_predictions["validation"]) == 24 for model_predictions in predictions.values())
    assert all(len(model_predictions["test"]) == 36 for model_predictions in predictions.values())


def test_comparison_tables_rank_models_by_each_split():
    tables = build_comparison_tables(PROJECT_ROOT)

    assert set(tables) == {"metrics", "residuals", "rankings"}
    assert len(tables["metrics"]) == 8
    assert set(tables["metrics"]["split"]) == {"validation", "test"}
    assert tables["rankings"].groupby("split").size().to_dict() == {
        "validation": 4,
        "test": 4,
    }
    assert tables["rankings"].groupby("split")["rmse_rank"].min().eq(1).all()


def test_comparison_outputs_are_persisted():
    paths = save_comparison_outputs(PROJECT_ROOT)

    assert all(path.exists() for path in paths.values())
    assert pd.read_csv(paths["metrics"]).shape[0] == 8