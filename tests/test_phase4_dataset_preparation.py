import json
from pathlib import Path

import pandas as pd

from src.features.phase4_data_loader import load_dnh_total_dataset
from src.features.phase4_dataset_preparation import (
    create_chronological_splits,
    describe_chronological_splits,
    prepare_shared_feature_matrix,
    save_prepared_artifacts,
)


CONFIG_PATH = Path("configs/synthetic_dnh_total_v2.json")


def test_shared_matrix_keeps_target_separate_and_is_complete():
    data = load_dnh_total_dataset(CONFIG_PATH)
    features, target = prepare_shared_feature_matrix(data)

    assert features.shape == (168, 28)
    assert target.name == "residential_water_demand_m3"
    assert "residential_water_demand_m3" not in features.columns
    assert features.index.equals(target.index)
    assert features.index.min() == pd.Timestamp("2011-01-01")
    assert features.index.max() == pd.Timestamp("2024-12-01")
    assert features.notna().all().all()
    assert target.notna().all()


def test_chronological_splits_are_contiguous_and_match_specification():
    data = load_dnh_total_dataset(CONFIG_PATH)
    features, target = prepare_shared_feature_matrix(data)
    splits = create_chronological_splits(features, target)
    summary = describe_chronological_splits(splits)

    assert summary["rows"].tolist() == [108, 24, 36]
    assert summary["start_date"].tolist() == ["2011-01-01", "2020-01-01", "2022-01-01"]
    assert summary["end_date"].tolist() == ["2019-12-01", "2021-12-01", "2024-12-01"]
    assert splits["train"][0].index[-1] + pd.offsets.MonthBegin(1) == splits["validation"][0].index[0]
    assert splits["validation"][0].index[-1] + pd.offsets.MonthBegin(1) == splits["test"][0].index[0]
    assert set(splits["train"][0].index).isdisjoint(splits["validation"][0].index)
    assert set(splits["validation"][0].index).isdisjoint(splits["test"][0].index)


def test_prepared_artifacts_are_reproducible(tmp_path):
    data = load_dnh_total_dataset(CONFIG_PATH)
    features, target = prepare_shared_feature_matrix(data)
    splits = create_chronological_splits(features, target)
    paths = save_prepared_artifacts(features, target, splits, output_dir=tmp_path)

    assert all(path.exists() for path in paths.values())
    saved_features = pd.read_csv(paths["features"], index_col="date", parse_dates=True)
    saved_target = pd.read_csv(paths["target"], index_col="date", parse_dates=True)[target.name]
    metadata = json.loads(paths["metadata"].read_text())

    pd.testing.assert_frame_equal(saved_features, features, check_dtype=False, check_freq=False)
    pd.testing.assert_series_equal(saved_target, target, check_dtype=False, check_names=True, check_freq=False)
    assert metadata["feature_count"] == 28
    assert metadata["complete_rows"] == 168
    assert len(metadata["split_row_ids"]["train"]) == 108
    assert len(metadata["split_row_ids"]["validation"]) == 24
    assert len(metadata["split_row_ids"]["test"]) == 36
    assert "final 36 complete months" in metadata["split_rationale"]
    assert metadata["random_shuffle"] is False
    assert metadata["future_value_fill"] is False
    assert paths["train_row_ids"].exists()
