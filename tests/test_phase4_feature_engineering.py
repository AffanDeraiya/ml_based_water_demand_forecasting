import json
from pathlib import Path

import pandas as pd

from src.features.phase4_data_loader import load_dnh_total_dataset
from src.features.phase4_feature_engineering import (
    LAGGED_PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    build_leakage_safe_features,
    feature_availability,
)


CONFIG_PATH = Path("configs/synthetic_dnh_total_v2.json")


def test_feature_plan_has_fixed_baseline_groups():
    data = load_dnh_total_dataset(CONFIG_PATH)
    features, target = build_leakage_safe_features(data)

    assert len(features) == len(data) == len(target)
    assert len(features.columns) == 28
    assert features.columns[:5].tolist() == [
        "month", "quarter", "year", "month_sin", "month_cos"
    ]
    assert target.name == TARGET_COLUMN
    assert set(f"{column}_lag_1" for column in LAGGED_PREDICTOR_COLUMNS).issubset(features.columns)
    assert feature_availability()["target_rolling"].startswith("prior observed demand")


def test_lags_and_rollings_use_only_prior_months():
    data = load_dnh_total_dataset(CONFIG_PATH)
    features, _ = build_leakage_safe_features(data)

    row_number = 24
    assert features.iloc[row_number]["demand_lag_1"] == data[TARGET_COLUMN].iloc[row_number - 1]
    assert features.iloc[row_number]["demand_lag_12"] == data[TARGET_COLUMN].iloc[row_number - 12]
    expected_mean = data[TARGET_COLUMN].iloc[row_number - 3:row_number].mean()
    assert features.iloc[row_number]["demand_rolling_mean_3"] == expected_mean
    assert features.iloc[0]["demand_lag_1"] != features.iloc[0]["demand_lag_1"]


def test_future_observation_changes_do_not_change_earlier_features():
    data = load_dnh_total_dataset(CONFIG_PATH)
    baseline_features, _ = build_leakage_safe_features(data)
    changed = data.copy()
    changed.loc[changed.index[60], TARGET_COLUMN] *= 10
    changed.loc[changed.index[60], "rainfall_mm"] *= 10
    changed_features, _ = build_leakage_safe_features(changed)

    pd.testing.assert_frame_equal(
        baseline_features.iloc[:61],
        changed_features.iloc[:61],
        check_dtype=False,
    )
    assert baseline_features.iloc[61]["demand_lag_1"] != changed_features.iloc[61]["demand_lag_1"]
    assert baseline_features.iloc[61]["rainfall_mm_lag_1"] != changed_features.iloc[61]["rainfall_mm_lag_1"]


def test_drop_incomplete_removes_only_warmup_rows():
    data = load_dnh_total_dataset(CONFIG_PATH)
    features, target = build_leakage_safe_features(data, drop_incomplete=True)

    assert len(features) == len(target) == len(data) - 12
    assert features.index.min() == data.index[12]
    assert features.notna().all().all()
    assert target.notna().all()
