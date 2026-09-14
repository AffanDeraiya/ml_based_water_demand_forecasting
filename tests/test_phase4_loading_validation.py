import json
from pathlib import Path

import pandas as pd
import pytest

from src.features.phase4_data_loader import (
    canonical_dataset_path,
    get_configured_dataset_paths,
    load_dnh_total_dataset,
    validate_dnh_total_dataframe,
)


@pytest.fixture
def config_path():
    return Path("configs/synthetic_dnh_total_v2.json")


def test_canonical_dataset_path_matches_config(config_path):
    config = json.loads(config_path.read_text())
    expected = Path(config["output_csv"])
    assert canonical_dataset_path(config_path) == expected
    assert expected.exists()


def test_get_configured_dataset_paths_returns_expected_values(config_path):
    paths = get_configured_dataset_paths(config_path)

    assert paths["dataset_path"] == Path("data/synthetic/raw/synthetic_dnh_total_monthly_v2.csv")
    assert paths["metadata_path"] == Path("data/synthetic/metadata/synthetic_dnh_total_monthly_v2_metadata.json")
    assert paths["dataset_path"].exists()
    assert paths["metadata_path"].exists()


def test_load_dnh_total_dataset_returns_validated_dataframe(config_path):
    df = load_dnh_total_dataset(config_path=config_path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 180
    assert df["date"].is_monotonic_increasing
    assert df["area_id"].eq("DNH_total").all()
    assert df["date"].nunique() == len(df)
    assert df["date"].dtype.kind in {"M", "O"}


def test_validate_dnh_total_dataframe_rejects_invalid_input():
    bad = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-02-01"],
            "area_id": ["DNH_total", "DNH_total"],
            "rainfall_mm": [10.0, None],
            "temp_max_c": [28.0, 30.0],
            "temp_min_c": [20.0, 25.0],
            "humidity_max_pct": [80.0, 85.0],
            "humidity_min_pct": [60.0, 65.0],
            "wind_speed_kmh": [5.0, 6.0],
            "solar_radiation_mj_m2": [120.0, 130.0],
            "sunshine_hours": [5.0, 4.0],
            "total_population": [1000, 1100],
            "urban_population": [300, 350],
            "total_households": [400, 450],
            "reservoir_level_m": [10.0, 12.0],
            "canal_discharge_cumecs": [1.0, 1.5],
            "groundwater_level_m_bgl": [8.0, 7.0],
            "residential_water_demand_m3": [50.0, 60.0],
        }
    )

    with pytest.raises(ValueError, match="null|non-finite|Missing|required"):
        validate_dnh_total_dataframe(bad)
