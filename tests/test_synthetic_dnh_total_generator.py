"""Tests for Phase 3 synthetic DNH-total generator and validator."""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from src.data_generation.synthetic_dnh_total_generator import SyntheticDNHTotalGenerator
from src.validation.validate_dnh_total_dataset import validate_dnh_total_dataset


@pytest.fixture
def config():
    """Load default config."""
    config_path = Path("configs/synthetic_dnh_total_v2.json")
    if not config_path.exists():
        pytest.skip("Config file not found")
    return json.loads(config_path.read_text())


def test_config_contains_required_brief_keys():
    """Config must expose the brief’s required top-level keys for reproducibility and transparency."""
    config_path = Path("configs/synthetic_dnh_total_v2.json")
    config = json.loads(config_path.read_text())

    required = [
        "schema_version", "start_date", "periods", "seed",
        "area_id", "output_csv", "metadata_output"
    ]
    for key in required:
        assert key in config, f"Missing required brief key: {key}"

    assert config["schema_version"] == "2.0"
    assert config["area_id"] == "DNH_total"
    assert config["periods"] == 180


def test_generator_creates_correct_shape_and_schema(config):
    """Test that default config produces exactly 180 rows and 17 columns in correct order."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()

    # Shape
    assert len(df) == 180, f"Expected 180 rows, got {len(df)}"
    assert len(df.columns) == 17, f"Expected 17 columns, got {len(df.columns)}"

    # Schema and order
    expected_cols = [
        "date", "area_id", "rainfall_mm", "temp_max_c", "temp_min_c",
        "humidity_max_pct", "humidity_min_pct", "wind_speed_kmh",
        "solar_radiation_mj_m2", "sunshine_hours", "total_population",
        "urban_population", "total_households", "reservoir_level_m",
        "canal_discharge_cumecs", "groundwater_level_m_bgl",
        "residential_water_demand_m3"
    ]
    assert list(df.columns) == expected_cols


def test_generator_is_deterministic(config):
    """Test that same config and seed produce identical output."""
    gen1 = SyntheticDNHTotalGenerator(config)
    df1 = gen1.generate_all()

    gen2 = SyntheticDNHTotalGenerator(config)
    df2 = gen2.generate_all()

    pd.testing.assert_frame_equal(df1, df2, check_dtype=False)


def test_different_seed_produces_different_output(config):
    """Test that different seed produces different stochastic output."""
    import copy
    config1 = copy.deepcopy(config)
    config2 = copy.deepcopy(config)
    config2["seed"] = int(config["seed"]) + 1
    if "randomization" in config2 and "seed" in config2["randomization"]:
        config2["randomization"]["seed"] = config2["seed"]

    gen1 = SyntheticDNHTotalGenerator(config1)
    df1 = gen1.generate_all()

    gen2 = SyntheticDNHTotalGenerator(config2)
    df2 = gen2.generate_all()

    # Should differ in stochastic fields
    assert not df1["rainfall_mm"].equals(df2["rainfall_mm"])
    assert not df1["residential_water_demand_m3"].equals(df2["residential_water_demand_m3"])


def test_dates_are_consecutive_monthly(config):
    """Test that dates are consecutive monthly starting from 2010-01-01."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()

    dates = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    expected = pd.date_range(start="2010-01-01", periods=180, freq="MS")
    pd.testing.assert_index_equal(dates, expected, check_names=False)


def test_area_id_always_dnh_total(config):
    """Test that all rows have area_id = 'DNH_total'."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()

    assert (df["area_id"] == "DNH_total").all()


def test_population_is_positive_and_growing(config):
    """Test that population is positive and shows overall growth."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()

    assert (df["total_population"] > 0).all()
    assert df["total_population"].iloc[-1] > df["total_population"].iloc[0]


def test_urban_population_within_total(config):
    """Test that urban_population <= total_population for all rows."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()

    assert (df["urban_population"] >= 0).all()
    assert (df["urban_population"] <= df["total_population"]).all()


def test_households_positive_and_growing(config):
    """Test that households is positive and shows overall growth."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()

    assert (df["total_households"] > 0).all()
    assert df["total_households"].iloc[-1] > df["total_households"].iloc[0]


def test_temperature_order(config):
    """Test that temp_min <= temp_max for all rows."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()

    assert (df["temp_min_c"] <= df["temp_max_c"]).all()


def test_humidity_bounds_and_order(config):
    """Test humidity is in [0,100] and min <= max."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()

    assert (df["humidity_max_pct"] >= 0).all()
    assert (df["humidity_max_pct"] <= 100).all()
    assert (df["humidity_min_pct"] >= 0).all()
    assert (df["humidity_min_pct"] <= 100).all()
    assert (df["humidity_min_pct"] <= df["humidity_max_pct"]).all()


def test_rainfall_non_negative(config):
    """Test that rainfall is non-negative."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()

    assert (df["rainfall_mm"] >= 0).all()


def test_demand_positive(config):
    """Test that residential demand is strictly positive."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()

    assert (df["residential_water_demand_m3"] > 0).all()


def test_demand_has_variance(config):
    """Test that demand has non-zero variance."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()

    assert df["residential_water_demand_m3"].std() > 0


def test_monsoon_rainfall_dominance(config):
    """Test that monsoon mean rainfall exceeds dry-season mean."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()

    df_check = df.copy()
    df_check["month"] = pd.to_datetime(df_check["date"]).dt.month
    monsoon_months = set(config["weather"]["monsoon_months"])

    monsoon = df_check[df_check["month"].isin(monsoon_months)]
    dry = df_check[~df_check["month"].isin(monsoon_months)]

    monsoon_mean = monsoon["rainfall_mm"].mean()
    dry_mean = dry["rainfall_mm"].mean()

    ratio = monsoon_mean / dry_mean if dry_mean > 0 else 0
    required_ratio = config["validation"]["monsoon_to_dry_rainfall_ratio"]
    assert ratio >= required_ratio * 0.8, f"Monsoon/dry rainfall ratio {ratio:.2f} below expected {required_ratio:.2f}"


def test_monsoon_humidity_higher(config):
    """Test that monsoon humidity is higher than dry season."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()

    df_check = df.copy()
    df_check["month"] = pd.to_datetime(df_check["date"]).dt.month
    monsoon_months = set(config["weather"]["monsoon_months"])

    monsoon = df_check[df_check["month"].isin(monsoon_months)]
    dry = df_check[~df_check["month"].isin(monsoon_months)]

    monsoon_hum = monsoon["humidity_max_pct"].mean()
    dry_hum = dry["humidity_max_pct"].mean()

    assert monsoon_hum > dry_hum


def test_monsoon_sunshine_lower(config):
    """Test that monsoon sunshine is lower than dry season."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()

    df_check = df.copy()
    df_check["month"] = pd.to_datetime(df_check["date"]).dt.month
    monsoon_months = set(config["weather"]["monsoon_months"])

    monsoon = df_check[df_check["month"].isin(monsoon_months)]
    dry = df_check[~df_check["month"].isin(monsoon_months)]

    monsoon_sun = monsoon["sunshine_hours"].mean()
    dry_sun = dry["sunshine_hours"].mean()

    assert monsoon_sun < dry_sun


def test_reservoir_has_variance(config):
    """Test that reservoir level has non-zero variance."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()

    assert df["reservoir_level_m"].std() > 0


def test_groundwater_has_variance(config):
    """Test that groundwater level has non-zero variance."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()

    assert df["groundwater_level_m_bgl"].std() > 0


def test_canal_discharge_non_negative(config):
    """Test that canal discharge is non-negative."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()

    assert (df["canal_discharge_cumecs"] >= 0).all()


def test_no_nulls_or_infinite_values(config):
    """Test that there are no nulls or infinite values."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()

    numeric_cols = [
        "rainfall_mm", "temp_max_c", "temp_min_c", "humidity_max_pct",
        "humidity_min_pct", "wind_speed_kmh", "solar_radiation_mj_m2",
        "sunshine_hours", "total_population", "urban_population",
        "total_households", "reservoir_level_m", "canal_discharge_cumecs",
        "groundwater_level_m_bgl", "residential_water_demand_m3"
    ]

    for col in numeric_cols:
        assert df[col].isnull().sum() == 0, f"{col} contains nulls"
        assert np.all(np.isfinite(df[col])), f"{col} contains infinite values"


def test_validator_accepts_valid_generated_data(config):
    """Test that validator accepts data generated by the generator."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()

    result = validate_dnh_total_dataset(df, config)
    assert result["ok"] is True, f"Validation failed: {result['errors']}"


def test_validator_rejects_missing_columns():
    """Test that validator rejects data with missing columns."""
    df = pd.DataFrame({"date": ["2010-01-01"], "area_id": ["DNH_total"]})

    result = validate_dnh_total_dataset(df)
    assert result["ok"] is False
    assert len(result["errors"]) > 0


def test_validator_rejects_wrong_area_id(config):
    """Test that validator rejects rows with wrong area_id."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()
    df.loc[0, "area_id"] = "zone_01"

    result = validate_dnh_total_dataset(df, config)
    assert result["ok"] is False


def test_validator_rejects_negative_rainfall(config):
    """Test that validator rejects negative rainfall."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()
    df.loc[0, "rainfall_mm"] = -1

    result = validate_dnh_total_dataset(df, config)
    assert result["ok"] is False


def test_validator_rejects_negative_demand(config):
    """Test that validator rejects negative demand."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()
    df.loc[0, "residential_water_demand_m3"] = -1

    result = validate_dnh_total_dataset(df, config)
    assert result["ok"] is False


def test_validator_rejects_temp_min_greater_than_max(config):
    """Test that validator rejects temp_min > temp_max."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()
    df.loc[0, "temp_min_c"] = 30
    df.loc[0, "temp_max_c"] = 20

    result = validate_dnh_total_dataset(df, config)
    assert result["ok"] is False


def test_validator_rejects_urban_exceeding_total_population(config):
    """Test that validator rejects urban > total population."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()
    df.loc[0, "urban_population"] = df.loc[0, "total_population"] + 1000

    result = validate_dnh_total_dataset(df, config)
    assert result["ok"] is False


def test_validator_rejects_humidity_out_of_bounds(config):
    """Test that validator rejects humidity outside [0, 100]."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()
    df.loc[0, "humidity_max_pct"] = 101

    result = validate_dnh_total_dataset(df, config)
    assert result["ok"] is False


def test_config_validation_rejects_invalid_monsoon_months():
    """Test that generator rejects invalid monsoon month configuration."""
    bad_config = {
        "schema_version": "2.0",
        "start_date": "2010-01-01",
        "periods": 180,
        "seed": 42,
        "area_id": "DNH_total",
        "output_csv": "dummy.csv",
        "metadata_output": "dummy.json",
        "dates": {"start_date": "2010-01-01", "periods": 180},
        "randomization": {"seed": 42},
        "geography": {"area_id": "DNH_total"},
        "output": {"csv_path": "dummy.csv", "metadata_path": "dummy.json"},
        "demographics": {
            "baseline_total_population": 865000,
            "baseline_urban_population": 250000,
            "baseline_total_households": 175000,
            "annual_population_growth_rate": 0.025,
        },
        "weather": {"monsoon_months": [13]},
        "demand": {
            "baseline_demand_lpd": 120,
            "seasonality": {"monsoon_multiplier": 0.85, "dry_multiplier": 1.10},
            "rainfall_elasticity": -0.0005,
            "autoregressive_strength": 0.4,
            "noise_std": 0.06,
        },
        "system_state": {
            "reservoir": {"baseline_level_m": 35, "max_level_m": 50, "min_level_m": 5, "recharge_response_rate": 0.3, "discharge_response_rate": 0.25},
            "groundwater": {"baseline_level_m_bgl": 8, "max_level_m_bgl": 25, "min_level_m_bgl": 2, "recharge_response_rate": 0.2, "discharge_response_rate": 0.15},
            "canal_discharge": {"baseline_discharge_cumecs": 2.5, "max_discharge_cumecs": 8, "min_discharge_cumecs": 0.3, "seasonal_variation": {"dry_multiplier": 1.2, "monsoon_multiplier": 0.9}, "demand_elasticity": 0.15},
        },
    }
    with pytest.raises(ValueError, match="monsoon_months"):
        SyntheticDNHTotalGenerator(bad_config)


def test_config_validation_rejects_invalid_bounds():
    """Test that generator rejects invalid reservoir/groundwater bounds."""
    bad_config = json.loads(json.dumps({
        "schema_version": "2.0",
        "start_date": "2010-01-01",
        "periods": 180,
        "seed": 42,
        "area_id": "DNH_total",
        "output_csv": "dummy.csv",
        "metadata_output": "dummy.json",
        **json.loads(Path("configs/synthetic_dnh_total_v2.json").read_text())
    }))
    bad_config["system_state"]["reservoir"]["min_level_m"] = 60
    bad_config["system_state"]["reservoir"]["max_level_m"] = 50
    with pytest.raises(ValueError, match="Reservoir"):
        SyntheticDNHTotalGenerator(bad_config)


def test_validator_rejects_duplicate_or_gapped_dates(config):
    """Test that validator rejects duplicate or gapped dates."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()
    df = pd.concat([df.iloc[:1], df], ignore_index=True)
    df.loc[0, "date"] = df.loc[1, "date"]
    result = validate_dnh_total_dataset(df, config)
    assert result["ok"] is False

    df2 = gen.generate_all().copy()
    df2 = df2.iloc[:-1].copy()
    df2.loc[len(df2), :] = df2.iloc[-1].copy()
    df2.loc[len(df2) - 1, "date"] = "2010-02-01"
    result2 = validate_dnh_total_dataset(df2, config)
    assert result2["ok"] is False


def test_validator_rejects_state_bound_violations(config):
    """Test that validator rejects out-of-range reservoir or groundwater state values."""
    gen = SyntheticDNHTotalGenerator(config)
    df = gen.generate_all()
    df.loc[0, "reservoir_level_m"] = -1
    df.loc[0, "groundwater_level_m_bgl"] = 100
    result = validate_dnh_total_dataset(df, config)
    assert result["ok"] is False


def test_generator_supports_configurable_periods(config):
    """Test that a custom periods value is respected end-to-end."""
    custom = json.loads(json.dumps(config))
    custom["start_date"] = "2020-01-01"
    custom["periods"] = 12
    custom["dates"]["start_date"] = "2020-01-01"
    custom["dates"]["periods"] = 12
    custom["seed"] = 7
    custom["randomization"]["seed"] = 7
    gen = SyntheticDNHTotalGenerator(custom)
    df = gen.generate_all()
    assert len(df) == 12
    assert pd.to_datetime(df["date"]).min() == pd.Timestamp("2020-01-01")
    assert pd.to_datetime(df["date"]).max() == pd.Timestamp("2020-12-01")


def test_state_variables_saturate_at_bounds(config):
    """Test that state variables remain clipped within configured bounds."""
    custom = json.loads(json.dumps(config))
    custom["system_state"]["reservoir"]["min_level_m"] = 10
    custom["system_state"]["reservoir"]["max_level_m"] = 12
    custom["system_state"]["groundwater"]["min_level_m_bgl"] = 4
    custom["system_state"]["groundwater"]["max_level_m_bgl"] = 5
    gen = SyntheticDNHTotalGenerator(custom)
    df = gen.generate_all()
    assert df["reservoir_level_m"].between(10, 12).all()
    assert df["groundwater_level_m_bgl"].between(4, 5).all()


def test_write_output_creates_files(config, tmp_path):
    """Test that write_output creates CSV and metadata files."""
    gen = SyntheticDNHTotalGenerator(config)
    gen.csv_path = tmp_path / "data.csv"
    gen.meta_path = tmp_path / "metadata.json"

    df = gen.generate_all()
    validation = validate_dnh_total_dataset(df, config)
    gen.write_output(df, validation_result=validation)

    assert gen.csv_path.exists()
    assert gen.meta_path.exists()

    # Verify CSV content
    df_read = pd.read_csv(gen.csv_path)
    assert len(df_read) == 180

    # Verify metadata
    meta = json.loads(gen.meta_path.read_text())
    assert meta["row_count"] == 180
    assert meta["synthetic"] is True
    assert "validation_summary" in meta
    assert "summary_statistics" in meta
    assert meta["validation_summary"]["ok"] is True
