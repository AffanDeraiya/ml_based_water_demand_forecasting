"""Canonical data-loading and validation utilities for Phase 4.

This module provides the single reusable path for reading the approved
DNH_total monthly dataset and validating it before feature engineering begins.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "date",
    "area_id",
    "rainfall_mm",
    "temp_max_c",
    "temp_min_c",
    "humidity_max_pct",
    "humidity_min_pct",
    "wind_speed_kmh",
    "solar_radiation_mj_m2",
    "sunshine_hours",
    "total_population",
    "urban_population",
    "total_households",
    "reservoir_level_m",
    "canal_discharge_cumecs",
    "groundwater_level_m_bgl",
    "residential_water_demand_m3",
]

NUMERIC_COLUMNS = [
    "rainfall_mm",
    "temp_max_c",
    "temp_min_c",
    "humidity_max_pct",
    "humidity_min_pct",
    "wind_speed_kmh",
    "solar_radiation_mj_m2",
    "sunshine_hours",
    "total_population",
    "urban_population",
    "total_households",
    "reservoir_level_m",
    "canal_discharge_cumecs",
    "groundwater_level_m_bgl",
    "residential_water_demand_m3",
]

DEFAULT_CONFIG_PATH = Path("configs/synthetic_dnh_total_v2.json")


def canonical_dataset_path(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Path:
    """Return the canonical dataset path recorded in the config file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    config = json.loads(config_file.read_text())
    dataset_path = Path(config["output_csv"])
    return dataset_path


def get_configured_dataset_paths(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Dict[str, Path]:
    """Return configured dataset and metadata paths defined for the project."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    config = json.loads(config_file.read_text())
    dataset_path = Path(config["output_csv"])
    metadata_path = Path(config["metadata_output"])
    return {
        "dataset_path": dataset_path,
        "metadata_path": metadata_path,
    }


def validate_dnh_total_dataframe(df: pd.DataFrame, *, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Validate a DNH_total DataFrame and return a cleaned, standardized copy.

    Raises ValueError with a descriptive message if any required validation fails.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")

    if data["date"].duplicated().any():
        raise ValueError("Column 'date' contains duplicate values.")
    if not data["date"].is_monotonic_increasing:
        raise ValueError("Column 'date' must be sorted chronologically.")
    data = data.reset_index(drop=True)

    if not (data["area_id"] == "DNH_total").all():
        raise ValueError("All rows must have area_id == 'DNH_total'.")

    if data.empty:
        raise ValueError("Dataset cannot be empty.")

    expected_months = pd.date_range(start=data["date"].min(), end=data["date"].max(), freq="MS")
    actual_months = pd.DatetimeIndex(data["date"])
    if len(actual_months) != len(expected_months) or not actual_months.equals(expected_months):
        raise ValueError("Dates must be a complete monthly series without gaps or duplicates.")

    for col in NUMERIC_COLUMNS:
        if data[col].isnull().any():
            raise ValueError(f"Column '{col}' contains null values.")
        if not np.isfinite(data[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Column '{col}' contains non-finite values.")

    if (data["rainfall_mm"] < 0).any():
        raise ValueError("rainfall_mm must be non-negative.")
    if (data["sunshine_hours"] < 0).any():
        raise ValueError("sunshine_hours must be non-negative.")
    if (data["wind_speed_kmh"] < 0).any():
        raise ValueError("wind_speed_kmh must be non-negative.")
    if (data["solar_radiation_mj_m2"] < 0).any():
        raise ValueError("solar_radiation_mj_m2 must be non-negative.")
    if (data["canal_discharge_cumecs"] < 0).any():
        raise ValueError("canal_discharge_cumecs must be non-negative.")
    if (data["residential_water_demand_m3"] <= 0).any():
        raise ValueError("residential_water_demand_m3 must be strictly positive.")

    if ((data["humidity_max_pct"] < 0) | (data["humidity_max_pct"] > 100)).any():
        raise ValueError("humidity_max_pct must be within [0, 100].")
    if ((data["humidity_min_pct"] < 0) | (data["humidity_min_pct"] > 100)).any():
        raise ValueError("humidity_min_pct must be within [0, 100].")
    if (data["humidity_min_pct"] > data["humidity_max_pct"]).any():
        raise ValueError("humidity_min_pct must be <= humidity_max_pct for all rows.")
    if (data["temp_min_c"] > data["temp_max_c"]).any():
        raise ValueError("temp_min_c must be <= temp_max_c for all rows.")

    if (data["total_population"] <= 0).any():
        raise ValueError("total_population must be positive.")
    if (data["total_households"] <= 0).any():
        raise ValueError("total_households must be positive.")
    if (data["urban_population"] < 0).any():
        raise ValueError("urban_population must be non-negative.")
    if (data["urban_population"] > data["total_population"]).any():
        raise ValueError("urban_population must be <= total_population.")

    if config is not None:
        for key in ["reservoir", "groundwater"]:
            if key not in config.get("system_state", {}):
                continue
            if key == "reservoir":
                cfg = config["system_state"]["reservoir"]
                low = float(cfg.get("min_level_m", -np.inf))
                high = float(cfg.get("max_level_m", np.inf))
                if ((data["reservoir_level_m"] < low) | (data["reservoir_level_m"] > high)).any():
                    raise ValueError("reservoir_level_m exceeds configured bounds.")
            if key == "groundwater":
                cfg = config["system_state"]["groundwater"]
                low = float(cfg.get("min_level_m_bgl", -np.inf))
                high = float(cfg.get("max_level_m_bgl", np.inf))
                if ((data["groundwater_level_m_bgl"] < low) | (data["groundwater_level_m_bgl"] > high)).any():
                    raise ValueError("groundwater_level_m_bgl exceeds configured bounds.")

    data = data.set_index("date", drop=False)
    data.index = pd.DatetimeIndex(data.index).to_period("M").to_timestamp()
    data.index.name = "date"
    return data


def load_dnh_total_dataset(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Load the canonical DNH_total dataset and validate it using the shared contract."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    if config is None:
        config = json.loads(config_file.read_text())

    dataset_path = Path(config["output_csv"])
    if not dataset_path.is_absolute():
        dataset_path = config_file.parent.parent / dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    return validate_dnh_total_dataframe(df, config=config)
