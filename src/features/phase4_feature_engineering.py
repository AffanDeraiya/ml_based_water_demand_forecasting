"""Leakage-safe baseline features for monthly DNH_total forecasting."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


TARGET_COLUMN = "residential_water_demand_m3"
TARGET_LAGS = (1, 2, 3, 6, 12)
ROLLING_WINDOWS = (3, 6)
LAGGED_PREDICTOR_COLUMNS = (
    "rainfall_mm",
    "temp_max_c",
    "temp_min_c",
    "humidity_max_pct",
    "humidity_min_pct",
    "wind_speed_kmh",
    "solar_radiation_mj_m2",
    "sunshine_hours",
    "reservoir_level_m",
    "canal_discharge_cumecs",
    "groundwater_level_m_bgl",
    "total_population",
    "urban_population",
    "total_households",
)


def _validate_input(data: pd.DataFrame) -> None:
    """Validate the minimum input contract required by the feature builder."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data must use a DatetimeIndex")
    if not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise ValueError("data index must be unique and chronologically ordered")
    required = {TARGET_COLUMN, *LAGGED_PREDICTOR_COLUMNS}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required feature input columns: {missing}")


def feature_availability() -> Dict[str, str]:
    """Return the documented availability policy for each feature family."""
    return {
        "calendar": "target month known at forecast origin",
        "target_lags": "prior observed demand only",
        "target_rolling": "prior observed demand only; current month excluded",
        "lagged_predictors": "one-month lag of observed weather, system, and demographic values",
    }


def build_leakage_safe_features(
    data: pd.DataFrame,
    *,
    drop_incomplete: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build baseline predictors using only information available before each target month.

    Parameters
    ----------
    data:
        Validated monthly data indexed by month start.
    drop_incomplete:
        If True, remove rows that lack the 12-month warm-up history. If False,
        retain the full index and expose leading NaN values explicitly.

    Returns
    -------
    features, target:
        A predictor dataframe and the separate target series with aligned indexes.
    """
    _validate_input(data)

    features = pd.DataFrame(index=data.index.copy())
    month_number = data.index.month
    features["month"] = month_number
    features["quarter"] = data.index.quarter
    features["year"] = data.index.year
    features["month_sin"] = np.sin(2 * np.pi * month_number / 12.0)
    features["month_cos"] = np.cos(2 * np.pi * month_number / 12.0)

    for lag in TARGET_LAGS:
        features[f"demand_lag_{lag}"] = data[TARGET_COLUMN].shift(lag)

    prior_demand = data[TARGET_COLUMN].shift(1)
    for window in ROLLING_WINDOWS:
        features[f"demand_rolling_mean_{window}"] = prior_demand.rolling(
            window=window,
            min_periods=window,
        ).mean()
        features[f"demand_rolling_std_{window}"] = prior_demand.rolling(
            window=window,
            min_periods=window,
        ).std()

    for column in LAGGED_PREDICTOR_COLUMNS:
        features[f"{column}_lag_1"] = data[column].shift(1)

    target = data[TARGET_COLUMN].rename(TARGET_COLUMN).copy()
    if drop_incomplete:
        complete_rows = features.notna().all(axis=1)
        features = features.loc[complete_rows]
        target = target.loc[complete_rows]

    return features, target
