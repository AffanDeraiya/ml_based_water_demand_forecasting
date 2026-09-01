"""Validator for Version 2 synthetic DNH-total dataset.

Implements structural, domain, and behavioral checks per PHASE_3_SYNTHETIC_DATA_GENERATION.md.
Returns a structured result with 'ok' boolean and 'errors' list.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


def validate_dnh_total_dataset(df: pd.DataFrame, config: Optional[Dict] = None) -> Dict:
    """
    Comprehensive validation of DNH-total dataset.

    Returns: {"ok": bool, "errors": [str...]}
    """
    errors: List[str] = []

    # ===== STRUCTURAL CHECKS =====
    required_cols = [
        "date", "area_id", "rainfall_mm", "temp_max_c", "temp_min_c",
        "humidity_max_pct", "humidity_min_pct", "wind_speed_kmh",
        "solar_radiation_mj_m2", "sunshine_hours", "total_population",
        "urban_population", "total_households", "reservoir_level_m",
        "canal_discharge_cumecs", "groundwater_level_m_bgl",
        "residential_water_demand_m3"
    ]

    # Check all required columns exist
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        return {"ok": False, "errors": errors}

    # Check column order
    actual_cols = list(df.columns)
    if actual_cols != required_cols:
        errors.append(f"Column order mismatch. Expected: {required_cols}")

    # Check row count (should be 180 for 15 years)
    if len(df) != 180:
        errors.append(f"Expected 180 rows (15 years), got {len(df)}")

    # Check area_id
    if not (df["area_id"] == "DNH_total").all():
        errors.append("All area_id values must be 'DNH_total'")

    # Check dates: parse, unique, ascending, consecutive months
    try:
        dates = pd.to_datetime(df["date"])
        df_check = df.copy()
        df_check["date_parsed"] = dates

        # Unique
        if dates.duplicated().any():
            errors.append("Dates must be unique")

        # Ascending and consecutive months
        if not dates.is_monotonic_increasing:
            errors.append("Dates must be in ascending order")

        sorted_dates = sorted(dates)
        expected_dates = pd.date_range(start=sorted_dates[0], periods=len(sorted_dates), freq="MS")
        if not (sorted_dates == expected_dates).all():
            errors.append("Dates must be consecutive monthly (no gaps)")

    except Exception as e:
        errors.append(f"Date parsing failed: {e}")
        return {"ok": False, "errors": errors}

    # Check for nulls and non-finite values
    numeric_cols = [
        "rainfall_mm", "temp_max_c", "temp_min_c", "humidity_max_pct",
        "humidity_min_pct", "wind_speed_kmh", "solar_radiation_mj_m2",
        "sunshine_hours", "total_population", "urban_population",
        "total_households", "reservoir_level_m", "canal_discharge_cumecs",
        "groundwater_level_m_bgl", "residential_water_demand_m3"
    ]

    for col in numeric_cols:
        if df[col].isnull().any():
            errors.append(f"Column {col} contains nulls")
        if not np.all(np.isfinite(df[col])):
            errors.append(f"Column {col} contains non-finite values (inf or nan)")

    # ===== DOMAIN CHECKS =====

    # Non-negativity checks
    non_negative_cols = [
        "rainfall_mm", "wind_speed_kmh", "solar_radiation_mj_m2",
        "sunshine_hours", "canal_discharge_cumecs", "residential_water_demand_m3"
    ]
    for col in non_negative_cols:
        if (df[col] < 0).any():
            errors.append(f"Column {col} must be non-negative")

    # Humidity bounds and order
    if (df["humidity_max_pct"] < 0).any() or (df["humidity_max_pct"] > 100).any():
        errors.append("humidity_max_pct must be in [0, 100]")
    if (df["humidity_min_pct"] < 0).any() or (df["humidity_min_pct"] > 100).any():
        errors.append("humidity_min_pct must be in [0, 100]")
    if (df["humidity_min_pct"] > df["humidity_max_pct"]).any():
        errors.append("humidity_min_pct must be <= humidity_max_pct for all rows")

    # Temperature order
    if (df["temp_min_c"] > df["temp_max_c"]).any():
        errors.append("temp_min_c must be <= temp_max_c for all rows")

    # Population relationships
    if (df["total_population"] <= 0).any():
        errors.append("total_population must be positive")
    if (df["total_households"] <= 0).any():
        errors.append("total_households must be positive")
    if (df["urban_population"] < 0).any():
        errors.append("urban_population must be non-negative")
    if (df["urban_population"] > df["total_population"]).any():
        errors.append("urban_population must be <= total_population")

    # Demand must be positive
    if (df["residential_water_demand_m3"] <= 0).any():
        errors.append("residential_water_demand_m3 must be strictly positive")

    # ===== BEHAVIORAL CHECKS =====

    # Monsoon rainfall dominance (if config provided)
    if config is not None:
        try:
            monsoon_months = set(config["weather"]["monsoon_months"])
            df_check = df.copy()
            df_check["month"] = pd.to_datetime(df_check["date"]).dt.month

            monsoon_data = df_check[df_check["month"].isin(monsoon_months)]
            dry_data = df_check[~df_check["month"].isin(monsoon_months)]

            monsoon_rain_mean = monsoon_data["rainfall_mm"].mean()
            dry_rain_mean = dry_data["rainfall_mm"].mean()

            if dry_rain_mean > 0:
                ratio = monsoon_rain_mean / dry_rain_mean
                required_ratio = config["validation"]["monsoon_to_dry_rainfall_ratio"]
                if ratio < required_ratio * 0.8:  # Allow 20% tolerance
                    errors.append(
                        f"Monsoon/dry rainfall ratio {ratio:.2f} is below expected {required_ratio:.2f}"
                    )

            # Monsoon humidity dominance
            monsoon_humidity_mean = monsoon_data["humidity_max_pct"].mean()
            dry_humidity_mean = dry_data["humidity_max_pct"].mean()
            if dry_humidity_mean > 0:
                humidity_ratio = monsoon_humidity_mean / dry_humidity_mean
                expected_ratio = config["validation"]["monsoon_to_dry_humidity_ratio"]
                if humidity_ratio < expected_ratio * 0.85:
                    errors.append(
                        f"Monsoon/dry humidity ratio {humidity_ratio:.2f} is below expected {expected_ratio:.2f}"
                    )

            # Monsoon sunshine reduction (should be lower in monsoon)
            monsoon_sunshine_mean = monsoon_data["sunshine_hours"].mean()
            dry_sunshine_mean = dry_data["sunshine_hours"].mean()
            if dry_sunshine_mean > 0:
                sunshine_ratio = monsoon_sunshine_mean / dry_sunshine_mean
                expected_ratio = config["validation"]["monsoon_to_dry_sunshine_ratio"]
                if sunshine_ratio > expected_ratio * 1.15:  # Allow 15% tolerance
                    errors.append(
                        f"Monsoon sunshine ratio {sunshine_ratio:.2f} is above expected {expected_ratio:.2f}"
                    )

        except Exception as e:
            errors.append(f"Monsoon validation check failed: {e}")

    # Population trend (should generally increase)
    pop_change = df["total_population"].iloc[-1] - df["total_population"].iloc[0]
    if pop_change <= 0:
        errors.append("Population should show positive overall trend over 15 years")

    # Household trend
    hh_change = df["total_households"].iloc[-1] - df["total_households"].iloc[0]
    if hh_change <= 0:
        errors.append("Households should show positive overall trend over 15 years")

    # Population growth reasonableness (month-to-month)
    if config is not None:
        try:
            max_tolerance = config["validation"]["population_growth_tolerance"]
            pop_pct_change = df["total_population"].pct_change().abs()
            if (pop_pct_change > max_tolerance).any():
                errors.append(
                    f"Population has month-to-month changes exceeding {max_tolerance*100:.1f}%"
                )
        except Exception:
            pass

    # Demand variance (should be non-trivial)
    if config is not None:
        try:
            min_variance = config["validation"]["demand_min_variance"]
            demand_var = df["residential_water_demand_m3"].var()
            if demand_var < min_variance:
                errors.append(
                    f"Demand variance {demand_var:.0f} is below minimum {min_variance:.0f}"
                )
        except Exception:
            pass

    # Demand should have non-zero variance
    if df["residential_water_demand_m3"].std() == 0:
        errors.append("Demand must have non-zero variance")

    # Reservoir and groundwater should not be constant
    if df["reservoir_level_m"].std() == 0:
        errors.append("Reservoir level must have non-zero variance")
    if df["groundwater_level_m_bgl"].std() == 0:
        errors.append("Groundwater level must have non-zero variance")

    # Respect configured state bounds when available.
    if config is not None:
        try:
            reservoir_cfg = config["system_state"]["reservoir"]
            gw_cfg = config["system_state"]["groundwater"]
            reservoir_min = float(reservoir_cfg["min_level_m"])
            reservoir_max = float(reservoir_cfg["max_level_m"])
            gw_min = float(gw_cfg["min_level_m_bgl"])
            gw_max = float(gw_cfg["max_level_m_bgl"])
            if (df["reservoir_level_m"] < reservoir_min).any() or (df["reservoir_level_m"] > reservoir_max).any():
                errors.append(f"Reservoir levels exceed configured bounds [{reservoir_min}, {reservoir_max}]")
            if (df["groundwater_level_m_bgl"] < gw_min).any() or (df["groundwater_level_m_bgl"] > gw_max).any():
                errors.append(f"Groundwater levels exceed configured bounds [{gw_min}, {gw_max}]")
        except Exception:
            pass

    # Canal discharge seasonal variation
    if config is not None:
        try:
            monsoon_months = set(config["weather"]["monsoon_months"])
            df_check = df.copy()
            df_check["month"] = pd.to_datetime(df_check["date"]).dt.month

            monsoon_canal = df_check[df_check["month"].isin(monsoon_months)]["canal_discharge_cumecs"].mean()
            dry_canal = df_check[~df_check["month"].isin(monsoon_months)]["canal_discharge_cumecs"].mean()

            if monsoon_canal > 0 and dry_canal > 0:
                if monsoon_canal == dry_canal:
                    errors.append("Canal discharge must have seasonal variation")

        except Exception:
            pass

    # No forbidden columns
    forbidden = [
        "zone", "village", "ward", "pateload", "hour", "agriculture",
        "water_consumption_m3", "area_water_usage_m3", "ph", "tds",
        "net_annual_groundwater_availability_mcm",
        "annual_groundwater_draft_mcm"
    ]
    for col in df.columns:
        if col.lower() in forbidden:
            errors.append(f"Forbidden column found: {col}")

    ok = len(errors) == 0
    return {"ok": ok, "errors": errors}
